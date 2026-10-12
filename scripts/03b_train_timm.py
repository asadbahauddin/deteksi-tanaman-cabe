"""Tahap 3b: training model timm generik (AdamW + cosine + label smoothing + AMP).

Contoh:
  python scripts/03b_train_timm.py --model mobilenetv3_large_100 --aug A
  python scripts/03b_train_timm.py --model efficientnet_b0 --aug B
Checkpoint terbaik dipilih berdasarkan val macro-F1; early stopping juga pada val macro-F1.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from common import ROOT, seed_everything
from data_utils import CsvDataset, build_eval_tf, build_train_tf, classes_from_csv
from notes_log import append_experiment


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    import random
    random.seed(s)


def make_model(name: str, tag: str, num_classes: int, drop_rate: float = 0.0, pretrained: bool = True):
    return timm.create_model(tag if pretrained else name, pretrained=pretrained,
                             num_classes=num_classes, drop_rate=drop_rate)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluasi dalam fp32 (AMP fp16 sempat menghasilkan logit overflow -> val loss NaN di epoch awal)."""
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits.append(model(x).float().cpu())
        ys.append(y)
    logits, ys = torch.cat(logits), torch.cat(ys)
    loss = torch.nn.functional.cross_entropy(logits, ys).item()
    pred = logits.argmax(1).numpy()
    return {"loss": loss, "acc": accuracy_score(ys, pred), "macro_f1": f1_score(ys, pred, average="macro")}, \
        logits.softmax(1).numpy(), ys.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/timm.yaml")
    ap.add_argument("--model", required=True, help="nama arsitektur timm, mis. mobilenetv3_large_100")
    ap.add_argument("--aug", choices=["A", "B"], default="A")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--name", type=str, help="nama folder run (default <model>_aug<A|B>)")
    ap.add_argument("--max-steps", type=int, default=0, help="batasi step/epoch (uji cepat)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for k in ("epochs", "batch", "lr", "workers"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg["model"], cfg["aug_name"] = args.model, args.aug
    cfg["pretrained_tag"] = cfg["pretrained_tags"].get(args.model, args.model)
    cfg["aug"] = cfg["aug"][args.aug]
    del cfg["pretrained_tags"]
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["amp"] and device.type == "cuda"

    run_dir = ROOT / cfg["runs_dir"] / (args.name or f"{args.model}_aug{args.aug}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    sp = ROOT / cfg["splits_dir"]
    classes = classes_from_csv(sp / "train.csv")
    model = make_model(args.model, cfg["pretrained_tag"], len(classes), cfg["drop_rate"])
    dcfg = timm.data.resolve_data_config({}, model=model)
    mean, std, size = dcfg["mean"], dcfg["std"], cfg["img_size"]
    cfg.update(mean=list(mean), std=list(std), classes=classes)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    g = torch.Generator().manual_seed(cfg["seed"])
    dl_kw = dict(num_workers=cfg["workers"], pin_memory=True, persistent_workers=cfg["workers"] > 0,
                 worker_init_fn=seed_worker)
    train_dl = DataLoader(CsvDataset(sp / "train.csv", classes, build_train_tf(cfg["aug"], size, mean, std)),
                          batch_size=cfg["batch"], shuffle=True, drop_last=True, generator=g, **dl_kw)
    val_ds = CsvDataset(sp / "val.csv", classes, build_eval_tf(size, mean, std))
    # worker val tidak persistent: 4 worker hidup bersamaan (~1.9 GB commit tiap worker) memicu WinError 1455
    val_dl = DataLoader(val_ds, batch_size=cfg["batch"] * 2, shuffle=False, **{**dl_kw, "persistent_workers": False})

    model = model.to(device)  # channels_last ~15x lebih lambat di RTX 2050 + cuDNN 9.24 (diukur)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = min(len(train_dl), args.max_steps or len(train_dl))
    total, warm = cfg["epochs"] * steps_per_epoch, cfg["warmup_epochs"] * steps_per_epoch
    min_ratio = cfg["min_lr"] / cfg["lr"]

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        t = (step - warm) / max(1, total - warm)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    crit = torch.nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    hist, best_f1, best_ep, bad = [], -1.0, 0, 0
    t_start = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_dl):
            if args.max_steps and step >= args.max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        train_s = time.time() - t0
        vm, probs, ys = evaluate(model, val_dl, device)
        row = {"epoch": ep, "train_loss": run_loss / n, "val_loss": vm["loss"], "val_acc": vm["acc"],
               "val_macro_f1": vm["macro_f1"], "lr": opt.param_groups[0]["lr"], "epoch_s": round(time.time() - t0, 1),
               "train_s": round(train_s, 1),
               "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if device.type == "cuda" else 0}
        hist.append(row)
        pd.DataFrame(hist).to_csv(run_dir / "history.csv", index=False)
        improved = vm["macro_f1"] > best_f1
        if improved:
            best_f1, best_ep, bad = vm["macro_f1"], ep, 0
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep, "val": vm}, run_dir / "best.pt")
            pd.DataFrame({"path": val_ds.df["path"], "label": ys, "pred": probs.argmax(1),
                          **{f"p_{c}": probs[:, i] for i, c in enumerate(classes)}}) \
                .to_csv(run_dir / "val_preds_best.csv", index=False)
        else:
            bad += 1
        print(f"[{ep:02d}/{cfg['epochs']}] loss {row['train_loss']:.4f} | val loss {vm['loss']:.4f} "
              f"acc {vm['acc']:.4f} F1 {vm['macro_f1']:.4f}{' *' if improved else ''} | "
              f"{row['epoch_s']}s | mem {row['gpu_mem_gb']}GB", flush=True)
        if bad >= cfg["patience"] and ep >= cfg.get("min_epochs", 0):
            print(f"Early stopping: {cfg['patience']} epoch tanpa peningkatan val macro-F1")
            break
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep}, run_dir / "last.pt")

    best_row = hist[best_ep - 1]
    res = {"model": args.model, "aug": args.aug, "val_macro_f1": best_f1, "val_acc": best_row["val_acc"],
           "best_epoch": best_ep, "epochs_run": ep, "train_min": round((time.time() - t_start) / 60, 1),
           "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "pretrained": cfg["pretrained_tag"], "run_dir": run_dir.relative_to(ROOT).as_posix()}
    (run_dir / "val_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    if not args.max_steps:
        append_experiment(res, cfg_desc=f"{cfg['pretrained_tag']}, {size}px, batch {cfg['batch']}, AdamW lr {cfg['lr']} "
                                        f"wd {cfg['weight_decay']}, cosine+warmup {cfg['warmup_epochs']}, LS "
                                        f"{cfg['label_smoothing']}, AMP, patience {cfg['patience']} (min_epochs {cfg.get('min_epochs', 0)})")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
