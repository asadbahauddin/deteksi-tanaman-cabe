"""Tahap 3a: training Ultralytics YOLO-cls; checkpoint terbaik & early stopping berdasarkan val macro-F1.

Ultralytics secara default memilih best.pt berdasarkan top-1 accuracy. Di sini ClassificationTrainer
di-subclass sehingga `fitness` = val macro-F1 (dipakai untuk best.pt dan patience).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

from common import ROOT, seed_everything  # noqa: I001  (harus sebelum ultralytics: set YOLO_AUTOINSTALL)

import pandas as pd
import torch
import ultralytics.data.utils as ul_data_utils
import yaml
from sklearn.metrics import accuracy_score, f1_score
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer

from notes_log import append_experiment

# Ultralytics "memperbaiki" JPEG tanpa marker FFD9 dengan menulis ulang file (re-encode q100), yang
# mengganti symlink di data/split dan membuat data YOLO berbeda dari model lain. 17 JPEG di dataset ini
# kehilangan marker tetapi terbaca penuh oleh PIL/cv2 (lihat audit), jadi penulisan dimatikan.
ul_data_utils._replace_image = lambda *a, **k: None


class MacroF1Trainer(ClassificationTrainer):
    def validate(self):
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        metrics.pop("fitness", None)
        pred = torch.cat(self.validator.pred)[:, 0].numpy()
        true = torch.cat(self.validator.targets).numpy()
        f1 = float(f1_score(true, pred, average="macro"))
        metrics["metrics/macro_f1"] = f1
        if self.best_fitness is None or self.best_fitness < f1:
            self.best_fitness = f1
        return metrics, f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/yolo11n_cls.yaml")
    ap.add_argument("--epochs", type=int, default=None, help="override (mis. uji cepat)")
    ap.add_argument("--name", type=str, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.name:
        cfg["name"] = args.name
    seed_everything(cfg["seed"])

    run_dir = ROOT / cfg["project"] / cfg["name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    model = YOLO(ROOT / cfg.pop("model"))
    t0 = time.time()
    train_kw = {**cfg, "data": str(ROOT / cfg["data"]), "project": str(ROOT / cfg["project"]), "exist_ok": True}
    model.train(trainer=MacroF1Trainer, **train_kw)
    train_min = (time.time() - t0) / 60

    # evaluasi ulang best.pt pada val (macro-F1 + accuracy) — test TIDAK disentuh
    best = YOLO(run_dir / "weights/best.pt")
    names = best.names
    cls_to_idx = {v: k for k, v in names.items()}
    files = sorted((ROOT / cfg["data"] / "val").glob("*/*"))
    y_true, y_pred = [], []
    for i in range(0, len(files), 64):
        chunk = files[i:i + 64]
        for f, r in zip(chunk, best.predict([str(f) for f in chunk], imgsz=cfg["imgsz"], verbose=False)):
            y_true.append(cls_to_idx[f.parent.name])
            y_pred.append(int(r.probs.top1))
    hist = pd.read_csv(run_dir / "results.csv")
    hist.columns = hist.columns.str.strip()
    res = {"model": "yolo11n-cls", "aug": "ultralytics-default", "val_macro_f1": f1_score(y_true, y_pred, average="macro"),
           "val_acc": accuracy_score(y_true, y_pred), "train_min": round(train_min, 1),
           "best_epoch": int(hist.loc[hist["metrics/macro_f1"].idxmax(), "epoch"]), "epochs_run": len(hist),
           "run_dir": run_dir.relative_to(ROOT).as_posix()}
    (run_dir / "val_metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump({**train_kw, "model": "yolo11n-cls.pt"}), encoding="utf-8")
    if not cfg["name"].startswith("_"):  # run uji (nama diawali "_") tidak dicatat
        append_experiment(res, cfg_desc=f"imgsz {cfg['imgsz']}, batch {cfg['batch']}, epochs≤{cfg['epochs']}, "
                                    f"patience {cfg['patience']} (macro-F1), optimizer {cfg['optimizer']}, AMP")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
