"""Tahap 5: ekspor model terbaik + verifikasi kesetaraan output.

- timm  -> ONNX via torch.onnx.export(dynamo=True) (exporter TorchScript lama sudah deprecated di torch 2.x),
           batch dinamis, metadata (classes, mean, std, img_size) disimpan di ONNX.
- YOLO  -> ONNX dan NCNN via exporter resmi Ultralytics.
Verifikasi memakai N gambar VAL (bukan test): selisih maksimum logit/probabilitas vs PyTorch-CPU fp32.

Contoh:
  python scripts/07_export.py --timm-run mobilenetv3_large_100_augB --yolo-run yolo11n_cls
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import torch

from common import ROOT, SEED, md_table, seed_everything
from data_utils import center_224, load_rgb
from models_io import OnnxTimmClassifier, TimmClassifier, load_classifier


def dir_size_mb(p: Path) -> float:
    files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1024**2, 2)


def export_timm(run: str, opset: int, imgs: list[np.ndarray]) -> dict:
    clf = TimmClassifier(ROOT / "runs" / run, device="cpu")
    model = clf.model.eval().float().cpu()
    out_dir = ROOT / "runs" / run / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{run}.onnx"
    x = clf.to_tensor(imgs[:2])
    prog = torch.onnx.export(model, (x,), dynamo=True, opset_version=opset, input_names=["input"],
                             output_names=["logits"], dynamic_shapes={"x": {0: torch.export.Dim("batch", max=256)}})
    prog.save(str(onnx_path))
    m = onnx.load(str(onnx_path))
    for k, v in {"classes": json.dumps(clf.classes), "mean": json.dumps(clf.cfg["mean"]), "std": json.dumps(clf.cfg["std"]),
                 "img_size": str(clf.size), "arch": clf.cfg["model"], "source_run": run}.items():
        m.metadata_props.add(key=k, value=v)
    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_path))

    with torch.no_grad():
        ref = model(clf.to_tensor(imgs)).numpy()
    sess = OnnxTimmClassifier(onnx_path)
    got = sess.sess.run(None, {sess.input_name: sess.to_array(imgs)})[0]
    p_ref = torch.from_numpy(ref).softmax(1).numpy()
    p_got = torch.from_numpy(got).softmax(1).numpy()
    return {"model": run, "format": f"ONNX (opset {opset}, dynamo)", "path": onnx_path.relative_to(ROOT).as_posix(),
            "size_MB": dir_size_mb(onnx_path), "max_abs_diff_logit": float(np.abs(ref - got).max()),
            "max_abs_diff_prob": float(np.abs(p_ref - p_got).max()),
            "top1_agree": f"{int((ref.argmax(1) == got.argmax(1)).sum())}/{len(imgs)}"}


def export_yolo(run: str, imgs: list[np.ndarray]) -> list[dict]:
    from ultralytics import YOLO

    best = ROOT / "runs" / run / "weights/best.pt"
    out_dir = ROOT / "runs" / run / "export"
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = load_classifier(run, device="cpu").predict_proba(imgs)
    rows = []
    for fmt in ("onnx", "ncnn"):
        exported = Path(YOLO(best).export(format=fmt, imgsz=224, device="cpu"))
        dst = out_dir / exported.name
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(str(exported), dst)
        got = load_classifier(dst.relative_to(ROOT).as_posix()).predict_proba(imgs)
        rows.append({"model": run, "format": "ONNX (Ultralytics)" if fmt == "onnx" else "NCNN (Ultralytics, fp32)",
                     "path": dst.relative_to(ROOT).as_posix(), "size_MB": dir_size_mb(dst),
                     "max_abs_diff_logit": np.nan, "max_abs_diff_prob": float(np.abs(ref - got).max()),
                     "top1_agree": f"{int((ref.argmax(1) == got.argmax(1)).sum())}/{len(imgs)}"})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timm-run", nargs="*", default=["mobilenetv3_large_100_augB"])
    ap.add_argument("--yolo-run", nargs="*", default=["yolo11n_cls"])
    ap.add_argument("--opset", type=int, default=18, help="opset ONNX untuk timm (18: default exporter dynamo)")
    ap.add_argument("--n-verify", type=int, default=64, help="jumlah gambar VAL untuk verifikasi")
    args = ap.parse_args()
    seed_everything(SEED)

    val = pd.read_csv(ROOT / "splits/val.csv").sample(args.n_verify, random_state=SEED)
    imgs = [center_224(load_rgb(ROOT / p, draft=None)) for p in val.path]
    rows = [export_timm(r, args.opset, imgs) for r in args.timm_run]
    for r in args.yolo_run:
        rows += export_yolo(r, imgs)
    for r in args.timm_run + args.yolo_run:
        pt = ROOT / "runs" / r / ("weights/best.pt" if (ROOT / "runs" / r / "weights").exists() else "best.pt")
        rows.append({"model": r, "format": "PyTorch checkpoint (referensi)", "path": pt.relative_to(ROOT).as_posix(),
                     "size_MB": dir_size_mb(pt), "max_abs_diff_logit": 0.0, "max_abs_diff_prob": 0.0, "top1_agree": "-"})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports/export.csv", index=False)
    show = df.assign(size_MB=df.size_MB.map("{:.1f}".format),
                     max_abs_diff_logit=df.max_abs_diff_logit.map(lambda v: "-" if pd.isna(v) else f"{v:.1e}"),
                     max_abs_diff_prob=df.max_abs_diff_prob.map("{:.1e}".format))
    txt = ("# Ekspor model\n\n"
           f"Verifikasi: {args.n_verify} gambar VAL acak (seed {SEED}), referensi = PyTorch CPU fp32. "
           "`max_abs_diff_prob` = selisih maksimum probabilitas softmax.\n\n"
           "Catatan: checkpoint timm = state_dict fp32 + config; checkpoint YOLO disimpan Ultralytics dalam fp16 "
           "(karena itu lebih kecil dari ONNX/NCNN fp32). YOLO mengeluarkan probabilitas langsung (tanpa logit).\n\n"
           + md_table(show) + "\n")
    (ROOT / "reports/export.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
