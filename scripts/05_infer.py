"""Tahap 5: inferensi pada satu gambar atau folder.

Output per gambar: label top-1, probabilitas semua kelas, dan waktu (preprocessing + inferensi, ms).
--model menerima: nama run di runs/ (PyTorch), path .onnx hasil 07_export.py (timm), atau ekspor YOLO (.onnx / *_ncnn_model).

Contoh:
  python scripts/05_infer.py data/raw/Curl_Virus/"Curl Virus_1002.png"
  python scripts/05_infer.py path/ke/folder --model runs/mobilenetv3_large_100_augB/export/mobilenetv3_large_100_augB.onnx --csv hasil.csv
  python scripts/05_infer.py path/ke/folder --model yolo11n_cls --device cpu
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from common import IMG_EXTS, ROOT
from data_utils import center_224, load_rgb
from models_io import load_classifier


def collect(inputs: list[str]) -> list[Path]:
    files = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            raise SystemExit(f"Tidak ditemukan: {p}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="path gambar dan/atau folder")
    ap.add_argument("--model", default="yolo11n_cls_augB")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="untuk model PyTorch")
    ap.add_argument("--csv", type=Path, help="simpan hasil ke CSV")
    ap.add_argument("--warmup", type=int, default=3, help="inferensi pemanasan (tidak dihitung)")
    args = ap.parse_args()

    files = collect(args.inputs)
    clf = load_classifier(args.model, device=args.device)
    cuda = getattr(clf, "device", None) is not None and str(clf.device).startswith(("cuda", "0")) and torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    print(f"Model: {args.model} ({clf.kind}) | {len(files)} gambar | kelas: {clf.classes}")
    if files:
        dummy = center_224(load_rgb(files[0], draft=None))
        for _ in range(args.warmup):
            clf.predict_proba([dummy])

    rows = []
    for f in files:
        t0 = time.perf_counter()
        img = center_224(load_rgb(f, draft=None))
        t1 = time.perf_counter()
        sync()
        prob = clf.predict_proba([img])[0]
        sync()
        t2 = time.perf_counter()
        k = int(prob.argmax())
        rows.append({"file": str(f), "pred": clf.classes[k], "conf": float(prob[k]),
                     "pre_ms": (t1 - t0) * 1000, "infer_ms": (t2 - t1) * 1000,
                     **{f"p_{c}": float(p) for c, p in zip(clf.classes, prob)}})
        probs = "  ".join(f"{c[:12]}={p:.3f}" for c, p in zip(clf.classes, prob))
        print(f"{f.name}: {clf.classes[k]} ({prob[k]:.3f}) | pre {rows[-1]['pre_ms']:.1f} ms, "
              f"infer {rows[-1]['infer_ms']:.1f} ms\n    {probs}")
    if rows:
        df = pd.DataFrame(rows)
        print(f"\nRata-rata: pre {df.pre_ms.mean():.1f} ms, infer {df.infer_ms.mean():.1f} ms "
              f"(p95 infer {df.infer_ms.quantile(0.95):.1f} ms)")
        if args.csv:
            out = args.csv if args.csv.is_absolute() else ROOT / args.csv
            df.to_csv(out, index=False)
            print(f"Tersimpan: {out}")


if __name__ == "__main__":
    main()
