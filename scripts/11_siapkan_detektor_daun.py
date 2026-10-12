"""Siapkan detektor daun open-vocabulary (YOLOE-26, Ultralytics) — cukup dijalankan SEKALI.

YOLOE diberi perintah teks (default "leaf"). Embedding teksnya dihitung sekali dengan encoder MobileCLIP2
(unduhan ~254 MB, jalankan saat koneksi bagus), lalu disimpan di dalam model sehingga
10_uji_layar.py tidak butuh internet / encoder teks lagi.

Ini detektor zero-shot (tanpa training di data cabai): kualitas kotak perlu dicek; untuk deploy nanti
detektor sebaiknya dilatih dengan anotasi daun dari foto lapangan.

Contoh:
  python scripts/11_siapkan_detektor_daun.py
  python scripts/11_siapkan_detektor_daun.py --model yoloe-26s-seg.pt --prompt leaf "chili leaf"
"""
import argparse
import os
from pathlib import Path

from common import ROOT  # noqa: F401  (set YOLO_AUTOINSTALL & encoding)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yoloe-26n-seg.pt", help="yoloe-26{n,s,m,l,x}-seg.pt")
    ap.add_argument("--prompt", nargs="+", default=["leaf"], help="teks kelas yang dideteksi")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/pretrained/yoloe-26n-seg-daun.pt")
    args = ap.parse_args()

    dst = ROOT / "runs/pretrained"
    dst.mkdir(parents=True, exist_ok=True)
    os.chdir(dst)  # Ultralytics mengunduh bobot & encoder teks ke folder kerja -> simpan di runs/pretrained
    from ultralytics import YOLOE

    print(f"Memuat {args.model} dan menghitung embedding untuk {args.prompt} "
          "(unduhan encoder teks MobileCLIP2 ~254 MB bila belum ada) ...", flush=True)
    m = YOLOE(args.model)
    m.set_classes(args.prompt)
    out = args.out if args.out.is_absolute() else ROOT / args.out
    m.save(str(out))
    print(f"Tersimpan: {out} | kelas: {YOLOE(str(out)).names}")
    print("Sekarang jalankan: python scripts/10_uji_layar.py   (mode deteksi aktif otomatis)")


if __name__ == "__main__":
    main()
