"""Tahap 5: demo webcam laptop (sanity check, BUKAN evaluasi).

Frame kamera -> center crop persegi -> resize 224 (preprocessing sama dengan evaluasi) -> klasifikasi.
Tampilan: kotak area yang diklasifikasi, label top-1, probabilitas semua kelas, FPS. Tekan 'q' / Esc untuk keluar.

Contoh:
  python scripts/06_webcam_demo.py
  python scripts/06_webcam_demo.py --model runs/yolo11n_cls/export/best_ncnn_model --camera 0
  python scripts/06_webcam_demo.py --max-frames 30 --no-show     (uji tanpa jendela)
"""
import argparse
import time

import cv2
import numpy as np

from common import ROOT  # noqa: F401  (set env & encoding)
from data_utils import center_224
from models_io import load_classifier

SHORT = {"Bacterial_Spot": "Bacterial Spot", "Cercospora_Leaf_Spot": "Cercospora", "Curl_Virus": "Curl Virus",
         "Healthy_Leaf": "Healthy", "Nutrition_Deficiency": "Nutrient Def.", "Powdery_Mildew": "Powdery Mildew"}


def draw(frame, classes, prob, fps, box):
    x0, y0, s = box
    cv2.rectangle(frame, (x0, y0), (x0 + s, y0 + s), (0, 255, 255), 2)
    k = int(prob.argmax())
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, f"{SHORT.get(classes[k], classes[k])}  {prob[k]:.2f}   |  {fps:.1f} FPS", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if prob[k] >= 0.6 else (0, 200, 255), 2)
    for i, (c, p) in enumerate(zip(classes, prob)):
        y = 56 + i * 22
        cv2.rectangle(frame, (8, y - 12), (8 + int(150 * p), y + 4), (0, 180, 0), -1)
        cv2.putText(frame, f"{SHORT.get(c, c)[:14]:14s} {p:.2f}", (165, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
    cv2.putText(frame, "Sanity check - bukan evaluasi lapangan", (8, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="yolo11n_cls_augB")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = sampai ditekan q")
    ap.add_argument("--no-show", action="store_true", help="tanpa jendela (uji)")
    args = ap.parse_args()

    clf = load_classifier(args.model, device=args.device)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.camera} tidak bisa dibuka")
    n, fps, t_prev, ema = 0, 0.0, time.perf_counter(), None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame gagal dibaca")
                break
            h, w = frame.shape[:2]
            s = min(h, w)
            box = ((w - s) // 2, (h - s) // 2, s)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            prob = clf.predict_proba([center_224(rgb)])[0]
            ema = prob if ema is None else 0.6 * ema + 0.4 * prob  # perataan ringan agar label tidak berkedip
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1 / max(now - t_prev, 1e-6)) if n else 1 / max(now - t_prev, 1e-6)
            t_prev = now
            n += 1
            if args.no_show:
                k = int(ema.argmax())
                print(f"frame {n}: {clf.classes[k]} {ema[k]:.2f} | {fps:.1f} FPS | {w}x{h}")
            else:
                draw(frame, clf.classes, ema, fps, box)
                cv2.imshow("Cabai - demo klasifikasi daun (q untuk keluar)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.max_frames and n >= args.max_frames:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Selesai: {n} frame, ~{fps:.1f} FPS")


if __name__ == "__main__":
    main()
