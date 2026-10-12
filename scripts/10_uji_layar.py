"""Uji live dari tangkapan layar: 2 model berdampingan + rekam video, CSV per frame, screenshot, ringkasan.

Pakai: buka gambar/video daun cabai di layar (mis. browser), jalankan script, lalu drag kotak di sekitar daun.
Tombol (fokus di jendela program):
  R  mulai/berhenti rekam (tiap mulai = klip video baru)
  S  simpan screenshot frame + prediksi
  A  pilih ulang area tangkapan
  Q / Esc  keluar (ringkasan ditulis otomatis)
Pastikan jendela program TIDAK menutupi area yang ditangkap (geser ke sisi lain / monitor lain).

Output: data/uji_internet/rekaman_layar/<tanggal_jam>/
  klip_01.mp4, ...   video (area tangkapan + panel prediksi kedua model)
  prediksi_per_frame.csv, screenshot_*.png, ringkasan.md

Contoh:
  python scripts/10_uji_layar.py
  python scripts/10_uji_layar.py --models mobilenetv3_large_100_augB yolo11n_cls --fps 10
  python scripts/10_uji_layar.py --region 100,200,600,600 --max-frames 30 --no-show --record   (uji tanpa jendela)
"""
import argparse
import datetime as dt
import time
from pathlib import Path

import cv2
import mss
import numpy as np
import pandas as pd

from common import ROOT
from data_utils import center_224
from models_io import load_classifier

SHORT = {"Bacterial_Spot": "Bacterial Spot", "Cercospora_Leaf_Spot": "Cercospora", "Curl_Virus": "Curl Virus",
         "Healthy_Leaf": "Healthy", "Nutrition_Deficiency": "Nutrient Def.", "Powdery_Mildew": "Powdery Mildew"}
NAMA = {"mobilenetv3_large_100_augB": "MobileNetV3-B", "yolo11n_cls": "YOLO11n-cls"}
H, PANEL_W = 540, 400  # tinggi video & lebar panel prediksi


def pilih_area(sct, mon) -> dict:
    full = cv2.cvtColor(np.asarray(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
    s = min(1.0, 1280 / full.shape[1], 720 / full.shape[0])
    small = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(small, "Drag kotak di sekitar daun, lalu ENTER/SPASI (C = batal)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    x, y, w, h = cv2.selectROI("Pilih area tangkapan", small, showCrosshair=True)
    cv2.destroyWindow("Pilih area tangkapan")
    if w < 10 or h < 10:
        raise SystemExit("Area tidak dipilih")
    return {"left": mon["left"] + int(x / s), "top": mon["top"] + int(y / s), "width": int(w / s), "height": int(h / s)}


def panel(results, classes, fps, rec, t_rec):
    p = np.full((H, PANEL_W, 3), 30, np.uint8)
    blk = (H - 40) // len(results)
    for m, (name, prob) in enumerate(results):
        y0 = m * blk + 10
        k = int(prob.argmax())
        col = (0, 220, 0) if prob[k] >= 0.6 else (0, 200, 255)
        cv2.putText(p, NAMA.get(name, name), (12, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(p, f"{SHORT.get(classes[k], classes[k])}  {prob[k]:.2f}", (12, y0 + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        for i, (c, v) in enumerate(zip(classes, prob)):
            yy = y0 + 76 + i * 22
            cv2.rectangle(p, (12, yy - 12), (12 + int(170 * v), yy + 4), (0, 160, 0) if i == k else (90, 90, 90), -1)
            cv2.putText(p, f"{SHORT.get(c, c)[:14]:14s} {v:.2f}", (190, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (230, 230, 230), 1)
        cv2.line(p, (0, y0 + blk - 6), (PANEL_W, y0 + blk - 6), (80, 80, 80), 1)
    status = f"{fps:4.1f} FPS"
    if rec:
        cv2.circle(p, (PANEL_W - 110, H - 18), 7, (0, 0, 255), -1)
        status += f"   REC {int(t_rec // 60):02d}:{int(t_rec % 60):02d}"
    cv2.putText(p, status, (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return p


def compose(cap_bgr, pan):
    s = H / cap_bgr.shape[0]
    w = max(1, int(cap_bgr.shape[1] * s))
    left = cv2.resize(cap_bgr, (w, H), interpolation=cv2.INTER_AREA)
    sq = min(left.shape[:2])  # tandai area persegi tengah yang benar-benar diklasifikasi (center crop)
    x0, y0 = (left.shape[1] - sq) // 2, (left.shape[0] - sq) // 2
    cv2.rectangle(left, (x0, y0), (x0 + sq - 1, y0 + sq - 1), (0, 255, 255), 1)
    frame = np.hstack([left, pan])
    if frame.shape[1] % 2:  # beberapa codec butuh lebar genap
        frame = frame[:, :-1]
    return frame


def ringkasan(df, out, clips, shots, area, names):
    L = [f"# Uji tangkapan layar — {out.name}", "",
         f"- Area tangkapan: {area['width']}x{area['height']} px di ({area['left']},{area['top']})",
         f"- Total frame diproses: {len(df)}; FPS rata-rata: {df.fps.mean():.1f}" if len(df) else "- Tidak ada frame",
         f"- Klip video: {', '.join(clips) or 'tidak ada'}; screenshot: {len(shots)}", "",
         "> Sanity check, bukan evaluasi: gambar yang tampil di layar belum tentu berlabel benar, dan satu gambar "
         "bisa muncul di banyak frame. Untuk akurasi gunakan `09_uji_eksternal.py` (gambar berlabel per folder).", ""]
    for judul, sub in (("frame saat merekam", df[df.rekam]), ("semua frame", df)):
        if sub.empty:
            continue
        L += [f"## {judul.capitalize()} (n={len(sub)})", ""]
        cols = [f"pred_{n}" for n in names]
        if len(names) == 2:
            setuju = (sub[cols[0]] == sub[cols[1]]).mean()
            L += [f"- Kedua model sepakat: **{setuju * 100:.1f}%** frame", ""]
        for n in names:
            vc = sub[f"pred_{n}"].value_counts(normalize=True)
            L += [f"- **{NAMA.get(n, n)}** — median confidence {sub[f'conf_{n}'].median():.2f}; prediksi: "
                  + ", ".join(f"{SHORT.get(k, k)} {v * 100:.0f}%" for k, v in vc.items())]
        L.append("")
    (out / "ringkasan.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["mobilenetv3_large_100_augB", "yolo11n_cls"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--monitor", type=int, default=1, help="1 = monitor utama")
    ap.add_argument("--region", type=str, help="x,y,w,h (lewati pemilihan area)")
    ap.add_argument("--fps", type=float, default=10, help="fps video rekaman")
    ap.add_argument("--out", type=Path, default=ROOT / "data/uji_internet/rekaman_layar")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-show", action="store_true", help="tanpa jendela (uji)")
    ap.add_argument("--record", action="store_true", help="langsung merekam sejak awal")
    args = ap.parse_args()

    clfs = [load_classifier(m, device=args.device) for m in args.models]
    classes = clfs[0].classes
    assert all(c.classes == classes for c in clfs), "urutan kelas antar model berbeda"
    names = [c.name for c in clfs]
    out = (args.out if args.out.is_absolute() else ROOT / args.out) / f"{dt.datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    with mss.MSS() as sct:
        mon = sct.monitors[args.monitor]
        if args.region:
            x, y, w, h = map(int, args.region.split(","))
            area = {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}
        else:
            area = pilih_area(sct, mon)
        print(f"Area: {area} | output: {out}")
        win = "Uji layar - R rekam, S screenshot, A area, Q keluar"
        rows, clips, shots = [], [], []
        writer, rec, t_rec0, n_written = None, False, 0.0, 0
        n, fps, t_prev, t0 = 0, 0.0, time.perf_counter(), time.perf_counter()

        def start_rec(frame):
            nonlocal writer, rec, t_rec0, n_written
            path = out / f"klip_{len(clips) + 1:02d}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, frame.shape[1::-1])
            clips.append(path.name)
            rec, t_rec0, n_written = True, time.perf_counter(), 0
            print(f"REKAM mulai: {path.name}")

        def stop_rec():
            nonlocal writer, rec
            if writer is not None:
                writer.release()
                print(f"REKAM berhenti: {clips[-1]} ({n_written} frame video)")
            writer, rec = None, False

        try:
            while True:
                cap = cv2.cvtColor(np.asarray(sct.grab(area)), cv2.COLOR_BGRA2BGR)
                img = center_224(cv2.cvtColor(cap, cv2.COLOR_BGR2RGB))
                results = [(c.name, c.predict_proba([img])[0]) for c in clfs]
                now = time.perf_counter()
                fps = 1 / max(now - t_prev, 1e-6) if n == 0 else 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
                n += 1
                frame = compose(cap, panel(results, classes, fps, rec, now - t_rec0))
                if args.record and n == 1:
                    start_rec(frame)
                if rec:
                    # tulis frame berulang sesuai waktu nyata agar durasi video = durasi rekaman
                    while n_written <= (now - t_rec0) * args.fps:
                        writer.write(frame)
                        n_written += 1
                rows.append({"waktu_s": round(now - t0, 3), "frame": n, "rekam": rec, "klip": clips[-1] if rec else "",
                             "fps": fps, **{k: v for name, prob in results for k, v in
                                            ((f"pred_{name}", classes[int(prob.argmax())]),
                                             (f"conf_{name}", float(prob.max())))}})
                key = -1
                if not args.no_show:
                    cv2.imshow(win, frame)
                    key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    stop_rec() if rec else start_rec(frame)
                if key == ord("s"):
                    p = out / f"screenshot_{len(shots) + 1:02d}.png"
                    cv2.imwrite(str(p), frame)
                    shots.append(p.name)
                    print(f"Screenshot: {p.name}")
                if key == ord("a"):
                    stop_rec()
                    area = pilih_area(sct, mon)
                if args.max_frames and n >= args.max_frames:
                    break
        finally:
            stop_rec()
            cv2.destroyAllWindows()
    df = pd.DataFrame(rows)
    df.to_csv(out / "prediksi_per_frame.csv", index=False)
    ringkasan(df, out, clips, shots, area, names)
    print(f"\nHasil tersimpan di: {out}")


if __name__ == "__main__":
    main()
