"""Uji live dari tangkapan layar: 2 model berdampingan + rekam video, CSV, screenshot, ringkasan.

Dua mode:
  - DETEKSI (otomatis bila runs/pretrained/yoloe-26n-seg-daun.pt ada; siapkan dengan 11_siapkan_detektor_daun.py):
      setiap daun di area diberi bounding box, lalu tiap daun diklasifikasi kedua model.
  - KLASIFIKASI (tanpa detektor): hanya kotak persegi terang di tengah area yang dibaca (1 daun);
      area di luarnya digelapkan agar jelas bagian mana yang dinilai.

Pakai: buka gambar daun cabai di layar (mis. browser), jalankan script, drag kotak di sekitar gambar, ENTER.
Jendela program otomatis TIDAK ikut tertangkap (Windows 10 2004+), jadi boleh menimpa area.
Tombol (klik jendela program dulu):
  R rekam mulai/berhenti   S screenshot   A ganti area   Q / Esc keluar

Output: data/uji_internet/rekaman_layar/<tanggal_jam>/  klip_XX.mp4, prediksi.csv, screenshot_XX.png, ringkasan.md

Contoh:
  python scripts/10_uji_layar.py
  python scripts/10_uji_layar.py --mode klasifikasi
  python scripts/10_uji_layar.py --region 100,200,600,600 --max-frames 30 --no-show --record   (uji tanpa jendela)
"""
import argparse
import ctypes
import datetime as dt
import sys
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
ABBR = {"Bacterial_Spot": "Bact", "Cercospora_Leaf_Spot": "Cerc", "Curl_Virus": "Curl", "Healthy_Leaf": "Sehat",
        "Nutrition_Deficiency": "Nutr", "Powdery_Mildew": "Powd"}
NAMA = {"mobilenetv3_large_100_augB": "MobileNetV3-B", "yolo11n_cls": "YOLO11n-cls (lama)", "yolo11n_cls_augB": "YOLO11n-cls-B"}
DETEKTOR_DEFAULT = ROOT / "runs/pretrained/yoloe-26n-seg-daun.pt"
H, PANEL_W = 600, 420
WIN = "Uji layar  |  R rekam  S screenshot  A ganti area  Q keluar"
SEL = "Langkah 1/2: drag kotak di sekitar gambar daun, lalu ENTER (C = batal)"


def dpi_aware():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:  # noqa: BLE001
            ctypes.windll.user32.SetProcessDPIAware()


def exclude_from_capture(title: str) -> bool:
    """Jendela program tidak ikut tertangkap (WDA_EXCLUDEFROMCAPTURE) -> tidak ada efek cermin berulang."""
    if sys.platform != "win32":
        return False
    hwnd = ctypes.windll.user32.FindWindowW(None, title)
    return bool(hwnd) and bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11))


def pilih_area(sct, mon, lama=None) -> dict:
    full = cv2.cvtColor(np.asarray(sct.grab(mon)), cv2.COLOR_BGRA2BGR)
    s = min(1.0, 1400 / full.shape[1], 800 / full.shape[0])
    small = cv2.resize(full, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.rectangle(small, (0, 0), (small.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(small, "Ini FOTO layar Anda (bukan live). Drag kotak di sekitar gambar daun, lalu ENTER.",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    while True:
        x, y, w, h = cv2.selectROI(SEL, small, showCrosshair=False)
        cv2.destroyWindow(SEL)
        if w >= 20 and h >= 20:
            return {"left": mon["left"] + int(x / s), "top": mon["top"] + int(y / s),
                    "width": int(w / s), "height": int(h / s)}
        if lama is not None:
            return lama
        print("Area belum dipilih. Coba lagi (drag kotak, lalu ENTER).")


def square_box(x1, y1, x2, y2, w, h, margin=0.12):
    """Kotak persegi di sekitar bbox daun (+margin), dipotong di batas gambar."""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    a, b = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    return a, b, int(min(w, cx + side / 2)), int(min(h, cy + side / 2))


class Detector:
    def __init__(self, path: Path, conf: float, device: str, max_det: int):
        from ultralytics import YOLOE

        self.m, self.conf, self.device, self.max_det = YOLOE(str(path)), conf, device, max_det

    def __call__(self, bgr):
        r = self.m.predict(bgr, conf=self.conf, imgsz=640, device=self.device, verbose=False, max_det=self.max_det,
                           agnostic_nms=True)[0]
        out = []
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if min(x2 - x1, y2 - y1) >= 24:
                out.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        return sorted(out, key=lambda b: (b[1] // 80, b[0]))  # urut baris lalu kolom -> nomor daun stabil


def draw_panel(mode, names, classes, leaves, fps, rec, t_rec, aspek_warn,
               bantuan="R rekam  S screenshot  A ganti area  Q keluar", tip_area="pilih 1 daun (A)"):
    p = np.full((H, PANEL_W, 3), 28, np.uint8)
    y = 26
    cv2.putText(p, f"Mode: {mode.upper()}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 22
    if mode == "klasifikasi":
        for t in ("Belum ada detektor daun: hanya kotak terang", "di tengah area yang dinilai (1 daun).",
                  "Siapkan detektor: 11_siapkan_detektor_daun.py"):
            cv2.putText(p, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
            y += 17
        if aspek_warn:
            cv2.putText(p, f"! Area terlalu lebar/tinggi - {tip_area}", (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 140, 255), 1)
            y += 17
        y += 6
        _, probs = leaves[0]
        for name, prob in zip(names, probs):
            k = int(prob.argmax())
            col = (0, 220, 0) if prob[k] >= 0.6 else (0, 200, 255)
            cv2.putText(p, NAMA.get(name, name), (12, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(p, f"{SHORT[classes[k]]}  {prob[k]:.2f}", (12, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 2)
            for i, (c, v) in enumerate(zip(classes, prob)):
                yy = y + 70 + i * 20
                cv2.rectangle(p, (12, yy - 11), (12 + int(170 * v), yy + 3), (0, 160, 0) if i == k else (90, 90, 90), -1)
                cv2.putText(p, f"{SHORT[c][:14]:14s} {v:.2f}", (192, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1)
            y += 70 + 6 * 20 + 6
        if len(names) == 2:
            same = probs[0].argmax() == probs[1].argmax()
            cv2.putText(p, "Kedua model: SEPAKAT" if same else "Kedua model: BERBEDA", (12, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0) if same else (0, 120, 255), 2)
    else:
        cv2.putText(p, f"{len(leaves)} daun terdeteksi", (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 26
        hdr = "  #  " + "   ".join(f"{NAMA.get(n, n)[:11]:11s}" for n in names)
        cv2.putText(p, hdr, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y += 8
        if not leaves:
            cv2.putText(p, "(tidak ada daun -> tidak ada prediksi)", (12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (160, 160, 160), 1)
        for j, (_, probs) in enumerate(leaves[:16]):
            y += 26
            ks = [int(pr.argmax()) for pr in probs]
            same = len(set(ks)) == 1
            txt = f"{j + 1:2d}   " + "   ".join(f"{ABBR[classes[k]]:5s} {pr[k]:.2f}" for k, pr in zip(ks, probs))
            cv2.putText(p, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0) if same else (0, 140, 255), 1)
        if len(leaves) > 16:
            cv2.putText(p, f"... +{len(leaves) - 16} daun", (12, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        cv2.putText(p, "hijau = kedua model sepakat, oranye = berbeda", (12, H - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (160, 160, 160), 1)
    cv2.putText(p, bantuan, (12, H - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    if fps is None:  # mode screenshot: tanpa FPS/REC
        return p
    status = f"{fps:4.1f} FPS"
    if rec:
        status += f"   REC {int(t_rec // 60):02d}:{int(t_rec % 60):02d}"
        cv2.circle(p, (PANEL_W - 20, H - 16), 7, (0, 0, 255), -1)
    cv2.putText(p, status, (12, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return p


def draw_left(cap, mode, boxes, leaves, classes):
    img = cap.copy()
    h, w = img.shape[:2]
    if mode == "klasifikasi":
        sq = min(h, w)
        x0, y0 = (w - sq) // 2, (h - sq) // 2
        dim = (img * 0.35).astype(np.uint8)
        dim[y0:y0 + sq, x0:x0 + sq] = img[y0:y0 + sq, x0:x0 + sq]
        img = dim
        cv2.rectangle(img, (x0, y0), (x0 + sq - 1, y0 + sq - 1), (0, 255, 255), 2)
    else:
        th = max(1, round(min(h, w) / 300))
        for j, ((x1, y1, x2, y2, _), (_, probs)) in enumerate(zip(boxes, leaves)):
            ks = [int(pr.argmax()) for pr in probs]
            col = (0, 220, 0) if len(set(ks)) == 1 else (0, 140, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 * th)
            lbl = f"{j + 1}: {ABBR[classes[ks[0]]]} {probs[0][ks[0]]:.2f}"
            fs = 0.5 * th
            (tw, tht), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            cv2.rectangle(img, (x1, max(0, y1 - tht - 8)), (x1 + tw + 6, max(tht + 8, y1)), col, -1)
            cv2.putText(img, lbl, (x1 + 3, max(tht + 3, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th)
    s = H / h
    left = cv2.resize(img, (max(2, int(w * s)), H), interpolation=cv2.INTER_AREA)
    return left


def ringkasan(df, out, clips, shots, area, names, mode, excluded):
    L = [f"# Uji tangkapan layar — {out.name}", "",
         f"- Mode: **{mode}**; area {area['width']}x{area['height']} px di ({area['left']},{area['top']}); "
         f"jendela program dikecualikan dari tangkapan: {'ya' if excluded else 'TIDAK (hindari menimpa area)'}",
         f"- Frame diproses: {df.frame.nunique() if len(df) else 0}; FPS rata-rata: {df.groupby('frame').fps.first().mean():.1f}"
         if len(df) else "- Tidak ada frame",
         f"- Klip video: {', '.join(clips) or 'tidak ada (tekan R untuk merekam)'}; screenshot: {len(shots)}", "",
         "> Sanity check, bukan evaluasi. Untuk akurasi gunakan `09_uji_eksternal.py` (gambar berlabel per folder).", ""]
    d = df[df.daun_id > 0] if mode == "deteksi" else df
    for judul, sub in (("frame saat merekam", d[d.rekam]), ("semua frame", d)):
        if sub.empty:
            continue
        L += [f"## {judul.capitalize()} (n={len(sub)} {'daun-frame' if mode == 'deteksi' else 'frame'})", ""]
        if mode == "deteksi":
            per = df[df.rekam] if judul.startswith("frame saat") else df
            L += [f"- Rata-rata daun terdeteksi per frame: {per.groupby('frame').daun_id.max().clip(lower=0).mean():.1f}"]
        if len(names) == 2:
            L += [f"- Kedua model sepakat: **{(sub[f'pred_{names[0]}'] == sub[f'pred_{names[1]}']).mean() * 100:.1f}%**"]
        for n in names:
            vc = sub[f"pred_{n}"].value_counts(normalize=True)
            L += [f"- **{NAMA.get(n, n)}** — median confidence {sub[f'conf_{n}'].median():.2f}; prediksi: "
                  + ", ".join(f"{SHORT.get(k, k)} {v * 100:.0f}%" for k, v in vc.items())]
        L.append("")
    (out / "ringkasan.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["yolo11n_cls_augB", "mobilenetv3_large_100_augB"])
    ap.add_argument("--mode", choices=["auto", "deteksi", "klasifikasi"], default="auto")
    ap.add_argument("--detektor", type=Path, default=DETEKTOR_DEFAULT)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--max-daun", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--monitor", type=int, default=1, help="1 = monitor utama")
    ap.add_argument("--region", type=str, help="x,y,w,h (lewati pemilihan area)")
    ap.add_argument("--fps", type=float, default=10, help="fps video rekaman")
    ap.add_argument("--out", type=Path, default=ROOT / "data/uji_internet/rekaman_layar")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-show", action="store_true", help="tanpa jendela (uji)")
    ap.add_argument("--record", action="store_true", help="langsung merekam sejak awal")
    args = ap.parse_args()
    dpi_aware()

    det_path = args.detektor if args.detektor.is_absolute() else ROOT / args.detektor
    mode = args.mode if args.mode != "auto" else ("deteksi" if det_path.exists() else "klasifikasi")
    if mode == "deteksi" and not det_path.exists():
        raise SystemExit(f"Detektor tidak ditemukan: {det_path}\nJalankan dulu: python scripts/11_siapkan_detektor_daun.py")
    print(f"MODE: {mode.upper()}" + ("" if mode == "deteksi" else
          "  (detektor daun belum disiapkan -> hanya kotak tengah yang dinilai; lihat 11_siapkan_detektor_daun.py)"))
    det = Detector(det_path, args.det_conf, 0 if args.device == "cuda" else "cpu", args.max_daun) if mode == "deteksi" else None
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
            print("Langkah 1/2: jendela berisi FOTO layar akan muncul -> drag kotak di sekitar gambar daun -> ENTER")
            area = pilih_area(sct, mon)
        print(f"Langkah 2/2: deteksi live berjalan. Area: {area} | output: {out}")
        rows, clips, shots = [], [], []
        writer, rec, t_rec0, n_written, excluded = None, False, 0.0, 0, False
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
                print(f"REKAM berhenti: {clips[-1]}")
            writer, rec = None, False

        try:
            while True:
                cap = cv2.cvtColor(np.asarray(sct.grab(area)), cv2.COLOR_BGRA2BGR)
                hh, ww = cap.shape[:2]
                rgb = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
                if mode == "deteksi":
                    boxes = det(cap)
                    crops = [center_224(rgb[b:d, a:c]) for a, b, c, d in
                             (square_box(x1, y1, x2, y2, ww, hh) for x1, y1, x2, y2, _ in boxes)]
                    per_model = [m.predict_proba(crops) if crops else np.zeros((0, len(classes))) for m in clfs]
                    leaves = [(boxes[j], [pm[j] for pm in per_model]) for j in range(len(boxes))]
                else:
                    boxes = []
                    leaves = [(None, [m.predict_proba([center_224(rgb)])[0] for m in clfs])]
                now = time.perf_counter()
                fps = 1 / max(now - t_prev, 1e-6) if n == 0 else 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
                t_prev = now
                n += 1
                aspek_warn = max(ww, hh) / max(1, min(ww, hh)) > 1.4
                left = draw_left(cap, mode, boxes, leaves, classes)
                frame = np.hstack([left, draw_panel(mode, names, classes, leaves, fps, rec, now - t_rec0, aspek_warn)])
                frame = frame[:, : frame.shape[1] - frame.shape[1] % 2]
                if args.record and n == 1:
                    start_rec(frame)
                if rec:
                    while n_written <= (now - t_rec0) * args.fps:  # durasi video = durasi nyata
                        writer.write(frame)
                        n_written += 1
                base = {"waktu_s": round(now - t0, 3), "frame": n, "rekam": rec, "klip": clips[-1] if rec else "",
                        "fps": fps, "mode": mode}
                if mode == "deteksi" and not leaves:
                    rows.append({**base, "daun_id": 0})
                for j, (b, probs) in enumerate(leaves):
                    rows.append({**base, "daun_id": j + 1 if mode == "deteksi" else 0,
                                 **({"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "det_conf": round(b[4], 3)} if b else {}),
                                 **{k: v for nm, pr in zip(names, probs) for k, v in
                                    ((f"pred_{nm}", classes[int(pr.argmax())]), (f"conf_{nm}", float(pr.max())))}})
                key = -1
                if not args.no_show:
                    cv2.imshow(WIN, frame)
                    if n == 1:
                        cv2.waitKey(1)
                        excluded = exclude_from_capture(WIN)
                        print("Jendela program dikecualikan dari tangkapan layar:", excluded)
                    key = cv2.waitKey(1) & 0xFF
                    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:  # jendela ditutup dengan tombol X
                        break
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
                    stop_rec()  # ukuran frame bisa berubah -> klip baru
                    area = pilih_area(sct, mon, lama=area)
                if args.max_frames and n >= args.max_frames:
                    break
        finally:
            stop_rec()
            cv2.destroyAllWindows()
    df = pd.DataFrame(rows)
    df.to_csv(out / "prediksi.csv", index=False)
    ringkasan(df, out, clips, shots, area, names, mode, excluded)
    print(f"\nHasil tersimpan di: {out}")


if __name__ == "__main__":
    main()
