"""Deteksi daun dari tampilan layar LIVE: pilih target dengan bounding box, lihat hasil, cari target lagi.

Alur (satu jendela):
  1. LIVE    : jendela menampilkan layar Anda secara live (jendela ini sendiri tidak ikut tertangkap).
               Buka browser, cari gambar daun cabai.
  2. PILIH   : saat target terlihat, drag kotak di tampilan live di sekitar daun -> frame dibekukan.
  3. HASIL   : prediksi 2 model (+ kotak per daun bila detektor daun sudah disiapkan).
               Opsional pilih "Label sebenarnya" -> akurasi dihitung di ringkasan.
  4. Klik "Cari target lagi" (atau Spasi) -> kembali LIVE.
Tombol lain: "Rekam" (video MP4 isi jendela), "Buka folder hasil", "Keluar".

Mode: DETEKSI bila runs/pretrained/yoloe-26n-seg-daun.pt ada (lihat 11_siapkan_detektor_daun.py),
selain itu KLASIFIKASI (kotak persegi tengah area yang dipilih yang dinilai -> pilih pas 1 daun).

Output: data/uji_internet/hasil_live/<tanggal_jam>/  NN_asli.png, NN_hasil.png, hasil.csv, ringkasan.md, rekaman_XX.mp4

Contoh:
  python scripts/12_deteksi_layar_live.py
  python scripts/12_deteksi_layar_live.py --skala 0.35          (jendela lebih kecil)
  python scripts/12_deteksi_layar_live.py --monitor 2           (tampilkan monitor ke-2)
"""
import argparse
import ctypes
import datetime as dt
import importlib
import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import mss
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

from common import ROOT
from data_utils import center_224
from models_io import load_classifier

L = importlib.import_module("10_uji_layar")  # Detector, square_box, draw_left, draw_panel, NAMA, SHORT
TIDAK_TAHU = "(tidak tahu)"


def exclude_tk(win: tk.Misc) -> bool:
    """Jendela tidak ikut tertangkap screen capture (WDA_EXCLUDEFROMCAPTURE)."""
    try:
        hwnd = ctypes.windll.user32.GetAncestor(win.winfo_id(), 2)  # GA_ROOT
        return bool(ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11))
    except Exception:  # noqa: BLE001
        return False


def info_panel(h: int, w: int, lines: list[tuple[str, tuple, float]]) -> np.ndarray:
    p = np.full((h, w, 3), 28, np.uint8)
    y = 30
    for text, col, fs in lines:
        cv2.putText(p, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2 if fs >= 0.6 else 1)
        y += int(40 * fs) + 8
    return p


class App:
    def __init__(self, args):
        self.args = args
        self.sct = mss.MSS()
        self.mon = self.sct.monitors[args.monitor]
        self.scale = args.skala
        self.pw, self.ph = int(self.mon["width"] * self.scale), int(self.mon["height"] * self.scale)
        self.panel_w = int(L.PANEL_W * self.ph / L.H)

        self.root = tk.Tk()
        self.root.title("Deteksi Cabai — LIVE")
        self.root.attributes("-topmost", args.topmost)
        self.root.resizable(False, False)
        self.canvas = tk.Canvas(self.root, width=self.pw + self.panel_w, height=self.ph, highlightthickness=0,
                                cursor="cross", bg="black")
        self.canvas.pack()
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)
        self.btn_lagi = ttk.Button(bar, text="Cari target lagi  (Spasi)", command=self.to_live, state="disabled")
        self.btn_lagi.pack(side="left")
        ttk.Label(bar, text="  Label sebenarnya:").pack(side="left")
        self.label_var = tk.StringVar(value=TIDAK_TAHU)
        self.cb = ttk.Combobox(bar, textvariable=self.label_var, state="disabled", width=22)
        self.cb.pack(side="left", padx=4)
        self.cb.bind("<<ComboboxSelected>>", lambda e: self.set_label())
        self.btn_rec = ttk.Button(bar, text="● Rekam", command=self.toggle_rec)
        self.btn_rec.pack(side="left", padx=(12, 0))
        ttk.Button(bar, text="Buka folder hasil", command=lambda: os.startfile(self.out)).pack(side="left", padx=6)
        ttk.Button(bar, text="Keluar", command=self.quit).pack(side="left")
        self.status = tk.StringVar(value="Memuat model ...")
        ttk.Label(self.root, textvariable=self.status, foreground="#444").pack(anchor="w", padx=10, pady=(0, 6))

        self.canvas.bind("<ButtonPress-1>", self.down)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.up)
        self.root.bind("<space>", lambda e: self.to_live())
        self.root.bind("<Escape>", lambda e: self.to_live())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self.state, self.start, self.sel = "memuat", None, None
        self.full = self.frozen = self.view = None
        self.n, self.rows = 0, []
        self.writer, self.n_written, self.t_rec0, self.clips = None, 0, 0.0, []
        self.out = (args.out if args.out.is_absolute() else ROOT / args.out) / f"{dt.datetime.now():%Y%m%d_%H%M%S}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.root.after(50, self.after_map)

    # ---------- persiapan ----------
    def after_map(self):
        self.excluded = exclude_tk(self.root)
        self.root.after(50, self.load)

    def load(self):
        a = self.args
        det_path = a.detektor if a.detektor.is_absolute() else ROOT / a.detektor
        self.mode = a.mode if a.mode != "auto" else ("deteksi" if det_path.exists() else "klasifikasi")
        if self.mode == "deteksi" and not det_path.exists():
            self.status.set(f"Detektor tidak ditemukan: {det_path} — jalankan 11_siapkan_detektor_daun.py")
            return
        self.det = L.Detector(det_path, a.det_conf, 0 if a.device == "cuda" else "cpu", a.max_daun) \
            if self.mode == "deteksi" else None
        self.clfs = [load_classifier(m, device=a.device) for m in a.models]
        self.classes = self.clfs[0].classes
        self.names = [c.name for c in self.clfs]
        self.cb.config(values=[TIDAK_TAHU] + list(self.classes))
        self.infer(np.full((256, 256, 3), 255, np.uint8))  # pemanasan
        self.live_panel = info_panel(self.ph, self.panel_w, [
            ("LIVE", (0, 220, 0), 0.9),
            (f"Mode: {self.mode.upper()}", (0, 255, 255), 0.6),
            ("1. Cari gambar daun di browser", (230, 230, 230), 0.5),
            ("2. Drag kotak di tampilan kiri", (230, 230, 230), 0.5),
            ("   di sekitar daun target", (230, 230, 230), 0.5),
            ("3. Hasil keluar otomatis", (230, 230, 230), 0.5),
            ("4. Spasi / tombol: cari lagi", (230, 230, 230), 0.5),
            ("", (0, 0, 0), 0.4),
            (("Kotak per daun aktif." if self.mode == "deteksi" else "Belum ada detektor daun:"), (180, 180, 180), 0.45),
            (("" if self.mode == "deteksi" else "pilih kotak pas di SATU daun."), (180, 180, 180), 0.45),
            ("", (0, 0, 0), 0.4),
            (f"Jendela ini tidak ikut tertangkap: {'ya' if self.excluded else 'TIDAK'}", (150, 150, 150), 0.42),
        ])
        self.to_live()
        self.tick()

    # ---------- loop tampilan ----------
    def tick(self):
        t0 = time.perf_counter()
        if self.state == "live":
            raw = self.sct.grab(self.mon)
            self.full = cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2BGR)
            self.view = np.hstack([cv2.resize(self.full, (self.pw, self.ph), interpolation=cv2.INTER_AREA),
                                   self.live_panel])
        self.render()
        dt_ms = (time.perf_counter() - t0) * 1000
        self.root.after(max(15, int(1000 / self.args.fps - dt_ms)), self.tick)

    def render(self):
        img = self.view.copy()
        if self.state == "memilih" and self.sel:
            a, b, c, d = self.sel
            cv2.rectangle(img, (a, b), (c, d), (0, 255, 255), 2)
        if self.writer is not None:
            cv2.circle(img, (self.pw - 18, 18), 8, (0, 0, 255), -1)
            now = time.perf_counter()
            w, h = self._rec_size
            while self.n_written <= (now - self.t_rec0) * self.args.fps:  # durasi video = durasi nyata
                self.writer.write(np.ascontiguousarray(img[:h, :w]))
                self.n_written += 1
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

    # ---------- interaksi ----------
    def down(self, e):
        if self.state != "live" or e.x >= self.pw:
            return
        self.frozen = self.full.copy()
        self.view = np.hstack([cv2.resize(self.frozen, (self.pw, self.ph), interpolation=cv2.INTER_AREA),
                               info_panel(self.ph, self.panel_w, [("MEMILIH TARGET", (0, 255, 255), 0.75),
                                                                 ("Lepas mouse = deteksi", (230, 230, 230), 0.5),
                                                                 ("Esc = batal", (230, 230, 230), 0.5)])])
        self.state, self.start, self.sel = "memilih", (e.x, e.y), None

    def drag(self, e):
        if self.state != "memilih":
            return
        x0, y0 = self.start
        x, y = min(max(e.x, 0), self.pw - 1), min(max(e.y, 0), self.ph - 1)
        self.sel = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))

    def up(self, e):
        if self.state != "memilih":
            return
        if not self.sel or self.sel[2] - self.sel[0] < 8 or self.sel[3] - self.sel[1] < 8:
            self.status.set("Kotak terlalu kecil — drag lebih besar di sekitar daun.")
            self.to_live()
            return
        a, b, c, d = (int(v / self.scale) for v in self.sel)
        self.detect((a, b, c, d))

    def to_live(self):
        if self.state == "memuat" and not hasattr(self, "live_panel"):
            return
        self.state, self.sel = "live", None
        self.btn_lagi.config(state="disabled")
        self.cb.config(state="disabled")
        self.status.set(f"LIVE — cari gambar daun lalu drag kotak di tampilan. Hasil disimpan di "
                        f"{self.out.relative_to(ROOT) if self.out.is_relative_to(ROOT) else self.out}")

    # ---------- inferensi ----------
    def infer(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        if self.mode == "deteksi":
            boxes = self.det(bgr)
            crops = [center_224(rgb[y1:y2, x1:x2]) for x1, y1, x2, y2 in
                     (L.square_box(*bx[:4], w, h) for bx in boxes)]
            pm = [m.predict_proba(crops) if crops else np.zeros((0, len(self.classes))) for m in self.clfs]
            leaves = [(boxes[j], [p[j] for p in pm]) for j in range(len(boxes))]
        else:
            boxes, leaves = [], [(None, [m.predict_proba([center_224(rgb)])[0] for m in self.clfs])]
        return boxes, leaves

    def detect(self, box):
        a, b, c, d = box
        crop = self.frozen[b:d, a:c].copy()
        boxes, leaves = self.infer(crop)
        ann = L.draw_left(crop, self.mode, boxes, leaves, self.classes)  # tinggi = L.H
        aspek = max(crop.shape[:2]) / max(1, min(crop.shape[:2])) > 1.4
        panel = L.draw_panel(self.mode, self.names, self.classes, leaves, None, False, 0, aspek,
                             bantuan="Spasi / tombol = cari target lagi", tip_area="pilih ulang pas 1 daun")
        # tampilan: frame beku + area terpilih diganti hasil anotasi
        shown = (self.frozen * 0.45).astype(np.uint8)
        shown[b:d, a:c] = cv2.resize(ann, (c - a, d - b), interpolation=cv2.INTER_AREA)
        cv2.rectangle(shown, (a, b), (c, d), (0, 255, 255), max(2, int(2 / self.scale)))
        self.view = np.hstack([cv2.resize(shown, (self.pw, self.ph), interpolation=cv2.INTER_AREA),
                               cv2.resize(panel, (self.panel_w, self.ph), interpolation=cv2.INTER_AREA)])
        self.state = "hasil"
        self.save(crop, np.hstack([ann, panel]), leaves, crop.shape)
        self.label_var.set(TIDAK_TAHU)
        self.cb.config(state="readonly")
        self.btn_lagi.config(state="normal")
        n_daun = len(leaves) if self.mode == "deteksi" else 1
        self.status.set(f"HASIL #{self.n:02d}" + (f" — {n_daun} daun terdeteksi" if self.mode == "deteksi" else "")
                        + ". Pilih label sebenarnya (opsional), lalu Spasi / 'Cari target lagi'.")

    # ---------- simpan ----------
    def save(self, crop, result_img, leaves, shape):
        self.n += 1
        cv2.imwrite(str(self.out / f"{self.n:02d}_asli.png"), crop)
        cv2.imwrite(str(self.out / f"{self.n:02d}_hasil.png"), result_img)
        h, w = shape[:2]
        for j, (bx, probs) in enumerate(leaves or [(None, None)]):
            r = {"no": self.n, "waktu": f"{dt.datetime.now():%H:%M:%S}", "mode": self.mode, "lebar": w, "tinggi": h,
                 "daun_id": j + 1 if bx else 0, "label_sebenarnya": ""}
            if bx:
                r.update(x1=bx[0], y1=bx[1], x2=bx[2], y2=bx[3], det_conf=round(bx[4], 3))
            if probs is not None:
                for nm, pr in zip(self.names, probs):
                    r[f"pred_{nm}"], r[f"conf_{nm}"] = self.classes[int(pr.argmax())], round(float(pr.max()), 4)
            self.rows.append(r)
        self.write_tables()

    def set_label(self):
        lab = "" if self.label_var.get() == TIDAK_TAHU else self.label_var.get()
        for r in self.rows:
            if r["no"] == self.n:
                r["label_sebenarnya"] = lab
        self.write_tables()
        self.status.set(f"Label #{self.n:02d} = {lab or 'tidak tahu'}. Spasi / 'Cari target lagi' untuk lanjut.")

    def write_tables(self):
        df = pd.DataFrame(self.rows)
        df.to_csv(self.out / "hasil.csv", index=False)
        Lm = [f"# Deteksi layar live — {self.out.name}", "",
              f"- Mode: **{self.mode}**; jumlah target: {self.n}; video: {', '.join(self.clips) or '-'}",
              "- Sanity check pada gambar layar, bukan evaluasi resmi. Akurasi hanya untuk target yang diberi label.", ""]
        col = f"pred_{self.names[0]}"
        d = df[df[col].notna()] if col in df else df.iloc[0:0]
        if len(d):
            if len(self.names) == 2:
                agree = (d[col] == d[f"pred_{self.names[1]}"]).mean() * 100
                Lm.append(f"- Kedua model sepakat: **{agree:.1f}%** ({len(d)} {'daun' if self.mode == 'deteksi' else 'target'})")
            for nm in self.names:
                Lm.append(f"- **{L.NAMA.get(nm, nm)}**: " + ", ".join(
                    f"{L.SHORT.get(k, k)} {v}" for k, v in d[f"pred_{nm}"].value_counts().items()))
            lab = d[d.label_sebenarnya != ""]
            if len(lab):
                Lm += ["", f"## Akurasi pada target berlabel (n={len(lab)})", ""]
                for nm in self.names:
                    ok = int((lab[f"pred_{nm}"] == lab.label_sebenarnya).sum())
                    Lm.append(f"- {L.NAMA.get(nm, nm)}: {ok / len(lab) * 100:.1f}% ({ok}/{len(lab)})")
                if self.mode == "deteksi":
                    Lm.append("- Catatan: label target diterapkan ke SEMUA daun yang terdeteksi pada target itu.")
        (self.out / "ringkasan.md").write_text("\n".join(Lm) + "\n", encoding="utf-8")

    # ---------- rekam ----------
    def toggle_rec(self):
        if self.writer is None:
            p = self.out / f"rekaman_{len(self.clips) + 1:02d}.mp4"
            w = (self.pw + self.panel_w) // 2 * 2
            self.writer = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), self.args.fps, (w, self.ph // 2 * 2))
            self.clips.append(p.name)
            self.n_written, self.t_rec0 = 0, time.perf_counter()
            self.btn_rec.config(text="■ Stop rekam")
            self._rec_size = (w, self.ph // 2 * 2)
        else:
            self.writer.release()
            self.writer = None
            self.btn_rec.config(text="● Rekam")
            self.write_tables()

    def quit(self):
        if self.writer is not None:
            self.writer.release()
        if self.rows:
            self.write_tables()
        self.sct.close()
        self.root.destroy()
        print(f"Hasil tersimpan di: {self.out}")

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["mobilenetv3_large_100_augB", "yolo11n_cls"])
    ap.add_argument("--mode", choices=["auto", "deteksi", "klasifikasi"], default="auto")
    ap.add_argument("--detektor", type=Path, default=L.DETEKTOR_DEFAULT)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--max-daun", type=int, default=20)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--monitor", type=int, default=1, help="monitor yang ditampilkan live (1 = utama)")
    ap.add_argument("--skala", type=float, default=0.45, help="ukuran tampilan live relatif terhadap layar")
    ap.add_argument("--fps", type=float, default=10)
    ap.add_argument("--topmost", action="store_true", help="jendela selalu di atas")
    ap.add_argument("--out", type=Path, default=ROOT / "data/uji_internet/hasil_live")
    args = ap.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Script ini khusus Windows (pengecualian jendela dari tangkapan layar).")
    L.dpi_aware()
    App(args).run()


if __name__ == "__main__":
    main()
