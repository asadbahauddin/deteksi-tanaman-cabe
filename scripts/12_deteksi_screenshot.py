"""Deteksi dari screenshot sesuai kebutuhan (seperti Snipping Tool).

Alur:
  1. Jalankan script -> muncul toolbar kecil "Deteksi Cabai" yang selalu di atas.
  2. Jelajahi internet seperti biasa. Saat menemukan gambar daun: klik "Ambil & Deteksi" atau tekan F8
     (berlaku dari aplikasi mana pun).
  3. Layar dibekukan -> drag kotak di sekitar daun (Esc = batal).
  4. Jendela hasil: kotak per daun (bila detektor sudah disiapkan) + prediksi 2 model.
     Opsional: pilih "Label sebenarnya" agar akurasi tercatat. Klik "Deteksi lagi" untuk target berikutnya.

Mode: DETEKSI bila runs/pretrained/yoloe-26n-seg-daun.pt ada (lihat 11_siapkan_detektor_daun.py),
selain itu KLASIFIKASI (kotak persegi tengah area yang dinilai -> pilih area pas di 1 daun).

Output: data/uji_internet/hasil_screenshot/<tanggal_jam>/
  NN_asli.png (area yang dipilih), NN_hasil.png (dengan kotak & prediksi), hasil.csv, ringkasan.md

Contoh:
  python scripts/12_deteksi_screenshot.py
  python scripts/12_deteksi_screenshot.py --hotkey F9 --mode klasifikasi
"""
import argparse
import ctypes
import ctypes.wintypes
import datetime as dt
import importlib
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import mss
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageTk

from common import ROOT
from data_utils import center_224
from models_io import load_classifier

L = importlib.import_module("10_uji_layar")  # pakai ulang Detector, square_box, draw_left, draw_panel
TIDAK_TAHU = "(tidak tahu)"


def hotkey_thread(vk: int, q: queue.Queue, tid_box: list):
    """Daftarkan hotkey global Windows; kirim event ke queue saat ditekan."""
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    tid_box.append(kernel32.GetCurrentThreadId())
    if not user32.RegisterHotKey(None, 1, 0x4000, vk):  # MOD_NOREPEAT
        q.put("hotkey_gagal")
        return
    msg = ctypes.wintypes.MSG()
    while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
        if msg.message == 0x0312:  # WM_HOTKEY
            q.put("ambil")
    user32.UnregisterHotKey(None, 1)


def exclude_tk(win: tk.Misc):
    """Jendela Tk tidak ikut tertangkap screenshot (WDA_EXCLUDEFROMCAPTURE)."""
    try:
        hwnd = ctypes.windll.user32.GetAncestor(win.winfo_id(), 2)  # GA_ROOT
        ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)
    except Exception:  # noqa: BLE001
        pass


class Selector(tk.Toplevel):
    """Overlay layar penuh berisi screenshot beku; drag untuk memilih area."""

    def __init__(self, root, shot: Image.Image, mon: dict, on_done):
        super().__init__(root)
        self.on_done, self.shot, self.start, self.rect = on_done, shot, None, None
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.geometry(f"{mon['width']}x{mon['height']}+{mon['left']}+{mon['top']}")
        self.bg = ImageTk.PhotoImage(ImageEnhance.Brightness(shot).enhance(0.55))
        self.bright = None
        self.c = tk.Canvas(self, width=mon["width"], height=mon["height"], highlightthickness=0, cursor="cross")
        self.c.pack()
        self.c.create_image(0, 0, image=self.bg, anchor="nw")
        self.c.create_rectangle(0, 0, mon["width"], 44, fill="black", outline="")
        self.c.create_text(16, 22, anchor="w", fill="#ffd400", font=("Segoe UI", 14, "bold"),
                           text="Drag kotak di sekitar daun (lepas mouse = deteksi).  Esc = batal")
        self.c.bind("<ButtonPress-1>", self.down)
        self.c.bind("<B1-Motion>", self.drag)
        self.c.bind("<ButtonRelease-1>", self.up)
        self.bind("<Escape>", lambda e: self.finish(None))
        self.focus_force()

    def down(self, e):
        self.start = (e.x, e.y)

    def drag(self, e):
        if not self.start:
            return
        x0, y0 = self.start
        a, b, c, d = min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y)
        self.c.delete("sel")
        if c - a > 2 and d - b > 2:
            self.bright = ImageTk.PhotoImage(self.shot.crop((a, b, c, d)))
            self.c.create_image(a, b, image=self.bright, anchor="nw", tags="sel")
        self.c.create_rectangle(a, b, c, d, outline="#ffd400", width=2, tags="sel")
        self.c.create_text(a + 4, max(50, b - 4), anchor="sw", fill="#ffd400", tags="sel",
                           font=("Segoe UI", 10), text=f"{c - a} x {d - b}")

    def up(self, e):
        if not self.start:
            return
        x0, y0 = self.start
        box = (min(x0, e.x), min(y0, e.y), max(x0, e.x), max(y0, e.y))
        self.finish(box if box[2] - box[0] >= 20 and box[3] - box[1] >= 20 else None)

    def finish(self, box):
        self.destroy()
        self.on_done(box)


class App:
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.title("Deteksi Cabai")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)
        sw = self.root.winfo_screenwidth()
        self.root.geometry(f"+{sw - 380}+40")
        self.status = tk.StringVar(value="Memuat model ...")
        ttk.Label(self.root, textvariable=self.status, wraplength=340, justify="left").pack(padx=10, pady=(10, 6))
        self.btn = ttk.Button(self.root, text=f"Ambil & Deteksi  ({args.hotkey})", command=self.trigger, state="disabled")
        self.btn.pack(fill="x", padx=10)
        row = ttk.Frame(self.root)
        row.pack(fill="x", padx=10, pady=(6, 10))
        ttk.Button(row, text="Buka folder hasil", command=lambda: os.startfile(self.out)).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Keluar", command=self.quit).pack(side="left", expand=True, fill="x", padx=(6, 0))
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.q, self.tid, self.busy, self.result_win = queue.Queue(), [], False, None
        self.n, self.rows = 0, []
        self.out = (args.out if args.out.is_absolute() else ROOT / args.out) / f"{dt.datetime.now():%Y%m%d_%H%M%S}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.root.after(50, lambda: exclude_tk(self.root))
        self.root.after(100, self.load)

    # ---------- model ----------
    def load(self):
        a = self.args
        det_path = a.detektor if a.detektor.is_absolute() else ROOT / a.detektor
        self.mode = a.mode if a.mode != "auto" else ("deteksi" if det_path.exists() else "klasifikasi")
        if self.mode == "deteksi" and not det_path.exists():
            self.status.set(f"Detektor tidak ditemukan: {det_path}. Jalankan 11_siapkan_detektor_daun.py")
            return
        self.det = L.Detector(det_path, a.det_conf, 0 if a.device == "cuda" else "cpu", a.max_daun) \
            if self.mode == "deteksi" else None
        self.clfs = [load_classifier(m, device=a.device) for m in a.models]
        self.classes = self.clfs[0].classes
        self.names = [c.name for c in self.clfs]
        # pemanasan agar deteksi pertama tidak lambat
        dummy = np.full((256, 256, 3), 255, np.uint8)
        self.process(dummy, simpan=False)
        vk = 0x70 + int(a.hotkey[1:]) - 1
        threading.Thread(target=hotkey_thread, args=(vk, self.q, self.tid), daemon=True).start()
        self.root.after(100, self.poll)
        info = ("DETEKSI: setiap daun diberi kotak." if self.mode == "deteksi" else
                "KLASIFIKASI: belum ada detektor daun, pilih area pas di SATU daun.")
        self.status.set(f"Siap. Mode {info}\nBuka gambar daun di layar, lalu klik tombol atau tekan {a.hotkey}.")
        self.btn.config(state="normal")

    def poll(self):
        try:
            while True:
                ev = self.q.get_nowait()
                if ev == "ambil":
                    self.trigger()
                elif ev == "hotkey_gagal":
                    self.status.set(self.status.get() + f"\n(Hotkey {self.args.hotkey} tidak bisa dipakai, gunakan tombol.)")
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    # ---------- ambil ----------
    def trigger(self):
        if self.busy:
            return
        self.busy = True
        if self.result_win is not None:
            self.close_result()
        self.root.withdraw()
        self.root.after(250, self.grab)

    def grab(self):
        with mss.MSS() as sct:
            self.mon = sct.monitors[self.args.monitor]
            raw = sct.grab(self.mon)
        self.shot = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        Selector(self.root, self.shot, self.mon, self.selected)

    def selected(self, box):
        self.root.deiconify()
        if box is None:
            self.busy = False
            self.status.set("Dibatalkan. Klik tombol / tekan hotkey untuk mencoba lagi.")
            return
        crop = cv2.cvtColor(np.asarray(self.shot.crop(box)), cv2.COLOR_RGB2BGR)
        self.status.set("Mendeteksi ...")
        self.root.update_idletasks()
        frame, rows = self.process(crop)
        self.show_result(frame, rows)
        self.busy = False

    # ---------- inferensi ----------
    def process(self, bgr, simpan=True):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = bgr.shape[:2]
        if self.mode == "deteksi":
            boxes = self.det(bgr)
            crops = [center_224(rgb[b:d, a:c]) for a, b, c, d in
                     (L.square_box(x1, y1, x2, y2, w, h) for x1, y1, x2, y2, _ in boxes)]
            pm = [m.predict_proba(crops) if crops else np.zeros((0, len(self.classes))) for m in self.clfs]
            leaves = [(boxes[j], [p[j] for p in pm]) for j in range(len(boxes))]
        else:
            boxes, leaves = [], [(None, [m.predict_proba([center_224(rgb)])[0] for m in self.clfs])]
        aspek = max(w, h) / max(1, min(w, h)) > 1.4
        frame = np.hstack([L.draw_left(bgr, self.mode, boxes, leaves, self.classes),
                           L.draw_panel(self.mode, self.names, self.classes, leaves, None, False, 0, aspek,
                                        bantuan=f"{self.args.hotkey} / tombol = deteksi lagi",
                                        tip_area="ambil ulang pas 1 daun")])
        if not simpan:
            return frame, []
        self.n += 1
        cv2.imwrite(str(self.out / f"{self.n:02d}_asli.png"), bgr)
        cv2.imwrite(str(self.out / f"{self.n:02d}_hasil.png"), frame)
        rows = []
        for j, (b, probs) in enumerate(leaves or [(None, None)]):
            r = {"no": self.n, "waktu": f"{dt.datetime.now():%H:%M:%S}", "mode": self.mode, "lebar": w, "tinggi": h,
                 "daun_id": j + 1 if (self.mode == "deteksi" and b) else 0, "label_sebenarnya": ""}
            if b:
                r.update(x1=b[0], y1=b[1], x2=b[2], y2=b[3], det_conf=round(b[4], 3))
            if probs is not None:
                for nm, pr in zip(self.names, probs):
                    r[f"pred_{nm}"], r[f"conf_{nm}"] = self.classes[int(pr.argmax())], round(float(pr.max()), 4)
            rows.append(r)
        self.rows += rows
        self.save_tables()
        return frame, rows

    # ---------- hasil ----------
    def show_result(self, frame, rows):
        win = tk.Toplevel(self.root)
        self.result_win = win
        win.title(f"Hasil #{self.n:02d} — Deteksi Cabai")
        win.attributes("-topmost", True)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        s = min(1.0, 0.9 * sw / frame.shape[1], 0.75 * sh / frame.shape[0])
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = img.resize((int(img.width * s), int(img.height * s)), Image.Resampling.LANCZOS)
        win.photo = ImageTk.PhotoImage(img)
        ttk.Label(win, image=win.photo).pack()
        bar = ttk.Frame(win)
        bar.pack(fill="x", padx=10, pady=8)
        ttk.Label(bar, text="Label sebenarnya (opsional):").pack(side="left")
        self.label_var = tk.StringVar(value=TIDAK_TAHU)
        cb = ttk.Combobox(bar, textvariable=self.label_var, state="readonly", width=24,
                          values=[TIDAK_TAHU] + list(self.classes))
        cb.pack(side="left", padx=6)
        cb.bind("<<ComboboxSelected>>", lambda e: self.set_label())
        ttk.Button(bar, text=f"Deteksi lagi ({self.args.hotkey})", command=self.trigger).pack(side="left", padx=6)
        ttk.Button(bar, text="Tutup", command=self.close_result).pack(side="left")
        hasil = self.out / f"{self.n:02d}_hasil.png"
        ttk.Label(win, text=f"Tersimpan: {hasil.relative_to(ROOT) if hasil.is_relative_to(ROOT) else hasil}",
                  foreground="#666").pack(pady=(0, 8))
        win.protocol("WM_DELETE_WINDOW", self.close_result)
        win.after(50, lambda: exclude_tk(win))
        n_daun = sum(1 for r in rows if r["daun_id"] > 0)
        self.status.set(f"Hasil #{self.n:02d}: " + (f"{n_daun} daun terdeteksi. " if self.mode == "deteksi" else "")
                        + f"Klik 'Deteksi lagi' atau tekan {self.args.hotkey} untuk target berikutnya.")

    def set_label(self):
        lab = "" if self.label_var.get() == TIDAK_TAHU else self.label_var.get()
        for r in self.rows:
            if r["no"] == self.n:
                r["label_sebenarnya"] = lab
        self.save_tables()

    def close_result(self):
        if self.result_win is not None:
            self.result_win.destroy()
            self.result_win = None

    # ---------- simpan ----------
    def save_tables(self):
        df = pd.DataFrame(self.rows)
        df.to_csv(self.out / "hasil.csv", index=False)
        Lm = [f"# Deteksi dari screenshot — {self.out.name}", "",
              f"- Mode: **{self.mode}**; jumlah pengambilan: {self.n}",
              "- Sanity check pada gambar layar, bukan evaluasi resmi. Akurasi hanya dihitung untuk pengambilan "
              "yang diberi 'Label sebenarnya'.", ""]
        d = df[df.get(f"pred_{self.names[0]}", pd.Series(dtype=str)).notna()] if len(df) else df
        if len(d):
            if len(self.names) == 2:
                Lm.append(f"- Kedua model sepakat: **{(d[f'pred_{self.names[0]}'] == d[f'pred_{self.names[1]}']).mean() * 100:.1f}%** "
                          f"({len(d)} {'daun' if self.mode == 'deteksi' else 'pengambilan'})")
            for nm in self.names:
                vc = d[f"pred_{nm}"].value_counts()
                Lm.append(f"- **{L.NAMA.get(nm, nm)}**: " + ", ".join(f"{L.SHORT.get(k, k)} {v}" for k, v in vc.items()))
            lab = d[d.label_sebenarnya != ""]
            if len(lab):
                Lm += ["", f"## Akurasi pada data berlabel (n={len(lab)})", ""]
                for nm in self.names:
                    Lm.append(f"- {L.NAMA.get(nm, nm)}: {(lab[f'pred_{nm}'] == lab.label_sebenarnya).mean() * 100:.1f}% "
                              f"({int((lab[f'pred_{nm}'] == lab.label_sebenarnya).sum())}/{len(lab)})")
                if self.mode == "deteksi":
                    Lm.append("- Catatan mode deteksi: label pengambilan diterapkan ke SEMUA daun di pengambilan itu.")
        (self.out / "ringkasan.md").write_text("\n".join(Lm) + "\n", encoding="utf-8")

    def quit(self):
        if self.tid:
            ctypes.windll.user32.PostThreadMessageW(self.tid[0], 0x0012, 0, 0)  # WM_QUIT ke thread hotkey
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
    ap.add_argument("--monitor", type=int, default=1)
    ap.add_argument("--hotkey", default="F8", choices=[f"F{i}" for i in range(1, 13)])
    ap.add_argument("--out", type=Path, default=ROOT / "data/uji_internet/hasil_screenshot")
    args = ap.parse_args()
    if sys.platform != "win32":
        raise SystemExit("Script ini khusus Windows (hotkey global & pengecualian jendela dari screenshot).")
    L.dpi_aware()
    App(args).run()


if __name__ == "__main__":
    main()
