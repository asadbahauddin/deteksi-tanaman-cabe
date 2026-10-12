"""Utilitas bersama untuk semua script tahap."""
import os
import random
import sys
from pathlib import Path

import numpy as np

# Ultralytics default mencoba `pip install` paket yang dianggap kurang (mis. onnxruntime-gpu) ke env aktif.
# Ini sempat menimpa sebagian onnxruntime 1.30.0 -> dimatikan agar env tetap sesuai requirements.txt.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

# Konsol/redirect Windows default cp1252 -> library yang mencetak emoji (torch.onnx, Ultralytics) bisa crash
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SEED = 42


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def list_images(class_dir: Path):
    """Semua file di bawah folder kelas (rekursif), termasuk yang bukan gambar."""
    return sorted(p for p in class_dir.rglob("*") if p.is_file())


def md_table(df, floatfmt: str = "{:.3f}") -> str:
    """DataFrame -> tabel markdown tanpa dependensi tabulate."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
