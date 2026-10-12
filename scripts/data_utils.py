"""Dataset & transformasi bersama (training timm, evaluasi, inferensi)."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from common import ROOT


def load_rgb(path, draft: int | None = 512) -> np.ndarray:
    """Baca gambar sebagai RGB uint8. Alpha dibuang (piksel di bawahnya putih, lihat audit).
    Untuk JPEG besar, `draft` memakai DCT scaling (decode jauh lebih cepat, tetap >= draft px)."""
    with Image.open(path) as im:
        if draft and im.format == "JPEG":
            im.draft("RGB", (draft, draft))
        return np.asarray(im.convert("RGB"))


def center_224(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """Preprocessing evaluasi/inferensi: resize sisi terpendek ke `size` (INTER_AREA) lalu center crop."""
    h, w = rgb.shape[:2]
    s = size / min(h, w)
    im = cv2.resize(rgb, (max(size, round(w * s)), max(size, round(h * s))), interpolation=cv2.INTER_AREA)
    h, w = im.shape[:2]
    y, x = (h - size) // 2, (w - size) // 2
    return np.ascontiguousarray(im[y:y + size, x:x + size])


def classes_from_csv(csv: Path) -> list[str]:
    return sorted(pd.read_csv(csv)["label"].unique())


def degrade_ops(d: dict, size: int) -> list:
    return [
        A.Downscale(scale_range=(d["downscale_px"][0] / size, d["downscale_px"][1] / size),
                    interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR},
                    p=d["downscale_p"]),
        A.OneOf([A.GaussianBlur(blur_range=(3, 5)), A.MotionBlur(blur_range=(3, 7))], p=d["blur_p"]),
        A.ImageCompression(quality_range=tuple(d["jpeg_quality"]), p=d["jpeg_p"]),
        A.GaussNoise(std_range=tuple(d["noise_std"]), p=d["noise_p"]),
    ]


def build_train_tf(aug: dict, size: int, mean, std) -> A.Compose:
    cj = aug["color_jitter"]
    ops = [
        A.RandomResizedCrop(size=(size, size), scale=tuple(aug["rrc_scale"]), interpolation=cv2.INTER_AREA),
        A.HorizontalFlip(p=aug["hflip"]),
        A.VerticalFlip(p=aug["vflip"]),
        A.Rotate(angle_range=(-aug["rotate_deg"], aug["rotate_deg"]), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ColorJitter(brightness_range=(1 - cj["brightness"], 1 + cj["brightness"]),
                      contrast_range=(1 - cj["contrast"], 1 + cj["contrast"]),
                      saturation_range=(1 - cj["saturation"], 1 + cj["saturation"]),
                      hue_range=(-cj["hue"], cj["hue"]), p=cj["p"]),
    ]
    if "degrade" in aug:
        ops += degrade_ops(aug["degrade"], size)
    return A.Compose(ops + [A.Normalize(mean=mean, std=std), ToTensorV2()])


def build_eval_tf(size: int, mean, std, degrade_px: int | None = None) -> A.Compose:
    """Resize sisi terpendek -> center crop (sama seperti transform val Ultralytics).
    degrade_px: jika diisi, gambar diturunkan ke degrade_px lalu dinaikkan lagi (uji robustness)."""
    ops = [A.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA), A.CenterCrop(size, size)]
    if degrade_px:
        ops += [A.Resize(degrade_px, degrade_px, interpolation=cv2.INTER_AREA),
                A.Resize(size, size, interpolation=cv2.INTER_LINEAR)]
    return A.Compose(ops + [A.Normalize(mean=mean, std=std), ToTensorV2()])


class CsvDataset(torch.utils.data.Dataset):
    def __init__(self, csv: Path, classes: list[str], tf, root: Path = ROOT):
        self.df = pd.read_csv(csv)
        self.paths = [root / p for p in self.df["path"]]
        self.idx = {c: i for i, c in enumerate(classes)}
        self.labels = [self.idx[c] for c in self.df["label"]]
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(image=load_rgb(self.paths[i]))["image"]
        return img, self.labels[i]
