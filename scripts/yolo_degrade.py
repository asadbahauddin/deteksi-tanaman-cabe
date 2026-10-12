"""Augmentasi degradasi (setara augmentasi B timm) untuk training YOLO-cls Ultralytics.

Gambar train dikecilkan dulu ke sisi terpendek `work_size` (default 256), lalu diberi degradasi
(downscale 48-128 px -> upscale, blur, JPEG q40-90, noise) dengan peluang sama seperti configs/timm.yaml aug B,
kemudian diteruskan ke augmentasi bawaan Ultralytics (random resized crop 224, flip, HSV, RandAugment, erasing).
Modul terpisah agar dapat di-import worker DataLoader di Windows (spawn).
"""
import albumentations as A
import cv2
import numpy as np
import ultralytics.data.utils as ul_data_utils
from PIL import Image
from ultralytics.data.dataset import ClassificationDataset

from data_utils import degrade_ops

# Jangan biarkan Ultralytics menulis ulang JPEG "corrupt" (mengganti symlink di data/split); lihat 03a.
ul_data_utils._replace_image = lambda *a, **k: None


class DegradeClassificationDataset(ClassificationDataset):
    def __init__(self, *args, degrade_cfg: dict, work_size: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        self.work_size = work_size
        self.degrade = A.Compose(degrade_ops(degrade_cfg, 224))

    def __getitem__(self, i: int) -> dict:
        f, j, _, _ = self.samples[i]
        im = self.imread(f)  # BGR
        h, w = im.shape[:2]
        s = self.work_size / min(h, w)
        if s < 1:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rgb = self.degrade(image=np.ascontiguousarray(rgb))["image"]
        return {"img": self.torch_transforms(Image.fromarray(rgb)), "cls": j}
