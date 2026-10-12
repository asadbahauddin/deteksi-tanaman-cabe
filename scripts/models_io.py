"""Memuat model terlatih (timm atau Ultralytics) dengan antarmuka seragam: RGB uint8 HxWx3 (sudah 224) -> probabilitas."""
from pathlib import Path

import cv2
import numpy as np
import torch

from common import ROOT


class TimmClassifier:
    kind = "timm"

    def __init__(self, run_dir: Path, device="cuda"):
        import timm

        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        self.cfg = ckpt["config"]
        self.classes = self.cfg["classes"]
        self.model = timm.create_model(self.cfg["model"], pretrained=False, num_classes=len(self.classes))
        self.model.load_state_dict(ckpt["model"])
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.eval().to(self.device)
        self.mean = np.array(self.cfg["mean"], dtype=np.float32) * 255
        self.std = np.array(self.cfg["std"], dtype=np.float32) * 255
        self.size = self.cfg["img_size"]
        self.name = run_dir.name

    def to_tensor(self, imgs: list[np.ndarray]) -> torch.Tensor:
        x = (np.stack(imgs).astype(np.float32) - self.mean) / self.std
        return torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def predict_proba(self, imgs: list[np.ndarray]) -> np.ndarray:
        return self.model(self.to_tensor(imgs).to(self.device)).float().softmax(1).cpu().numpy()

    def cam_target(self):
        """(model -> logit, layer target Grad-CAM). Dipilih di val: EfficientNet=bn2, MobileNetV3=blocks[-1]."""
        layer = self.model.bn2 if "efficientnet" in self.cfg["model"] else self.model.blocks[-1]
        return self.model, layer


class _YoloLogits(torch.nn.Module):
    """Head Classify Ultralytics saat eval mengembalikan (probabilitas, logit); Grad-CAM butuh logit."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        y = self.net(x)
        return y[1] if isinstance(y, (tuple, list)) else y


class YoloClassifier:
    kind = "yolo"

    def __init__(self, run_dir: Path, device="cuda"):
        from ultralytics import YOLO

        self.model = YOLO(run_dir / "weights/best.pt")
        names = self.model.names
        self.classes = [names[i] for i in range(len(names))]
        self.device = 0 if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.size = 224
        self.name = run_dir.name

    def to_tensor(self, imgs: list[np.ndarray]) -> torch.Tensor:
        """Preprocessing YOLO-cls: RGB / 255 (tanpa normalisasi mean/std)."""
        return torch.from_numpy(np.stack(imgs).astype(np.float32) / 255).permute(0, 3, 1, 2).contiguous()

    def cam_target(self):
        """(model -> logit, layer target Grad-CAM) = konvolusi 1x1 terakhir di head Classify (fitur spasial 7x7)."""
        dev = "cuda" if (self.device == 0 and torch.cuda.is_available()) else "cpu"
        net = self.model.model.float().eval().to(dev)
        for p_ in net.parameters():
            p_.requires_grad_(True)
        return _YoloLogits(net), net.model[-1].conv

    def predict_proba(self, imgs: list[np.ndarray]) -> np.ndarray:
        # Ultralytics menerima BGR; input sudah 224x224 sehingga resize/crop internal = identitas
        bgr = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in imgs]
        if getattr(self, "per_image", False):  # ekspor ONNX/NCNN Ultralytics: batch statis 1
            res = [self.model.predict(b, imgsz=self.size, device=self.device, verbose=False)[0] for b in bgr]
        else:
            res = self.model.predict(bgr, imgsz=self.size, device=self.device, verbose=False)
        return np.stack([r.probs.data.float().cpu().numpy() for r in res])


class OnnxTimmClassifier:
    """Model timm hasil 07_export.py; kelas, mean, std dibaca dari metadata ONNX."""
    kind = "onnx-timm"

    def __init__(self, path: Path, threads: int = 0):
        import json

        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        meta = self.sess.get_modelmeta().custom_metadata_map
        self.classes = json.loads(meta["classes"])
        self.mean = np.array(json.loads(meta["mean"]), dtype=np.float32) * 255
        self.std = np.array(json.loads(meta["std"]), dtype=np.float32) * 255
        self.size = int(meta["img_size"])
        self.input_name = self.sess.get_inputs()[0].name
        self.name = path.stem

    def to_array(self, imgs: list[np.ndarray]) -> np.ndarray:
        x = (np.stack(imgs).astype(np.float32) - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def predict_proba(self, imgs: list[np.ndarray]) -> np.ndarray:
        logits = self.sess.run(None, {self.input_name: self.to_array(imgs)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)


def load_classifier(run: str, device="cuda"):
    """`run`: nama folder di runs/, path .onnx hasil 07_export.py (timm), atau path ekspor YOLO (.onnx / *_ncnn_model)."""
    p = Path(run)
    if p.suffix == ".onnx" or p.name.endswith("_ncnn_model"):
        p = p if p.is_absolute() else ROOT / p
        if p.suffix == ".onnx":
            import onnx

            meta = {m.key for m in onnx.load(str(p), load_external_data=False).metadata_props}
            if "classes" in meta:
                return OnnxTimmClassifier(p)
        clf = YoloClassifier.__new__(YoloClassifier)  # ekspor Ultralytics (ONNX/NCNN), dijalankan di CPU
        from ultralytics import YOLO

        clf.model, clf.size, clf.device, clf.name = YOLO(str(p), task="classify"), 224, "cpu", p.name
        clf.per_image = True
        # predictor dibangun di CPU dulu; mengakses .names lebih awal membuat predictor dengan device default (CUDA)
        clf.model.predict(np.zeros((224, 224, 3), np.uint8), imgsz=224, device="cpu", verbose=False)
        clf.classes = [clf.model.names[i] for i in range(len(clf.model.names))]
        return clf
    run_dir = p if p.is_absolute() else ROOT / "runs" / run
    if (run_dir / "weights/best.pt").exists():
        return YoloClassifier(run_dir, device)
    if (run_dir / "best.pt").exists():
        return TimmClassifier(run_dir, device)
    raise FileNotFoundError(f"Tidak ada checkpoint di {run_dir}")
