"""Tahap 0: laporan spesifikasi sistem, versi library, dan status CUDA."""
import argparse
import importlib
import json
import platform
import subprocess
import time
from pathlib import Path

LIBS = [
    "torch", "torchvision", "ultralytics", "timm", "albumentations", "sklearn",
    "imagehash", "PIL", "pandas", "numpy", "matplotlib", "seaborn", "pytorch_grad_cam",
    "onnx", "onnxruntime", "cv2", "yaml",
]


def lib_versions():
    out = {}
    for name in LIBS:
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[name] = f"TIDAK TERPASANG ({type(e).__name__})"
    return out


def cuda_report():
    import torch

    rep = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version()}
    if not rep["cuda_available"]:
        return rep
    p = torch.cuda.get_device_properties(0)
    rep.update(gpu=p.name, vram_gb=round(p.total_memory / 1024**3, 2),
               compute_capability=f"{p.major}.{p.minor}")
    # tensor test: matmul GPU vs CPU
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    rep["matmul_2048_x10_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    rep["gpu_vs_cpu_max_abs_diff"] = float((c.cpu() - a.cpu() @ b.cpu()).abs().max())
    with torch.autocast("cuda", dtype=torch.float16):
        rep["amp_fp16_dtype"] = str((a @ b).dtype)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None, help="simpan laporan JSON")
    args = ap.parse_args()

    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        smi = "nvidia-smi tidak ditemukan"
    rep = {"os": platform.platform(), "python": platform.python_version(),
           "cpu": platform.processor(), "nvidia_smi": smi,
           "libs": lib_versions(), "cuda": cuda_report()}
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
