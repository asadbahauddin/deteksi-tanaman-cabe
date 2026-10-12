"""Tahap 5: benchmark latensi CPU (proxy kasar untuk Raspberry Pi 5).

Mengukur forward pass murni, batch 1, input 1x3x224x224 (tanpa decode/preprocessing gambar):
  - PyTorch CPU (fp32)  vs  ONNX Runtime CPU  untuk model timm dan YOLO
  - NCNN (pyncnn) untuk YOLO
Thread dibatasi (default 4, sama dengan jumlah core Cortex-A76 di Pi 5). Warm-up lalu N run; dilaporkan mean, p50, p95.
Catatan: CPU laptop (x86) jauh berbeda dari Pi 5 (ARM); angka ini hanya untuk perbandingan relatif antar backend.
"""
import argparse
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

from common import ROOT, SEED, md_table, seed_everything


def bench(fn, warmup: int, runs: int) -> dict:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return {"mean_ms": t.mean(), "p50_ms": np.percentile(t, 50), "p95_ms": np.percentile(t, 95), "std_ms": t.std(),
            "FPS_mean": 1000 / t.mean()}


def ort_session(path: Path, threads: int):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timm-run", default="mobilenetv3_large_100_augB")
    ap.add_argument("--yolo-run", default="yolo11n_cls")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    seed_everything(SEED)
    torch.set_num_threads(args.threads)
    x = np.random.RandomState(SEED).rand(1, 3, 224, 224).astype(np.float32)
    xt = torch.from_numpy(x)
    rows = []

    def add(model, backend, fn):
        r = bench(fn, args.warmup, args.runs)
        rows.append({"model": model, "backend": backend, **r})
        print(f"{model:28s} {backend:22s} mean {r['mean_ms']:7.2f} ms | p95 {r['p95_ms']:7.2f} ms", flush=True)

    # --- timm ---
    from models_io import TimmClassifier

    tm = TimmClassifier(ROOT / "runs" / args.timm_run, device="cpu").model.eval()
    with torch.inference_mode():
        add(args.timm_run, "PyTorch CPU fp32", lambda: tm(xt))
    s = ort_session(ROOT / "runs" / args.timm_run / "export" / f"{args.timm_run}.onnx", args.threads)
    add(args.timm_run, "ONNX Runtime CPU", lambda: s.run(None, {"input": x}))

    # --- YOLO ---
    from ultralytics import YOLO

    ym = YOLO(ROOT / "runs" / args.yolo_run / "weights/best.pt").model.float().eval().cpu()
    with torch.inference_mode():
        add(args.yolo_run, "PyTorch CPU fp32", lambda: ym(xt))
    exp = ROOT / "runs" / args.yolo_run / "export"
    sy = ort_session(exp / "best.onnx", args.threads)
    add(args.yolo_run, "ONNX Runtime CPU", lambda: sy.run(None, {sy.get_inputs()[0].name: x}))
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = args.threads
    net.opt.use_vulkan_compute = False
    net.load_param(str(exp / "best_ncnn_model/model.ncnn.param"))
    net.load_model(str(exp / "best_ncnn_model/model.ncnn.bin"))
    mat = ncnn.Mat(x[0]).clone()

    def ncnn_run():
        with net.create_extractor() as ex:
            ex.input("in0", mat)
            ex.extract("out0")

    add(args.yolo_run, "NCNN CPU (fp32)", ncnn_run)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reports/benchmark_cpu.csv", index=False)
    info = (f"CPU: {platform.processor()} | threads={args.threads} | warm-up {args.warmup}, {args.runs} run | "
            f"torch {torch.__version__}, onnxruntime {ort.__version__}, ncnn {ncnn.__version__}")
    txt = ("# Benchmark latensi CPU (proxy Pi 5)\n\n" + info + "\n\n"
           "Forward pass murni batch 1 (1x3x224x224), tanpa decode/preprocessing gambar.\n"
           "**Angka laptop x86 ini bukan latensi Pi 5**; gunakan hanya untuk perbandingan relatif antar backend. "
           "Ukur ulang di Pi 5.\n\n" + md_table(df, "{:.2f}") + "\n")
    (ROOT / "reports/benchmark_cpu.md").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
