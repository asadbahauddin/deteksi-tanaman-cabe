# Benchmark latensi CPU (proxy Pi 5)

CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel | threads=4 | warm-up 20, 100 run | torch 2.14.1+cu130, onnxruntime 1.30.0, ncnn 1.0.20260526

Forward pass murni batch 1 (1x3x224x224), tanpa decode/preprocessing gambar.
**Angka laptop x86 ini bukan latensi Pi 5**; gunakan hanya untuk perbandingan relatif antar backend. Ukur ulang di Pi 5.

| model | backend | mean_ms | p50_ms | p95_ms | std_ms | FPS_mean |
|---|---|---|---|---|---|---|
| mobilenetv3_large_100_augB | PyTorch CPU fp32 | 11.47 | 11.46 | 13.81 | 1.31 | 87.17 |
| mobilenetv3_large_100_augB | ONNX Runtime CPU | 2.80 | 2.81 | 3.38 | 0.34 | 357.73 |
| yolo11n_cls | PyTorch CPU fp32 | 6.25 | 6.28 | 7.76 | 0.84 | 160.01 |
| yolo11n_cls | ONNX Runtime CPU | 2.08 | 2.02 | 2.66 | 0.32 | 481.75 |
| yolo11n_cls | NCNN CPU (fp32) | 3.56 | 3.54 | 4.11 | 0.35 | 280.84 |
