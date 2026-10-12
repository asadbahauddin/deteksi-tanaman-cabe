# Ekspor model

Verifikasi: 64 gambar VAL acak (seed 42), referensi = PyTorch CPU fp32. `max_abs_diff_prob` = selisih maksimum probabilitas softmax.

Catatan: checkpoint timm = state_dict fp32 + config; checkpoint YOLO disimpan Ultralytics dalam fp16 (karena itu lebih kecil dari ONNX/NCNN fp32). YOLO mengeluarkan probabilitas langsung (tanpa logit).

| model | format | path | size_MB | max_abs_diff_logit | max_abs_diff_prob | top1_agree |
|---|---|---|---|---|---|---|
| mobilenetv3_large_100_augB | ONNX (opset 18, dynamo) | runs/mobilenetv3_large_100_augB/export/mobilenetv3_large_100_augB.onnx | 16.3 | 1.0e-05 | 1.7e-06 | 64/64 |
| yolo11n_cls | ONNX (Ultralytics) | runs/yolo11n_cls/export/best.onnx | 5.9 | - | 1.5e-06 | 64/64 |
| yolo11n_cls | NCNN (Ultralytics, fp32) | runs/yolo11n_cls/export/best_ncnn_model | 5.9 | - | 5.4e-07 | 64/64 |
| mobilenetv3_large_100_augB | PyTorch checkpoint (referensi) | runs/mobilenetv3_large_100_augB/best.pt | 16.2 | 0.0e+00 | 0.0e+00 | - |
| yolo11n_cls | PyTorch checkpoint (referensi) | runs/yolo11n_cls/weights/best.pt | 3.0 | 0.0e+00 | 0.0e+00 | - |
| yolo11n_cls_augB | ONNX (Ultralytics) | runs/yolo11n_cls_augB/export/best.onnx | 5.9 | - | 1.7e-06 | 64/64 |
| yolo11n_cls_augB | NCNN (Ultralytics, fp32) | runs/yolo11n_cls_augB/export/best_ncnn_model | 5.9 | - | 8.9e-07 | 64/64 |
| yolo11n_cls_augB | PyTorch checkpoint (referensi) | runs/yolo11n_cls_augB/weights/best.pt | 3.0 | 0.0e+00 | 0.0e+00 | - |
