# Evaluasi test set

- n = 1245; waktu 2026-10-09 16:53; runs: yolo11n_cls, mobilenetv3_large_100_augA, mobilenetv3_large_100_augB, efficientnet_b0_augA, efficientnet_b0_augB
- Input identik untuk semua model: resize sisi terpendek 224 (INTER_AREA) + center crop 224.

## Ringkasan

| run | accuracy | macro_f1 | F1_Bact | F1_Cerc | F1_Curl | F1_Healthy | F1_Nutr | F1_Powd |
|---|---|---|---|---|---|---|---|---|
| yolo11n_cls | 0.9775 | 0.9793 | 0.9681 | 0.9811 | 0.9708 | 0.9751 | 0.9810 | 1.0000 |
| mobilenetv3_large_100_augA | 0.9855 | 0.9868 | 0.9778 | 0.9887 | 0.9800 | 0.9840 | 0.9905 | 1.0000 |
| mobilenetv3_large_100_augB | 0.9855 | 0.9866 | 0.9704 | 0.9925 | 0.9782 | 0.9909 | 0.9874 | 1.0000 |
| efficientnet_b0_augA | 0.9880 | 0.9891 | 0.9780 | 0.9925 | 0.9819 | 0.9885 | 0.9937 | 1.0000 |
| efficientnet_b0_augB | 0.9831 | 0.9845 | 0.9728 | 0.9905 | 0.9766 | 0.9796 | 0.9873 | 1.0000 |

## Robustness (macro-F1)

| run | asli(224) | 128 | 96 | 64 |
|---|---|---|---|---|
| efficientnet_b0_augA | 0.9891 | 0.9560 | 0.8466 | 0.7570 |
| efficientnet_b0_augB | 0.9845 | 0.9809 | 0.9823 | 0.9830 |
| mobilenetv3_large_100_augA | 0.9868 | 0.9858 | 0.9796 | 0.9137 |
| mobilenetv3_large_100_augB | 0.9866 | 0.9859 | 0.9859 | 0.9858 |
| yolo11n_cls | 0.9793 | 0.9484 | 0.9253 | 0.8870 |

## Per gaya/sub-sumber

| run | style | n | salah | accuracy | kelas |
|---|---|---|---|---|---|
| yolo11n_cls | JPG-640 | 112 | 14 | 0.8750 | Bact,Curl,Healthy |
| yolo11n_cls | JPG-foto>=3000 | 73 | 13 | 0.8219 | Cerc,Curl,Healthy,Nutr |
| yolo11n_cls | JPG-lain | 238 | 0 | 1.0000 | Cerc |
| yolo11n_cls | PNG-1000-RGBA | 236 | 1 | 0.9958 | Bact,Healthy,Nutr |
| yolo11n_cls | PNG-256-RGBA | 516 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| yolo11n_cls | PNG-lain-RGBA | 70 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |
| mobilenetv3_large_100_augA | JPG-640 | 112 | 10 | 0.9107 | Bact,Curl,Healthy |
| mobilenetv3_large_100_augA | JPG-foto>=3000 | 73 | 7 | 0.9041 | Cerc,Curl,Healthy,Nutr |
| mobilenetv3_large_100_augA | JPG-lain | 238 | 0 | 1.0000 | Cerc |
| mobilenetv3_large_100_augA | PNG-1000-RGBA | 236 | 1 | 0.9958 | Bact,Healthy,Nutr |
| mobilenetv3_large_100_augA | PNG-256-RGBA | 516 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| mobilenetv3_large_100_augA | PNG-lain-RGBA | 70 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |
| mobilenetv3_large_100_augB | JPG-640 | 112 | 11 | 0.9018 | Bact,Curl,Healthy |
| mobilenetv3_large_100_augB | JPG-foto>=3000 | 73 | 6 | 0.9178 | Cerc,Curl,Healthy,Nutr |
| mobilenetv3_large_100_augB | JPG-lain | 238 | 0 | 1.0000 | Cerc |
| mobilenetv3_large_100_augB | PNG-1000-RGBA | 236 | 1 | 0.9958 | Bact,Healthy,Nutr |
| mobilenetv3_large_100_augB | PNG-256-RGBA | 516 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| mobilenetv3_large_100_augB | PNG-lain-RGBA | 70 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |
| efficientnet_b0_augA | JPG-640 | 112 | 9 | 0.9196 | Bact,Curl,Healthy |
| efficientnet_b0_augA | JPG-foto>=3000 | 73 | 6 | 0.9178 | Cerc,Curl,Healthy,Nutr |
| efficientnet_b0_augA | JPG-lain | 238 | 0 | 1.0000 | Cerc |
| efficientnet_b0_augA | PNG-1000-RGBA | 236 | 0 | 1.0000 | Bact,Healthy,Nutr |
| efficientnet_b0_augA | PNG-256-RGBA | 516 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| efficientnet_b0_augA | PNG-lain-RGBA | 70 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |
| efficientnet_b0_augB | JPG-640 | 112 | 12 | 0.8929 | Bact,Curl,Healthy |
| efficientnet_b0_augB | JPG-foto>=3000 | 73 | 6 | 0.9178 | Cerc,Curl,Healthy,Nutr |
| efficientnet_b0_augB | JPG-lain | 238 | 0 | 1.0000 | Cerc |
| efficientnet_b0_augB | PNG-1000-RGBA | 236 | 2 | 0.9915 | Bact,Healthy,Nutr |
| efficientnet_b0_augB | PNG-256-RGBA | 516 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| efficientnet_b0_augB | PNG-lain-RGBA | 70 | 1 | 0.9857 | Bact,Healthy,Nutr,Powd |

## Pasangan salah klasifikasi

**yolo11n_cls**: Bact→Curl 8; Cerc→Healthy 3; Curl→Cerc 3; Curl→Bact 2; Bact→Healthy 2; Healthy→Cerc 2; Curl→Healthy 2; Cerc→Nutr 1; Curl→Nutr 1; Healthy→Nutr 1; Nutr→Bact 1; Nutr→Cerc 1; Nutr→Healthy 1

**mobilenetv3_large_100_augA**: Bact→Curl 6; Bact→Healthy 2; Curl→Cerc 2; Healthy→Cerc 2; Curl→Healthy 2; Cerc→Curl 1; Bact→Nutr 1; Healthy→Nutr 1; Nutr→Cerc 1

**mobilenetv3_large_100_augB**: Bact→Curl 7; Curl→Cerc 3; Curl→Bact 2; Bact→Healthy 2; Cerc→Nutr 1; Bact→Nutr 1; Healthy→Nutr 1; Nutr→Healthy 1

**efficientnet_b0_augA**: Bact→Curl 6; Curl→Bact 2; Healthy→Cerc 2; Bact→Healthy 1; Cerc→Curl 1; Curl→Healthy 1; Healthy→Nutr 1; Nutr→Cerc 1

**efficientnet_b0_augB**: Bact→Curl 8; Nutr→Healthy 3; Cerc→Healthy 2; Curl→Cerc 2; Bact→Healthy 1; Cerc→Curl 1; Bact→Nutr 1; Curl→Healthy 1; Healthy→Bact 1; Healthy→Curl 1

## Grad-CAM

Bagian ini DIGANTI oleh `gradcam_summary.md` dan `gradcam_*.png` (dijalankan ulang 16:57 dengan layer target yang dipilih di val: EfficientNet=`bn2`, MobileNetV3=`blocks[-1]`). Versi pertama memakai `blocks[-1]` untuk EfficientNet yang menghasilkan 12–46% CAM kosong. Metrik klasifikasi di atas tidak berubah.
