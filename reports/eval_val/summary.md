# Evaluasi val set

- n = 1181; waktu 2026-10-12 16:29; runs: yolo11n_cls, yolo11n_cls_augB, mobilenetv3_large_100_augB
- Input identik untuk semua model: resize sisi terpendek 224 (INTER_AREA) + center crop 224.

## Ringkasan

| run | accuracy | macro_f1 | F1_Bact | F1_Cerc | F1_Curl | F1_Healthy | F1_Nutr | F1_Powd |
|---|---|---|---|---|---|---|---|---|
| yolo11n_cls | 0.9890 | 0.9897 | 0.9847 | 0.9859 | 0.9847 | 0.9955 | 0.9876 | 1.0000 |
| yolo11n_cls_augB | 0.9865 | 0.9877 | 0.9899 | 0.9817 | 0.9762 | 0.9909 | 0.9875 | 1.0000 |
| mobilenetv3_large_100_augB | 0.9915 | 0.9923 | 0.9898 | 0.9879 | 0.9890 | 0.9932 | 0.9938 | 1.0000 |

## Robustness (macro-F1)

| run | asli(224) | 128 | 96 | 64 |
|---|---|---|---|---|
| mobilenetv3_large_100_augB | 0.9923 | 0.9915 | 0.9908 | 0.9910 |
| yolo11n_cls | 0.9897 | 0.9770 | 0.9449 | 0.8921 |
| yolo11n_cls_augB | 0.9877 | 0.9882 | 0.9851 | 0.9861 |

## Per gaya/sub-sumber

| run | style | n | salah | accuracy | kelas |
|---|---|---|---|---|---|
| yolo11n_cls | JPG-640 | 64 | 5 | 0.9219 | Bact,Cerc,Curl,Healthy |
| yolo11n_cls | JPG-foto>=3000 | 78 | 8 | 0.8974 | Bact,Cerc,Curl,Healthy,Nutr |
| yolo11n_cls | JPG-lain | 220 | 0 | 1.0000 | Bact,Cerc |
| yolo11n_cls | PNG-1000-RGBA | 226 | 0 | 1.0000 | Bact,Healthy,Nutr |
| yolo11n_cls | PNG-256-RGBA | 521 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| yolo11n_cls | PNG-lain-RGBA | 72 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |
| yolo11n_cls_augB | JPG-640 | 64 | 2 | 0.9688 | Bact,Cerc,Curl,Healthy |
| yolo11n_cls_augB | JPG-foto>=3000 | 78 | 14 | 0.8205 | Bact,Cerc,Curl,Healthy,Nutr |
| yolo11n_cls_augB | JPG-lain | 220 | 0 | 1.0000 | Bact,Cerc |
| yolo11n_cls_augB | PNG-1000-RGBA | 226 | 0 | 1.0000 | Bact,Healthy,Nutr |
| yolo11n_cls_augB | PNG-256-RGBA | 521 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| yolo11n_cls_augB | PNG-lain-RGBA | 72 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |
| mobilenetv3_large_100_augB | JPG-640 | 64 | 3 | 0.9531 | Bact,Cerc,Curl,Healthy |
| mobilenetv3_large_100_augB | JPG-foto>=3000 | 78 | 6 | 0.9231 | Bact,Cerc,Curl,Healthy,Nutr |
| mobilenetv3_large_100_augB | JPG-lain | 220 | 1 | 0.9955 | Bact,Cerc |
| mobilenetv3_large_100_augB | PNG-1000-RGBA | 226 | 0 | 1.0000 | Bact,Healthy,Nutr |
| mobilenetv3_large_100_augB | PNG-256-RGBA | 521 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| mobilenetv3_large_100_augB | PNG-lain-RGBA | 72 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |

## Pasangan salah klasifikasi

**yolo11n_cls**: Bact→Curl 3; Bact→Cerc 2; Cerc→Curl 2; Nutr→Healthy 2; Curl→Cerc 2; Bact→Nutr 1; Cerc→Nutr 1

**yolo11n_cls_augB**: Cerc→Curl 6; Bact→Curl 2; Healthy→Cerc 2; Bact→Nutr 1; Curl→Cerc 1; Healthy→Curl 1; Nutr→Bact 1; Nutr→Curl 1; Nutr→Healthy 1

**mobilenetv3_large_100_augB**: Bact→Curl 2; Curl→Cerc 2; Bact→Nutr 1; Bact→Cerc 1; Cerc→Curl 1; Cerc→Healthy 1; Healthy→Cerc 1; Nutr→Healthy 1

## Grad-CAM: fraksi massa CAM di dalam daun (subset berlatar putih, mask = piksel bukan putih)

Bila model fokus pada daun, `cam_on_leaf` jauh di atas `leaf_area` (baseline acak = luas daun). `cam_pojok_kiri_atas` = massa CAM di blok 32x32 kiri-atas (2% area; baseline acak ~0.02).

| run | correct | n | cam_kosong | leaf_area | cam_on_leaf | cam_on_leaf_median | cam_pojok_kiri_atas |
|---|---|---|---|---|---|---|---|
| mobilenetv3_large_100_augB | False | 1 | 0 | 0.655 | 0.976 | 0.976 | 0.000 |
| mobilenetv3_large_100_augB | True | 1056 | 0 | 0.418 | 0.847 | 0.860 | 0.001 |
| yolo11n_cls | False | 1 | 0 | 0.731 | 0.849 | 0.849 | 0.011 |
| yolo11n_cls | True | 1056 | 0 | 0.418 | 0.740 | 0.745 | 0.005 |
| yolo11n_cls_augB | True | 1057 | 0 | 0.418 | 0.754 | 0.770 | 0.004 |
