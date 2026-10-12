# Evaluasi val set

- n = 1181; waktu 2026-10-09 16:48; runs: mobilenetv3_large_100_augB
- Input identik untuk semua model: resize sisi terpendek 224 (INTER_AREA) + center crop 224.

## Ringkasan

| run | accuracy | macro_f1 | F1_Bact | F1_Cerc | F1_Curl | F1_Healthy | F1_Nutr | F1_Powd |
|---|---|---|---|---|---|---|---|---|
| mobilenetv3_large_100_augB | 0.9915 | 0.9923 | 0.9898 | 0.9879 | 0.9890 | 0.9932 | 0.9938 | 1.0000 |

## Robustness (macro-F1)

| run | asli(224) | 96 |
|---|---|---|
| mobilenetv3_large_100_augB | 0.9923 | 0.9908 |

## Per gaya/sub-sumber

| run | style | n | salah | accuracy | kelas |
|---|---|---|---|---|---|
| mobilenetv3_large_100_augB | JPG-640 | 64 | 3 | 0.9531 | Bact,Cerc,Curl,Healthy |
| mobilenetv3_large_100_augB | JPG-foto>=3000 | 78 | 6 | 0.9231 | Bact,Cerc,Curl,Healthy,Nutr |
| mobilenetv3_large_100_augB | JPG-lain | 220 | 1 | 0.9955 | Bact,Cerc |
| mobilenetv3_large_100_augB | PNG-1000-RGBA | 226 | 0 | 1.0000 | Bact,Healthy,Nutr |
| mobilenetv3_large_100_augB | PNG-256-RGBA | 521 | 0 | 1.0000 | Curl,Healthy,Nutr,Powd |
| mobilenetv3_large_100_augB | PNG-lain-RGBA | 72 | 0 | 1.0000 | Bact,Healthy,Nutr,Powd |

## Pasangan salah klasifikasi

**mobilenetv3_large_100_augB**: Bact→Curl 2; Curl→Cerc 2; Bact→Nutr 1; Bact→Cerc 1; Cerc→Curl 1; Cerc→Healthy 1; Healthy→Cerc 1; Nutr→Healthy 1

## Grad-CAM: fraksi massa CAM di dalam daun (subset berlatar putih, mask = piksel bukan putih)

Bila model fokus pada daun, `cam_on_leaf` jauh di atas `leaf_area` (baseline acak = luas daun).

| run | correct | n | leaf_area | cam_on_leaf | cam_on_leaf_median |
|---|---|---|---|---|---|
| mobilenetv3_large_100_augB | False | 1 | 0.655 | 0.976 | 0.976 |
| mobilenetv3_large_100_augB | True | 1056 | 0.418 | 0.847 | 0.860 |
