# Log eksperimen

| waktu | model | augmentasi | config | best epoch | waktu train (menit) | val macro-F1 | val acc | run |
|---|---|---|---|---|---|---|---|---|
| 2026-10-09 09:52 | yolo11n-cls | ultralytics-default | imgsz 224, batch 64, epochs≤60, patience 12 (macro-F1), optimizer auto, AMP | 53 | 72.6 | 0.9905 | 0.9898 | `runs/yolo11n_cls` |
| 2026-10-09 10:08 | mobilenetv3_large_100 | A | mobilenetv3_large_100.ra4_e3600_r224_in1k, 224px, batch 64, AdamW lr 0.0005 wd 0.05, cosine+warmup 2, LS 0.1, AMP, patience 8 | 8 | 16.0 | 0.9846 | 0.9831 | `runs/_v1_mobilenetv3_large_100_augA` — **TIDAK DIPAKAI**: early stopping prematur (patience 8, berhenti ep 16/40 saat LR masih tinggi); diulang dengan min_epochs 25, patience 10 |
| 2026-10-09 10:58 | mobilenetv3_large_100 | A | mobilenetv3_large_100.ra4_e3600_r224_in1k, 224px, batch 64, AdamW lr 0.0005 wd 0.05, cosine+warmup 2, LS 0.1, AMP, patience 10 (min_epochs 25) | 37 | 44.1 | 0.9919 | 0.9915 | `runs/mobilenetv3_large_100_augA` |
| 2026-10-09 11:40 | mobilenetv3_large_100 | B | mobilenetv3_large_100.ra4_e3600_r224_in1k, 224px, batch 64, AdamW lr 0.0005 wd 0.05, cosine+warmup 2, LS 0.1, AMP, patience 10 (min_epochs 25) | 26 | 41.0 | 0.9925 | 0.9915 | `runs/mobilenetv3_large_100_augB` |
| 2026-10-09 12:20 | efficientnet_b0 | A | efficientnet_b0.ra4_e3600_r224_in1k, 224px, batch 64, AdamW lr 0.0005 wd 0.05, cosine+warmup 2, LS 0.1, AMP, patience 10 (min_epochs 25) | 23 | 39.5 | 0.9914 | 0.9907 | `runs/efficientnet_b0_augA` |
| 2026-10-09 16:34 | efficientnet_b0 | B | efficientnet_b0.ra4_e3600_r224_in1k, 224px, batch 64, AdamW lr 0.0005 wd 0.05, cosine+warmup 2, LS 0.1, AMP, patience 10 (min_epochs 25) | 12 | 38.6 | 0.9886 | 0.9873 | `runs/efficientnet_b0_augB` |
