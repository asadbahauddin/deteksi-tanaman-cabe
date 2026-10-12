# Grad-CAM (layer: EfficientNet=bn2, MobileNetV3=blocks[-1])

| run | correct | n | cam_kosong | leaf_area | cam_on_leaf | cam_on_leaf_median | cam_pojok_kiri_atas |
|---|---|---|---|---|---|---|---|
| efficientnet_b0_augA | False | 4 | 0 | 0.700 | 0.682 | 0.741 | 0.276 |
| efficientnet_b0_augA | True | 1095 | 0 | 0.439 | 0.448 | 0.441 | 0.215 |
| efficientnet_b0_augB | False | 9 | 0 | 0.528 | 0.685 | 0.810 | 0.007 |
| efficientnet_b0_augB | True | 1090 | 0 | 0.439 | 0.586 | 0.590 | 0.007 |
| mobilenetv3_large_100_augA | False | 6 | 0 | 0.596 | 0.876 | 0.883 | 0.001 |
| mobilenetv3_large_100_augA | True | 1093 | 0 | 0.439 | 0.800 | 0.828 | 0.002 |
| mobilenetv3_large_100_augB | False | 8 | 0 | 0.631 | 0.900 | 0.919 | 0.002 |
| mobilenetv3_large_100_augB | True | 1091 | 0 | 0.438 | 0.853 | 0.869 | 0.001 |
