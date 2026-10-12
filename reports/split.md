# Split dataset (Tahap 2)

- Config: `configs/split.yaml`; seed 42; StratifiedGroupKFold 20 fold (stratifikasi: class × style); grup = pHash ≤ 4 (invarian rotasi/flip) + MD5
- File awal (terbaca): 8817; dipakai: **7924** dalam **5663** grup
- Materialisasi: {'symlink': 7924}

## Tindakan per file

| action | Bacterial_Spot | Cercospora_Leaf_Spot | Curl_Virus | Healthy_Leaf | Nutrition_Deficiency | Powdery_Mildew |
|---|---|---|---|---|---|---|
| kept | 1273 | 1683 | 1578 | 1480 | 1067 | 843 |
| removed_exact_dup | 349 | 215 | 8 | 166 | 140 | 0 |
| removed_extra_copy | 0 | 0 | 0 | 0 | 0 | 3 |
| removed_label_conflict | 7 | 0 | 4 | 1 | 0 | 0 |

## Jumlah per kelas per split

| class | train | val | test | total |
|---|---|---|---|---|
| Bacterial_Spot | 867 (68.1%) | 199 (15.6%) | 207 (16.3%) | 1273 |
| Cercospora_Leaf_Spot | 1172 (69.6%) | 247 (14.7%) | 264 (15.7%) | 1683 |
| Curl_Virus | 1077 (68.3%) | 227 (14.4%) | 274 (17.4%) | 1578 |
| Healthy_Leaf | 1042 (70.4%) | 220 (14.9%) | 218 (14.7%) | 1480 |
| Nutrition_Deficiency | 748 (70.1%) | 161 (15.1%) | 158 (14.8%) | 1067 |
| Powdery_Mildew | 592 (70.2%) | 127 (15.1%) | 124 (14.7%) | 843 |
| TOTAL | 5498 (69.4%) | 1181 (14.9%) | 1245 (15.7%) | 7924 |

| split | grup |
|---|---|
| train | 4029 |
| val | 855 |
| test | 779 |

## Kelas × gaya per split

| class | style | train | val | test |
|---|---|---|---|---|
| Bacterial_Spot | JPG-640 | 187 | 39 | 39 |
| Bacterial_Spot | JPG-foto>=3000 | 0 | 1 | 0 |
| Bacterial_Spot | JPG-lain | 0 | 1 | 0 |
| Bacterial_Spot | PNG-1000-RGBA | 550 | 117 | 128 |
| Bacterial_Spot | PNG-lain-RGBA | 130 | 41 | 40 |
| Cercospora_Leaf_Spot | JPG-640 | 0 | 1 | 0 |
| Cercospora_Leaf_Spot | JPG-foto>=3000 | 126 | 27 | 26 |
| Cercospora_Leaf_Spot | JPG-lain | 1046 | 219 | 238 |
| Curl_Virus | JPG-640 | 107 | 21 | 70 |
| Curl_Virus | JPG-foto>=3000 | 112 | 23 | 21 |
| Curl_Virus | PNG-256-RGBA | 858 | 183 | 183 |
| Healthy_Leaf | JPG-640 | 19 | 3 | 3 |
| Healthy_Leaf | JPG-foto>=3000 | 99 | 21 | 21 |
| Healthy_Leaf | JPG-lain | 8 | 0 | 0 |
| Healthy_Leaf | PNG-1000-RGBA | 337 | 72 | 72 |
| Healthy_Leaf | PNG-256-RGBA | 490 | 105 | 104 |
| Healthy_Leaf | PNG-lain-RGBA | 89 | 19 | 18 |
| Nutrition_Deficiency | JPG-foto>=3000 | 28 | 6 | 5 |
| Nutrition_Deficiency | PNG-1000-RGBA | 169 | 37 | 36 |
| Nutrition_Deficiency | PNG-256-RGBA | 546 | 116 | 114 |
| Nutrition_Deficiency | PNG-lain-RGBA | 5 | 2 | 3 |
| Powdery_Mildew | JPG-foto>=3000 | 1 | 0 | 0 |
| Powdery_Mildew | PNG-256-RGBA | 546 | 117 | 115 |
| Powdery_Mildew | PNG-lain-RGBA | 45 | 10 | 9 |

## Verifikasi tanpa irisan

| cek | jumlah_irisan |
|---|---|
| md5 train∩val | 0 |
| group train∩val | 0 |
| md5 train∩test | 0 |
| group train∩test | 0 |
| md5 val∩test | 0 |
| group val∩test | 0 |
| pasangan pHash<=4 lintas split | 0 |
| file di data/split/train == csv | 0 |
| file di data/split/val == csv | 0 |
| file di data/split/test == csv | 0 |

**Status: LULUS**

Pasangan near-duplicate yang berada di split berbeda (di atas threshold, untuk informasi):

| dist | pasangan_lintas_split |
|---|---|
| 6 | 1672 |
| 8 | 6547 |
| 10 | 22785 |
| 12 | 68935 |
