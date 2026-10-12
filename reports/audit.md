# Audit Dataset — Chilli Leaf Disease (Mendeley tm3v4zmh7c)

- Root data: `data/raw` (read-only)
- Seed: 42; pHash 64-bit (hash_size=8) dihitung pada thumbnail 256 px, jarak = Hamming minimum atas 8 transformasi rotasi/flip; threshold near-duplicate = **4**

## 1. Struktur & jumlah per kelas

Folder kelas: Bacterial_Spot, Cercospora_Leaf_Spot, Curl_Virus, Healthy_Leaf, Nutrition_Deficiency, Powdery_Mildew. File non-gambar: 0. File lepas di root: tidak ada.

| class | jumlah_file | deskripsi | selisih |
|---|---|---|---|
| Bacterial_Spot | 1629 | 1629 | 0 |
| Cercospora_Leaf_Spot | 1898 | 1898 | 0 |
| Curl_Virus | 1590 | 1590 | 0 |
| Healthy_Leaf | 1647 | 1647 | 0 |
| Nutrition_Deficiency | 1207 | 1207 | 0 |
| Powdery_Mildew | 846 | 843 | 3 |

Total: **8817** file; deskripsi: **8814**.

## 2. File rusak / tidak terbaca

0 file.

## 3. Format, mode warna, resolusi

### Ekstensi per kelas

| class | .jpg | .png |
|---|---|---|
| Bacterial_Spot | 275 | 1354 |
| Cercospora_Leaf_Spot | 1898 | 0 |
| Curl_Virus | 366 | 1224 |
| Healthy_Leaf | 175 | 1472 |
| Nutrition_Deficiency | 78 | 1129 |
| Powdery_Mildew | 1 | 845 |

### Mode warna per kelas

| class | RGB | RGBA |
|---|---|---|
| Bacterial_Spot | 275 | 1354 |
| Cercospora_Leaf_Spot | 1898 | 0 |
| Curl_Virus | 366 | 1224 |
| Healthy_Leaf | 175 | 1472 |
| Nutrition_Deficiency | 78 | 1129 |
| Powdery_Mildew | 1 | 845 |

File dengan kanal alpha yang benar-benar transparan: 6024. File dengan tag EXIF orientation != 0/1: 0.

### Resolusi

| res | n |
|---|---|
| 256x256 | 3577 |
| 1000x1000 | 2028 |
| 640x640 | 416 |
| 4000x3000 | 398 |
| 3000x4000 | 127 |
| 517x640 | 6 |
| 3048x4064 | 5 |
| 4064x3048 | 4 |
| 1314x1314 | 4 |
| 473x640 | 4 |

| class | 1000x1000 | 256x256 | 3000x4000 | 4000x3000 | 517x640 | 640x640 | lainnya |
|---|---|---|---|---|---|---|---|
| Bacterial_Spot | 1143 | 0 | 0 | 0 | 0 | 219 | 267 |
| Cercospora_Leaf_Spot | 0 | 0 | 44 | 116 | 0 | 0 | 1738 |
| Curl_Virus | 0 | 1224 | 48 | 103 | 6 | 171 | 38 |
| Healthy_Leaf | 638 | 700 | 24 | 115 | 0 | 26 | 144 |
| Nutrition_Deficiency | 247 | 872 | 10 | 64 | 0 | 0 | 14 |
| Powdery_Mildew | 0 | 781 | 1 | 0 | 0 | 0 | 64 |

### Gaya / sub-sumber (format + resolusi) per kelas

Distribusi gaya yang berbeda antar kelas = risiko *shortcut learning* (model belajar sumber, bukan penyakit).

| style | Bacterial_Spot | Cercospora_Leaf_Spot | Curl_Virus | Healthy_Leaf | Nutrition_Deficiency | Powdery_Mildew | Total |
|---|---|---|---|---|---|---|---|
| JPG-640 | 273 | 1 | 210 | 26 | 0 | 0 | 510 |
| JPG-foto>=3000 | 1 | 180 | 156 | 141 | 78 | 1 | 557 |
| JPG-lain | 1 | 1717 | 0 | 8 | 0 | 0 | 1726 |
| PNG-1000-RGBA | 1143 | 0 | 0 | 638 | 247 | 0 | 2028 |
| PNG-256-RGBA | 0 | 0 | 1224 | 700 | 872 | 781 | 3577 |
| PNG-lain-RGBA | 211 | 0 | 0 | 134 | 10 | 64 | 419 |
| Total | 1629 | 1898 | 1590 | 1647 | 1207 | 846 | 8817 |

### Ukuran file median (KB) per kelas & ekstensi

| class | ext | median_KB |
|---|---|---|
| Bacterial_Spot | .jpg | 24.000 |
| Bacterial_Spot | .png | 646.000 |
| Cercospora_Leaf_Spot | .jpg | 155.000 |
| Curl_Virus | .jpg | 22.000 |
| Curl_Virus | .png | 54.000 |
| Healthy_Leaf | .jpg | 1308.000 |
| Healthy_Leaf | .png | 302.000 |
| Nutrition_Deficiency | .jpg | 1409.000 |
| Nutrition_Deficiency | .png | 71.000 |
| Powdery_Mildew | .jpg | 4961.000 |
| Powdery_Mildew | .png | 53.000 |

### Statistik warna rata-rata per kelas (thumbnail 256 px, skala 0–255)

`frac_white` = fraksi piksel hampir putih (min RGB > 235), `frac_black` = hampir hitam (max RGB < 20).

| class | brightness | contrast | mean_r | mean_g | mean_b | frac_white | frac_black |
|---|---|---|---|---|---|---|---|
| Bacterial_Spot | 217.549 | 48.995 | 222.713 | 228.063 | 201.872 | 0.614 | 0.000 |
| Cercospora_Leaf_Spot | 171.455 | 86.873 | 176.915 | 185.854 | 151.596 | 0.503 | 0.000 |
| Curl_Virus | 203.355 | 55.729 | 200.165 | 212.088 | 197.811 | 0.497 | 0.001 |
| Healthy_Leaf | 211.530 | 52.780 | 209.427 | 221.272 | 203.891 | 0.601 | 0.000 |
| Nutrition_Deficiency | 202.112 | 57.209 | 202.317 | 215.614 | 188.406 | 0.535 | 0.000 |
| Powdery_Mildew | 220.764 | 49.800 | 223.706 | 229.620 | 208.966 | 0.640 | 0.000 |

### Nama file tidak standar

| relpath | md5 | width | height |
|---|---|---|---|
| Powdery_Mildew/Powdery Mildew_187-2VytwV.png | 418751967ce5359716826c496e84d96c | 256 | 256 |
| Powdery_Mildew/Powdery Mildew_188-f4NSlF.png | 278b2d771bccda801dd31d7ad9e410c8 | 256 | 256 |
| Powdery_Mildew/Powdery Mildew_265-ttly2A.png | a2f58cd69a7e13e407fc81f9f7ab164d | 256 | 256 |

Stem nama sama dengan ekstensi berbeda dalam satu kelas: 0.

## 4. Duplikat

### Exact (MD5)

- Grup MD5 dengan >1 file: **787** (1668 file, 881 redundan)
- Grup exact **lintas kelas**: **0**

### Near-duplicate (pHash, invarian rotasi/flip) — sensitivitas threshold

`file_redundan` = file yang akan terbuang jika tiap grup disisakan 1.

| threshold | pasangan | pasangan_lintas_kelas | pasangan_via_rotasi_flip | file_dalam_grup | grup | file_redundan |
|---|---|---|---|---|---|---|
| 0 | 3308 | 5 | 1583 | 2517 | 897 | 1620 |
| 1 | 3308 | 5 | 1583 | 2517 | 897 | 1620 |
| 2 | 6059 | 17 | 3358 | 3281 | 1083 | 2198 |
| 3 | 6059 | 17 | 3358 | 3281 | 1083 | 2198 |
| 4 | 9981 | 203 | 5676 | 4441 | 1286 | 3155 |
| 5 | 9981 | 203 | 5676 | 4441 | 1286 | 3155 |
| 6 | 17989 | 1246 | 10372 | 6100 | 1127 | 4973 |
| 7 | 17989 | 1246 | 10372 | 6100 | 1127 | 4973 |
| 8 | 37898 | 6941 | 22776 | 7631 | 478 | 7153 |
| 9 | 37898 | 6941 | 22776 | 7631 | 478 | 7153 |
| 10 | 98347 | 33423 | 64790 | 8534 | 113 | 8421 |
| 11 | 98347 | 33423 | 64790 | 8534 | 113 | 8421 |
| 12 | 280327 | 134342 | 202378 | 8775 | 12 | 8763 |

### Grup pada threshold terpilih (4, termasuk exact)

- Grup: **1286**, file di dalam grup: **4441**, ukuran grup maks: 111
- Grup **lintas kelas**: **48** (480 file)

| class | file | dalam_grup_dup | nn_dist_median | unik_setelah_dedup |
|---|---|---|---|---|
| Bacterial_Spot | 1629 | 1367 | 0.000 | 506 |
| Cercospora_Leaf_Spot | 1898 | 737 | 6.000 | 1438 |
| Curl_Virus | 1590 | 635 | 6.000 | 1168 |
| Healthy_Leaf | 1647 | 903 | 4.000 | 1012 |
| Nutrition_Deficiency | 1207 | 546 | 6.000 | 895 |
| Powdery_Mildew | 846 | 253 | 6.000 | 698 |

Detail: `reports/dup_groups.csv`, `reports/near_dup_pairs.csv`. Distribusi jarak tetangga terdekat: `reports/nn_dist_hist.png`.

## 5. Visual

Grid sampel: `samples_Bacterial_Spot.png`, `samples_Cercospora_Leaf_Spot.png`, `samples_Curl_Virus.png`, `samples_Healthy_Leaf.png`, `samples_Nutrition_Deficiency.png`, `samples_Powdery_Mildew.png`

Grid pasangan duplikat: `dup_pairs_d00-00.png`, `dup_pairs_d01-02.png`, `dup_pairs_d03-04.png`, `dup_pairs_d05-06.png`, `dup_pairs_d07-08.png`, `dup_pairs_d09-10.png`, `dup_pairs_d11-12.png`, `dup_pairs_cross_class.png`, `dup_pairs_rotflip.png`
