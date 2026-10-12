# Hasil Baseline — Klasifikasi Penyakit Daun Cabai

**Dataset:** Mendeley Data tm3v4zmh7c v1 (Chilli Leaf Disease, 6 kelas, CC BY 4.0)
**Tanggal evaluasi test:** 2026-10-09 (test set dipakai satu kali; log di `reports/eval/TEST_USED`)
**Detail angka:** `reports/eval/summary.md`, `reports/eval/gradcam_summary.md`, `reports/eval/*.png`

> **Penting:** Semua angka di bawah adalah performa **in-distribution**, yaitu pada gambar dari sumber, kamera, dan
> gaya foto yang sama dengan data latih. Angka ini **bukan** perkiraan performa di lapangan dengan kamera statis
> di Raspberry Pi 5. Data lapangan sendiri belum dikumpulkan dan belum diuji.

---

## 1. Ringkasan

- Kelima model mencapai **macro-F1 0,979–0,989** di test (n = 1.245). Selisih antarmodel kecil: 15–28 gambar salah.
- **Hampir semua kesalahan (94–100%) terjadi pada 185 gambar berlatar berwarna** (foto mentah berlatar abu-abu dan
  JPG-640 berlatar biru). Akurasi di subset ini hanya **85–92%**, sedangkan gambar berlatar putih (n = 1.060)
  nyaris sempurna (99,7–100%).
- **Augmentasi degradasi (B) sangat efektif untuk input beresolusi rendah.** Pada resolusi efektif 64 px,
  macro-F1 EfficientNet-B0 turun ke 0,757 tanpa B dan tetap 0,983 dengan B; MobileNetV3 turun ke 0,914 tanpa B
  dan tetap 0,986 dengan B. Pada gambar bersih, B tidak menurunkan performa secara berarti.
- **Grad-CAM:** MobileNetV3 paling fokus pada daun. EfficientNet-B0-A memperlihatkan **perhatian tidak wajar ke
  pojok kiri atas gambar** (indikasi *shortcut*), dan pola ini tidak muncul pada EfficientNet-B0-B.
- **Kandidat terkuat untuk tahap berikutnya: MobileNetV3-Large-B.** Akurasinya setara di gambar bersih, paling
  tahan resolusi rendah, dan fokusnya paling konsisten pada daun. YOLO11n-cls paling ringan (1,5 M parameter), tetapi
  paling lemah di gambar berlatar berwarna dan di resolusi rendah.

---

## 2. Data & protokol (ringkas)

| Tahap | Keputusan |
|---|---|
| Audit | 8.817 file; 0 rusak; 881 exact duplicate (MD5); salinan rotasi/flip terdeteksi dengan pHash invarian dihedral; 12 file konflik label (daun sama di dua kelas); 3 salinan artefak unduhan |
| Dedup | Exact duplicate dan konflik label dibuang, sehingga tersisa **7.924 gambar** |
| Split | Group-aware (near-duplicate pHash ≤ 4 dalam satu grup), stratified kelas × gaya, 70/15/15, seed 42 |
| Verifikasi | 0 irisan MD5, grup, maupun pasangan pHash ≤ 4 antar split |
| Training | Pretrained ImageNet, input 224, AMP; checkpoint dan early stopping berdasarkan **val macro-F1** |
| Test | Input identik untuk semua model: resize sisi terpendek 224 (INTER_AREA) lalu center crop 224 |

**Augmentasi timm.**
- A: random resized crop, flip, rotasi ±15°, color jitter ringan.
- B: A ditambah downscale acak ke 48–128 px lalu upscale ke 224, blur Gaussian/gerak, JPEG q40–90, dan noise ringan.

YOLO11n-cls memakai augmentasi bawaan Ultralytics.

---

## 3. Metrik test

| Model | Aug | Test acc | **Test macro-F1** | Val macro-F1 | Salah (dari 1.245) | Parameter |
|---|---|---|---|---|---|---|
| YOLO11n-cls | Ultralytics | 0,9775 | 0,9793 | 0,9905 | 28 | 1,53 M |
| MobileNetV3-Large | A | 0,9855 | 0,9868 | 0,9919 | 18 | 4,21 M |
| MobileNetV3-Large | B | 0,9855 | 0,9866 | 0,9925 | 18 | 4,21 M |
| EfficientNet-B0 | A | **0,9880** | **0,9891** | 0,9914 | **15** | 4,02 M |
| EfficientNet-B0 | B | 0,9831 | 0,9845 | 0,9886 | 21 | 4,02 M |

Di semua model, macro-F1 test sedikit lebih rendah daripada val (−0,2 sampai −1,1 poin). Salah satu penyebabnya,
test berisi lebih banyak gambar JPG-640 (112, sedangkan val 64). Hal ini terjadi karena grup near-duplicate yang
besar harus berada utuh di satu split (lihat Checkpoint 2).

### F1 per kelas (test)

| Model | Bacterial | Cercospora | Curl Virus | Healthy | Nutrition Def. | Powdery |
|---|---|---|---|---|---|---|
| YOLO11n-cls | 0,968 | 0,981 | 0,971 | 0,975 | 0,981 | 1,000 |
| MobileNetV3-A | 0,978 | 0,989 | 0,980 | 0,984 | 0,991 | 1,000 |
| MobileNetV3-B | 0,970 | 0,993 | 0,978 | 0,991 | 0,987 | 1,000 |
| EfficientNet-B0-A | 0,978 | 0,993 | 0,982 | 0,989 | 0,994 | 1,000 |
| EfficientNet-B0-B | 0,973 | 0,991 | 0,977 | 0,980 | 0,987 | 1,000 |

Precision/recall per kelas: `reports/eval/classification_report_<model>.csv`.
Confusion matrix (absolut dan ternormalisasi): `reports/eval/cm_<model>.png`.

---

## 4. Analisis error

### 4.1 Kesalahan terkonsentrasi pada gambar berlatar berwarna

| Model | Salah di latar putih (n = 1.060) | Salah di latar berwarna (n = 185) | Akurasi latar berwarna |
|---|---|---|---|
| YOLO11n-cls | 1 | 27 | 0,854 |
| MobileNetV3-A | 1 | 17 | 0,908 |
| MobileNetV3-B | 1 | 17 | 0,908 |
| EfficientNet-B0-A | 0 | 15 | 0,919 |
| EfficientNet-B0-B | 3 | 18 | 0,903 |

"Latar berwarna" mencakup gaya **JPG-640** (latar biru/ungu, n = 112) dan **foto mentah ≥ 3000 px berlatar
abu-abu** (n = 73). Foto mentah adalah subset yang paling mirip kondisi kamera nyata. Akurasi di sana 82–92%,
jauh di bawah angka keseluruhan 98–99%. Akurasi tinggi pada gambar berlatar putih kemungkinan besar dibantu oleh
keseragaman gaya per sumber. Kelas Cercospora, misalnya, hampir seluruhnya berupa JPG dengan latar sudah dihapus
(lihat audit).

### 4.2 Pasangan salah klasifikasi

- **Bacterial Spot → Curl Virus** adalah error terbanyak di **semua** model (6–8 gambar), dan **seluruhnya pada
  gaya JPG-640** (daun kecil di atas latar biru muda). Contoh visualnya ada di `errors_<model>.png`.
- Terdapat **9 gambar yang salah di kelima model**, 7 di antaranya Bacterial_Spot JPG-640 (mis. `_571`, `_611`,
  `_684`, `_703`). Pada audit, `Bacterial_Spot_684` sudah tercatat sangat mirip dengan beberapa gambar Curl Virus
  dari sumber yang sama. Perlu dicek oleh ahli apakah labelnya benar atau gejalanya memang ambigu.
- **Curl Virus ↔ Nutrition Deficiency** hampir tidak tertukar: 0 kasus di keempat model timm dan 1 kasus di YOLO.
  Kekhawatiran awal bahwa dua kelas ini mudah tertukar **tidak terbukti pada data ini**. Namun, sebagian besar
  gambar kedua kelas berasal dari gaya PNG-256 yang seragam, sehingga kemungkinan ini tetap perlu diuji ulang di
  data lapangan.
- Error pada foto mentah berlatar abu-abu tersebar (Healthy → Cercospora, Curl → Cercospora, Nutrition → Healthy).
  Daun di foto ini gelap dan pencahayaannya datar.

---

## 5. Grad-CAM

**Metode.** Grad-CAM (pytorch-grad-cam 1.5.7) dihitung terhadap kelas prediksi. Layer target dipilih di **val**:
- MobileNetV3: `blocks[-1]`.
- EfficientNet-B0: `bn2`. Dengan `blocks[-1]`/`conv_head`, 12–46% peta CAM kosong.

Pengukuran kuantitatif dilakukan pada 1.099 gambar test berlatar putih. Mask daun didefinisikan sebagai piksel yang
bukan putih.

> Koreksi audit: kanal alpha pada PNG di dataset ini **tidak transparan** (semua alpha ≥ 128). Latar belakangnya
> putih buram, sehingga alpha tidak bisa dipakai sebagai mask.

| Model | Luas daun rata-rata | Massa CAM di daun | Massa CAM di pojok kiri atas (2% area) |
|---|---|---|---|
| MobileNetV3-A | 0,44 | **0,80** | 0,002 |
| MobileNetV3-B | 0,44 | **0,85** | 0,001 |
| EfficientNet-B0-A | 0,44 | 0,45 | **0,215** |
| EfficientNet-B0-B | 0,44 | 0,59 | 0,007 |

Angka di atas untuk prediksi yang benar. Baseline acak: massa CAM di daun ≈ luas daun, dan pojok ≈ 0,02.

**Kesimpulan:**
- **MobileNetV3 (A dan B) fokus pada daun.** Hampir dua kali lipat dari baseline acak. Visualnya di
  `gradcam_mobilenetv3_*.png`: panas berada di area daun dan sering di lesi.
- **EfficientNet-B0-A menunjukkan indikasi *shortcut learning*.** Sekitar 21% massa CAM jatuh di blok kecil pojok kiri
  atas, apa pun isi gambarnya (10 kali baseline acak), dan massa di daun hanya setara baseline acak. Pola yang sama
  muncul di val dengan layer berbeda, jadi ini perilaku model, bukan artefak layer. Akurasi test-nya tertinggi,
  tetapi **akurasi itu sebagian mungkin bersandar pada isyarat non-daun**, dan ini berisiko di lapangan.
- **EfficientNet-B0-B tidak memperlihatkan pola pojok tersebut.** Augmentasi degradasi tampaknya menghilangkan isyarat
  tersebut. Fokusnya ke daun sedang (0,59).
- YOLO11n-cls belum dianalisis dengan Grad-CAM karena prompt hanya meminta Grad-CAM untuk model timm.

---

## 6. Robustness: input beresolusi rendah

Simulasi crop daun kecil dari kamera lapangan: gambar test 224 px diturunkan ke N px lalu dinaikkan lagi ke 224.

| Model | Asli (224) | 128 px | 96 px | 64 px | Penurunan di 64 px |
|---|---|---|---|---|---|
| YOLO11n-cls | 0,979 | 0,948 | 0,925 | 0,887 | −0,092 |
| MobileNetV3-A | 0,987 | 0,986 | 0,980 | 0,914 | −0,073 |
| **MobileNetV3-B** | 0,987 | **0,986** | **0,986** | **0,986** | **−0,001** |
| EfficientNet-B0-A | 0,989 | 0,956 | 0,847 | 0,757 | −0,232 |
| **EfficientNet-B0-B** | 0,985 | **0,981** | **0,982** | **0,983** | **−0,001** |

Nilai di atas adalah macro-F1 (grafik: `reports/eval/robustness.png`).

**Kesimpulan A/B:** augmentasi degradasi membuat performa nyaris tidak berubah sampai resolusi efektif 64 px,
sementara versi A turun 7–23 poin. Untuk target crop daun 60–120 px dari kamera statis, **model B jelas lebih tepat**.
Ini adalah robustness terhadap *resolusi*. Robustness terhadap perbedaan pencahayaan, latar, dan kamera lapangan
belum teruji (lihat Bagian 7).

---

## 7. Keterbatasan

1. **In-distribution.** Train, val, dan test berasal dari dataset yang sama (Bangladesh, kamera smartphone,
   sebagian besar latar putih hasil penghapusan latar). Angka 98–99% tidak boleh dikutip sebagai performa sistem
   kamera di kebun.
2. **Confound gaya/sumber vs kelas.** Format, resolusi, dan latar berkorelasi kuat dengan kelas (mis. Cercospora
   hampir semuanya JPG berlatar dihapus; Powdery hampir semuanya PNG-256). Akurasi sempurna di subset latar putih
   bisa sebagian berasal dari isyarat gaya. Subset paling realistis (foto mentah, n = 73) hanya mencapai 82–92%.
3. **Kemiripan sisa antar split.** Masih ada 1.672 pasangan pHash berjarak 6 yang berada di split berbeda. Dari
   inspeksi visual sebagian besar adalah false positive, tetapi sebagian mungkin foto berurutan dari daun yang sama.
   Kondisi ini cenderung *menaikkan* skor test.
4. **Ketidakpastian statistik.** Selisih jumlah salah antarmodel hanya 0–13 gambar dari 1.245. Selisih sekecil itu
   tidak bisa dipakai untuk menyatakan satu model lebih unggul. Interval kepercayaan belum dihitung.
5. **Kualitas label belum diverifikasi ahli.** Ada konflik label yang sudah dibuang (12 file) dan 9 gambar yang
   salah di semua model (mayoritas Bacterial JPG-640). Perlu verifikasi oleh ahli penyakit tanaman.
6. **Grad-CAM adalah indikasi, bukan bukti.** Metrik "massa di daun" bergantung pada mask berbasis warna putih
   dan hanya berlaku untuk subset berlatar putih.
7. **Preprocessing YOLO saat evaluasi** memakai cv2 INTER_AREA (diseragamkan dengan model lain), bukan resize
   bawaan Ultralytics. Di val efeknya −0,0008 macro-F1.
8. **Satu seed.** Variasi antar-seed belum diukur.

---

## 8. Rekomendasi ke tahap berikutnya

- **Model utama: MobileNetV3-Large-B.** Akurasi bersihnya setara dengan yang terbaik, paling tahan resolusi
  rendah, dan fokus Grad-CAM-nya konsisten pada daun. **YOLO11n-cls** tetap diekspor (ONNX + NCNN) sebagai pembanding
  latensi di Pi 5.
- **Prioritas data:** kumpulkan foto daun cabai lokal dengan kamera dan posisi yang sama seperti rencana deploy, lalu
  uji model tanpa fine-tuning terlebih dahulu. Hasil itulah yang menjadi angka "lapangan" yang sebenarnya.
- **Untuk training ulang:** kurangi isyarat gaya, misalnya dengan augmentasi penggantian latar (latar putih ↔ tanah
  atau daun lain) dan normalisasi format JPEG untuk semua gambar.
