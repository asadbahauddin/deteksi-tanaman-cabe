# Deteksi Tanaman Cabe — Baseline klasifikasi penyakit daun cabai

Ini adalah baseline *classifier* penyakit daun cabai (6 kelas) yang dilatih di laptop dengan dataset publik Mendeley
tm3v4zmh7c, sebagai tahap awal sistem deteksi penyakit berbasis kamera di Raspberry Pi 5.
Hasil lengkapnya ada di [`reports/hasil_baseline.md`](reports/hasil_baseline.md). **Semua angka di sana in-distribution, bukan performa lapangan.**

## Dataset & lisensi

- Dataset **tidak disertakan** di repo ini. Unduh dari Mendeley Data: *Chilli Leaf Disease Image Dataset for
  Classification and Early Diagnosis in Agriculture*, DOI [10.17632/tm3v4zmh7c.1](https://doi.org/10.17632/tm3v4zmh7c.1)
  (lisensi CC BY 4.0). Ekstrak ke `data/raw/<kelas>/`, lalu jalankan `01_audit.py` dan `02_split.py`.
- Gambar contoh di `reports/` (grid sampel, error, Grad-CAM) berasal dari dataset tersebut (CC BY 4.0).
- Library utama: Ultralytics (AGPL-3.0), AlbumentationsX (AGPL-3.0), timm (Apache-2.0), PyTorch (BSD).

## Setup

```bash
conda create -n cabai python=3.11 -y
conda activate cabai
pip install -r requirements.txt        # torch 2.14.1+cu130 dari index PyTorch (sudah tercantum di file)
# atau: conda env create -f environment.yml
```

Semua perintah dijalankan dari root proyek, di dalam env `cabai`. Path relatif terhadap root.
Seed 42 tetap untuk semua tahap. Config tersimpan di `configs/`.

## Alur & perintah

| Tahap | Perintah | Output |
|---|---|---|
| 0. Cek env | `python scripts/00_check_env.py --out reports/env.json` | spesifikasi, versi, status CUDA |
| 1. Audit | `python scripts/01_audit.py` (`--reuse` untuk memakai hash yang sudah ada, `--phash-threshold N`) | `reports/audit.md`, `samples_*.png`, `dup_*` |
| 2. Split | `python scripts/02_split.py` (config: `configs/split.yaml`) | `splits/*.csv`, `data/split/` (symlink), `reports/split.md` |
| 3a. YOLO | `python scripts/03a_train_yolo_cls.py` (config: `configs/yolo11n_cls.yaml`) | `runs/yolo11n_cls/` |
| 3b. timm | `python scripts/03b_train_timm.py --model mobilenetv3_large_100 --aug B` (config: `configs/timm.yaml`) | `runs/<model>_aug<A/B>/` |
| 3. Semua | `bash scripts/run_stage3.sh` (hentikan dengan aman: buat file `runs/_logs/STOP`) | log di `runs/_logs/` |
| 4. Evaluasi test | `python scripts/04_evaluate.py --runs <run> ...` (dikunci: hanya sekali, lihat `reports/eval/TEST_USED`) | `reports/eval/` |
| 5. Inferensi | `python scripts/05_infer.py <gambar/folder> [--model ...] [--device cpu] [--csv out.csv]` | label, probabilitas, waktu |
| 5. Webcam | `python scripts/06_webcam_demo.py [--model ...] [--camera 0]` (tekan `q` untuk keluar) | jendela real-time |
| 5. Ekspor | `python scripts/07_export.py` | ONNX/NCNN di `runs/<run>/export/`, `reports/export.md` |
| 5. Benchmark | `python scripts/08_benchmark_cpu.py [--threads 4]` | `reports/benchmark_cpu.md` |

`--model` pada script 05 dan 06 menerima:
- nama run (`mobilenetv3_large_100_augB`, `yolo11n_cls`);
- file ONNX timm (`runs/mobilenetv3_large_100_augB/export/mobilenetv3_large_100_augB.onnx`);
- ekspor YOLO (`runs/yolo11n_cls/export/best.onnx`, `runs/yolo11n_cls/export/best_ncnn_model`).

## Struktur

```
data/raw/        dataset asli (read-only; satu-satunya salinan lokal)
data/split/      symlink train/val/test/<kelas>/ ke data/raw
splits/          train/val/test.csv (path, label, md5, group_id, style), duplicates.csv
configs/         dataset.json, split.yaml, yolo11n_cls.yaml, timm.yaml
scripts/         00–08 + common.py, data_utils.py, models_io.py, notes_log.py, run_stage3.sh
runs/            checkpoint, history, ekspor; _logs/ berisi log; run berawalan _ = arsip/gagal/uji
reports/         audit, split, eval (test), hasil_baseline.md, export.md, benchmark_cpu.md
notes/           eksperimen.md (log semua run)
```

## Catatan teknis (Windows + RTX 2050 4 GB + RAM 7,7 GB)

- `workers: 2` dan worker val tidak persistent. Lebih dari itu memicu WinError 1455 (commit memory habis).
- `channels_last` tidak dipakai karena terukur ~15× lebih lambat di RTX 2050 + cuDNN 9.24.
- `YOLO_AUTOINSTALL=False` di-set di `scripts/common.py`, karena Ultralytics sempat mencoba meng-install `onnxruntime-gpu` ke env.
- Penulisan ulang JPEG "corrupt" oleh Ultralytics dimatikan di `03a`. 17 JPEG tanpa marker FFD9 tetap terbaca penuh.
- `albumentations` asli sudah tidak dipelihara (sejak Juni 2025), sehingga diganti `albumentationsx` (API sama, lisensi AGPL-3.0).
- `torch.onnx.export` memakai exporter dynamo (default di torch 2.x), yang membutuhkan `onnxscript`.
