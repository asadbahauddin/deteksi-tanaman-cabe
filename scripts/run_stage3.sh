#!/usr/bin/env bash
# Tahap 3: jalankan semua training berurutan (GPU 4 GB -> satu per satu). Log: runs/_logs/<run>.log
# Pakai: bash scripts/run_stage3.sh            (dari root proyek, env conda "cabai")
set -u
cd "$(dirname "$0")/.."
mkdir -p runs/_logs
export PYTHONUNBUFFERED=1
PY="conda run --no-capture-output -n cabai python -u"

run() {  # run <nama_log> <script> [args...]
  local name=$1; shift
  if [ -f runs/_logs/STOP ]; then echo "[$(date '+%F %T')] STOP file ada, lewati $name"; return; fi
  echo "[$(date '+%F %T')] START $name"
  if $PY "$@" > "runs/_logs/$name.log" 2>&1; then
    echo "[$(date '+%F %T')] DONE  $name"
  else
    echo "[$(date '+%F %T')] FAIL  $name (lihat runs/_logs/$name.log)"
  fi
}

# run yolo11n_cls              scripts/03a_train_yolo_cls.py
# run mobilenetv3_large_100_augA scripts/03b_train_timm.py --model mobilenetv3_large_100 --aug A
# run mobilenetv3_large_100_augB scripts/03b_train_timm.py --model mobilenetv3_large_100 --aug B
# run efficientnet_b0_augA     scripts/03b_train_timm.py --model efficientnet_b0 --aug A
run efficientnet_b0_augB     scripts/03b_train_timm.py --model efficientnet_b0 --aug B
echo "[$(date '+%F %T')] ALL DONE"
