#!/usr/bin/env bash
# GPU 메모리 사용량을 1초마다 기록 (사용법: bash gpu_watch.sh 로그파일 &)
while true; do
  echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
  sleep 1
done >> "$1"
