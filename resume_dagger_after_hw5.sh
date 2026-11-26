#!/usr/bin/env bash
# (11-26 14:1x, 사용자 결정: 스윕 자리에 과제5) 과제5 창(~/hw5_sim/window1.sh)이 끝나면 멈춰 둔 run_dagger_r1a.sh 를 다시 돌린다.
#   bash resume_dagger_after_hw5.sh <run_dagger_r1a PID> <window1 PID>
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
D=$1; W=$2
while kill -0 "$W" 2>/dev/null; do sleep 20; done
log "과제5 창 끝 ($(tail -1 ~/hw5_sim/logs/window1.log | cut -c1-200)) → run_dagger_r1a.sh($D) 다시 돌림"
kill -CONT "$D"
