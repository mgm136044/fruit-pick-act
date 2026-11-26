#!/usr/bin/env bash
# (11-26 14:1x, 사용자 결정 '스윕 자리에 과제5' 의 2차 시도) DAgger 시험(round 0)이 끝나면 run_dagger_r1a.sh 를 멈춘 채로
# 과제5 창(~/hw5_sim/window1.sh, 빠진 파일 so101_chain.json·fruit_pts·orange_recover.py 복사함)을 돌리고, 끝나면(실패해도) R1a 쪽을 다시 돌린다.
#   bash hw5_after_dagger_test.sh <run_dagger_r1a PID>
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
D=$1
trap 'kill -CONT "$D" 2>/dev/null; log "run_dagger_r1a.sh($D) 다시 돌림 → DAgger 시험 판정 후 R1a"' EXIT
while pgrep -f "dagger_collect.py .*--round 0" >/dev/null; do sleep 20; done
log "DAgger 시험 python 끝 → 과제5 창 다시 시작 (R1a 는 멈춘 채)"
cd ~/hw5_sim && bash window1.sh > logs/window1c.out 2>&1
cd ~/fruit_datagen
log "과제5 창 끝: $(tail -1 ~/hw5_sim/logs/window1.log | cut -c1-200)"
