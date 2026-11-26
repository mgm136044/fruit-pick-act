#!/usr/bin/env bash
# (11-26 14:1x) 스윕 자리에 과제5 를 넣어서 P0 스윕 남은 7조건을 R1a 뒤로 옮김. xy6·xy10 은 14:09 까지 16회 끝남(*_partial_xy6xy10.log).
# R1a 가 정상으로 끝났을 때(DAGGER_R1A_DONE)만 돈다. DAgger 시험 실패로 R1a 가 안 돌았으면 GPU 를 수정·재시험에 쓰도록 그냥 끝낸다.
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q DAGGER_R1A_DONE logs/run_dagger_r1a.log 2>/dev/null; do
  pgrep -f "^bash run_dagger_r1a.sh" >/dev/null || { sleep 5; grep -q DAGGER_R1A_DONE logs/run_dagger_r1a.log 2>/dev/null && break; log "sweep_after_R1a: R1a 가 완료 표시 없이 끝남 → 스윕 안 함"; exit 1; }
  sleep 30
done
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 COURSE_ASSETS=$HOME/fruit_datagen/assets_c4
CONDS="yaw45:yaw=45;yaw180:yaw=180;rob5:rob=5/15;rob10:rob=10/30;plate6:plate=0.06;cnt23:count=2-3;cnt4:count=4"
log "P0 스윕 나머지 시작: mix2 10만 te0.01, 7조건 × 8"
python eval_sweep.py --checkpoint te_models/act_mix2_100k_color_te --conds "$CONDS" --per-cond 8 \
  --gif-dir report_html/assets/sweep_mix2_te01 --out logs/sweep_act_mix2_100k_color_te01_rest7.json --viz none > logs/sweep_act_mix2_100k_color_te01_rest7.log 2>&1
log "P0 스윕 나머지: $(grep -a '^SWEEP' logs/sweep_act_mix2_100k_color_te01_rest7.log | cut -c1-400) $(grep -aE 'Traceback' logs/sweep_act_mix2_100k_color_te01_rest7.log | tail -1 | cut -c1-200)"
echo SWEEP_REST_DONE
