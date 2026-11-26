#!/usr/bin/env bash
# (11-26 09:5x, Fable 검토 반영) A 생성이 끝나면 te −0.05 대신 P0 단독 스윕 72회: 9조건 × 8 (과일당 2), mix2 10만 te 0.01.
# 나머지 조건 표본은 DAgger R1 의 평가형 롤아웃이 채운다.
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null; do
  pgrep -f "^bash run_mix3.sh A" >/dev/null || { sleep 5; grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null && break; log "after_A3: A 생성이 완료 표시 없이 끝남 → 중단"; exit 1; }
  sleep 15
done
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 COURSE_ASSETS=$HOME/fruit_datagen/assets_c4
CONDS="xy6:xy=0.06;xy10:xy=0.10;yaw45:yaw=45;yaw180:yaw=180;rob5:rob=5/15;rob10:rob=10/30;plate6:plate=0.06;cnt23:count=2-3;cnt4:count=4"
log "P0 스윕 시작: mix2 10만 te0.01, 9조건 × 8"
python eval_sweep.py --checkpoint te_models/act_mix2_100k_color_te --conds "$CONDS" --per-cond 8 \
  --gif-dir report_html/assets/sweep_mix2_te01 --out logs/sweep_act_mix2_100k_color_te01.json --viz none > logs/sweep_act_mix2_100k_color_te01.log 2>&1
log "P0 스윕: $(grep -a '^SWEEP' logs/sweep_act_mix2_100k_color_te01.log | cut -c1-400) $(grep -aE 'Traceback|Error' logs/sweep_act_mix2_100k_color_te01.log | tail -1 | cut -c1-200)"
echo SWEEP_DONE
