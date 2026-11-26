#!/usr/bin/env bash
# P0 스윕(run_after_A3.sh)이 끝나면: DAgger 시험 16회(저장 시험 포함) → 넘겨받은 것 중 전문가 성공 ≥ 60% 면 R1a 본 수집.
# R1a: mix2 10만 te0.01 이 운전, 10조건(9 + mix), 조건당 36(같은 과일·mix 1.5배), 저장 260 에서 멈춤. 조건이 섞여 돌아 중간에 멈춰도 고름.
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q SWEEP_DONE logs/run_after_A3.log 2>/dev/null; do
  pgrep -f "^bash run_after_A3.sh" >/dev/null || { sleep 5; grep -q SWEEP_DONE logs/run_after_A3.log 2>/dev/null && break; log "dagger_r1a: 스윕이 완료 표시 없이 끝남 → 중단"; exit 1; }
  sleep 15
done
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 COURSE_ASSETS=$HOME/fruit_datagen/assets_c4
DRV=te_models/act_mix2_100k_color_te
log "DAgger 시험 16회 시작 (운전 $DRV)"
python dagger_collect.py --checkpoint $DRV --dataset ~/datasets/fruit_dagger_test --attempts 2 --round 0 --seed 5 \
  --conds "cnt23:count=2-3;cnt4:count=4;rob10:rob=10/30;xy10:xy=0.10;yaw180:yaw=180;plate6:plate=0.06;mix:count=2-4,rob=10/30,plate=0.06;xy6:xy=0.06" \
  --early-frac 0.3 --control-frac 0 --gif-dir report_html/assets/dagger_test --gif-n 16 --viz none > logs/dagger_test.log 2>&1
SM=$(grep -a '^SUMMARY' logs/dagger_test.log)
log "DAgger 시험: ${SM:-SUMMARY 없음} $(grep -aE 'Traceback|Error' logs/dagger_test.log | tail -1 | cut -c1-200)"
RATE=$(python3 -I - <<'PY'
import json
rs=[json.loads(l) for l in open('/home/mingyeongmin/fruit_datagen/logs/dagger_fruit_dagger_test.jsonl')]
tk=[r for r in rs if r.get('result') in ('saved','expert_fail')]
print(round(sum(r['result']=='saved' for r in tk)/len(tk),2) if tk else 0)
PY
)
log "DAgger 시험: 넘겨받은 것 중 전문가 성공률 $RATE"
echo "DAGGER_TEST_DONE rate=$RATE"
python3 -I -c "import sys; sys.exit(0 if float('$RATE') >= 0.6 else 1)" || { log "DAgger 시험 성공률 $RATE < 0.6 → R1a 시작 안 함 (GIF 확인 후 수정)"; exit 1; }
log "DAgger R1a 시작: 운전 $DRV, 조건당 36(cnt23·cnt4·mix 1.5배), 최대 저장 260"
python dagger_collect.py --checkpoint $DRV --dataset ~/datasets/fruit_dagger_r1a --attempts 36 --round 1 --seed 100 \
  --cond-weights cnt23=1.5,cnt4=1.5,mix=1.5 --max-saved 260 --gif-dir report_html/assets/dagger_r1a --gif-n 20 --viz none > logs/dagger_r1a.log 2>&1
log "DAgger R1a: $(grep -a '^SUMMARY' logs/dagger_r1a.log | cut -c1-400) $(grep -aE 'Traceback|Error' logs/dagger_r1a.log | tail -1 | cut -c1-200)"
echo DAGGER_R1A_DONE
