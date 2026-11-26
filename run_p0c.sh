#!/usr/bin/env bash
# P0-c 로컬 보정 + P0-d 그리퍼 재매핑 진단 (STRATEGY_W1.md 4·8장)
#   W0(after_gpu_free.sh) 가 끝나면(로그 문구로 판단) 시작:
#   1) 10만 모델 te0.01 사본(config 만 바꿈)으로 W0 와 같은 장면·시드 48 → 같은 과일 4개 16  (n100 과 짝 비교)
#   2) 진단: te + 그리퍼 재매핑 1.5 로 기본·같은 과일 16 (제출물 아님)
set -uo pipefail
cd ~/fruit_datagen
NAME=act_mix2_100k_color
TE=te_models/${NAME}_te
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q AFTER_GPU_EVAL_DONE logs/after_gpu_free.log 2>/dev/null; do
  pgrep -f "^bash after_gpu_free.sh" >/dev/null || { log "P0-c: W0 대기 스크립트가 끝났는데 완료 표시가 없음 → 중단"; exit 1; }
  sleep 30
done
sleep 20
[ -d "$TE" ] || { log "P0-c: $TE 없음 → 중단"; exit 1; }
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
log "P0-c 시작: $NAME te0.01 (config 사본) W0 와 같은 장면 48 + 4개 16"
python eval_proxy.py --checkpoint "$TE" --per-scene 8 --gif-dir "report_html/assets/policy_${NAME}_te01" --gif-per-scene 2 \
  --out "logs/eval_proxy_${NAME}_te01.json" --viz none > "logs/eval_proxy_${NAME}_te01.log" 2>&1
log "P0-c te 48: $(grep -E '^SCORE' logs/eval_proxy_${NAME}_te01.log | cut -c1-300) $(grep -aE 'Traceback' logs/eval_proxy_${NAME}_te01.log | tail -1)"
COURSE_ASSETS=~/fruit_datagen/assets_c4 python eval_proxy.py --checkpoint "$TE" --scenes count4 --per-scene 16 \
  --gif-dir "report_html/assets/policy_${NAME}_te01_c4" --gif-per-scene 2 \
  --out "logs/eval_proxy_${NAME}_c4te01.json" --viz none > "logs/eval_proxy_${NAME}_c4te01.log" 2>&1
log "P0-c te 4개 16: $(grep -E '^SCORE' logs/eval_proxy_${NAME}_c4te01.log | cut -c1-200) $(grep -aE 'Traceback' logs/eval_proxy_${NAME}_c4te01.log | tail -1)"
python eval_proxy.py --checkpoint "$TE" --scenes default,count --per-scene 8 --grip-remap 1.5 \
  --out "logs/diag_griprmp_${NAME}_te01.json" --viz none > "logs/diag_griprmp_${NAME}_te01.log" 2>&1
log "P0-d 진단(te + 그리퍼 재매핑 1.5, 기본·같은 과일 16, 제출물 아님): $(grep -E '^SCORE' logs/diag_griprmp_${NAME}_te01.log | cut -c1-200)"
echo P0C_DONE
