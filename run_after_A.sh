#!/usr/bin/env bash
# mix3 A 생성이 끝나면(logs/run_mix3_A.log 의 MIX3_A_DONE) 바로:
#   1) P3 로컬 선별: 10만 모델 te −0.05 사본으로 P0-c 와 같은 서버형 40장면 (방해 과일 제외, 같은 시드 123 → 같은 장면)
#   2) mix3 B 생성 시작 (A 와 같은 옵션·과일 비중, 시드 2000~)
# A 가 완료 표시 없이 끝나면 아무것도 하지 않는다.
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null; do
  pgrep -f "^bash run_mix3.sh A" >/dev/null || { sleep 5; grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null && break; log "after_A: A 생성이 완료 표시 없이 끝남 → 중단"; exit 1; }
  sleep 20
done
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
TE=te_models/act_mix2_100k_color_tem05
log "P3 선별 시작: $TE (te −0.05) 서버형 40장면, P0-c 와 같은 시드"
python eval_proxy.py --checkpoint "$TE" --scenes default,count,robot,light,mixed --per-scene 8 \
  --gif-dir report_html/assets/policy_act_mix2_100k_color_tem05 --gif-per-scene 2 \
  --out logs/eval_proxy_act_mix2_100k_color_tem05.json --viz none > logs/eval_proxy_act_mix2_100k_color_tem05.log 2>&1
log "P3 te −0.05 40: $(grep -E '^SCORE' logs/eval_proxy_act_mix2_100k_color_tem05.log | cut -c1-300) $(grep -aE 'Traceback' logs/eval_proxy_act_mix2_100k_color_tem05.log | tail -1)"
echo P3_DONE
bash run_mix3.sh B 165 138 138 110 2000 > logs/run_mix3_B.log 2>&1
echo AFTER_A_DONE
