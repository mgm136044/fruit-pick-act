#!/usr/bin/env bash
# (11-26 09:4x 개정) 91점 대응 지시에 따라 B 자동 생성은 뺐다. A 가 끝나면 P3 te −0.05 로컬 선별 40회만 돌린다.
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null; do
  pgrep -f "^bash run_mix3.sh A" >/dev/null || { sleep 5; grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null && break; log "after_A2: A 생성이 완료 표시 없이 끝남 → 중단"; exit 1; }
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
