#!/usr/bin/env bash
# run_eval.sh 가 끝나면: 같은 체크포인트로 실행 설정만 바꿔 기본 장면 + 같은 과일 여러 개 장면 비교 (재학습 없음)
#   bash run_variants.sh [이름=act_scripted_v12] [장면당 에피소드=8]
set -uo pipefail
NAME="${1:-act_scripted_v12}"; PER="${2:-8}"
cd ~/fruit_datagen
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
until grep -q EVAL_DONE "logs/run_eval_${NAME#act_scripted_}.log" 2>/dev/null || grep -q EVAL_DONE logs/run_eval_v12.log 2>/dev/null; do sleep 30; done
CK="train_runs/act_scripted_v12/outputs/$NAME/checkpoints/last/pretrained_model"
for V in "n50:--n-action-steps 50" "n20:--n-action-steps 20" "te01:--temporal-ensemble 0.01"; do
  TAG="${V%%:*}"; OPT="${V#*:}"
  python eval_proxy.py --checkpoint "$CK" --scenes default,count --per-scene "$PER" $OPT \
    --gif-dir "report_html/assets/policy_${NAME}_${TAG}" --gif-per-scene 2 \
    --out "logs/eval_proxy_${NAME}_${TAG}.json" --viz none > "logs/eval_proxy_${NAME}_${TAG}.log" 2>&1
  echo "[$(date '+%F %H:%M')] 변형 실행설정 ${TAG}: $(grep -E '^SCORE' logs/eval_proxy_${NAME}_${TAG}.log | cut -c1-300)" | tee -a progress.log
done
echo VARIANTS_DONE
