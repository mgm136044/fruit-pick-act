#!/usr/bin/env bash
# 2차 학습이 끝나면: run_eval.sh(run_policy 24 + eval_proxy 장면당 8 + GIF + 보고서)
#   → 같은 과일 4개 장면(서버의 '바나나 4개 중 하나', assets_c4 사본) 16에피소드
#   → run_variants.sh(n_action_steps 50·20, temporal ensemble)
# 대기는 로그 파일 문구(TRAIN2_EXIT)로만 판단한다 (pgrep 이 실행한 셸 명령줄과 일치해 멈춘 after_v3.sh 의 교훈)
set -uo pipefail
cd ~/fruit_datagen
NAME=act_scripted_v3mix
until grep -q "TRAIN2_EXIT" logs/after_v3.log 2>/dev/null; do sleep 60; done
grep -q "TRAIN2_EXIT=0" logs/after_v3.log || { echo "[$(date '+%F %H:%M')] 2차 학습 비정상 종료: $(grep TRAIN2_EXIT logs/after_v3.log) → 평가 안 함" | tee -a progress.log; exit 1; }
kill "$(cat logs/gpu_watch_train2.pid)" 2>/dev/null
echo "[$(date '+%F %H:%M')] 2차 학습 완료 → 2차 평가 시작" | tee -a progress.log
bash run_eval.sh "$NAME" > "logs/run_eval_v3mix.log" 2>&1
grep -q EVAL_DONE logs/run_eval_v3mix.log || { echo "[$(date '+%F %H:%M')] 2차 평가 중단: $(tail -2 logs/run_eval_v3mix.log | tr '\n' ' ')" | tee -a progress.log; exit 1; }

source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
CK="train_runs/$NAME/outputs/$NAME/checkpoints/last/pretrained_model"
COURSE_ASSETS=~/fruit_datagen/assets_c4 python eval_proxy.py --checkpoint "$CK" --scenes count4 --per-scene 16 \
  --gif-dir "report_html/assets/policy_${NAME}_c4" --gif-per-scene 4 \
  --out "logs/eval_proxy_${NAME}_c4.json" --viz none > "logs/eval_proxy_${NAME}_c4.log" 2>&1
echo "[$(date '+%F %H:%M')] 같은 과일 4개 장면(assets_c4) 16: $(grep -E '^SCORE' logs/eval_proxy_${NAME}_c4.log | cut -c1-300) $(grep -aE 'Traceback|Error' logs/eval_proxy_${NAME}_c4.log | tail -1 | cut -c1-200)" | tee -a progress.log

bash run_variants.sh "$NAME" 8 > "logs/run_variants_v3mix.log" 2>&1
echo "AFTER_TRAIN2_DONE"
