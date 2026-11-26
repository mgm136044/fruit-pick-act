#!/usr/bin/env bash
# 학습이 끝나면 평가: 키트 run_policy(기본 환경, 24에피소드) → eval_proxy(서버형 변형 장면 + 다른 종류 과일 참고) → HTML 보고서 갱신
#   bash run_eval.sh [이름=act_scripted_v12] [학습 폴더=train_runs/<이름>]
#   (pgrep 은 python 으로 시작하는 프로세스만 본다: 실행한 셸의 명령줄과 일치해 멈추는 것 방지)
set -uo pipefail
NAME="${1:-act_scripted_v12}"
cd ~/fruit_datagen
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
D="${2:-train_runs/$NAME}"   # 학습 폴더 (1차: train_runs/act_scripted_v12, 2차: train_runs/act_scripted_v3mix)
until [ -f "$D/submission_${NAME}.zip" ]; do
  pgrep -f "^[^ ]*python[^ ]* .*train_act.py.*--name $NAME" >/dev/null || pgrep -f "^[^ ]*python[^ ]* .*lerobot_train.*job_name=$NAME" >/dev/null || { [ -f "$D/submission_${NAME}.zip" ] || { echo "학습 프로세스가 없는데 submission zip 도 없음 → 중단"; exit 1; }; }
  sleep 30
done
CK="$D/outputs/$NAME/checkpoints/last/pretrained_model"
echo "[$(date '+%F %H:%M')] 학습 완료 확인: $CK" | tee -a progress.log
nohup bash gpu_watch.sh logs/gpu_eval_${NAME}.log >/dev/null 2>&1 & GW=$!
python "$ROBOT_KIT/isaac/scripts/run_policy.py" --checkpoint "$CK" --episodes 24 --viz none > "logs/eval_run_policy_${NAME}.log" 2>&1
echo "[$(date '+%F %H:%M')] run_policy 24: $(grep -E '^성공률' logs/eval_run_policy_${NAME}.log)" | tee -a progress.log
python eval_proxy.py --checkpoint "$CK" --per-scene 8 --gif-dir "report_html/assets/policy_${NAME}" --gif-per-scene 4 \
  --out "logs/eval_proxy_${NAME}_n100.json" --viz none > "logs/eval_proxy_${NAME}_n100.log" 2>&1
echo "[$(date '+%F %H:%M')] eval_proxy (n_action_steps 기본): $(grep -E '^SCORE' logs/eval_proxy_${NAME}_n100.log | cut -c1-400)" | tee -a progress.log
kill $GW 2>/dev/null
python make_report.py --no-gif > /dev/null 2>&1 && echo "보고서 갱신 (1차·2차 비교)"
echo EVAL_DONE
