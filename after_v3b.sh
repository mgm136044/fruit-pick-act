#!/usr/bin/env bash
# v3 생성이 끝나면: GPU 기록 중지 → v3 검증 → 2차 학습 (4만 스텝)
# after_v3.sh 의 대기(pgrep) 부분이 실행한 셸 자신과 일치해 멈춰서, 대기 없이 이어서 실행하는 사본
set -uo pipefail
cd ~/fruit_datagen
kill "$(cat logs/gpu_watch_v3.pid)" 2>/dev/null
echo "[$(date '+%F %H:%M')] v3 완료: $(grep SUMMARY logs/gen_fruit_scripted_v3.log | cut -c1-400)" | tee -a progress.log
source ~/robot_course/activate.sh
python verify_dataset.py --dataset ~/datasets/fruit_scripted_v3 --log logs/gen_fruit_scripted_v3.jsonl --out check_v3 > logs/verify_v3.log 2>&1
echo "[$(date '+%F %H:%M')] v3 검증: $(grep -E 'features 비교|디코딩' logs/verify_v3.log | tr '\n' ' ' | cut -c1-300)" | tee -a progress.log
grep -q "features 비교: 동일" logs/verify_v3.log || { echo "v3 형식이 기준과 다름 → 학습 안 함"; exit 1; }
nohup bash gpu_watch.sh logs/gpu_train_v3mix.log > /dev/null 2>&1 & echo $! > logs/gpu_watch_train2.pid
echo "[$(date '+%F %H:%M')] 2차 학습 시작: run_training2.sh 40000 8" | tee -a progress.log
bash run_training2.sh 40000 8 > logs/train_v3mix.log 2>&1
echo "TRAIN2_EXIT=$?"
