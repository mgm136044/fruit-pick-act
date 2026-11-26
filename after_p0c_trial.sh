#!/usr/bin/env bash
# P0-c·P0-d 가 끝나면(로그 문구) mix3 시험 생성 (과일당 10회, 저장 안 함). A 본 생성은 결과를 보고 따로 시작한다.
cd ~/fruit_datagen
until grep -q P0C_DONE logs/run_p0c.log 2>/dev/null; do
  pgrep -f "^bash run_p0c.sh" >/dev/null || { echo "run_p0c.sh 가 완료 표시 없이 끝남 → 시험 안 함"; exit 1; }
  sleep 30
done
sleep 15
bash run_mix3.sh trial 10 10 10 10 900
