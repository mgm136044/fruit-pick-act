#!/usr/bin/env bash
# 0단계 기준 데이터: 키트 record.py --fake-leader 로 1 에피소드 (형식 기준용, 창 없이)
#   bash make_ref.sh [기록 초=6] [추가 record.py 인자...]
# record.py 는 stdin 으로 Enter(기록 시작) → Enter(정지) → y(저장) 를 받는다. FIFO 로 로그를 보며 넣는다.
set -uo pipefail
source ~/robot_course/activate.sh
export PYTHONUNBUFFERED=1   # 로그를 바로 보려고 (FIFO 입력 시점 판단)
REC_SEC="${1:-6}"; shift || true
OUT=~/fruit_datagen/ref_fake
LOG=~/fruit_datagen/logs/ref_fake.log
mkdir -p ~/fruit_datagen/logs
FIFO=~/fruit_datagen/logs/ref.fifo; rm -f "$FIFO"; mkfifo "$FIFO"
python "$ROBOT_KIT/isaac/scripts/record.py" --fake-leader --dataset "$OUT" --episodes 1 --no-pack --fruit orange "$@" \
  < "$FIFO" > "$LOG" 2>&1 &
PID=$!
exec 3> "$FIFO"
wait_for() { until grep -q "$1" "$LOG"; do kill -0 $PID 2>/dev/null || { echo "record.py 종료됨"; return 1; }; sleep 1; done; }
wait_for "\[준비\]" && sleep 3 && echo "" >&3 \
  && wait_for "\[기록 중\]" && sleep "$REC_SEC" && echo "" >&3 \
  && wait_for "y + Enter 저장" && echo "y" >&3
wait $PID; RC=$?
exec 3>&-; rm -f "$FIFO"
echo "record.py 종료 코드 $RC"
tail -5 "$LOG"
