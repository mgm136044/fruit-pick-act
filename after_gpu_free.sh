#!/usr/bin/env bash
# 학교 GPU 모델(act_mix2_100k_color) 평가: GPU 가 비면 바로 시작 (사용자 요청 11-26 04:0x)
#   GPU 가 빈 상태 = 계산 프로세스 없음(gnome-remote-desktop 제외) + 사용 메모리 1000MiB 미만
#                    + ~/jobs/*/run.sh 대기열 없음 → 3분 연속이면 시작 (다른 학습과 6GB 를 나눠 쓰지 않게)
#   → run_eval.sh (run_policy 24 + eval_proxy 장면당 8 + GIF + 보고서) → 같은 과일 4개 장면 16 → 보고서 다시
# pgrep 패턴은 실행 경로(/bin/bash ~/jobs/...)로 고정해 이 스크립트 자신과 일치하지 않게 함 (after_v3.sh 교훈)
set -uo pipefail
cd ~/fruit_datagen
NAME=act_mix2_100k_color
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
gpu_free() {
  local used apps
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  apps=$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -v gnome-remote-desktop | grep -c .)
  [ "${used:-99999}" -lt 1000 ] && [ "$apps" -eq 0 ] && ! pgrep -f "^/bin/bash $HOME/jobs/[^ ]*/run.sh" >/dev/null
}
wait_gpu() { local ok=0; until [ $ok -ge 3 ]; do if gpu_free; then ok=$((ok + 1)); else ok=0; fi; sleep 60; done; }

log "$NAME 평가 대기 시작 (after_gpu_free.sh): GPU 가 3분 연속 비면 시작"
wait_gpu
log "GPU 빔 → $NAME 평가 시작 (run_policy 24 → eval_proxy 장면당 8 → 같은 과일 4개 16, 약 1.5시간)"
bash run_eval.sh "$NAME" > "logs/run_eval_${NAME}.log" 2>&1
grep -q EVAL_DONE "logs/run_eval_${NAME}.log" || { log "$NAME 평가 중단: $(tail -2 logs/run_eval_${NAME}.log | tr '\n' ' ')"; exit 1; }

wait_gpu  # 그사이 다른 작업이 GPU 를 잡았으면 끝날 때까지 기다림
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
CK="train_runs/$NAME/outputs/$NAME/checkpoints/last/pretrained_model"
COURSE_ASSETS=~/fruit_datagen/assets_c4 python eval_proxy.py --checkpoint "$CK" --scenes count4 --per-scene 16 \
  --gif-dir "report_html/assets/policy_${NAME}_c4" --gif-per-scene 4 \
  --out "logs/eval_proxy_${NAME}_c4.json" --viz none > "logs/eval_proxy_${NAME}_c4.log" 2>&1
log "$NAME 같은 과일 4개 장면(assets_c4) 16: $(grep -E '^SCORE' logs/eval_proxy_${NAME}_c4.log | cut -c1-300) $(grep -aE 'Traceback' logs/eval_proxy_${NAME}_c4.log | tail -1 | cut -c1-200)"
python make_report.py --no-gif > /dev/null 2>&1 && log "보고서 갱신 (1차·2차·학교 10만 비교)" || log "보고서 갱신 실패 (평가 결과는 logs/ 에 있음)"
echo "AFTER_GPU_EVAL_DONE"
