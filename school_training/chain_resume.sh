#!/usr/bin/env bash
# 학교 GPU 세션 연결 실행 (사람·에이전트가 없어도 이어 감):
#   앞 학습(--prev)이 끝나고, 그 자동 업로드(auto_finish_upload.py --name <prev>)도 끝나면
#   → outputs/<prev> 를 outputs/<new> 로 복사(원본은 그대로) → 새 데이터로 이어 학습(train.sh 가 저장 지점을 보고 --resume 으로 이어 감)
#   → 학습이 끝나면 auto_finish_upload.py 로 zip·형식 검사·비공개 업로드·서버 대조
#
# 학습 폴더(train.sh·train_act.py·train.log·auto_finish_upload.py·새 데이터 zip 이 있는 곳)에서, HF 로그인이 되는 셸로:
#   nohup bash chain_resume.sh --prev act_mix3A_100k --prev-steps 100000 --new act_mix3A_R1a_170k \
#       --data fruit_mix3_A_R1a.zip --steps 170000 --expect-episodes 802 --repo mgm136044/act_mix3 \
#       --zips "170000:plain:submission_act_mix3A_R1a_170k,170000:te:submission_act_mix3A_R1a_170k_te,150000:te:submission_act_mix3A_R1a_150k_te" \
#       --note "..." > chain_act_mix3A_R1a_170k.log 2>&1 &
# 로그 마지막 줄이 CHAIN_DONE 이면 성공. ✘ 줄에서 멈췄으면 그 줄이 이유다.
# 시험용: CHAIN_POLL(기다리기 간격, 기본 60초), CHAIN_CHECK_WAIT(이어 학습 시작 확인까지, 기본 300초), --dry-run(업로드 안 함)
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
HERE=$PWD
PREV= PREV_STEPS=100000 NEW= DATA= STEPS= EXPECT= REPO= ZIPS= NOTE= DRY=
while [ $# -gt 0 ]; do
  case "$1" in
    --prev) PREV=$2 ;; --prev-steps) PREV_STEPS=$2 ;; --new) NEW=$2 ;; --data) DATA=$2 ;; --steps) STEPS=$2 ;;
    --expect-episodes) EXPECT=$2 ;; --repo) REPO=$2 ;; --zips) ZIPS=$2 ;; --note) NOTE=$2 ;;
    --dry-run) DRY=--dry-run; shift; continue ;;
    *) echo "모르는 옵션: $1"; exit 2 ;;
  esac
  shift 2
done
POLL=${CHAIN_POLL:-60}; CHECK_WAIT=${CHAIN_CHECK_WAIT:-300}
log() { echo "[$(date '+%F %T')] $*"; }
die() { log "✘ $1"; exit "${2:-2}"; }
for v in PREV NEW DATA STEPS REPO ZIPS; do [ -n "${!v}" ] || die "--${v,,} 가 없음"; done

# 0) 띄우는 순간 확인 (몇 시간 기다린 뒤 실패하지 않게)
PY=$(grep -o '학습 시작 · .*' train.log 2>/dev/null | tail -1 | sed 's/^학습 시작 · //')
[ -n "$PY" ] && [ -x "$PY" ] || die "train.log 에서 학습 python 을 못 찾음"
"$PY" -c "import lerobot, huggingface_hub" 2>/dev/null || die "$PY 에 lerobot/huggingface_hub 없음"
[ -f "$DATA" ] || die "새 데이터 $DATA 없음"
[ -f auto_finish_upload.py ] || die "auto_finish_upload.py 없음"
[ -d "outputs/$PREV" ] || die "outputs/$PREV 없음"
[ -e "outputs/$NEW" ] && die "outputs/$NEW 가 이미 있음 (지우지 않는다. 사람이 확인)"
WHO=$(env -u HF_HUB_OFFLINE "$PY" -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])" 2>/dev/null) \
  || die "HF 로그인 안 됨 → 사람이 토큰을 넣은 같은 셸에서 다시 띄울 것"
log "시작: $PREV($PREV_STEPS) 끝 → $NEW 로 $DATA 이어 학습 $STEPS → $REPO 업로드${DRY:+ (DRY-RUN)} · HF 사용자 $WHO · python $PY"

# 1) 앞 학습 끝 + 앞 자동 업로드 끝 기다리기
PSTEP=$(printf "%06d" "$PREV_STEPS")
miss=0; last_note=0
while :; do
  trn=$( { pgrep -f -- "--output_dir=$HERE/outputs/$PREV( |\$)"; pgrep -f -- "train_act.py .*--name $PREV( |\$)"; } | sort -u | tr '\n' ' ')
  af=$(pgrep -f -- "auto_finish_upload.py --name $PREV( |\$)" | tr '\n' ' ')
  ready=0; [ -d "outputs/$PREV/checkpoints/$PSTEP/training_state" ] && ready=1
  if [ -z "$trn" ] && [ -z "$af" ] && [ $ready = 1 ]; then break; fi
  if [ -z "$trn" ] && [ $ready = 0 ]; then
    miss=$((miss + 1)); [ $miss -ge 5 ] && die "앞 학습($PREV)이 없는데 $PREV_STEPS 저장 지점도 없음 → 앞 학습이 멈춘 것 같다. train.log 확인" 3
  else
    miss=0
  fi
  if [ $(( $(date +%s) - last_note )) -ge 1800 ]; then
    log "기다리는 중: 앞 학습 pid [${trn}] · 앞 자동 업로드 pid [${af}] · $PREV_STEPS 저장 지점 $([ $ready = 1 ] && echo 있음 || echo 아직)"
    last_note=$(date +%s)
  fi
  sleep "$POLL"
done
log "앞 학습 끝. 앞 자동 업로드 로그 끝 줄: $(tail -1 "auto_finish_$PREV.log" 2>/dev/null || echo '(로그 없음 — 앞 학습 zip 은 따로 확인)')"

# 2) 복사 → 이어 학습 (끊기면 마지막 저장 지점부터 2번까지 다시)
need=$(( $(du -sb "outputs/$PREV" | cut -f1) / 1000000000 + 12 ))
free=$(( $(df -B1 --output=avail . | tail -1) / 1000000000 ))
[ "$free" -ge "$need" ] || die "디스크 여유 ${free}GB < 필요 ${need}GB (복사 + 새 저장 지점)"
cp -a "outputs/$PREV" "outputs/$NEW" || die "복사 실패"
log "복사: outputs/$PREV → outputs/$NEW (원본 그대로)"
rc=1
for try in 1 2 3; do
  bash train.sh --data "$DATA" --steps "$STEPS" --name "$NEW" >> "train_$NEW.out" 2>&1 &
  TP=$!
  if [ $try = 1 ]; then
    sleep "$CHECK_WAIT"
    ok=1
    grep -q "이어서 학습: $PREV_STEPS → $STEPS" "train_$NEW.out" || ok=0
    epl=$(grep -m1 -o "에피소드 [0-9]*개, 프레임 [0-9]*개" "train_$NEW.out" || true)
    [ -n "$EXPECT" ] && ! echo "$epl" | grep -q "에피소드 ${EXPECT}개" && ok=0
    if [ $ok = 0 ] || ! kill -0 $TP 2>/dev/null; then
      kill $TP 2>/dev/null; pkill -f -- "--output_dir=$HERE/outputs/$NEW( |\$)"; pkill -f -- "train_act.py .*--name $NEW( |\$)"
      die "이어 학습 시작 확인 실패 (기대: '이어서 학습: $PREV_STEPS → $STEPS', 에피소드 ${EXPECT:-?}개 / 실제: '$(grep -m1 '이어서 학습\|학습 시작:' "train_$NEW.out")', '$epl') → train_$NEW.out 확인" 4
    fi
    log "이어 학습 시작 확인: $(grep -m1 '이어서 학습' "train_$NEW.out") · $epl"
  fi
  wait $TP; rc=$?
  [ $rc = 0 ] && break
  log "학습 종료 코드 $rc (시도 $try/3) → 60초 뒤 마지막 저장 지점부터 다시"
  sleep 60
done
[ $rc = 0 ] || die "학습이 3번 모두 멈춤 → train_$NEW.out 확인" 5
log "학습 끝: $(grep -o '완료: .*' "train_$NEW.out" | tail -1)"

# 3) zip · 형식 검사 · 업로드
"$PY" auto_finish_upload.py --name "$NEW" --repo "$REPO" --zips "$ZIPS" --note "$NOTE" --poll 30 $DRY > "auto_finish_$NEW.log" 2>&1
rc=$?
log "자동 업로드 종료 코드 $rc · 끝 줄: $(tail -1 "auto_finish_$NEW.log")"
[ $rc = 0 ] || die "자동 업로드 실패 → auto_finish_$NEW.log 확인" 6
echo CHAIN_DONE
