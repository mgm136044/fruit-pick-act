#!/usr/bin/env bash
# mix3 생성 (STRATEGY_W1.md 5장): 과일별로 따로 실행(비중·시드), 같은 옵션, 종류당 4개 장면 자산(assets_c4)
#   bash run_mix3.sh <묶음 A|B|trial> <딸기> <오렌지> <레몬> <바나나> <시드 시작>
#   trial 이면 --no-save (과일마다 시도 수 = 주어진 수), 데이터셋을 만들지 않음
# 과일 하나가 끝나면 logs/mix3_<묶음>_<과일>.done → 다시 실행하면 끝난 과일은 건너뜀.
# 끝나지 않은 과일 데이터셋 폴더가 남아 있으면 지우지 않고 이름을 바꿔 두고 그 과일을 처음부터 다시 만듦.
set -uo pipefail
cd ~/fruit_datagen
PART=$1; SEED=$6
declare -A N=([strawberry]=$2 [orange]=$3 [lemon]=$4 [banana]=$5)
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 COURSE_ASSETS=$HOME/fruit_datagen/assets_c4
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
OPTS=(--vary-prob 0.9 --vary-use count=0.56,extra=0,robot=0.45 --allow-plain-vary --count3-prob 0.5 --count4-prob 0.3
      --single-random-prob 0.5 --single-mid-prob 1.0 --light-prob 0.33 --dart-prob 0.3 --start-perturb-prob 0.3
      --nearest-margin 0.025 --plate-prob 0.33 --plate-range 0.06 --far-prob 0.2 --far-min 0.30 --params params_mix3.json)
[ -f "logs/gen_scripted_mix3${PART}_snapshot.py" ] || cp gen_scripted.py "logs/gen_scripted_mix3${PART}_snapshot.py"
cp params_mix3.json "logs/params_mix3${PART}.json"
log "mix3 $PART 시작: 딸기 ${N[strawberry]} · 오렌지 ${N[orange]} · 레몬 ${N[lemon]} · 바나나 ${N[banana]}, 시드 $SEED~, 옵션 ${OPTS[*]}"
i=0
for f in strawberry orange lemon banana; do
  s=$((SEED + i)); i=$((i + 1))
  [ "${N[$f]}" -gt 0 ] || continue
  [ -f "logs/mix3_${PART}_$f.done" ] && { echo "$f 끝남, 건너뜀"; continue; }
  D=$HOME/datasets/fruit_mix3_${PART}_$f
  if [ "$PART" = trial ]; then
    EXTRA=(--no-save --per-fruit "${N[$f]}" --max-attempts "${N[$f]}" --snap 2)
  else
    if [ -e "$D" ]; then mv "$D" "${D}.partial_$(date +%H%M)"; log "끝나지 않은 $D → .partial 로 이름 바꿈"; fi
    EXTRA=(--per-fruit "${N[$f]}" --max-attempts $((N[$f] * 2)))
  fi
  python gen_scripted.py --dataset "$D" --fruits "$f" --seed "$s" "${OPTS[@]}" "${EXTRA[@]}" \
    --log "logs/gen_fruit_mix3_${PART}_$f.jsonl" --viz none > "logs/gen_fruit_mix3_${PART}_$f.log" 2>&1
  sm=$(grep -a "^SUMMARY" "logs/gen_fruit_mix3_${PART}_$f.log" | cut -c1-300)
  if [ -n "$sm" ]; then
    touch "logs/mix3_${PART}_$f.done"
    log "mix3 $PART $f: $sm"
  else
    log "mix3 $PART $f: SUMMARY 없음 (중단?) $(grep -aE 'Traceback|Error' logs/gen_fruit_mix3_${PART}_$f.log | tail -1 | cut -c1-200)"
    exit 1
  fi
done
echo "MIX3_${PART}_DONE"
