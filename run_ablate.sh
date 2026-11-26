#!/usr/bin/env bash
# mix3 잡기 값 분리 실험 (no-save): bash run_ablate.sh <과일당 시도> <변형 이름...>
#   변형: V0 (params 없음 = v3 잡기) | params_abl/<이름>.json
#   장면 옵션은 run_mix3.sh 와 같다(--params 만 다름). 시드는 과일별 고정(900 딸기, 901 오렌지, 902 레몬, 903 바나나) → 변형끼리 비슷한 장면
set -uo pipefail
cd ~/fruit_datagen
N=$1; shift
FRUITS=${FRUITS:-"strawberry orange"}
source ~/robot_course/activate.sh
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 COURSE_ASSETS=$HOME/fruit_datagen/assets_c4
OPTS=(--vary-prob 0.9 --vary-use count=0.56,extra=0,robot=0.45 --allow-plain-vary --count3-prob 0.5 --count4-prob 0.3
      --single-random-prob 0.5 --single-mid-prob 1.0 --light-prob 0.33 --dart-prob 0.3 --start-perturb-prob 0.3
      --nearest-margin 0.025 --plate-prob 0.33 --plate-range 0.06 --far-prob 0.2 --far-min 0.30)
declare -A SEED=([strawberry]=900 [orange]=901 [lemon]=902 [banana]=903)
for V in "$@"; do
  PR=(); [ "$V" = V0 ] || PR=(--params "params_abl/$V.json")
  for f in $FRUITS; do
    python gen_scripted.py --dataset "$HOME/datasets/abl_${V}_$f" --fruits "$f" --seed "${SEED[$f]}" "${OPTS[@]}" "${PR[@]}" \
      --no-save --per-fruit "$N" --max-attempts "$N" --snap 2 \
      --log "logs/abl_${V}_$f.jsonl" --viz none > "logs/abl_${V}_$f.log" 2>&1
    echo "[$(date '+%F %H:%M')] abl $V $f: $(grep -a '^SUMMARY' logs/abl_${V}_$f.log | cut -c1-200) $(grep -aE 'Traceback' logs/abl_${V}_$f.log | tail -1)" | tee -a progress.log
  done
done
echo ABL_DONE
