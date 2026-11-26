#!/usr/bin/env bash
# v2 생성이 끝난 뒤: 학습 데이터 만들기 → 묶음 → (속도 점검) → 학습. 키트 train.sh / train_act.py 를 그대로 사용.
#   bash run_training.sh <스텝 수> [배치] [이름]     예: bash run_training.sh 100 8 speedtest  /  bash run_training.sh 20000 8
set -euo pipefail
STEPS="${1:-20000}"; BATCH="${2:-8}"; NAME="${3:-act_scripted_v12}"
RUN=act_scripted_v12
source ~/robot_course/activate.sh
cd ~/fruit_datagen
DS=~/datasets/fruit_scripted_act
if [ ! -d "$DS" ]; then
  # 자동 생성 데이터만: v1(다른 종류 방해 과일 제외본) + v2(같은 필터)
  python make_mix.py --out "$DS" \
    --src "$HOME/datasets/fruit_scripted_v1_act:::" \
    --src "$HOME/datasets/fruit_scripted_v2:logs/gen_fruit_scripted_v2.jsonl:all:no_other_fruit" 2>&1 | grep -v -i -E "svt|warn" | tail -30
fi
[ -f "${DS}_train/$(basename $DS).zip" ] || python "$ROBOT_KIT/isaac/scripts/pack_dataset.py" --dataset "$DS"
D=train_runs/$RUN
mkdir -p "$D"
cp "$ROBOT_KIT/train/train.sh" "$ROBOT_KIT/train/train_act.py" "$D/"
ln -sf "${DS}_train/$(basename $DS).zip" "$D/$(basename $DS).zip"
cd "$D"
TRAIN_PYTHON="$(which python)" bash train.sh --steps "$STEPS" --batch-size "$BATCH" --name "$NAME"
