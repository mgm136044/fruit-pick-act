#!/usr/bin/env bash
# 2차 학습: v3 전체 + v1·v2 변형 장면 + v2 기본 장면 일부(과일별 13) → 기본 장면 비율 약 20% 이하
# --augment 는 안 씀: LeRobot 기본 증강에 RandomAffine(이동 5%, 회전 5°)이 있어 정면 카메라로 위치를 찾는 데 해가 될 수 있음. 조명 변화는 v3 데이터에 실제로 들어 있음
#   bash run_training2.sh <스텝> [배치=8] [이름=act_scripted_v3mix]
set -euo pipefail
STEPS="${1:-40000}"; BATCH="${2:-8}"; NAME="${3:-act_scripted_v3mix}"
source ~/robot_course/activate.sh
cd ~/fruit_datagen
DS=~/datasets/fruit_scripted_mix2
if [ ! -d "$DS" ]; then
  python make_mix.py --out "$DS" \
    --src "$HOME/datasets/fruit_scripted_v3:logs/gen_fruit_scripted_v3.jsonl:all:" \
    --src "$HOME/datasets/fruit_scripted_v2:logs/gen_fruit_scripted_v2.jsonl:all:vary_only" \
    --src "$HOME/datasets/fruit_scripted_v1:logs/gen_fruit_scripted_v1.jsonl:all:vary_only" \
    --src "$HOME/datasets/fruit_scripted_v2:logs/gen_fruit_scripted_v2.jsonl:all:default_only:13" 2>&1 | grep -v -i -E "svt|warn" | tail -40
fi
[ -f "${DS}_train/$(basename $DS).zip" ] || python "$ROBOT_KIT/isaac/scripts/pack_dataset.py" --dataset "$DS"
D=train_runs/$NAME
mkdir -p "$D"
cp "$ROBOT_KIT/train/train.sh" "$ROBOT_KIT/train/train_act.py" "$D/"
ln -sf "${DS}_train/$(basename $DS).zip" "$D/$(basename $DS).zip"
cd "$D"
TRAIN_PYTHON="$(which python)" bash train.sh --steps "$STEPS" --batch-size "$BATCH" --name "$NAME"
