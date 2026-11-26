#!/usr/bin/env bash
# 체크포인트(pretrained_model 폴더) → 실행 설정만 바꾼 사본 폴더 + 제출용 zip + 키트 형식 검사 + sha256
#   bash make_te_zip.sh <pretrained_model 폴더> <이름> [te 계수=0.01 | nNN]
#   예) bash make_te_zip.sh train_runs/act_mix2_100k_color/outputs/act_mix2_100k_color/checkpoints/last/pretrained_model act_mix2_100k_color_te
#       bash make_te_zip.sh <폴더> act_x_n20 n20        # temporal ensemble 없이 n_action_steps 20
# 결과: te_models/<이름>/ (로컬 평가에 그대로 쓰는 폴더), te_models/<이름>.zip (파일을 zip 맨 위에)
# config.json 의 n_action_steps·temporal_ensemble_coeff 두 값만 바꾼다. 가중치·정규화 파일은 그대로 복사.
set -euo pipefail
SRC="$1"; NAME="$2"; MODE="${3:-0.01}"
cd ~/fruit_datagen
OUT="te_models/$NAME"
[ -e "$OUT" ] && { echo "이미 있음: $OUT (지우지 않음, 다른 이름을 쓰세요)"; exit 1; }
mkdir -p "$OUT"
cp -a "$SRC"/. "$OUT"/
python3 - "$OUT/config.json" "$MODE" <<'EOF'
import json, sys
p, mode = sys.argv[1], sys.argv[2]
c = json.load(open(p))
if mode.startswith("n"):
    c["n_action_steps"], c["temporal_ensemble_coeff"] = int(mode[1:]), None
else:
    c["n_action_steps"], c["temporal_ensemble_coeff"] = 1, float(mode)
json.dump(c, open(p, "w"), indent=4)
print("config:", {k: c[k] for k in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff")})
EOF
(cd "$OUT" && zip -q -X -r "../$NAME.zip" .)
source ~/robot_course/activate.sh >/dev/null 2>&1
(cd ~/student_kit && python -I check_submission.py ~/fruit_datagen/te_models/$NAME.zip 2>&1 | grep -E "^zip|^config|성공|OK|통과|실패")
sha256sum "te_models/$NAME.zip"
