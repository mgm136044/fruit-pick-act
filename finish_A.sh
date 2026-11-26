#!/usr/bin/env bash
# mix3 A 마무리 (CPU 만): A 생성 완료(MIX3_A_DONE) → 과일별 검증 → 합치기(scene_index 포함) → 키트 pack_dataset → sha256·README
#   nohup setsid bash finish_A.sh > logs/finish_A.log 2>&1 &
# 결과: ~/datasets/fruit_mix3_A_train/{fruit_mix3_A.zip, train.sh, train_act.py, README.md} — HF 업로드는 사용자가 한다.
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null; do
  pgrep -f "^bash run_mix3.sh A" >/dev/null || { sleep 5; grep -q MIX3_A_DONE logs/run_mix3_A.log 2>/dev/null && break; log "finish_A: A 생성이 완료 표시 없이 끝남 → 중단"; exit 1; }
  sleep 30
done
source ~/robot_course/activate.sh
FR="strawberry orange lemon banana"
for f in $FR; do
  python -I verify_dataset.py --dataset ~/datasets/fruit_mix3_A_$f --log logs/gen_fruit_mix3_A_$f.jsonl --out check_mix3A_$f \
    > logs/verify_mix3A_$f.log 2>&1 || { log "finish_A: verify $f 실패 → 중단 (logs/verify_mix3A_$f.log)"; exit 1; }
  log "finish_A verify $f: $(grep -aE 'OK|NaN|에피소드|frames|불일치|다름' logs/verify_mix3A_$f.log | tail -3 | tr '\n' ' ' | cut -c1-250)"
done
SRC=()
for f in $FR; do SRC+=(--src "$HOME/datasets/fruit_mix3_A_$f:logs/gen_fruit_mix3_A_$f.jsonl:all:"); done
python -I make_mix.py --out ~/datasets/fruit_mix3_A "${SRC[@]}" > logs/make_mix_mix3A.log 2>&1 || { log "finish_A: make_mix 실패 → 중단"; exit 1; }
log "finish_A merge: $(grep -a '^MERGED' logs/make_mix_mix3A.log)"
python -I ~/student_kit/isaac/scripts/pack_dataset.py --dataset ~/datasets/fruit_mix3_A > logs/pack_mix3A.log 2>&1 || { log "finish_A: pack 실패 → 중단"; exit 1; }
T=~/datasets/fruit_mix3_A_train
ls -la "$T" >> logs/pack_mix3A.log
python3 -I mix3_stats.py logs/gen_fruit_mix3_A_*.jsonl --md logs/mix3A_stats.md > logs/mix3A_stats.log 2>&1
SHA=$(sha256sum "$T/fruit_mix3_A.zip" | cut -d' ' -f1); SIZE=$(stat -c %s "$T/fruit_mix3_A.zip")
EP=$(python3 -I -c "import json;d=json.load(open('$HOME/datasets/fruit_mix3_A/meta/info.json'));print(d['total_episodes'],d['total_frames'])")
cat > "$T/README.md" <<EOR
# fruit_mix3_A (비공개 · 스크립트 자동 시연, 사람 시연 아님)

- 에피소드·프레임: $EP (30fps, front·wrist 480x640, state/action 6)
- \`fruit_mix3_A.zip\`: $SIZE bytes, sha256 \`$SHA\`
- 생성: \`gen_scripted.py\` mix3 옵션(\`run_mix3.sh A 165 138 138 110 1000\`), 잡기 값 \`params_mix3.json\` 2판(가운데 맞춤, v3 쥐는 깊이), \`--keep-kind-on-relayout\`
- 구성: 같은 과일 2~4개 약 55%(4개 약 15~20%), 가장 가까운 것을 집음(둘째와 2.5cm 이상 차이), 다른 종류 과일 0%, 로봇 이동·접시 이동·조명·DART 겹침
- 과일 비중: 딸기 165 · 오렌지 138 · 레몬 138 · 바나나 110 (목표)
- 에피소드별 장면 기록: zip 안 \`scene_index.json\`
- 학습 코드: \`train.sh\`, \`train_act.py\` (수업 키트, 수정 없음)

## 구성 통계
$(cat logs/mix3A_stats.md)
EOR
log "finish_A 완료: $T (zip $SIZE bytes, sha256 ${SHA:0:16}…, $EP)"
echo FINISH_A_DONE
