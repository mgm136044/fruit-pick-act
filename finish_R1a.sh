#!/usr/bin/env bash
# DAgger R1a 마무리 (CPU): R1a 완료(DAGGER_R1A_DONE) → 품질 검사 → A(과일별 4개 + 생성 로그) + R1a(+ DAgger 로그) 합치기 → pack → sha256·README
#   nohup setsid bash finish_R1a.sh > logs/finish_R1a.log 2>&1 &
set -uo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
until grep -q DAGGER_R1A_DONE logs/run_dagger_r1a.log 2>/dev/null; do
  pgrep -f "^bash run_dagger_r1a.sh" >/dev/null || { sleep 5; grep -q DAGGER_R1A_DONE logs/run_dagger_r1a.log 2>/dev/null && break; log "finish_R1a: R1a 가 완료 표시 없이 끝남 → 중단"; exit 1; }
  sleep 60
done
source ~/robot_course/activate.sh
D=~/datasets/fruit_dagger_r1a
python -I quality_check.py --dataset $D --log logs/dagger_fruit_dagger_r1a.jsonl --out logs/quality_dagger_r1a.json --md logs/quality_dagger_r1a.md > logs/quality_dagger_r1a.log 2>&1
log "finish_R1a 품질: $(head -1 logs/quality_dagger_r1a.md) · 표시 $(python3 -I -c "import json;print(json.load(open('logs/quality_dagger_r1a.json'))['summary']['flagged_any'])")"
SRC=()
for f in strawberry orange lemon banana; do SRC+=(--src "$HOME/datasets/fruit_mix3_A_$f:logs/gen_fruit_mix3_A_$f.jsonl:all:"); done
SRC+=(--src "$D:logs/dagger_fruit_dagger_r1a.jsonl:all:")
python -I make_mix.py --out ~/datasets/fruit_mix3_A_R1a "${SRC[@]}" > logs/make_mix_mix3A_R1a.log 2>&1 || { log "finish_R1a: make_mix 실패 → 중단"; exit 1; }
log "finish_R1a merge: $(grep -a '^MERGED' logs/make_mix_mix3A_R1a.log)"
python -I ~/student_kit/isaac/scripts/pack_dataset.py --dataset ~/datasets/fruit_mix3_A_R1a > logs/pack_mix3A_R1a.log 2>&1 || { log "finish_R1a: pack 실패 → 중단"; exit 1; }
T=~/datasets/fruit_mix3_A_R1a_train
SHA=$(sha256sum "$T/fruit_mix3_A_R1a.zip" | cut -d' ' -f1); SIZE=$(stat -c %s "$T/fruit_mix3_A_R1a.zip")
EP=$(python3 -I -c "import json;d=json.load(open('$HOME/datasets/fruit_mix3_A_R1a/meta/info.json'));print(d['total_episodes'],d['total_frames'])")
NR=$(python3 -I -c "import json;d=json.load(open('$D/meta/info.json'));print(d['total_episodes'],d['total_frames'])")
cat > "$T/README.md" <<EOR
# fruit_mix3_A_R1a (비공개 · 스크립트 자동 시연 + DAgger 라운드 1, 사람 시연 아님)

- 전체 에피소드·프레임: $EP · 그중 DAgger R1a: $NR
- \`fruit_mix3_A_R1a.zip\`: $SIZE bytes, sha256 \`$SHA\`
- 구성: \`fruit_mix3_A\`(과일별 4개 합침) 다음에 DAgger R1a 를 이어 붙임. 에피소드별 출처·장면·DAgger 정보는 zip 안 \`scene_index.json\`
- DAgger R1a: mix2 10만 te0.01 정책이 몰고, 실패 조짐(빗나감·빈손 들기·떨어뜨림·멈춤·늦음) 또는 무작위 이른 시점에 스크립트 전문가가 넘겨받아 성공까지. 넘겨받은 프레임부터 저장, 성공만. retreat 에서 끝(휴식 자세로 돌아가기 없음)
- 학습 코드: \`train.sh\`, \`train_act.py\` (수업 키트, 수정 없음)

## DAgger 품질 검사
$(cat logs/quality_dagger_r1a.md)
EOR
log "finish_R1a 완료: $T (zip $SIZE bytes, sha256 ${SHA:0:16}…, 전체 $EP, R1a $NR)"
echo FINISH_R1A_DONE
