#!/usr/bin/env bash
# R1a 다시 묶기 (CPU): 품질 검사에서 팔 명령 튐(jump > 10 단위, 정상 최대 7.6)이 난 DAgger 에피소드 9개를 빼고 A + R1a 합치기 → pack → README
#   처음 묶음(뺀 것 없음)은 지우지 않고 *_v0jump 로 이름만 바꿔 둔다.
set -euo pipefail
cd ~/fruit_datagen
log() { echo "[$(date '+%F %H:%M')] $*" | tee -a progress.log; }
EX="1+59+82+106+122+135+140+165+175"
[ -e ~/datasets/fruit_mix3_A_R1a_v0jump ] || mv ~/datasets/fruit_mix3_A_R1a ~/datasets/fruit_mix3_A_R1a_v0jump
[ -e ~/datasets/fruit_mix3_A_R1a_v0jump_train ] || mv ~/datasets/fruit_mix3_A_R1a_train ~/datasets/fruit_mix3_A_R1a_v0jump_train
source ~/robot_course/activate.sh
D=~/datasets/fruit_dagger_r1a
SRC=()
for f in strawberry orange lemon banana; do SRC+=(--src "$HOME/datasets/fruit_mix3_A_$f:logs/gen_fruit_mix3_A_$f.jsonl:all:"); done
SRC+=(--src "$D:logs/dagger_fruit_dagger_r1a.jsonl:all:::ex=$EX")
python -I make_mix.py --out ~/datasets/fruit_mix3_A_R1a "${SRC[@]}" > logs/make_mix_mix3A_R1a_clean.log 2>&1 || { log "R1a 다시 묶기: make_mix 실패"; exit 1; }
log "R1a 다시 묶기 merge: $(grep -a '^MERGED' logs/make_mix_mix3A_R1a_clean.log)"
python -I ~/student_kit/isaac/scripts/pack_dataset.py --dataset ~/datasets/fruit_mix3_A_R1a > logs/pack_mix3A_R1a_clean.log 2>&1 || { log "R1a 다시 묶기: pack 실패"; exit 1; }
T=~/datasets/fruit_mix3_A_R1a_train
SHA=$(sha256sum "$T/fruit_mix3_A_R1a.zip" | cut -d' ' -f1); SIZE=$(stat -c %s "$T/fruit_mix3_A_R1a.zip")
EP=$(python3 -I -c "import json;d=json.load(open('$HOME/datasets/fruit_mix3_A_R1a/meta/info.json'));print(d['total_episodes'],d['total_frames'])")
NR=$(python3 -I -c "import json;s=json.load(open('$HOME/datasets/fruit_mix3_A_R1a/scene_index.json'));e=[r for r in s['episodes'] if r['src']=='fruit_dagger_r1a'];print(len(e))")
cat > "$T/README.md" <<EOR
# fruit_mix3_A_R1a (비공개 · 스크립트 자동 시연 + DAgger 라운드 1, 사람 시연 아님)

- 전체 에피소드·프레임: $EP · 그중 DAgger R1a: $NR 에피소드
- \`fruit_mix3_A_R1a.zip\`: $SIZE bytes, sha256 \`$SHA\`
- 구성: \`fruit_mix3_A\`(551, 과일별 4개 합침) 다음에 DAgger R1a 를 이어 붙임. 에피소드별 출처·장면·DAgger 정보는 zip 안 \`scene_index.json\`
- DAgger R1a: mix2 10만 te0.01 정책이 몰고, 실패 조짐(빗나감·빈손 들기·떨어뜨림·멈춤·늦음) 또는 무작위 이른 시점에 스크립트 전문가가 넘겨받아 성공까지. 넘겨받은 프레임부터 저장, 성공만. retreat 에서 끝(휴식 자세로 돌아가기 없음)
- **품질로 뺀 것**: 저장한 260개 중 9개(원본 번호 $EX). 모두 정책이 출발한 직후(20~31스텝) 이른 넘겨받기에서 첫 1~4프레임 팔 명령이 프레임당 10.5~15.8 단위로 튐(정상 시연 최대 7.6). 나머지 표시 3개(추종 오차만 15.9~19.0, 넘겨받는 순간 또는 끝 retreat)는 남김
- 학습 코드: \`train.sh\`, \`train_act.py\` (수업 키트, 수정 없음)

## DAgger 품질 검사 (빼기 전 260개)
$(cat logs/quality_dagger_r1a.md)
EOR
log "R1a 다시 묶기 완료: $T (zip $SIZE bytes, sha256 ${SHA:0:16}…, 전체 $EP, R1a $NR)"
echo R1A_CLEAN_DONE
