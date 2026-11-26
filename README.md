# 과일 집기 ACT — 시연 자동 생성·평가·DAgger (로봇제어AI 텀프로젝트, 팀 공유용 · 비공개)

Isaac Sim 6.2 / Isaac Lab 3.0 / LeRobot 0.6.1 ACT. SO-101 로봇이 지정된 과일(딸기·오렌지·레몬·바나나)을 접시로 옮긴다. **사람 시연 없이** 스크립트 상태머신(IK 계획)으로 시연을 만들고, 그 데이터로 학습한다.

> 데이터셋·모델 zip·GIF·학습 결과(약 16GB)는 저장소에 없다. 데이터와 모델은 HF 비공개 저장소(`mgm136044/...`)에 있다. 이 저장소는 코드·문서·결과 기록만 담는다.

## 먼저 읽을 문서

| 문서 | 내용 |
|---|---|
| `STRATEGY_W1.md` | 1주차 전략 전체: 진단 H1~H8, 우선순위, mix3 설계, 학습·실행 설정, 일정, 제출 운영, 위험 |
| `ROADMAP.md` | 진행 상태. 2부 개정 = 91점 대응 지시 반영 |
| `reports_w1/` | 단계별 보고서 01~07 (`index.html` 은 한 페이지 모음) |
| `REPORT.md` | 1부(11-24~25) 데이터 생성·1·2차 학습 보고서 |
| `logs/submissions.md` | 서버 제출 기록 |
| `prompts/` | 학교 GPU 학습 에이전트에게 준 지시문(그대로 실행 가능) |
| `school_training/README.md` | 학교 GPU 학습 명령과 기록 |

## 코드

| 파일 | 하는 일 |
|---|---|
| `gen_scripted.py` | **시연 생성기(상태머신)**: 과일 위 → 내려가기 → 쥐기 → 들기 → 접시 → 놓기 → 휴식 자세. IK(`kin.py`, `so101_chain.json`), 중력·적분 보정, DART 흔들림, 장면 변형, 같은 과일 여러 개(가장 가까운 것), 접시·로봇 이동, 조명. 성공한 것만 LeRobot 형식으로 저장 |
| `params_mix3.json` | mix3 잡기 값 2판(가운데 맞춤). 1판(깊게 닫기)은 실패해 뺐다(보고서 05) |
| `run_mix3.sh` | mix3 묶음 생성: `bash run_mix3.sh <A|B|trial> <딸> <오> <레> <바> <시드>` |
| `make_mix.py` | 여러 데이터셋 합치기 + `scene_index.json`(에피소드별 출처·장면·DAgger 정보) |
| `verify_dataset.py`, `quality_check.py`, `mix3_stats.py` | 형식 검증, 에피소드별 품질 검사, 장면 구성 통계 |
| `eval_proxy.py` | 로컬 대리 평가(서버와 같은 `lerobot_policy` 실행). 장면: 기본·같은 과일·로봇 이동·조명·혼합 |
| `eval_sweep.py` | P0 조건 스윕: 위치·회전·로봇·접시 범위별 성공률과 실패 7유형 자동 분류, 실패 GIF |
| `dagger_collect.py` | **개입형 DAgger**: 정책이 몰다가 실패 조짐(T1~T5)이나 무작위 이른 시점에 전문가가 지금 상태에서 넘겨받아 성공까지. 성공만 저장 |
| `make_te_zip.sh` | 체크포인트 → 실행 설정만 바꾼 제출 zip + 키트 형식 검사 |
| `build_scene_c4.py`, `assets_c4/` | 종류당 과일 4개 장면 사본(`COURSE_ASSETS=assets_c4`) |
| `kin.py`, `so101_chain.json` | SO-101 FK/IK |
| `logs/fruit_pts_*.npy` | 과일 메시 점(손가락 간섭 검사용, 생성기가 읽음) |
| `run_*.sh`, `after_*.sh`, `finish_*.sh` | 단계 연결 스크립트(대기 → 실행 → 기록) |

## 실행 예 (키트 환경: `source ~/robot_course/activate.sh`)

```bash
# 시연 생성 (시험: 저장 안 함)
COURSE_ASSETS=$PWD/assets_c4 python gen_scripted.py --dataset ~/datasets/test --fruits strawberry --no-save --per-fruit 5 \
  --max-attempts 5 --params params_mix3.json --keep-kind-on-relayout --viz none
# 로컬 평가 (서버와 같은 실행 방식, 체크포인트 config 그대로)
python eval_proxy.py --checkpoint <pretrained_model 폴더> --per-scene 8 --viz none
# 조건 스윕
COURSE_ASSETS=$PWD/assets_c4 python eval_sweep.py --checkpoint <폴더> --per-cond 8 --viz none
# DAgger 수집
COURSE_ASSETS=$PWD/assets_c4 python dagger_collect.py --checkpoint <폴더> --dataset ~/datasets/fruit_dagger_x --attempts 24 --viz none
```

## 지금까지 핵심 결과 (11-26)

- 서버 최고 54.2(13/24) = mix2 10만 + temporal ensemble 0.01. 서버는 같은 파일에 같은 점수를 준다(고정 24장면).
- 로컬 같은 장면 비교: te 24/40 대 n100 12/40. te 의 실패는 모두 잡기 실패이고, 다시 잡으러 가지 않는다 → DAgger 회복 데이터로 겨냥한다.
- 깊게 닫는 잡기는 이 시뮬레이터에서 과일을 짜내 실패한다. 가운데 맞춤만 쓴다.
- 사람 시연 데이터는 이 저장소와 HF 에 올리지 않는다.

## 저장소에 없는 것과 있는 곳

| 항목 | 있는 곳 |
|---|---|
| 학습 데이터셋 zip (스크립트 자동 시연) | HF 비공개: `mgm136044/fruit_scripted_mix2_train`(mix2 589ep), `mgm136044/fruit_mix3_train`(mix3 A, A+DAgger) |
| 학습한 모델 / 제출 zip | HF 비공개: `mgm136044/act_mix2_100k_color`, `act_mix2_200k_color`, `act_mix2_c50`, `act_mix3` |
| 로컬 원본 데이터셋·체크포인트·평가 GIF·1부 HTML 보고서(GIF 포함, 파일당 45~77MB) | 메인 PC `~/datasets`, `~/fruit_datagen/{train_runs,te_models,report_html}` (용량 때문에 제외) |
| 데이터셋을 만든 시점의 코드 사본 | 이 저장소 `logs/*_snapshot.py`, `logs/params_mix3*.json` |
| 팀원 사람 시연 데이터 | 올리지 않음 |
| 수업 자료(PDF·키트 zip) | 올리지 않음 (키트는 각자 설치) |
