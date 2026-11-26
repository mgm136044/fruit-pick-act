# 과일 집기 시연 자동 생성 보고서 (로봇제어인공지능 텀프로젝트)

> **사람 시연 아님.** 이 보고서의 `fruit_scripted_*` 데이터는 스크립트 정책(상태머신)이 시뮬레이터에서 자동으로 만든 것이다. 가이드에는 "리더 암으로 직접 조작해 시연 데이터를 모은다"고 되어 있다. 자동 생성 데이터가 과제 규칙상 허용되는지는 2026-11-25 에 사용자가 확인했다(허용). 학교 GPU 학습을 위한 HF 업로드는 사용자가 직접 한다(비공개 저장소 권장). 서버 등록·모델 제출은 하지 않았다.

작성: 2026-11-25 · 실행 PC: RTX 3050 6GB

---

## 1. 결론

**(중간 결론, 2026-11-25 07:40 — 2차 학습·평가 뒤 갱신)**

1. **데이터는 사람 없이 만들 수 있다.** 수업 키트 환경에서 상태머신으로 `record.py` 와 형식이 같은 시연을 자동 생성했다. v2 는 207번 시도해 200개 저장(97%), 과일별 50개, 에피소드 평균 9.8초. 형식·디코딩·키트 묶기·키트 학습까지 검증했다.
2. **그 데이터만으로 학습한 ACT 는 '집어서 접시에 놓기' 동작을 배웠다.** 키트 `run_policy`(서버 사전 테스트와 같은 기본 환경)에서 **54.2%**(바나나 100%, 오렌지 50%, 딸기·레몬 33%).
3. **하지만 배치가 바뀌면 거의 실패한다.** 서버형 변형 장면 점수 20, 특히 서버 평가의 주 장면인 '같은 과일 여러 개'는 0/8. 롤아웃 영상을 보면 과일 위치를 보지 않고 기본 장면 궤적을 외웠다 — 학습 데이터의 73%가 기본 장면이었기 때문으로 판단.
4. 따라서 가설은 **절반만 입증**: 동작 학습은 되지만 고품질(일반화)은 데이터 다양성에 달렸다. 무작위 배치 위주 v3 로 2차 학습을 해서 확인 중.
5. 자동 생성 데이터는 규칙상 허용된다고 사용자가 확인했다(11-25). 학교 GPU 학습용 묶음 `~/datasets/fruit_scripted_mix2_train`(README 포함)은 사용자가 HF 에 올린다. 모델 제출은 하지 않았다.

---

## 2. 근거

### 2-1. 실행 환경

| 항목 | 값 |
|---|---|
| GPU / 드라이버 | RTX 3050 6GB / 595.99.02 (사용자가 580.178.04 → 595 업데이트) |
| 수업 키트 환경 | `~/robot_course` (isaacsim 6.2.0.0 · isaaclab 3.0.0 · lerobot 0.6.1 · torch 2.11.0+cu130) |
| 렌더링 | **기본(`default`)**, 창 없이 `--viz none`. dlss 계열은 쓰지 않음 (평가 서버와 화질 동일) |
| `check_env.py` | 스텝 87.4ms(11.4Hz), 4과일 성공 판정 정상 |
| GPU 메모리 최고치 | 생성 4,327MiB(v1) / 4,325MiB(v2), 학습(ACT 배치 8) 4,597MiB (6GB 중) |
| RAM 최고치 | 13.1GB (`check_env.py`, `/usr/bin/time`) · 스왑 불필요 |
| 에피소드당 시간 | 시도당 평균 v1 34.4 / v2 37.0초 (시뮬레이션 + 영상 인코딩), 에피소드 길이 평균 약 10초 |
| 디스크 | 에피소드당 약 5.8MB. v1 584MB, v2 1.2GB, 1차 학습 묶음 zip 1.25GB(+학습 폴더에 푼 것). 여유 약 30GB |

설치 중 발견한 문제: 키트 `setup/install.sh` 1/5 단계가 이 PC에서 종료 코드 141로 멈춘다. `set -o pipefail` 상태에서 `ldd --version | head -1`이 SIGPIPE를 받기 때문이다(5회 모두 재현). 원본은 고치지 않고 `setup_fix/install_fixed.sh`(3줄 수정: `awk 'NR==1'`로 교체 등)로 설치했다. 같은 Ubuntu 22.04 학생 PC에서도 생길 수 있어 조교에게 알릴 만하다.

### 2-2. 생성 방법 (`gen_scripted.py`)

- 키트 원본은 고치지 않고 `course_env`(PickFruitEnvCfg, mdp, so101, paths)를 import 해서 쓴다. `record.py`의 장면 배치 코드(put/place_robot/place_fruits)는 모듈 수준에서 argparse를 실행해 import 할 수 없어서 그대로 옮겨 썼다(같은 상수: 조리대 영역, 접시 0.14m·로봇 0.17m·과일 0.075m(바나나 0.11m) 간격, 로봇 ±10cm·±30°).
- **기록 형식은 `record.py`와 같다**: `LeRobotDataset.create(fps=30, robot_type="so101_follower", use_videos=True)`, state = 실제 관절(rad→LeRobot 단위, `so101.rad_to_motor`), action = 그 스텝에 보낸 관절 목표(LeRobot 단위), front·wrist 640×480, task = `task_text(fruit)`, `save_episode(parallel_encoding=False)`.
- **기구학**: 로봇 USD의 회전 관절 프레임을 뽑아 순기구학을 직접 만들고(`kin.py`, `so101_chain.json`), 시뮬레이터 링크 자세와 무작위 12자세에서 오차 0으로 일치함을 확인했다. IK는 감쇠 최소제곱(위치 우선, 방향은 가중치를 단계적으로 낮춤). 손목 굽힘 한계(95°) 때문에 먼 곳·높은 곳은 기울여 잡는다.
- **상태머신** (30fps 기준 프레임 수, 에피소드마다 속도 ×0.9~1.1, 구간마다 ±5%):
  휴식 자세 → 과일 위 7cm(60, 그리퍼 열기) → 잠깐 멈춤(5) → 수직 하강(30) → 쥐기(15) → 유지(6) → 10cm 들기(25) → 접시 위(50) → 내려놓기(25) → 놓기(10) → 6cm 위로(15) → 휴식 자세(50) → 휴식 자세에 닿으면 바로 종료. 성공(접시 위)한 것만 저장.
- **추적 보정**: 관절 강성이 17.8 N·m/rad로 낮아 팔이 처진다. PhysX 중력 보상 토크 ÷ 강성을 목표에 더하고(피드포워드), 내려가기·쥐기·놓기 구간에서 관절 오차 적분 보정을 더했다. 과일 위 TCP 오차 평균 1~2mm.
- **과일별 값** (키트 메시를 직접 잼, 프롬프트의 YCB 수치와 다름):

| 과일 | 메시 크기 (mm) | 질량 | 잡는 방법 | 그리퍼 열기/쥐기 (LeRobot 단위) |
|---|---|---|---|---|
| 딸기 | 46 높이, 수평 54.6×41.9 | 30g | v1: 둥글게 / v2: 짧은 축(41.9) 가로질러 | 42 / v1 25 → v2 23 |
| 오렌지 | 54×56×54 (구) | 120g | 손목 기본 방향 | 48 / 33 |
| 레몬 | 45 높이, 수평 58.5×45.6 (대각선 방향) | 80g | v1: 둥글게 / v2: 짧은 축(45.6) 가로질러 | 46 / v1 29 → v2 25.5 |
| 바나나 | 93×152×31, 가운데 단면 32~34 | 120g | 긴 축에 수직으로 손목 회전 (v1: 가운데, v2: 중심선이 곧은 y=-20mm 지점) | 36 / 16 |

  쥐기 값은 그리퍼 메시로 계산한 '손가락 사이 폭 = 과일 폭'이 되는 각도보다 약 4단위 더 닫은 값이다(사람 오렌지 시연의 명령-실제 차이 2.5~4와 비슷). 그리퍼 힘 상한이 1.0 N·m라 더 닫아도 힘은 같다.
- **다양성** (`record.py` 범위 안): 기본 장면 약 절반, 나머지는 같은 과일 2~3개 / 다른 과일 1~2종 × 1~2개 / 로봇 이동(±10cm, ±30°)을 각각 60% 확률로 조합. 시작 자세 25%는 휴식 자세에서 조금 바꿈(기록 전). 접근 위치 ±8mm, 잡는 점 ±3mm, 놓는 점 ±12mm, 속도 ±10%. 같은 과일이 여러 개면 로봇에 가장 가까운 것을 집음(일관성). 도달할 수 없는 배치는 다시 뽑음.

### 2-3. 결과

**파일럿** (과일별 5개, `~/fruit_datagen/pilot_v1`)

| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 기본/변형 장면 | 데이터셋 | 확인 이미지 |
|---|---|---|---|---|---|---|---|---|
| 딸기 | 5 | 5 | 100% | 5 | 1,491 | 2/3 | `~/fruit_datagen/pilot_v1` | `check_pilot/sheet_strawberry.png` |
| 오렌지 | 5 | 5 | 100% | 5 | 1,388 | 0/5 | `~/fruit_datagen/pilot_v1` | `check_pilot/sheet_orange.png` |
| 레몬 | 5 | 5 | 100% | 5 | 1,439 | 3/2 | `~/fruit_datagen/pilot_v1` | `check_pilot/sheet_lemon.png` |
| 바나나 | 5 | 5 | 100% | 5 | 1,468 | 2/3 | `~/fruit_datagen/pilot_v1` | `check_pilot/sheet_banana.png` |
| 합계 | 20 | 20 | 100% | 20 | 5,786 | | | |

**본 생성 v1** (과일별 25개, seed 2)

| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 기본/변형 장면 | 데이터셋 | 확인 이미지 |
|---|---|---|---|---|---|---|---|---|
| 딸기 | 26 | 25 | 96% | 25 | 7,276 | 14/11 | `~/datasets/fruit_scripted_v1` | `check_v1/sheet_strawberry.png` |
| 오렌지 | 25 | 25 | 100% | 25 | 7,385 | 17/8 | `~/datasets/fruit_scripted_v1` | `check_v1/sheet_orange.png` |
| 레몬 | 30 | 25 | 83% | 25 | 7,487 | 16/9 | `~/datasets/fruit_scripted_v1` | `check_v1/sheet_lemon.png` |
| 바나나 | 34 | 25 | 74% | 25 | 7,266 | 16/9 | `~/datasets/fruit_scripted_v1` | `check_v1/sheet_banana.png` |
| 합계 | 115 | 100 | 87% | 100 | 29,414 | | | |

**개선 v2** (과일별 50개, seed 3, 아래 2-4의 수정 반영)

| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 기본/변형 장면 | 데이터셋 | 확인 이미지 |
|---|---|---|---|---|---|---|---|---|
| 딸기 | 52 | 50 | 96% | 50 | 14,820 | 25/25 | `~/datasets/fruit_scripted_v2` | `check_v2/sheet_strawberry.png` |
| 오렌지 | 50 | 50 | 100% | 50 | 14,643 | 25/25 | `~/datasets/fruit_scripted_v2` | `check_v2/sheet_orange.png` |
| 레몬 | 53 | 50 | 94% | 50 | 14,671 | 25/25 | `~/datasets/fruit_scripted_v2` | `check_v2/sheet_lemon.png` |
| 바나나 | 52 | 50 | 96% | 50 | 14,782 | 25/25 | `~/datasets/fruit_scripted_v2` | `check_v2/sheet_banana.png` |
| 합계 | 207 | 200 | 97% | 200 | 58,916 | | | |

**다양화 v3** (과일별 100개, seed 4, 1차 평가에서 "기본 궤적을 외움"이 확인된 뒤 2-7 참고. 사람 시연 아님)

| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 기본/변형 장면 | 데이터셋 | 확인 이미지 |
|---|---|---|---|---|---|---|---|---|
| 딸기 | 101 | 100 | 99% | 100 | 29,829 | 0/100 (전부 무작위 배치) | `~/datasets/fruit_scripted_v3` | `check_v3/sheet_strawberry.png` |
| 오렌지 | 108 | 100 | 93% | 100 | 29,629 | 0/100 | `~/datasets/fruit_scripted_v3` | `check_v3/sheet_orange.png` |
| 레몬 | 101 | 100 | 99% | 100 | 29,587 | 0/100 | `~/datasets/fruit_scripted_v3` | `check_v3/sheet_lemon.png` |
| 바나나 | 119 | 100 | 84% | 100 | 29,960 | 0/100 | `~/datasets/fruit_scripted_v3` | `check_v3/sheet_banana.png` |
| 합계 | 429 | 400 | 93% | 400 | 119,005 | | | |

v3 에서 바꾼 것 (기본값은 v2 와 같게 두고 옵션으로 추가, 코드 `logs/gen_scripted_v3_snapshot.py`):
- `--single-random-prob 1.0`: 과일이 1개인 장면도 기본 자리 대신 무작위 위치에 놓는다. 생성 로그의 "기본" 60개는 개수·다른 과일·로봇 변화가 없다는 뜻이고, 자리는 무작위다.
- `--count3-prob 0.6`: 같은 과일 3개 장면을 늘린다.
- `--light-prob 0.5`: 조명 밝기 ×0.5~1.5, 색 채널 ×0.85~1.15 (`/World/Light`).
- `--dart-prob 0.5`: 바로잡기(DART). 과일 위로 가는 동안 실제로 보내는 관절 명령에만 OU 흔들림(σ 2~2.5°)을 넣는다. 기록되는 action 은 흔들림 없는 깨끗한 명령이다. 흔들린 자세에서 돌아오는 장면을 학습시키기 위해서다. 시험 1에서 흔들림이 하강까지 남아 레몬이 35mm 위에서 막혔다. 그래서 과일 위로 가는 동안만 흔들고, 멈춤 구간 앞 40%에서 0으로 줄이고, 멈춤을 10프레임 늘렸다. 시험 2는 9/10.
- 실패 29건(바나나 19)은 DART 와 상관이 없다. 바나나는 흔들림 없이 12/63, 흔들림 있을 때 7/56 실패했다. 실패한 시도는 저장하지 않는다.

| 항목 | v1 | v2 | v3 |
|---|---|---|---|
| 장면 (저장 기준) | 기본 63 / 변형 37 | 기본 100 / 변형 100 (`balance_kinds`) | 기본 자리 0 / 무작위 400 |
| 같은 과일 개수 1/2/3 | 79/15/6 | 136/38/26 | 128/103/169 |
| 다른 종류 과일 포함 / 로봇 이동 / 시작 자세 변화 | 27 / 25 / 25 | 57 / 56 / 55 | 105 / 172 / 125 |
| 조명 변화 / DART | 0 / 0 | 0 / 0 | 199 / 194 |
| 시도당 실제 시간 | 평균 34.4초 (총 65.9분) | 평균 37.0초 (총 127.7분) | 평균 36.2초 (총 259.2분) |
| 에피소드 길이 | 평균 9.8초, 최대 11.0초 | 평균 9.8초, 최대 11.1초 | 평균 9.9초, 최대 11.6초 |
| 내려갈 때 과일이 5mm 넘게 밀림 | 11/100 | 8/200 | 20/400 |
| 놓인 위치 (접시 중심에서) | 평균 19mm, 최대 60mm | 평균 21mm, 최대 65mm | 평균 20mm, 최대 89mm |
| 디스크 | 584MB | 1.2GB | 2.4GB |
| GPU 메모리 최고 | 4,327MiB | 4,325MiB | 4,327MiB |

**1차 학습 데이터** `~/datasets/fruit_scripted_act`: v1 중 다른 종류 과일이 없는 73개 + v2 중 다른 종류 과일이 없는 143개 = **216 에피소드, 63,294 프레임** (`make_mix.py`, 형식 동일). 같은 과일 2~3개 장면은 33개(15%)뿐이다.

### 2-4. 실패 유형, 바꾼 파라미터와 이유

**v1 실패 15건과 v2 에서 바꾼 것**

| 과일 | v1 실패 | 원인 (로그·메시 측정으로 확인) | v2 에서 바꾼 것 | v2 결과 |
|---|---|---|---|---|
| 바나나 | 놓침 9 | 가운데(y=0) 단면을 잡았는데 그곳 중심선이 로컬 y 축에서 12.3° 돌아가 있어, 고정 손가락이 바나나 위에 얹힘. 벌린 손가락이 옆 과일에 걸린 경우도 섞임 | 잡는 점을 중심선이 곧은 y=-20mm(폭 31.6mm)로, 긴 축 [0.06,1,0], 고정 손가락 여유 5→7mm, 폭 34→32mm | 74% → 96% |
| 레몬 | 놓침 2, 미끄러짐 1, 접시 밖 2 | 레몬 메시가 실제로는 58.5×45.6mm 로 길쭉한데(대각선 방향) 둥근 과일로 다뤄 긴 축으로 쥔 경우 미끄러짐. '접시 밖' 2건은 놓을 때 레몬이 손가락 사이에 끼어 휴식 자세까지 딸려감 | 짧은 축(45.6mm)을 가로질러 쥠(긴 축 [0.77,0.63,0], 폭 51→46mm, 쥐기 29→25.5). 놓을 때 그리퍼를 접근 때보다 6 더 엶(43→52) | 83% → 94% |
| 딸기 | 놓침 1 | 딸기도 54.6×41.9mm 로 길쭉함 | 짧은 축 가로질러 쥠(긴 축 [-0.73,-0.68,0], 폭 45→42mm, 쥐기 25→23) | 96% → 96% |
| 공통 | — | 내려갈 자리에 다른 과일이 있으면 벌린 손가락이 밀어냄. v1 은 변형 장면 실패가 많아 저장 기준 변형 비율이 37% 로 줄어듦 | 손가락 간섭 검사(`finger_check`, 여유 8mm 미만이면 손목을 0/±45/90° 돌려 보고 안 되면 다시 배치), 저장 기준으로 기본/변형 반반(`balance_kinds`) | 변형 장면 100/200 |

**v2 실패 7건** (모두 v2 그대로 기록, 생성 중 파라미터는 바꾸지 않음)
- 레몬 4·44, 바나나 18: 로봇을 옮긴 장면에서 과일이 조리대 끝(팔이 닿는 범위 가장자리)에 있음 → 그리퍼를 기울여 잡다가 손가락이 과일 위에 얹혀 18~53mm 위에서 하강이 막힘.
- 레몬 40 (기본 장면): 팔 추적 오차는 6mm 안이었는데 쥘 때 레몬이 57mm 튕겨 나감. 길쭉한 레몬을 쥐는 방향이 조금 어긋나 타원이 돌며 빠진 것으로 추정(미확인).
- 딸기 8: 운반 중 미끄러짐(로봇 이동 장면, 하강 중 45mm 위에서 막힌 뒤 얕게 쥠). 딸기 41: 레몬이 함께 있는 장면에서 딸기가 30mm 밀림. 바나나 37: 로봇 23° 회전 장면에서 놓침.
- 다음 생성 때 줄이는 방법: IK 기울기가 큰 배치(팔 범위 가장자리)를 미리 거르기.

**v3 실패 29건** (딸기 1, 오렌지 8, 레몬 1, 바나나 19. 생성 중 파라미터는 바꾸지 않음)
- 유형: 놓침 22, 미끄러짐 4, 접시 밖 3. DART(흔들림)와는 상관이 없다. 바나나는 흔들림 없을 때 12/63, 있을 때 7/56 실패했다.
- 거리와 상관이 있다(로그의 처음 위치로 계산).
  - 로봇 받침에서 33cm 넘게 떨어진 과일: 바나나 13번 중 7번(54%), 오렌지 7번 중 3번(43%) 실패.
  - 그보다 가까운 과일: 실패율 바나나 11%, 오렌지 5%.
  - v3 는 모든 과일을 조리대 영역 전체에 무작위로 놓아서, v2 보다 팔이 닿는 범위 끝 배치가 많아졌다.
- 영향: 실패한 시도는 저장하지 않는다. 그래서 학습 데이터에는 먼 위치 예시가 상대적으로 적다(선택 편향). 서버가 과일을 먼 곳에 놓으면 그 장면에서 약할 수 있다. 미확인, 2차 평가의 과일 위치별 결과로 확인한다.

### 2-5. 검증 (2단계)

| 항목 | v1 | v2 | 1차 학습 데이터 | v3 |
|---|---|---|---|---|
| `meta/info.json` features 를 `record.py --fake-leader` 기준(`ref_info.json`)과 비교 | 동일 | 동일 | 동일 | 동일 |
| LeRobotDataset + pyav 로 읽기·영상 디코딩 | OK | OK (처음·끝 프레임 15.1초, 에피소드 0 전체 276프레임) | OK | OK (처음·끝 9.6초, 에피소드 0 전체 320프레임 3.7초) |
| 에피소드 / 프레임 | 100 / 29,414 | 200 / 58,916 | 216 / 63,294 | 400 / 119,005 |
| NaN, 값 범위 | 없음, 범위 정상 | 없음, 범위 정상 | — | 없음, 범위 정상 (action−state 평균 차 0.3~3.3) |
| 확인 이미지 (과일별 3에피소드 × 시작·잡기·놓기·끝 × 정면·손목) | `check_v1/` | `check_v2/` (눈으로 확인: 잡기·접시 위 놓기·끝 장면 정상) | — | `check_v3/` (눈으로 확인: 무작위 배치·같은 과일 여러 개에서 목표 과일을 잡아 접시에 놓음) |
| 키트 `pack_dataset.py` | `fruit_scripted_v1_train/` (zip 611MB) | (묶지 않음, 학습 데이터에 포함) | `fruit_scripted_act_train/` (zip 1,252MB) | (2차 학습 데이터에 포함) |
| 키트 학습 코드 몇 스텝 | — | — | 100스텝 정상 완료, `submission_speedtest.zip` 생성(제출 안 함). RTX 3050 에서 1.88 step/s, GPU 4,597MiB, GPU 사용률 99.7% | — |

업로드·등록·제출은 하지 않았다.

### 2-6. 팀원 데이터셋 (`~/다운로드/datasets`) 점검

`~/다운로드/datasets` (원본은 읽기만 함, `check_team_dataset.py` → `team_check/`)

| 항목 | 레몬 | 딸기 |
|---|---|---|
| 에피소드 | 60 | 60 |
| 형식 (기준과 비교) | 동일 | 동일 |
| 작업 문구 | Pick the lemon and put it on the plate. | Pick the strawberry and put it on the plate. |
| 길이 | 4.3~16.5초 (중앙 8.2초) | 1.8~15.6초 (중앙 6.3초) |
| 끝 프레임 | 60개 모두 과일이 접시 위, 팔 휴식 자세 | ep55 만 불량 (시작부터 딸기가 접시 위, 1.8초 정지) |
| 장면 구성 | 0~19 과일 1개, 20~39 같은 과일 2개, 40~59 3개. 다른 종류 과일·로봇 이동 없음 | 같음 |

- 딸기 ep55 를 뺀 복사본: `~/datasets/team_strawberry_clean` (59 에피소드, 12,537 프레임, 형식 동일, `team_clean.py`).
- 쓸 수 있다. 다만 1차 학습은 '사람 녹화 없이 만든 데이터만으로 되는가'를 보려고 섞지 않았다. 사람 시연은 자동 데이터보다 빠르고(중앙 6~8초 대 9.8초) 같은 과일 여러 개 장면이 2/3 라, 서버 평가 장면(같은 과일 여러 개)에 특히 도움이 될 수 있다 → 사람+자동 혼합은 다음 비교 실험 후보.

### 2-7. 학습·평가 (가설 검증)

**1차 학습** (`train_runs/act_scripted_v12/`, 키트 `train.sh`/`train_act.py` 그대로)

| 항목 | 값 |
|---|---|
| 데이터 | `~/datasets/fruit_scripted_act`: v1 73 + v2 143 에피소드(다른 종류 과일 장면 제외) = 216 에피소드, 63,294 프레임. 기본 장면 158개(73%) |
| 설정 | ACT, resnet18, chunk 100, `n_action_steps` 100, 이미지 증강 없음, 20,000 스텝 · 배치 8 (데이터 약 2.5회) |
| 시간 · 자원 | 약 3시간, RTX 3050 1.86 step/s, GPU 메모리 3.7GB(최고 4.6GB), 사용률 99.7% |
| loss | 6.745(200) → 0.501(4K) → 0.209(8K) → 0.157(12K) → 0.124(16K) → 0.106(20K) |

**1차 평가**

| 평가 | 결과 |
|---|---|
| 키트 `run_policy.py` 기본 환경 24 에피소드 (서버 사전 테스트와 같은 방식) | **13/24 = 54.2%** · 딸기 2/6 · 오렌지 3/6 · 레몬 2/6 · 바나나 6/6. 성공 13개 중 다수는 약 6~7초 만에 한 번에 성공(중앙 202스텝), 일부는 16~27초(다시 잡기로 추정) |
| `eval_proxy.py` 서버형 장면 5종 × 8 | **점수 20.0** · 기본 4/8 · 같은 과일 여러 개 **0/8** · 로봇 이동 1/8 · 조명 2/8 · 혼합 1/8 (참고: 다른 종류 과일 1/8) |
| 실패 유형 (eval_proxy 48개 중 실패 39) | 못 들어올림(잡기 실패) 26 · 들었지만 접시 밖 13 |
| 오프라인 점검 (`offline_check.py`, 학습 데이터 프레임에 그대로 넣기) | 그리퍼 쥐기 값은 정확(딸기 정답 23.0 → 예측 22.0~22.7). 팔 관절 앞 30스텝 평균 오차 2~6 단위 |

**롤아웃 영상으로 본 원인** (`report_html/assets/policy_act_scripted_v12/`)
- 같은 과일 여러 개 장면: 실제 과일이 아니라 **기본 장면에서 과일이 있던 자리**로 가서 빈손으로 쥐고 접시까지 감 → 카메라로 위치를 찾지 않고 기본 궤적을 외웠다. 학습 데이터의 73%가 기본 장면(기준 자리 ±3cm)이었고, 나머지도 대부분 기준 자리 근처였다.
- 기본 장면의 딸기·레몬: 잡는 위치가 1~2cm 치우쳐 가장자리만 스침 → 빈손으로 접시까지 → 휴식 자세 → 다시 시도하다 시간 초과. 100스텝(3.3초)을 다 실행한 뒤에야 다시 보므로 어긋남을 바로잡지 못함.

**2차 (진행 중)**: v3 = 무작위 배치 85%(같은 과일 2~3개 + 과일 1개도 영역 전체 무작위) + 조명 무작위 50% + DART(접근 중 흔들림, 기록은 바로잡는 명령) 50% + 로봇 이동, 과일별 100. 학습은 기본 장면 비율을 20% 이하로 낮춘다(`--augment` 는 기본 증강에 화면 이동·회전이 있어 위치 파악에 해가 될 수 있어 안 씀; 조명 변화는 v3 데이터에 실제로 있음). 결과는 미확인.

### 2-8. 3단계: 오렌지 사람 시연으로 다른 과일 만들기

**진행 전.** 학습이 GPU 를 쓰는 동안(6GB) 시뮬레이터를 함께 띄울 수 없어 학습 뒤로 미뤘다. 준비한 것:
- HF `LightwheelAI/leisaac-pick-orange` 의 parquet·meta 만 받음(3.1MB, `hf_orange/`). 영상은 받지 않음.
- `orange_recover.py`: 그리퍼 상태가 명령보다 1.2 이상 덜 닫힌 구간(15프레임 이상)을 '쥔 구간'으로 보고 151개 검출. 순기구학으로 쥔 순간의 오렌지 중심과 놓는 지점을 복원. 과일과 접시가 12cm 안으로 가까운 28개는 제외 예정.
- 목표-실제 지연이 2프레임으로 우리 30Hz 데이터와 같아 30Hz 기록으로 추정, 1프레임씩 재생.
- `orange_replay.py` (작성, 미실행): 복원한 위치에 과일·접시를 놓고 사람 관절 궤적을 재생. 다른 과일은 그리퍼 값을 접촉 폭 차이만큼 옮기고, 높이는 IK 로 보정.
- 결과(성공률, 1단계와 비교)는 미확인.

---

## 3. 리더보드 전략 (근거와 함께)

근거가 있는 것부터 적는다.

1. **서버 평가 장면에 맞춘 데이터** — 사용자 정보(11-25): 서버에는 다른 종류 과일이 섞인 장면이 없고 '같은 과일 여러 개 중 하나'(예: 바나나 4개, 레몬 3개)만 있다. 성공 판정은 사본 중 아무거나 접시에 오르면 성공(`fruits_on_plate`가 사본을 OR). 1차 학습 데이터에는 이런 장면이 15%뿐이라 **v3 는 같은 과일 2~3개 장면 위주**(`--vary-prob 0.8 --vary-use count=0.9,extra=0.4,robot=0.6 --count3-prob 0.6`, 약 70%)로 만든다. 여러 개일 때는 '로봇에 가장 가까운 것'을 집는 일관된 규칙이라 정책이 배우기 쉽다. 키트 장면은 종류당 3개라 4개는 로컬에서 못 만든다(4개로 일반화되는지는 미확인).
2. **다른 종류 과일 장면은 A/B 로 판단** — ACT 는 작업 문구를 입력으로 쓰지 않아 다른 종류가 섞이면 무엇을 집을지 알 수 없다(키트 `lerobot_policy.py` 확인). 그래도 사용자는 일반화를 위해 계속 넣길 원함 → 1차(제외)와 2차(포함)를 같은 평가로 비교한다.
3. **실행 설정 조정** — 서버는 체크포인트 `config.json` 을 그대로 쓴다. ACT 기본은 100 스텝 묶음을 다 실행(`n_action_steps=100`)하므로, 학습 뒤 `eval_proxy.py --n-action-steps 20/50` 이나 `--temporal-ensemble 0.01` 로 로컬 비교 후 좋은 값을 `config.json` 에 적어 낸다(재학습 불필요).
4. **변형 환경 대비** — 비공개 환경이 무엇이 다른지는 모른다. 조명 변화 장면(`eval_proxy` light)으로 보고, 약하면 생성 때 조명 무작위(`--light-prob`)를 늘린다. 키트 `--augment` 는 LeRobot 기본 증강에 화면 이동(5%)·회전(5°)이 섞여 있어 위치 파악에 해가 될 수 있으므로, 쓰려면 색 변화만 남기는 설정이 필요(미확인).
5. **복구 동작 데이터** — 스크립트 시연은 늘 깔끔해서, 정책이 조금 어긋나면 돌아오는 법을 못 배울 수 있다. 실패 사례를 보고 필요하면 궤적 중간에 작은 흔들림을 넣고 바로잡는 동작을 기록(DART 방식).
6. **제출 횟수 아끼기** — 하루 5회 제한. `eval_proxy.py` 로 로컬에서 먼저 고르고, 서버는 `test_submit.py` 사전 테스트부터.
7. **규칙 확인이 먼저** — 가이드는 리더 암 시연을 전제로 한다. 월요일 데이터셋 제출분이 그 주 모델 학습에 쓴 데이터여야 하므로, 자동 생성 데이터 허용 여부를 조교에게 확인한 뒤 제출 전략을 정한다.

---

## 4. 다음 단계

진행 상태는 `ROADMAP.md` 가 기준이다.

1. **v3 생성** (진행 중, 과일별 100): 무작위 배치 위주 + 조명 + DART. 끝나면 검증(`verify_dataset.py`, 시트).
2. **2차 학습**: v3 전체 + v1·v2 의 변형 장면 + 기본 장면 일부(기본 비율 20% 이하), 4만 스텝(약 6시간). `--augment` 는 안 씀.
3. **2차 평가**: `run_policy` 24 + `eval_proxy` 6장면 × 8 (1차와 같은 시드) → 1차와 비교. 특히 '같은 과일 여러 개'.
4. **실행 설정**: 2차 모델로 `run_variants.sh` (`n_action_steps` 50·20, temporal ensemble 0.01) → 좋은 값을 `config.json` 에.
5. 그래도 같은 과일 여러 개 장면이 약하면: 데이터 더 늘리기 / 학습 더 길게 / 팀원 사람 시연 섞기(같은 과일 여러 개 장면이 2/3) 비교.
6. 3단계(오렌지 사람 시연 전이)는 선택 — 위 작업 뒤 GPU 가 비면.
7. **사용자 확인**: 자동 생성 데이터 허용 여부(조교), 제출 여부·시점.

---

## 5. 재현 명령 전체

```bash
source ~/robot_course/activate.sh && cd ~/fruit_datagen
# 0) 설치 (원본 install.sh 대신 3줄 고친 복사본)
bash setup_fix/install_fixed.sh
# 기준 데이터 (record.py --fake-leader 1에피소드) → ref_info.json
bash make_ref.sh
# 장면·기구학 탐색 (so101_chain_raw.json, scene_geom.json, 메시 점)
python explore_scene.py --viz none
# 1) 파일럿 / v1 / v2  (v1 코드: logs/gen_scripted_v1_snapshot.py, v2 코드: logs/gen_scripted_v2_snapshot.py)
python gen_scripted.py --dataset pilot_v1 --fruits all --per-fruit 5 --seed 1 --log logs/gen_pilot_v1.jsonl --viz none
python logs/gen_scripted_v1_snapshot.py --dataset ~/datasets/fruit_scripted_v1 --fruits all --per-fruit 25 --max-attempts 75 --seed 2 --log logs/gen_fruit_scripted_v1.jsonl --viz none
python gen_scripted.py --dataset ~/datasets/fruit_scripted_v2 --fruits all --per-fruit 50 --max-attempts 120 --seed 3 --log logs/gen_fruit_scripted_v2.jsonl --viz none
# 2) 검증·요약·묶기
python verify_dataset.py --dataset ~/datasets/fruit_scripted_v2 --log logs/gen_fruit_scripted_v2.jsonl --out check_v2
python3 summarize.py logs/gen_fruit_scripted_v2.jsonl --dataset ~/datasets/fruit_scripted_v2 --sheets check_v2 --gpu-log logs/gpu_v2.log
python $ROBOT_KIT/isaac/scripts/pack_dataset.py --dataset ~/datasets/fruit_scripted_v1
# 팀원 데이터 점검·정리
python check_team_dataset.py   # → team_check/
python team_clean.py           # → ~/datasets/team_strawberry_clean
# 학습 데이터 (다른 종류 과일 장면 제외)
python make_mix.py --out ~/datasets/fruit_scripted_v1_act --src "$HOME/datasets/fruit_scripted_v1:logs/gen_fruit_scripted_v1.jsonl:all:no_other_fruit"
#   (run_training.sh 가 v1_act + v2 no_other_fruit → ~/datasets/fruit_scripted_act 를 만들고 묶은 뒤 키트 train.sh 실행)
bash run_training.sh 100 8 speedtest      # 속도 측정
bash run_training.sh 20000 8              # 1차 학습 → train_runs/act_scripted_v12/outputs/act_scripted_v12
# 평가
python $ROBOT_KIT/isaac/scripts/run_policy.py --checkpoint train_runs/act_scripted_v12/outputs/act_scripted_v12/checkpoints/last/pretrained_model --episodes 24 --viz none
python eval_proxy.py --checkpoint train_runs/act_scripted_v12/outputs/act_scripted_v12/checkpoints/last/pretrained_model --per-scene 8 --viz none
#   (bash run_eval.sh act_scripted_v12 이 위 두 개 + GIF + 보고서 갱신을 한 번에)
# 3) v3 생성 (코드: logs/gen_scripted_v3_snapshot.py) → 검증 → 2차 학습 → 2차 평가
python gen_scripted.py --dataset ~/datasets/fruit_scripted_v3 --fruits all --per-fruit 100 --max-attempts 200 --vary-prob 0.85 \
  --vary-use count=0.75,extra=0.3,robot=0.5 --count3-prob 0.6 --single-random-prob 1.0 --light-prob 0.5 --dart-prob 0.5 \
  --start-perturb-prob 0.3 --seed 4 --log logs/gen_fruit_scripted_v3.jsonl --viz none
python verify_dataset.py --dataset ~/datasets/fruit_scripted_v3 --log logs/gen_fruit_scripted_v3.jsonl --out check_v3
python3 summarize.py logs/gen_fruit_scripted_v3.jsonl --dataset ~/datasets/fruit_scripted_v3 --sheets check_v3 --gpu-log logs/gpu_v3.log
bash run_training2.sh 40000 8             # v3 전체 + v1·v2 변형 + v2 기본 과일별 13 → ~/datasets/fruit_scripted_mix2 → train_runs/act_scripted_v3mix
bash run_eval.sh act_scripted_v3mix       # run_policy 24 + eval_proxy 장면당 8 + GIF + 보고서
bash run_variants.sh act_scripted_v3mix   # 실행 설정 n_action_steps 50·20, temporal ensemble 0.01
#   (after_v3b.sh / after_train2.sh 가 위 순서를 자동으로 이어 붙임)
# 보고서 (HTML, 로컬)
python make_report.py
```

---

## 6. 미확인 · 주의

- 자동 생성 데이터 허용 여부: 사용자가 확인함(11-25, 허용).
- 비공개 평가 환경이 기본 환경과 무엇이 다른지 **미확인**. 서버는 같은 과일 여러 개 장면만 있다는 것은 사용자 정보다. 4개 장면은 키트의 `COURSE_ASSETS` 로 장면 사본(`assets_c4`)을 만들어 재현한다. 시뮬레이터 동작 확인은 아직 하지 않았다.
- 성공 판정은 과일 root 위치 기준. YCB 메시 원점이 기하 중심인지 **미확인** (잡는 점은 메시를 직접 재서 정했다).
- 레몬 40 실패의 원인(쥐는 방향 어긋남)은 추정, **미확인**.
- 3단계(오렌지 사람 시연 → 다른 과일) 결과 **미확인** (미실행).
- 학습된 정책의 성공률은 학습·평가가 끝나야 나온다 (아래 2-8).
