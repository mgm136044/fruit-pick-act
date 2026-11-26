# 작업: 수업 과일 집기 환경에서 과일 시연 데이터를 사람 녹화 없이 생성

작업 폴더는 `~/fruit_datagen`이다.

## 가장 먼저: 보낸 파일 전부 읽기 (다 읽기 전에는 설치나 코드 작업을 시작하지 말 것)
사용자가 Mac에서 이 폴더로 자료를 보냈다. 아래 파일을 모두 읽는다. PDF는 `pdftotext -layout 파일.pdf -`로 읽는다.

1. `00_PROMPT.md`(이 파일): 끝까지 읽는다.
2. `01_리서치_요약.md`: 앞서 조사한 근거. 끝까지 읽는다.
3. `materials/로봇제어AI_세팅가이드_Ubuntu.pdf`: 과제 설명, 설치, 수집, 학습, 제출, 평가. 끝까지 읽는다.
4. `materials/robot_course_kit.zip`: `cd ~ && python3 -m zipfile -e ~/fruit_datagen/materials/robot_course_kit.zip .`로 `~/student_kit`에 풀고, 다음 코드를 읽는다.
   - `NOTICE.txt`
   - `setup/install.sh`
   - `isaac/course_env/` 아래 전부: `pick_fruit.py`, `mdp.py`, `so101.py`, `paths.py`, `__init__.py`
   - `isaac/scripts/` 아래 전부: `record.py`, `run_policy.py`, `pack_dataset.py`, `check_env.py`
   - `train/train.sh`, `train/train_act.py`, `checkpoint_spec.py`
   - `isaac/assets/scenes/kitchen_fruit/objects.json`
5. `materials/12. 시뮬레이션 기반 인공지능 모델 학습.pdf`: 강의 교안. 데이터 수집 방식(텔레오퍼레이션, 스테이트 머신, MimicGen) 부분을 중심으로 읽는다.
6. `materials/SO-101 시뮬레이션 모방학습 실습_261107.html`과 `materials/sim_bridge_261107.zip`: 12주차 실습과 평가 브리지. `~/sim_bridge`에 풀어서 `README.md`를 읽는다. 참고용이다.

다 읽었으면 `~/fruit_datagen/progress.log`에 다음을 짧게 적고 0단계로 넘어간다.
- 파일별로 읽은 핵심 (한두 줄씩)
- 이 프롬프트의 설명과 실제 파일 내용이 다른 곳 (있으면, 실제 파일을 따른다)

## 배경
- 수업(로봇제어인공지능) 텀프로젝트: Isaac Sim 속 SO-101이 과일을 접시에 놓도록 ACT를 학습해 제출한다.
  - 에피소드마다 딸기, 오렌지, 레몬, 바나나 중 하나가 조리대에 놓이고, 30초 안에 접시에 놓으면 성공이다.
  - 평가는 데이터를 모은 환경과 다르게 만든 비공개 환경에서 하고, 네 과일을 같은 횟수씩 본다.
- 원래 방식은 리더 암 텔레오퍼레이션(`record.py`)으로 시연을 모으는 것이다. 사용자는 딸기·레몬·바나나 시연을 사람 녹화 없이 만들 수 있는지 확인하고 싶어 한다.
- 이 PC가 쓰는 것은 수업 키트 환경이다(`materials/robot_course_kit.zip`, Isaac Sim 6.2 + Isaac Lab 3.0 + LeRobot 0.6.1). 리서치 요약에 나오는 LeIsaac(Isaac Sim 5.1)은 참고용일 뿐이다.

## 목표
수업 키트 환경에서 `record.py`가 만드는 것과 형식이 똑같은 LeRobot 데이터셋을 사람 조작 없이 만든다.
- 주 경로(1단계): 스크립트 정책(상태머신)으로 네 과일 시연을 생성한다.
- 실험 경로(3단계): 오렌지 사람 시연을 다른 과일로 옮길 수 있는지와 수율만 확인한다.

## 규칙 확인이 끝나지 않은 부분
- 가이드에는 "리더 암으로 직접 조작해 시연 데이터를 모은다"고 돼 있다. 또 데이터셋은 매주 월요일 Hugging Face(비공개)에 올려 서버에 등록한다.
- 자동 생성 데이터가 허용되는지는 아직 확인되지 않았다.
- 그래서 만든 데이터는 로컬에만 둔다. HF 업로드, 서버 등록, 모델 제출은 하지 않는다. 보고서에는 "사람 시연 아님"을 분명히 적는다.

## 자료 (`~/fruit_datagen/materials`)
### robot_course_kit.zip (수업 키트)
가이드대로 홈 폴더에 풀면 `~/student_kit`이 된다. 주요 파일:
- `setup/install.sh`
  - `~/robot_course`에 uv 가상환경으로 설치한다(sudo 불필요).
  - 설치 전에 드라이버 595.58 이상, glibc 2.35 이상, 디스크 여유 40GB 이상인지 검사하고 아니면 실패한다.
- `isaac/course_env/pick_fruit.py`(환경)
  - 카메라: front·wrist 640×480
  - 제어 30Hz(물리 60Hz, decimation 2)
  - action: 관절 목표 6개(rad 절대값)
  - 에피소드마다 과일 하나, 과일·접시 위치는 ±3cm 무작위, 에피소드 30초
  - 그리퍼 힘 상한 1.0 N·m: 끝까지 조이면 과일이 튕겨 나간다는 주석이 있다.
  - 관련 함수: `fix_fruit()`, `env.fixed_fruit`, `set_episode_events()`, `task_text(fruit)` = "Pick the {fruit} and put it on the plate."
- `isaac/course_env/mdp.py`
  - 성공: 과일 중심이 접시 중심에서 x·y ±10cm, 높이 ±7cm 안에 있을 때
  - 안 쓰는 과일은 PARK 위치(바닥 구석)로 치운다.
- `isaac/course_env/so101.py`: rad와 LeRobot 정규화 단위 사이 변환, 휴식 자세 `REST_POSE_DEG`
- `isaac/course_env/paths.py`: `FRUITS = (Strawberry, Orange, Lemon, Banana)`, 장면의 `objects.json`
- `isaac/scripts/record.py`
  - 텔레오퍼레이션 녹화 도구다. LeRobotDataset 생성(268~279행), `add_frame`(487행), `save_episode`(542행)가 들어 있다.
  - 장면 변형 옵션: `--count`, `--add`, `--robot`. 그 밖에 `--fake-leader`가 있다.
  - 자동 생성기는 이 파일의 데이터셋 기록 방식을 그대로 재사용해서 형식을 맞춘다.
- 그 밖의 스크립트
  - `isaac/scripts/pack_dataset.py`: 학습 서버용 묶음을 만든다.
  - `run_policy.py`: 로컬 점수를 낸다(`--viz none`이면 창 없이).
  - `check_env.py`: 설치를 확인한다.
- `train/train.sh`, `train/train_act.py`: 학습. `NOTICE.txt`: 과일 모델은 YCB 메시.

### 나머지 자료
- `로봇제어AI_세팅가이드_Ubuntu.pdf`: 과제 설명, 설치, 수집, 학습, 제출, 평가가 다 들어 있다. 특히 5-3 "좋은 데이터를 모으는 법"을 지킨다.
  - 네 과일을 골고루, 과일당 15~25개부터 시작
  - 일관된 시연(같은 방향에서 접근, 비슷한 속도, 망설임 없이)
  - 정지 전에 오래 가만히 있지 않기
  - 시작 자세, 과일 개수, 방해 과일, 로봇 위치를 다양하게
  - 그리퍼는 과일이 손가락 사이에 들어오면 닫기
- `12. 시뮬레이션 기반 인공지능 모델 학습.pdf`: 강의 교안. 데이터 수집 방식으로 텔레오퍼레이션, 스테이트 머신, MimicGen을 다룬다.
- `SO-101 시뮬레이션 모방학습 실습_261107.html`, `sim_bridge_261107.zip`: 12주차 실습(LeIsaac + Isaac Sim 5.1)과 조교가 만든 평가 브리지다. 텀프로젝트 평가는 키트의 `run_policy.py`와 같은 방식이라 참고용이다. 필요하면 `~/sim_bridge`에 푼다.

## 이 PC에서 이미 확인한 것 (2026-11-24 22:26)
- GPU: RTX 3050 6GB, 드라이버 580.178.04
- RAM 30GB, 스왑 없음, `/` 디스크 여유 63GB
- conda `env_isaaclab`(Isaac Sim 5.1.0 + Isaac Lab 0.54.3 + mimic)는 다른 프로젝트용이다. 건드리지 않는다. 수업 키트는 `~/robot_course`에 따로 설치한다.
- 가이드 요구 사항: 드라이버 595.58 이상, GPU 메모리 8GB 이상(권장 12GB). 이 PC는 둘 다 못 미친다.

## 0단계: 설치와 실행 가능성 확인 (막히면 여기서 멈추고 보고)
1. `~/student_kit`이 있는지 확인한다. 자료를 읽을 때 풀었으면 건너뛴다.
2. `nvidia-smi`로 드라이버 버전을 본다. 595.58 미만이면 멈추고 사용자에게 알린다.
   - 사용자가 할 일: `sudo ubuntu-drivers install`(또는 595.58 이상 nvidia-driver 패키지) 후 재부팅
   - 직접 재부팅하지 않는다.
3. 드라이버가 준비되면 설치하고 확인한다. 첫 실행은 셰이더 준비로 10분 이상 걸린다.
   ```
   cd ~/student_kit && bash setup/install.sh
   source ~/robot_course/activate.sh
   python $ROBOT_KIT/isaac/scripts/check_env.py --out ~/fruit_datagen/check_out
   ```
   EULA 동의를 물으면 Yes로 답한다.
4. GPU 메모리가 6GB라 실행이 어려울 수 있다.
   - 먼저 기본 렌더링에 창 없이 돌려본다.
   - 안 되면 `COURSE_RENDER=dlss_perf_lite`로 다시 돌려본다. 다만 이 설정은 평가 서버와 화질이 다르다(pick_fruit.py 주석 참고). 그래서 생성 데이터는 기본 렌더링으로 만드는 것이 원칙이다. dlss 계열로 만든 데이터는 "화질 다름"으로 표시한다.
   - GPU 메모리 최고 사용량을 기록한다.
5. RAM이 부족하면 16GB 스왑파일을 만든다. sudo 비밀번호가 필요하면 사용자에게 묻는다.
6. 기준 데이터 하나를 만든다: `record.py --fake-leader --dataset ~/fruit_datagen/ref_fake --episodes 1 --no-pack`. 창 없이 되는 옵션을 찾아서 쓴다. 만들어진 `meta/info.json`을 형식 기준으로 저장해 둔다.

## 1단계: 스크립트 정책(상태머신) 생성기 (주 경로)
**만드는 방식**
- 새 스크립트 `~/fruit_datagen/gen_scripted.py`를 만든다. 키트 원본은 고치지 말고 import해서 쓴다.
- 환경은 `PickFruitEnvCfg`를 쓰고, 과일을 고정해서 과일별로 생성한다.

**정책**
- 시뮬레이터에서 이번 에피소드 과일과 접시의 root 위치를 읽는다. 바나나는 회전도 읽는다.
- 휴식 자세 → 과일 위로 접근 → 내려가기 → 쥐기 → 들기 → 접시 위로 → 내려놓기 → 놓기 → 팔 정리 순서로 관절 목표를 만든다.
- IK는 Isaac Lab의 미분 IK를 써도 되고 직접 짜도 된다.
- 참고 구현은 LeIsaac `pick_orange.py` 상태머신이다(리서치 요약 5번).

**과일별 주의**
- 크기(YCB): 오렌지 지름 73mm, 레몬 54×68mm, 딸기 43.8×55mm, 바나나 36×190mm. 잡는 높이와 그리퍼 닫힘값을 과일마다 정한다.
- 그리퍼 힘 상한이 1.0 N·m라서 너무 조이면 튕겨 나간다. 과일 폭보다 조금 더 닫는 값을 과일별로 찾는다.
- 바나나는 yaw에 맞춰 wrist_roll을 돌려서 짧은 축을 가로질러 쥔다.

**다양성** (평가가 다른 비공개 환경이므로)
- `record.py`가 지원하는 장면 변형 범위 안에서 무작위로 바꾼다: 같은 과일 1~3개, 방해 과일, 로봇 위치와 회전.
- 변형 없는 기본 장면을 절반쯤 섞는다.
- 접근 위치와 속도에 작은 무작위를 준다. 단, 일관성(가이드 5-3)은 지킨다.

**기록 형식**
- `record.py`와 똑같이 맞춘다.
  - state = 실제 관절 위치, action = 그 스텝에 보낸 관절 목표. 둘 다 LeRobot 정규화 단위(`so101.py`로 변환)
  - 30fps, front·wrist 640×480
  - 에피소드별 문구는 `task_text(fruit)`
- 과일이 접시에 놓이고 팔을 정리한 직후에 에피소드를 끝낸다. 성공한 것만 저장하고, 30초 안에 끝나야 한다.

**진행 순서**
1. 파일럿: 과일별 5개를 만든다. 성공률과 실패 유형(놓침, 미끄러짐, 접시 밖, 시간 초과)을 기록하고 조정한다.
2. 본 생성: 네 과일 각 25개를 `~/datasets/fruit_scripted_v1`에 만든다.
3. 시간이 남으면 과일별 50개 버전을 따로 만든다.

## 2단계: 검증
- 0단계 기준 데이터와 `meta/info.json`의 features를 비교한다: 키, shape, dtype, fps, robot_type, 코덱, names.
- LeRobotDataset(0.6.1)으로 불러오고 영상을 디코딩해 본다. 가이드대로 pyav 백엔드를 쓴다.
- 과일별 확인 이미지를 만든다: 에피소드 3개 × (첫 프레임, 쥐는 순간, 놓는 순간, 마지막) × front·wrist를 접촉 시트 PNG 한 장으로.
- `pack_dataset.py`로 학습 서버용 묶음이 만들어지는지 확인한다(업로드는 하지 않음).
- 선택: 키트 학습 코드가 이 PC에서 몇 스텝 도는지만 본다. 본 학습과 제출은 하지 않는다.

## 3단계: 오렌지 시연 기반 실험 (시간이 남으면, 레몬 먼저)
목적은 오렌지 사람 시연으로 다른 과일 시연을 만들 수 있는지와 그 수율을 보는 것이다.

소스는 HF `LightwheelAI/leisaac-pick-orange`이다(LeIsaac 환경, 오렌지 3개를 연속으로 집음, 60 에피소드, v2.1). 수업 환경은 LeIsaac의 장면, 로봇, 카메라, 단위 변환을 바탕으로 했지만, 과일 모델(YCB)과 배치, 에피소드 구성(과일 1개)이 다르다.

1. 오렌지 위치 복원
   - 에피소드마다 쥐는 시점 3개와 놓는 시점 3개를 찾는다. 그리퍼 값 패턴은 리서치 요약 2번을 참고한다.
   - 그 시점 관절값을 rad로 바꿔 순기구학을 계산하고, 그리퍼 끝 위치에서 오렌지와 접시 위치(로봇 베이스 기준)를 구한다.
2. 오렌지로 먼저 검증
   - 수업 환경에서 로봇 베이스 기준 같은 상대 위치에 오렌지와 접시를 놓고 집기 구간을 재생한다.
   - 첫 구간은 휴식 자세에서 시작한다. 2·3번째 구간은 휴식 자세에서 구간 시작 자세까지 보간을 붙인다.
   - 성공률이 30% 미만이면 위치 복원이 부정확한 것이니 원인만 정리하고 이 단계를 멈춘다.
3. 같은 구간을 레몬, 딸기, 바나나로 재생한다.
   - 쥐는 구간의 그리퍼 명령을 과일 폭에 맞게 고친다.
   - 쥐는 높이는 과일 중심 높이 차이만큼 옮긴다.
   - 성공한 것만 저장한다.
4. 과일별 수율을 1단계 결과와 비교한다.

## 산출물
- `~/fruit_datagen/REPORT.md`: 한국어, 결론 → 근거 → 다음 단계 순서
  - 과일별 표: 시도 수, 성공 수, 성공률, 에피소드 수, 프레임 수, 데이터셋 경로, 확인 이미지 경로
  - 실패 유형 통계, 바꾼 파라미터와 바꾼 이유
  - 실행 환경: GPU 메모리 최고치, 에피소드당 시간, 디스크 사용량, 렌더링 설정
  - 3단계 결과
  - 재현 명령 전체
  - "사람 시연 아님, 규칙 확인 필요" 표시
- `~/fruit_datagen/progress.log`: 단계마다 시각과 한 줄 상태
- 생성 스크립트와 데이터셋

## 규칙
- HF 업로드, 평가 서버 제출, 데이터셋 등록, git push, 기존 파일 삭제는 하지 않는다.
- 키트 원본 파일은 고치지 않는다. 복사본이나 별도 스크립트로 작업한다.
- conda `env_isaaclab`와 기존 프로젝트 폴더는 건드리지 않는다.
- 멈추고 사용자에게 물어볼 때: sudo가 필요할 때(드라이버, 스왑), 재부팅이 필요할 때, 디스크 여유가 15GB 아래로 내려갈 때.
- 확인하지 않은 것은 "미확인"으로 적는다.
- 같은 오류로 3번 실패하면 방법을 바꾼다. 그래도 안 되면 그 단계를 [막힘]으로 표시하고 다음 단계로 넘어간다.
