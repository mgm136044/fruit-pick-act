# 작업 (학교 GPU 세션, 처음 보는 에이전트용): 새 데이터 mix3 A 로 ACT 10만 스텝 학습 (변형 C, chunk 100)

## 배경
- 수업 과제: Isaac Sim 과일 집기(SO-101 로봇) ACT 정책을 학습해 평가 서버에 낸다. **평가 서버 제출은 하지 않는다**(사람이 직접 한다).
- 지금 가장 좋은 모델은 mix2 데이터(589 에피소드)로 10만 스텝 학습한 것이다. 서버 54.2 점, 설정은 chunk 100, 색 증강만, temporal ensemble 0.01 이다.
- 새 데이터 `fruit_mix3_A` 는 스크립트 자동 시연이다(사람 시연 아님). 약 550 에피소드이고 다음을 바꿨다.
  - 과일을 두 손가락 가운데로 잡는다.
  - 같은 과일 2~4개 장면이 약 55% 다. 가장 가까운 것을 집는다.
  - 접시 이동, 로봇 이동, 과일 위치·회전을 넓혔다.
  - 다른 종류 과일 장면은 0% 다.
- 이 학습은 팀 지시서의 **변형 C(새 데이터만, 비교 기준)** 다. 나중에 같은 데이터에 DAgger 데이터를 더한 변형 A·B 와 비교한다.
- 데이터는 이미 올라가 있다: 에피소드·프레임 {EP}, `fruit_mix3_A.zip` {SIZE} bytes, sha256 `{SHA}`.

## 0. 접근 권한 (사람이 준비)
- Hugging Face 토큰은 사람이 세션에 직접 넣는다: `export HF_TOKEN=...`
  - **토큰을 파일·로그·README 에 쓰지 않는다.**
  - 토큰이 없으면 멈추고 요청한다.
- 학습 코드: 비공개 데이터셋 저장소 `mgm136044/fruit_scripted_mix2_train` 의 `train.sh`, `train_act.py`, `README.md`. mix2 zip 은 받지 않아도 된다.
- 새 데이터: 비공개 데이터셋 저장소 `mgm136044/fruit_mix3_train` 의 `fruit_mix3_A.zip` 과 `README.md`. README 에 sha256 이 적혀 있다.
- 결과 올릴 곳: 비공개 모델 저장소 `mgm136044/act_mix3`.
  - 없으면 **비공개로** 만들고, 비공개인지 확인한다.
  - 쓰기 권한이 없으면 zip 을 세션에 두고 보고한다.

## 1. 준비
```bash
mkdir -p /home/work/act_mix3 && cd /home/work/act_mix3
huggingface-cli download mgm136044/fruit_scripted_mix2_train train.sh train_act.py --repo-type dataset --local-dir .
huggingface-cli download mgm136044/fruit_mix3_train fruit_mix3_A.zip README.md --repo-type dataset --local-dir .
sha256sum fruit_mix3_A.zip   # {SHA} 와 같아야 함
```
- `nvidia-smi`, `nproc`, `df -h /home/work` 를 확인한다.
- LeRobot 0.6.1 환경이 없으면 `train.sh` 첫 부분 안내대로 만든다. 코드는 고치지 않는다.
- 크기나 sha256 이 다르면 멈추고 보고한다.

## 2. 학습
```bash
cd /home/work/act_mix3
nohup bash train.sh --data fruit_mix3_A.zip --steps 100000 --batch-size 16 --save-freq 10000 --name act_mix3A_100k \
  --extra --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}}' \
  > train_A.out 2>&1 &
```
- `--augment` 는 쓰지 않는다. 쓰면 선명도·위치 이동 증강까지 켜진다.
- **시작 1~2분 뒤 설정 출력을 확인한다.** 다르면 멈추고 outputs 를 지운 뒤 다시 시작한다.
  - chunk_size 100, n_action_steps 100
  - `image_transforms.enable: true`
  - tfs 네 개만(brightness·contrast·saturation·hue)
  - seed 1000, batch 16
- 첫 저장(1만) 뒤 `checkpoints/010000/pretrained_model/train_config.json` 으로 다시 확인한다.
- `outputs/act_mix3A_100k` 의 저장 지점(`training_state` 포함)은 지우지 않는다. 나중에 이어 학습할 수 있다.

## 3. zip 만들기
- `pretrained_model` 안의 파일들을 zip 맨 위에 넣는다.
- 10만: `submission_act_mix3A_100k.zip`(config 그대로), `submission_act_mix3A_100k_te.zip`
- 6만·8만: `submission_act_mix3A_60k_te.zip`, `submission_act_mix3A_80k_te.zip`
- `_te` 는 사본 config.json 의 `"n_action_steps": 1`, `"temporal_ensemble_coeff": 0.01` 두 값만 바꾼 것이다.
- 형식 검사: LeRobot 로 불러 무작위 입력(front·wrist 480x640x3, state 6)으로 `select_action` 을 120번 부른다. 숫자 6개, NaN 없음.
- sha256 을 적는다.
- `mgm136044/act_mix3`(비공개)에 zip 4개와 README.md 를 올린다. README 에 적을 것:
  - 데이터(저장소·zip 이름·에피소드·프레임 수, 사람 시연 아님)
  - 학습 설정
  - 파일 표(이름·스텝·실행 설정·sha256)

## 4. 보고
- 시작 시각, step/s, 예상 종료 시각, 실제 증강 설정(train_config.json 의 image_transforms 전문)
- 1만 스텝마다 loss
- zip 이름·sha256·형식 검사 결과, 업로드 확인

## 지킬 것
- 평가 서버에는 아무것도 올리지 않는다.
- 데이터·코드는 고치지 않는다. HF 는 비공개로만, 다른 저장소는 건드리지 않는다.
- 디스크 여유가 10GB 아래면 멈추고 보고한다.
