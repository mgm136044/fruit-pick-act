# 작업 (학교 GPU 세션, 처음 보는 에이전트용): mix2 데이터로 chunk_size 50 ACT 를 10만 스텝 학습하고 제출용 zip 만들기

## 배경
- 수업 과제: Isaac Sim 과일 집기(SO-101 로봇) ACT 정책을 학습해 평가 서버에 낸다. **평가 서버 제출은 하지 않는다**(사람이 직접 한다).
- 지금 가장 좋은 모델은 `act_mix2_100k_color` 이고, 서버 점수는 54.2 다.
  - 데이터 `fruit_scripted_mix2`: 589 에피소드, 스크립트 자동 시연(사람 시연 아님)
  - 학습: 수업 키트 `train.sh`/`train_act.py`(LeRobot 0.6.1), 10만 스텝, 배치 16, seed 1000, 색 증강만, **chunk_size 100**
  - 제출 설정: temporal ensemble(`n_action_steps` 1, `temporal_ensemble_coeff` 0.01)
- 이번 실험은 **chunk_size 만 50 으로 줄인다.** 데이터·스텝·나머지 설정은 같다.
  - 로컬 분석 결과, temporal ensemble 은 정책이 놓친 뒤 다시 잡으러 가는 전환을 지운다.
  - chunk 가 짧으면 평균에 섞이는 예측이 줄어 전환이 빨라질 수 있다.
  - chunk_size 는 모델 구조라 학습 뒤에는 바꿀 수 없다.

## 0. 접근 권한 (사람이 준비)
- Hugging Face 토큰은 사람이 이 세션에 직접 넣는다: `export HF_TOKEN=...`
  - **토큰을 파일·로그·README 에 쓰지 않는다.**
  - 토큰이 없으면 멈추고 요청한다.
- 받을 곳: 비공개 데이터셋 저장소 `mgm136044/fruit_scripted_mix2_train`. 파일은 `fruit_scripted_mix2.zip`(약 3.7GB), `train.sh`, `train_act.py`, `README.md` 다.
- 올릴 곳: 비공개 모델 저장소 `mgm136044/act_mix2_c50`.
  - 없으면 **비공개로** 만들고, 비공개인지 확인한다.
  - 쓰기 권한이 없으면 zip 을 세션에 두고 보고한다.

## 1. 준비
```bash
mkdir -p /home/work/act_mix2_c50 && cd /home/work/act_mix2_c50
huggingface-cli download mgm136044/fruit_scripted_mix2_train --repo-type dataset --local-dir . 
sha256sum fruit_scripted_mix2.zip   # 앞부분이 eca47d3d 인지
```
- `nvidia-smi`, `nproc`, `df -h /home/work` 를 확인한다. 디스크는 압축 해제를 포함해 약 10GB 가 필요하다.
- LeRobot 0.6.1 환경이 없으면 `README.md` 와 `train.sh` 첫 부분의 안내대로 만든다. 코드(`train.sh`, `train_act.py`)는 고치지 않는다.

## 2. 학습
```bash
cd /home/work/act_mix2_c50
nohup bash train.sh --data fruit_scripted_mix2.zip --steps 100000 --batch-size 16 --save-freq 10000 --name act_mix2_c50 \
  --extra --policy.chunk_size=50 --policy.n_action_steps=50 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}}' \
  > train_c50.out 2>&1 &
```
- `--extra` 뒤의 옵션은 lerobot-train 에 그대로 넘어간다.
- `--augment` 는 쓰지 않는다. 쓰면 선명도·위치 이동 증강까지 켜진다.
- **시작 1~2분 뒤 설정 출력을 확인한다.** 아래와 다르면 멈추고, outputs 폴더를 지운 뒤 다시 시작한다.
  - `chunk_size: 50`, `n_action_steps: 50`
  - `image_transforms.enable: true`
  - tfs 키가 brightness·contrast·saturation·hue **네 개만**
  - seed 1000, batch 16
- 첫 저장(1만) 뒤 `checkpoints/010000/pretrained_model/config.json`·`train_config.json` 으로 다시 확인한다.

## 3. 10만에서 제출용 zip (2개)
- `pretrained_model` 안의 파일들을 zip 맨 위에 넣는다. 폴더째 넣지 않는다.
- `submission_act_mix2_c50_100k.zip`: config 그대로 (n_action_steps 50)
- `submission_act_mix2_c50_100k_te.zip`: 사본의 `config.json` 에서 `"n_action_steps": 1`, `"temporal_ensemble_coeff": 0.01` 두 값만 바꾼다.
- 형식 검사: LeRobot `ACTPolicy.from_pretrained` + `make_pre_post_processors` 로 불러 무작위 입력으로 `select_action` 을 120번 부른다.
  - 입력: front·wrist 480x640x3, state 6
  - 출력이 숫자 6개이고 NaN 이 없어야 한다.
- 각 zip 의 sha256 을 적는다.
- 6만·8만 저장 지점도 te zip 으로 만들어 둔다: `…_60k_te.zip`, `…_80k_te.zip`. 로컬에서 체크포인트를 고를 때 쓴다.
- 4개 zip 과 README.md 를 `mgm136044/act_mix2_c50`(비공개)에 올린다. README 에 적을 것:
  - 데이터, 설정, 파일 표(이름·스텝·실행 설정·sha256)
  - 사람 시연 아님

## 4. 보고
- 시작 시각, step/s, 예상·실제 종료 시각
- 실제 설정: config.json 의 chunk_size·n_action_steps, train_config.json 의 image_transforms 전문
- 1만 스텝마다 loss
- zip 이름·sha256·형식 검사 결과, 업로드 확인

## 지킬 것
- 평가 서버에는 아무것도 올리지 않는다.
- 데이터·코드는 고치지 않는다. 비공개로만 다룬다.
- 디스크 여유가 10GB 아래면 멈추고 보고한다.
