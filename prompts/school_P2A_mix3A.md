# 작업: 새 데이터 mix3 A 로 ACT 10만 스텝 학습하고 제출용 zip 만들기

## 배경
- 앞의 작업(`act_mix2_200k_color`, mix2 이어 학습)과 같은 과제다. 평가 서버 제출은 사람이 직접 한다. **너는 제출하지 않는다.**
- 새 데이터 `fruit_mix3_A`는 잡기 방식을 고친 스크립트 자동 시연이다(사람 시연 아님).
  - 과일을 두 손가락 가운데로 잡는다. 쥐는 깊이는 이전 데이터와 같다.
  - 같은 과일 2~4개, 접시 이동, 과일 위치·회전을 넓혔고, 다른 종류 과일 장면은 없다.
  - 에피소드는 약 550개다. 정확한 수는 zip 안 `meta/info.json` 을 본다.
- 목적: 같은 스텝(10만)에서 mix2 모델(서버 54.2점)보다 나은지 서버에서 비교한다.
- 이 학습은 나중에 B 묶음을 더한 데이터(A+B)로 20만 스텝까지 **이어 학습**한다. 그러니 `outputs/act_mix3A_100k` 의 저장 지점(`training_state` 포함)을 지우지 않는다.

## 할 일

### 1. 준비
- 데이터 받기: Hugging Face 비공개 데이터셋 저장소 `mgm136044/fruit_mix3_train` 의 `fruit_mix3_A.zip`(+ `README.md`). 받은 뒤 크기와 sha256 을 저장소 README 의 값과 대조한다.
- 새 작업 폴더 `/home/work/act_mix3` 를 만든다.
  - `/home/work/act_mix2` 의 `train.sh`, `train_act.py` 를 복사한다(같은 키트 코드, 고치지 않음).
  - zip 을 이 폴더에 둔다.
- GPU 에서 다른 학습(예: `act_mix2_200k_color` 이어 학습)이 아직 돌고 있으면 멈추지 않는다. 대신 아래를 보고한다.
  - `nvidia-smi` 로 메모리 여유를 본다.
  - 둘을 같이 돌릴 수 있는지 판단한다. 메모리가 충분하면 같이 돌린다. 속도는 둘 다 느려진다.
  - 메모리가 부족하면 앞 학습이 끝날 때까지 기다렸다가 시작한다.

### 2. 학습 (색 증강만 — 앞 학습과 똑같이)
```bash
cd /home/work/act_mix3
nohup bash train.sh --data fruit_mix3_A.zip --steps 100000 --batch-size 16 --save-freq 10000 --name act_mix3A_100k \
  --extra --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}}' \
  > train_A.out 2>&1 &
```
- **증강은 '색만'이어야 한다**: 밝기 0.8~1.2, 대비 0.8~1.2, 채도 0.5~1.5, 색조 ±0.05, 한 번에 최대 3개.
  - `--augment` 만 주면 LeRobot 기본 증강(선명도, 위치 이동·회전 affine)까지 켜지므로 쓰지 않는다.
  - 위 `--dataset.image_transforms.tfs` JSON 이 색 증강 4개만 남기는 옵션이다(LeRobot 설정 파서로 확인함). `/home/work/act_mix2/train.log` 의 10만 학습 명령줄과도 비교해 보고 다르면 보고한다.
- 시작 1~2분 뒤 `outputs/act_mix3A_100k/checkpoints/` 가 생기기 전이라도 학습 로그의 설정 출력을 본다. 아래를 확인한다.
  - `image_transforms.enable=true`
  - `tfs` 에 brightness·contrast·saturation·hue **네 개만** 있다
  - seed 1000, batch 16
  - 다르면 멈추고 고친 뒤 다시 시작한다(그때는 outputs 폴더를 지우고 처음부터).
- 첫 1만 스텝 저장 뒤 `checkpoints/010000/pretrained_model/train_config.json` 의 `dataset.image_transforms` 를 다시 확인한다.

### 3. 10만에서 제출용 zip (2개)
앞 작업과 같은 방식이다. `pretrained_model` 안의 파일을 zip 맨 위에 넣는다.
- `submission_act_mix3A_100k.zip`: config 그대로 (n_action_steps 100)
- `submission_act_mix3A_100k_te.zip`: 사본의 config.json 에서 `"n_action_steps": 1`, `"temporal_ensemble_coeff": 0.01` 두 값만 바꾼다
- 키트 `check_submission.py`(없으면 LeRobot 로 불러 무작위 입력 120번)로 형식을 검사한다. `sha256sum` 을 적는다.

### 4. Hugging Face 비공개 저장소에 올리기
- 저장소 `mgm136044/act_mix3`, **비공개**, 종류 model. 없으면 비공개로 만들고 비공개인지 확인한다.
- zip 2개와 README.md 를 올린다. README 에 들어갈 내용:
  - 데이터: `mgm136044/fruit_mix3_train` 의 `fruit_mix3_A.zip`, 에피소드·프레임 수, 사람 시연 아님
  - 학습: 처음부터 10만 스텝, 배치 16, seed 1000, 색 증강만
  - 파일 표: 이름 · 스텝 · 실행 설정 · sha256

### 5. 보고
- 시작 시각, 속도(step/s), 예상 종료 시각, 실제 사용된 증강 설정(train_config.json 의 image_transforms 부분 전문)
- 1만 스텝마다 loss 표
- zip 이름·sha256·형식 검사 결과, HF 업로드 확인
- `outputs/act_mix3A_100k/checkpoints/last/training_state` 가 있는지 (다음 이어 학습에 필요)

## 지킬 것
- 평가 서버에는 아무것도 올리지 않는다.
- `/home/work/act_mix2` 안의 것(outputs, 데이터, 코드)은 고치거나 지우지 않는다.
- HF 저장소는 비공개로만. 다른 저장소는 건드리지 않는다.
- 디스크 여유가 10GB 아래면 멈추고 보고한다.
