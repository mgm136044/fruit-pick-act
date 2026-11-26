# 작업 (팀원 학교 GPU 세션): mix2 데이터로 chunk_size 50 ACT 를 10만 스텝 학습하고 제출용 zip 만들기

## 배경 (처음 보는 사람 기준)
- 수업 과제: Isaac Sim 과일 집기(SO-101 로봇) ACT 정책을 학습해 평가 서버에 낸다. **평가 서버 제출은 하지 않는다**(팀에서 정한 사람이 한다).
- 지금까지 가장 좋은 모델은 `act_mix2_100k_color`이고, 서버 54.2점이다.
  - 데이터 `fruit_scripted_mix2`: 589 에피소드, 스크립트 자동 시연(사람 시연 아님)
  - 학습: 수업 키트 `train.sh`/`train_act.py`(LeRobot 0.6.1), 10만 스텝, 배치 16, seed 1000, 색 증강만, **chunk_size 100**
  - 제출할 때 config 를 temporal ensemble 0.01 로 바꿨다: `n_action_steps` 1, `temporal_ensemble_coeff` 0.01
- 이번 실험: **동작 묶음 길이(chunk_size)만 50으로 줄여** 같은 데이터·스텝으로 학습한다.
  - temporal ensemble 은 겹치는 예측을 평균한다. chunk 100(30fps에서 3.3초)이면 오래된 예측이 많이 섞여 하강 중 보정이 느릴 수 있다. chunk 50(1.7초)이 더 빨리 반응하는지 본다.
  - chunk_size 는 모델 구조라 학습 뒤에는 못 바꾼다.

## 할 일

### 1. 준비
- 데이터: Hugging Face 비공개 데이터셋 저장소 `mgm136044/fruit_scripted_mix2_train`. 들어 있는 파일은 `fruit_scripted_mix2.zip`(약 3.7GB), `train.sh`, `train_act.py`, `README.md` 이다. **받는 방법(접근 권한)은 팀원이 알려 준다.**
  - 받은 뒤 zip 의 sha256 앞부분이 `eca47d3d` 인지 확인한다.
- 작업 폴더 `/home/work/act_mix2_c50` 를 만들고 네 파일을 둔다. 코드는 고치지 않는다.
- `nvidia-smi`, `nproc`, `df -h /home/work` 를 확인한다. 디스크는 압축 해제를 포함해 약 10GB 가 필요하다.

### 2. 학습
```bash
cd /home/work/act_mix2_c50
nohup bash train.sh --data fruit_scripted_mix2.zip --steps 100000 --batch-size 16 --save-freq 10000 --name act_mix2_c50 \
  --extra --policy.chunk_size=50 --policy.n_action_steps=50 \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}}' \
  > train_c50.out 2>&1 &
```
- `--extra` 뒤의 옵션은 lerobot-train 에 그대로 넘어간다. `--augment` 는 쓰지 않는다(선명도·위치 이동 증강까지 켜짐).
- **시작 직후 확인**: 1~2분 뒤 로그의 설정 출력에서 아래를 본다. 다르면 멈추고 outputs 폴더를 지운 뒤 다시 시작한다.
  - `chunk_size: 50`, `n_action_steps: 50`
  - `image_transforms.enable: true`
  - `tfs` 키가 brightness·contrast·saturation·hue **네 개만**
  - seed 1000, batch 16
- 첫 저장(1만) 뒤 `checkpoints/010000/pretrained_model/config.json`·`train_config.json` 으로 다시 확인한다.
- 예상 시간은 약 7시간이다(H200 절반 기준). 실제 step/s 로 끝나는 시각을 보고한다.

### 3. 10만에서 제출용 zip (2개)
- `pretrained_model` 안의 파일들을 zip 맨 위에 넣는다(폴더째 넣지 않는다).
- `submission_act_mix2_c50_100k.zip`: config 그대로 (n_action_steps 50)
- `submission_act_mix2_c50_100k_te.zip`: 사본의 `config.json` 에서 `"n_action_steps": 1`, `"temporal_ensemble_coeff": 0.01` 두 값만 바꾼다
- 형식 검사:
  - 키트 `check_submission.py` 가 있으면 쓴다.
  - 없으면 LeRobot `ACTPolicy.from_pretrained` + `make_pre_post_processors` 로 불러 무작위 입력(front·wrist 480x640x3, state 6)으로 `select_action` 120번. 숫자 6개, NaN 없음.
- 각 zip 의 sha256 을 적는다.
- **전달**: 결과 zip 을 팀이 정한 곳(비공개 저장소 등)에 올린다. 공개 저장소에는 올리지 않는다.

### 4. 보고
- 시작 시각, step/s, 예상·실제 종료 시각
- 실제 설정(config.json 의 chunk_size·n_action_steps, train_config.json 의 image_transforms 전문)
- 1만 스텝마다 loss
- zip 이름·sha256·형식 검사 결과

## 지킬 것
- 평가 서버에는 아무것도 올리지 않는다.
- 데이터·코드는 고치지 않는다. 비공개로만 다룬다.
- 디스크 여유가 10GB 아래면 멈추고 보고한다.
