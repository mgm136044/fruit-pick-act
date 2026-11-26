# 작업: 학교 GPU 에서 ACT 모델 mix2 를 10만 → 20만 스텝까지 이어 학습하고 제출용 zip 만들기

## 배경 (처음 보는 사람 기준)
- 수업 과제: Isaac Sim 과일 집기(SO-101 로봇) ACT 정책을 학습해 평가 서버에 낸다. 평가 서버 제출은 사람이 직접 한다. **너는 제출하지 않는다.**
- 이 서버에서 이미 학습을 한 번 끝냈다. 수업 키트의 `train.sh`/`train_act.py`(LeRobot 0.6.1)로 데이터 `fruit_scripted_mix2`(589 에피소드)를 10만 스텝 학습했다. 설정은 배치 16, seed 1000, 색 증강만(밝기·대비·채도·색조)이다.
  - 그때 경로(모델 설정 파일에 남은 값): 작업 폴더 `/home/work/act_mix2`, 데이터 `/home/work/act_mix2/data/fruit_scripted_mix2`, 결과 `/home/work/act_mix2/outputs/act_mix2_100k_color`
  - 결과 zip 은 Hugging Face 비공개 저장소 `mgm136044/act_mix2_100k_color` 에 올라가 있다.
- 서버에서 이 10만 모델에 실행 설정 temporal ensemble 0.01 을 주었을 때 점수가 가장 좋았다(54.2점). 이번에는 **같은 데이터로 더 오래(20만 스텝) 학습하면 나아지는지** 보려고 한다.

## 할 일

### 1. 상태 확인 (먼저 보고 시작)
아래를 확인해서 짧게 적어라. 경로가 다르면 실제 위치를 찾아 쓴다.
- `ls /home/work/act_mix2`
  - `train.sh`, `train_act.py`, mix2 데이터셋 zip(이름 확인), `data/`, `outputs/` 가 있는지
- `ls /home/work/act_mix2/outputs/act_mix2_100k_color/checkpoints/`
  - 저장 지점 목록(010000 … 100000, `last`)을 본다.
  - `last` 가 100000 을 가리키는지 본다.
- `ls /home/work/act_mix2/outputs/act_mix2_100k_color/checkpoints/last/`
  - `pretrained_model/` 과 `training_state/` 가 둘 다 있는지 본다. `training_state` 가 있어야 이어 학습이 된다.
- `head -c 3000 /home/work/act_mix2/train.log`
  - 10만 학습 때 실제로 실행된 명령줄(`$ ... lerobot_train ...` 로 시작하는 줄)을 **그대로 복사해서 보고**한다.
  - 특히 `--dataset.image_transforms...` 같은 증강 옵션이 어떻게 들어갔는지 본다. 다음 학습에서 '색 증강만'을 똑같이 재현하는 데 필요하다.
- `nvidia-smi`, `nproc`, `df -h /home/work`

### 2. 10만 결과를 보존하고 새 이름으로 이어 학습
원본 `outputs/act_mix2_100k_color` 는 **절대 지우거나 고치지 않는다.**

```bash
cd /home/work/act_mix2
cp -a outputs/act_mix2_100k_color outputs/act_mix2_200k_color      # -a: 'last' 심볼릭 링크 보존
nohup bash train.sh --data <mix2 데이터셋 zip> --steps 200000 --name act_mix2_200k_color > train_200k.out 2>&1 &
```
- `train_act.py` 는 같은 이름의 outputs 에 저장 지점이 있으면 "이어서 학습: 100000 → 200000" 을 출력한다. 그다음 `--config_path=.../train_config.json --resume=true --steps=200000` 으로 이어 간다.
  - 저장 간격(1만 스텝)과 색 증강 설정은 저장된 설정을 그대로 따른다. `--batch-size` 는 무시된다.
- 시작 뒤 1~2분 안에 `train_200k.out` 과 `train.log` 끝을 보고 확인한다.
  - "이어서 학습 100000 → 200000" 이 찍혔는지
  - 스텝이 100000 이후부터 올라가는지
  - loss 가 10만 시점 근처(약 0.05~0.1)에서 이어지는지
- **이어 학습이 안 될 때**(`training_state` 없음, resume 오류): 바로 멈추고 오류 전문을 보고한다. 임의로 처음부터 다시 학습하지 않는다.
- 세션이 끊겨도 돌도록 `nohup`(또는 tmux)로 돌린다. 예상 시간은 10만 학습 때와 비슷한 약 7시간이다. 실제 속도(step/s)를 재서 끝나는 시각을 보고한다.

### 3. 15만·20만 체크포인트로 제출용 zip 만들기 (각 2개씩)
`checkpoints/150000/pretrained_model`(15만), `checkpoints/200000/pretrained_model`(20만)이 생길 때마다 아래를 한다. 폴더 이름은 `ls` 로 확인한다.
- **zip 안에는 `pretrained_model` 폴더 안의 파일들을 맨 위에 그대로 넣는다**(폴더째 넣지 않는다). 파일은 config.json, model.safetensors, policy_preprocessor.json, policy_postprocessor.json, 정규화 .safetensors 2개, train_config.json 이다.
- 기본판: `submission_act_mix2_150k_color.zip` (config 그대로)
- te판: `submission_act_mix2_150k_color_te.zip`
  - pretrained_model 을 임시 폴더에 복사한다.
  - 사본의 `config.json` 에서 두 값만 바꾼다: `"n_action_steps": 1`, `"temporal_ensemble_coeff": 0.01`. 다른 값과 다른 파일은 손대지 않는다.
  - 사본으로 묶는다.
- 20만도 같은 방식으로 만든다(`..._200k_color.zip`, `..._200k_color_te.zip`).
- **형식 검사**:
  - 수업 키트의 `check_submission.py` 가 이 서버에 있으면 각 zip 에 돌리고 "형식 검사 통과" 를 확인한다.
  - 없으면 LeRobot 으로 직접 불러 본다: `ACTPolicy.from_pretrained` + `make_pre_post_processors`. 그다음 무작위 입력(front·wrist 480x640x3, state 6)으로 `select_action` 을 120번 부르고, 출력이 숫자 6개이며 NaN 이 없는지 확인한다.
- 각 zip 의 `sha256sum` 을 적는다.

### 4. Hugging Face 비공개 저장소에 올리기
- 저장소: `mgm136044/act_mix2_200k_color`, **반드시 비공개(private)**, 종류 model.
  - 없으면 비공개로 만든다. 만든 뒤 비공개인지 확인한다.
  - 10만 때처럼 `hf` CLI 를 쓴다.
- 15만 zip 2개는 생기는 대로 먼저 올린다. 20만 zip 2개는 학습이 끝나면 올린다.
- README.md 를 함께 올린다. 내용은 아래와 같다.
  - 학습 데이터: `mgm136044/fruit_scripted_mix2_train`(비공개), 589 에피소드·174,781 프레임, 스크립트 자동 시연(사람 시연 아님)
  - 학습: 10만 모델(`act_mix2_100k_color`)에서 이어 학습, 총 20만 스텝, 배치 16, seed 1000, 색 증강만
  - 파일 표: 파일 이름 · 스텝 · 실행 설정(기본 = n_action_steps 100 / te = n_action_steps 1 + temporal_ensemble_coeff 0.01) · sha256
- 올린 뒤 저장소 파일 목록과 크기를 확인한다.

### 5. 보고 (끝날 때와 15만 때 두 번)
- 1의 확인 결과(특히 10만 학습 명령줄 전문)
- 이어 학습 시작 시각·속도·예상 종료 시각
- 1만 스텝마다의 loss 표(110000~200000)
- zip 이름·sha256·형식 검사 결과, HF 업로드 확인
- 문제나 이상한 점

## 지킬 것
- 평가 서버(학교 와이파이 주소)에는 아무것도 올리지 않는다. 제출은 사람이 한다.
- `outputs/act_mix2_100k_color`, 데이터셋, `train.sh`·`train_act.py` 는 고치거나 지우지 않는다.
- HF 저장소는 비공개로만 둔다. 다른 저장소는 건드리지 않는다.
- 다른 사람의 작업이나 GPU 를 쓰는 다른 프로세스가 보이면 건드리지 말고 보고한다.
- 디스크 여유가 10GB 아래로 내려가면 멈추고 보고한다.
