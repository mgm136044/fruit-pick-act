# 작업 (학교 GPU 세션): 변형 B1 — C50 모델(act_mix3A_c50_100k, chunk 50)을 A+DAgger 데이터로 7만 스텝 이어 학습

## 배경
- 앞의 작업(변형 C50: `fruit_mix3_A` 로 `act_mix3A_c50_100k` 10만 스텝, chunk 50)과 **같은 세션·같은 폴더(`/home/work/act_mix3_c50`)** 에서 이어 간다. **평가 서버 제출은 하지 않는다.**
- 새 데이터 `fruit_mix3_A_R1a` = `fruit_mix3_A` 전체 + **DAgger 라운드 1** 에피소드.
  - DAgger 에피소드: 기존 정책이 로봇을 몰다가 실패 조짐(빗나감·빈손으로 들기·떨어뜨림·멈춤)이 보이면 스크립트 전문가가 넘겨받아 성공까지 한 시연이다. 정책이 실패하는 상태에서 회복하는 법을 가르친다.
  - 사람 시연 아님.
  - 에피소드·프레임 {EP2}, `fruit_mix3_A_R1a.zip` {SIZE2} bytes, sha256 `{SHA2}`
- 목적: 금요일 오전 제출. 처음부터가 아니라 C50 의 10만 저장 지점에서 **17만까지 이어** 학습한다.
- 학습 설정(chunk 50, 배치 16, 색 증강만, lr)은 C50 의 `train_config.json` 에서 그대로 이어받는다.

## 할 일
### 1. 준비
```bash
cd /home/work/act_mix3_c50
huggingface-cli download mgm136044/fruit_mix3_train fruit_mix3_A_R1a.zip README_R1a.md --repo-type dataset --local-dir .
sha256sum fruit_mix3_A_R1a.zip   # {SHA2}
ls outputs/act_mix3A_c50_100k/checkpoints/last/training_state   # 있어야 함
```
- C50 학습이 아직 돌고 있으면 끝날 때까지 기다린다(10만 저장과 zip 4개 업로드까지).

### 2. 이어 학습 (원본 C 폴더는 건드리지 않는다)
```bash
cd /home/work/act_mix3_c50
cp -a outputs/act_mix3A_c50_100k outputs/act_mix3A_c50_R1a_170k
nohup bash train.sh --data fruit_mix3_A_R1a.zip --steps 170000 --name act_mix3A_c50_R1a_170k > train_B1.out 2>&1 &
```
- `train.sh` 는 같은 이름의 저장 지점이 있으면 `--resume=true` 로 이어 가고, 데이터 경로만 새 zip 으로 바꾼다(키트 `train_act.py` 189~192행).
- **시작 직후 확인**
  - 로그에 "이어서 학습: 100000 → 170000"
  - 새 데이터의 에피소드 수({EP2} 의 앞 숫자)
  - chunk_size 50, image_transforms 가 색 4개만
  - 다르면 멈추고 보고한다.

### 3. zip (te 설정)
- 15만·17만: `submission_act_mix3A_c50_R1a_150k_te.zip`, `submission_act_mix3A_c50_R1a_170k_te.zip`
  - 사본 config.json 의 `"n_action_steps": 1`, `"temporal_ensemble_coeff": 0.01` 두 값만 바꾼다.
- 17만은 config 그대로인 것도 만든다: `submission_act_mix3A_c50_R1a_170k.zip`
- 형식 검사(무작위 입력 120번, 숫자 6개, NaN 없음)를 하고 sha256 을 적는다.
- `mgm136044/act_mix3`(비공개)에 올리고, README 표에 줄을 더한다.

### 4. 보고
시작 시각, step/s, 예상 종료, 1만마다 loss, zip·sha256·검사·업로드 확인.

## 지킬 것
- 평가 서버에는 아무것도 올리지 않는다.
- `outputs/act_mix3A_c50_100k` 는 고치거나 지우지 않는다.
- HF 는 비공개로만.
- 디스크 여유가 10GB 아래면 멈추고 보고한다.
