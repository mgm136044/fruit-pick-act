# 학교 GPU 학습 (ACT, LeRobot 0.6.1)

`train.sh`, `train_act.py` 는 **수업 키트 원본 그대로**다(`student_kit/train/`, 수정 없음). 학교 서버에서는 이 두 파일과 데이터셋 zip 을 한 폴더에 두고 돌린다. 데이터셋 zip 은 `pack_dataset.py`(키트)로 만든 것이고, HF 비공개 저장소에 있다.

## 공통 설정

- 배치 16, seed 1000, 1만 스텝마다 저장, lr 은 키트 기본값
- **이미지 증강은 색만 쓴다.** `--augment` 는 쓰지 않는다. 쓰면 선명도와 위치 이동(affine) 증강까지 켜진다.

```bash
--extra --dataset.image_transforms.enable=true \
  --dataset.image_transforms.tfs='{"brightness": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"brightness": [0.8, 1.2]}}, "contrast": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"contrast": [0.8, 1.2]}}, "saturation": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"saturation": [0.5, 1.5]}}, "hue": {"weight": 1.0, "type": "ColorJitter", "kwargs": {"hue": [-0.05, 0.05]}}}'
```

- **제출 실행 설정은 temporal ensemble 이다.** zip 사본의 `config.json` 에서 `"n_action_steps": 1`, `"temporal_ensemble_coeff": 0.01` 두 값만 바꾼다. 로컬에서는 `../make_te_zip.sh <pretrained_model> <이름> [계수|nNN]` 로 만든다.
- **학습 끝 자동 마무리**: `auto_finish_upload.py`(내가 만든 것, 키트 아님). 학습 폴더에서 학습용 python 으로 nohup 실행하면 학습이 끝나기를 기다렸다가 zip(기본·te) → 서버 방식 형식 검사 → sha256 → HF 비공개 업로드 → 서버 값 대조까지 한다. 학교 세션은 `mgm136044/fruit_mix3_train` 의 `tools/` 에서 받는다. 사용법은 파일 머리말.
- **이어 학습**: 저장 폴더를 복사한 뒤, 새 데이터 zip 과 새 이름으로 `train.sh` 를 다시 돌린다. `train_act.py` 189~192행에서 `--resume=true` 와 새 `--dataset.root` 로 이어 간다.

## 학습 기록

| 이름 | 데이터 | 스텝 | chunk | 상태 / 서버 점수 | 지시 프롬프트 |
|---|---|---|---|---|---|
| act_mix2_100k_color | fruit_scripted_mix2 (589ep) | 10만 | 100 | n100 37.5 · n30 45.8 · n20 50.0 · **te0.01 54.2** | (초기 수동) |
| act_mix2_200k_color | 위에서 이어 학습 | 20만 | 100 | 진행 중 (P0-a) | `../prompts/school_P0a_mix2_200k.md` |
| act_mix2_c50 | mix2 | 10만 | 50 | 진행 중 (11-26 친구 세션) | `../prompts/friend1_T1_mix2_chunk50.md` |
| act_mix3A_100k (변형 C) | fruit_mix3_A (~550ep) | 10만 | 100 | A 생성 뒤 | `../prompts/draft_C_mix3A_chunk100.md` |
| act_mix3A_c50_100k (C50) | fruit_mix3_A | 10만 | 50 | A 생성 뒤 | `../prompts/draft_C50_mix3A_chunk50.md` |
| act_mix3A_R1a_170k (A1) | fruit_mix3_A + DAgger R1a | C 에서 +7만 | 100 | R1a 뒤 | `../prompts/draft_A1_resume_C_with_R1a.md` |
| act_mix3A_c50_R1a_170k (B1) | 같음 | C50 에서 +7만 | 50 | R1a 뒤 | `../prompts/draft_B1_resume_C50_with_R1a.md` |

제출 기록 전체는 `../logs/submissions.md` 에 있다.
