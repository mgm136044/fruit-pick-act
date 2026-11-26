# 03 · P1-code 생성기 mix3 수정

- 기간: 2026-11-26 05:35 ~ 05:55
- 상태: ✅ 코드 완료 · 시험 생성(P1-test)은 P0-d 뒤 자동 실행
- 로컬 보고서

## 1. 연구 주제

잡기 방식과 장면 구성을 바꾼 데이터 mix3 를 만들려고 생성기와 보조 스크립트를 고쳤다.

| 파일 | 내용 | 수정 전 사본 |
|---|---|---|
| `gen_scripted.py` | mix3 옵션 추가. 기본값은 v3 동작 그대로 | `logs/gen_scripted_pre_mix3_snapshot.py` |
| `params_mix3.json` | 새 잡기 값 | (새 파일) |
| `make_mix.py` | 합칠 때 `scene_index.json` 기록 | `logs/make_mix_pre_mix3_snapshot.py` |
| `run_mix3.sh` | 과일별 실행 | (새 파일) |
| `mix3_stats.py` | 구성·잡기 지표 통계 | (새 파일) |
| `after_p0c_trial.sh` | P0-d 뒤 시험 생성 연결 | (새 파일) |

## 2. 바꾼 것

### 2-1. 잡기 값 (`params_mix3.json`, `--params` 로 적용)

| 과일 | close (지금 → mix3) | open (지금 → mix3) | 고정 손가락 여유 (지금 → mix3) |
|---|---|---|---|
| 딸기 | 23 → **8** | 42 → 45 | 5 → **11mm** |
| 오렌지 | 33 → **15** | 48 → 51 | 5 → **8mm** |
| 레몬 | 25.5 → **10** | 46 → 47.5 | 5 → **11mm** |
| 바나나 | 16 → **4** | 36 → 37.5 | 7 → **11mm** |

```json
"common": {"T": {"close": 24}, "finger_thick": 0.011, "gap_ext": [[44.0, 71.6], [47.2, 77.9]]}
```

### 2-2. 장면 옵션 (`gen_scripted.py`, 기본값 0 = v3 와 같은 동작·같은 난수)

```python
ap.add_argument("--nearest-margin", ...)   # 같은 과일 여러 개: 가장 가까운 것만, 둘째와 거리 차 < 값이면 다시 뽑음
ap.add_argument("--single-mid-prob", ...)  # 과일 1개: 기준 자리 ±8cm + 회전
ap.add_argument("--plate-prob", ...)       # 접시 ±6cm (변형 장면만, 로봇 받침과 0.22m 이상)
ap.add_argument("--far-prob", ...)         # 목표를 로봇에서 0.30m 넘게 (전체 무작위·여러 개 장면만)
ap.add_argument("--allow-plain-vary", ...) # 개수·로봇 변형이 둘 다 안 뽑혀도 그대로 둠
```

가장 가까운 것만 고르는 부분:

```python
cand_iter = cand
if args.nearest_margin > 0 and cand:  # mix3: 가장 가까운 것만. 둘째와 거리 차가 작으면 다시 뽑음
    cand_iter = cand[:1] if (len(cand) < 2 or cand[1][0] - cand[0][0] >= args.nearest_margin) else []
```

새로 기록하는 값:
- `layout.single_mid`, `layout.plate`(cm), `layout.want_far`, `layout.copies_dist`(사본별 로봇 거리)
- `grasp_shift_mm`: 하강 끝에서 쥐기 끝까지 과일이 움직인 거리. 튕김을 확인하는 지표다.

### 2-3. 실행 구성 (`run_mix3.sh`)

```
--vary-prob 0.9 --vary-use count=0.56,extra=0,robot=0.45 --allow-plain-vary --count3-prob 0.5 --count4-prob 0.3
--single-random-prob 0.5 --single-mid-prob 1.0 --light-prob 0.33 --dart-prob 0.3 --start-perturb-prob 0.3
--nearest-margin 0.025 --plate-prob 0.33 --plate-range 0.06 --far-prob 0.2 --far-min 0.30 --params params_mix3.json
```

- 종류당 4개 자산(`COURSE_ASSETS=assets_c4`)으로 돌린다.
- 과일별로 따로 실행하고, 시드도 과일마다 다르게 준다.
- 끝난 과일은 `.done` 표시로 건너뛴다. 끝나지 않은 폴더는 지우지 않고 이름을 바꿔 둔다.

### 2-4. 합치기 기록 (`make_mix.py` → `scene_index.json`)

- 합친 에피소드 번호마다 원본 이름·번호·과일·장면 기록을 남긴다. 학습 때 `--dataset.episodes` 로 일부만 고를 수 있게 하려는 것이다.
- 합친 순서는 길이로 대조한다. 어긋나면 멈춘다.
- **시험**: pilot_v1(과일별 2개) + v3 레몬 2개 → 10개. 순서·과일·길이·장면 종류가 모두 일치했다.

## 3. 왜

- **잡기 값**: 정책 오차 1~2cm에서도 잡히게 하려는 것이다.
  - 깊게 닫아도 힘 상한(1.0 N·m) 때문에 쥐는 힘은 같다. 대신 중심이 17~22mm 어긋나도 최대 힘으로 잡힌다(지금은 약 4.5mm부터 힘이 준다).
  - 과일이 두 손가락 가운데 오게 했다. 벌림을 +10으로 넓히면 옆 과일에 닿아서 그렇게 하지 않았다(Fable 메시 계산).
- **쥐는 시간**: 깊게 닫으면 닿는 속도가 빨라져 튕길 수 있어서 늘렸다.
- **가장 가까운 것만**: ACT는 지시문을 안 쓴다. 화면만 보고 정답이 하나로 정해져야 한다.
- **`--allow-plain-vary`**: 기존 코드는 변형이 하나도 안 뽑히면 '다른 종류 과일'까지 포함한 후보에서 강제로 하나를 켰다. 그러면 다른 종류 과일 0%와 '로봇 기본 + 과일 무작위' 장면을 만들 수 없었다.
- **접시 이동·먼 목표·±8cm**: 비공개 환경 대비(`set_episode_events` 가 물체별 범위를 받음)와, v3 생성 실패로 먼 목표가 줄어든 선택 편향을 보완하려는 것이다.

## 4. 기대효과

- '못 들어올림·들다 놓침'이 줄어든다(로컬 실패의 70%). 예상 효과는 +10~20%p [추측].
- 같은 과일 여러 개·4개 장면에서 목표가 하나로 정해진다.
- 에피소드별 장면 기록으로 나중에 부분 학습과 분석을 할 수 있다.

## 5. 결론 및 추후 시사점

- 코드는 완료했다. 컴파일 통과, 합치기 시험 통과, 통계 스크립트는 v3 로그로 시험했다.
- **시뮬레이터에서는 아직 확인하지 않았다.** P1-test(과일당 10회)에서 아래를 v3와 비교해 값을 확정한다.
  - 성공률
  - 쥐는 동안 이동(`grasp_shift_mm`)과 하강 끝 밀림
  - 옆 과일 간섭(다시 뽑기 수)
  - 구성 비율
- 문제가 생기면 쥐기·시간 변경만 남기고 벌림은 v3 값으로 되돌린다.
