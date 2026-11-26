| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 평균 길이(초) | 데이터셋 | 확인 이미지 |
|---|---|---|---|---|---|---|---|---|
| 딸기 | 26 | 25 | 96% | 25 | 7276 | 9.7 | `/home/mingyeongmin/datasets/fruit_scripted_v1` | `/home/mingyeongmin/fruit_datagen/check_v1/sheet_strawberry.png` |
| 오렌지 | 25 | 25 | 100% | 25 | 7385 | 9.8 | `/home/mingyeongmin/datasets/fruit_scripted_v1` | `/home/mingyeongmin/fruit_datagen/check_v1/sheet_orange.png` |
| 레몬 | 30 | 25 | 83% | 25 | 7487 | 10.0 | `/home/mingyeongmin/datasets/fruit_scripted_v1` | `/home/mingyeongmin/fruit_datagen/check_v1/sheet_lemon.png` |
| 바나나 | 34 | 25 | 74% | 25 | 7266 | 9.7 | `/home/mingyeongmin/datasets/fruit_scripted_v1` | `/home/mingyeongmin/fruit_datagen/check_v1/sheet_banana.png` |
| 합계 | 115 | 100 | 87% | 100 | 29414 | | | |

실패 유형: {"Strawberry": {"놓침": 1}, "Lemon": {"접시 밖": 2, "놓침": 2, "미끄러짐": 1}, "Banana": {"놓침": 9}}

장면: {'변형 장면': 37, '기본 장면': 63}, 같은 과일 개수 {1: 79, 2: 15, 3: 6}, 방해 과일 있음 27, 로봇 이동 25, 시작 자세 변화 25
시도당 실제 시간: 평균 34.4s (최소 13, 최대 46), 총 65.9분
에피소드 길이: 평균 9.8s, 최대 11.0s (30초 제한)
내려갈 때 과일 밀림 > 5mm: 11/100
놓인 위치(접시 중심 기준) |dxy| 평균 19mm, 최대 60mm (성공 기준 x·y 각 100mm)
디스크: 584M	/home/mingyeongmin/datasets/fruit_scripted_v1
GPU 메모리 최고 4327 MiB / 6144 MiB
