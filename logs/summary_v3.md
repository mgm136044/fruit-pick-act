| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 평균 길이(초) | 데이터셋 | 확인 이미지 |
|---|---|---|---|---|---|---|---|---|
| 딸기 | 101 | 100 | 99% | 100 | 29829 | 9.9 | `/home/mingyeongmin/datasets/fruit_scripted_v3` | `/home/mingyeongmin/fruit_datagen/check_v3/sheet_strawberry.png` |
| 오렌지 | 108 | 100 | 93% | 100 | 29629 | 9.9 | `/home/mingyeongmin/datasets/fruit_scripted_v3` | `/home/mingyeongmin/fruit_datagen/check_v3/sheet_orange.png` |
| 레몬 | 101 | 100 | 99% | 100 | 29587 | 9.9 | `/home/mingyeongmin/datasets/fruit_scripted_v3` | `/home/mingyeongmin/fruit_datagen/check_v3/sheet_lemon.png` |
| 바나나 | 119 | 100 | 84% | 100 | 29960 | 10.0 | `/home/mingyeongmin/datasets/fruit_scripted_v3` | `/home/mingyeongmin/fruit_datagen/check_v3/sheet_banana.png` |
| 합계 | 429 | 400 | 93% | 400 | 119005 | | | |

실패 유형: {"Strawberry": {"놓침": 1}, "Orange": {"놓침": 8}, "Lemon": {"놓침": 1}, "Banana": {"놓침": 12, "미끄러짐": 4, "접시 밖": 3}}

장면: {'기본 장면': 60, '변형 장면': 340}, 같은 과일 개수 {1: 128, 2: 103, 3: 169}, 방해 과일 있음 105, 로봇 이동 172, 시작 자세 변화 125
시도당 실제 시간: 평균 36.2s (최소 15, 최대 44), 총 259.2분
에피소드 길이: 평균 9.9s, 최대 11.6s (30초 제한)
내려갈 때 과일 밀림 > 5mm: 20/400
놓인 위치(접시 중심 기준) |dxy| 평균 20mm, 최대 89mm (성공 기준 x·y 각 100mm)
디스크: 2.4G	/home/mingyeongmin/datasets/fruit_scripted_v3
GPU 메모리 최고 4327 MiB / 6144 MiB
