| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 평균 길이(초) | 데이터셋 | 확인 이미지 |
|---|---|---|---|---|---|---|---|---|
| 딸기 | 52 | 50 | 96% | 50 | 14820 | 9.9 | `/home/mingyeongmin/datasets/fruit_scripted_v2` | `/home/mingyeongmin/fruit_datagen/check_v2/sheet_strawberry.png` |
| 오렌지 | 50 | 50 | 100% | 50 | 14643 | 9.8 | `/home/mingyeongmin/datasets/fruit_scripted_v2` | `/home/mingyeongmin/fruit_datagen/check_v2/sheet_orange.png` |
| 레몬 | 53 | 50 | 94% | 50 | 14671 | 9.8 | `/home/mingyeongmin/datasets/fruit_scripted_v2` | `/home/mingyeongmin/fruit_datagen/check_v2/sheet_lemon.png` |
| 바나나 | 52 | 50 | 96% | 50 | 14782 | 9.9 | `/home/mingyeongmin/datasets/fruit_scripted_v2` | `/home/mingyeongmin/fruit_datagen/check_v2/sheet_banana.png` |
| 합계 | 207 | 200 | 97% | 200 | 58916 | | | |

실패 유형: {"Strawberry": {"미끄러짐": 1, "놓침": 1}, "Lemon": {"놓침": 3}, "Banana": {"놓침": 2}}

장면: {'변형 장면': 100, '기본 장면': 100}, 같은 과일 개수 {1: 136, 2: 38, 3: 26}, 방해 과일 있음 57, 로봇 이동 56, 시작 자세 변화 55
시도당 실제 시간: 평균 37.0s (최소 15, 최대 54), 총 127.7분
에피소드 길이: 평균 9.8s, 최대 11.1s (30초 제한)
내려갈 때 과일 밀림 > 5mm: 8/200
놓인 위치(접시 중심 기준) |dxy| 평균 21mm, 최대 65mm (성공 기준 x·y 각 100mm)
디스크: 1.2G	/home/mingyeongmin/datasets/fruit_scripted_v2
GPU 메모리 최고 4325 MiB / 6144 MiB
