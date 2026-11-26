"""팀원 딸기 데이터셋에서 불량 에피소드(55: 시작부터 딸기가 접시 위, 1.8초 동안 정지)를 뺀 복사본을 만든다. 원본은 읽기만."""
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

src = LeRobotDataset("local/dataset_strawberry", root="/home/mingyeongmin/다운로드/datasets/dataset_strawberry", video_backend="pyav")
out = delete_episodes(src, [55], output_dir="/home/mingyeongmin/datasets/team_strawberry_clean", repo_id="local/team_strawberry_clean")
print("DONE", out.meta.total_episodes, out.meta.total_frames)
