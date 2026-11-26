"""학습된 ACT 를 학습 데이터 프레임에 그대로 넣어 예측 동작 묶음(100스텝)과 정답 action 을 비교 (CPU, 시뮬레이터 없음).

덜 배운 것(학습 데이터에서도 오차 큼)인지, 실제로 돌릴 때 오차가 쌓이는 문제인지 가르는 용도.
  source ~/robot_course/activate.sh
  python offline_check.py --checkpoint <pretrained_model> --dataset ~/datasets/fruit_scripted_act --per-fruit 3
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

KIT = os.environ.get("ROBOT_KIT", os.path.expanduser("~/student_kit"))
sys.path.insert(0, KIT)
ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--per-fruit", type=int, default=3)
ap.add_argument("--every", type=int, default=30, help="몇 프레임마다 예측할지")
ap.add_argument("--horizon", type=int, default=30, help="비교할 앞쪽 스텝 수 (묶음 100 중)")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=None)
a = ap.parse_args()
torch.set_num_threads(a.threads)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot_policy import load_policy  # noqa: E402

pol = load_policy(os.path.abspath(os.path.expanduser(a.checkpoint)), "cpu", ["act"])
root = os.path.expanduser(a.dataset)
ds = LeRobotDataset(f"local/{os.path.basename(root)}", root=root, video_backend="pyav")
eps = ds.meta.episodes
by_fruit = {}
for i in range(len(eps)):
    f = eps[i]["tasks"][0].split()[2]
    by_fruit.setdefault(f, []).append(i)
JN = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
rep = {}
for fruit, idxs in by_fruit.items():
    rng = np.random.default_rng(0)
    pick = rng.choice(idxs, size=min(a.per_fruit, len(idxs)), replace=False)
    errs, grip_rows = [], []
    for ei in pick:
        e = eps[int(ei)]
        s0, s1 = e["dataset_from_index"], e["dataset_to_index"]
        acts = np.stack([ds.hf_dataset[j]["action"].numpy() for j in range(s0, s1)])
        for t in range(0, s1 - s0 - a.horizon, a.every):
            x = ds[s0 + t]
            batch = {"observation.images.front": x["observation.images.front"].unsqueeze(0),
                     "observation.images.wrist": x["observation.images.wrist"].unsqueeze(0),
                     "observation.state": x["observation.state"].unsqueeze(0), "task": [x["task"]]}
            with torch.no_grad():
                pol.policy.reset()
                chunk = pol.policy.predict_action_chunk(pol.pre(batch))  # (1, 100, 6) 정규화 단위
                pred = np.stack([pol.post(chunk[:, k]).squeeze(0).float().numpy() for k in range(a.horizon)])
            gt = acts[t:t + a.horizon]
            err = np.abs(pred - gt).mean(0)
            errs.append(err)
            grip_rows.append((int(ei), t, round(float(gt[:, 5].min()), 1), round(float(pred[:, 5].min()), 1)))
    E = np.mean(errs, 0)
    rep[fruit] = {"mae_per_joint": dict(zip(JN, np.round(E, 2).tolist())), "queries": len(errs),
                  "grip_min_gt_vs_pred(ep,t,gt,pred)": grip_rows}
    print(fruit, "관절별 평균 절대 오차(LeRobot 단위, 앞", a.horizon, "스텝):", dict(zip(JN, np.round(E, 2).tolist())), flush=True)
out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "offline_check.json")
json.dump(rep, open(out, "w"), indent=1, ensure_ascii=False)
print("→", out)
