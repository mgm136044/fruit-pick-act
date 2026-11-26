"""팀원 데이터셋(사람 텔레오퍼레이션 추정) 점검: 형식·디코딩·값·작업 문구·에피소드 길이·확인 이미지.
원본은 읽기만 한다.

  python check_team_dataset.py ~/다운로드/datasets/dataset_lemon --out team_check/lemon
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--out", required=True)
ap.add_argument("--ref", default=os.path.join(HERE, "ref_info.json"))
a = ap.parse_args()
root = os.path.abspath(os.path.expanduser(a.dataset))
out = os.path.abspath(a.out)
os.makedirs(out, exist_ok=True)
rep = {"dataset": root}

info = json.load(open(os.path.join(root, "meta", "info.json")))
ref = json.load(open(a.ref))
diffs = []
for k in ("codebase_version", "fps", "robot_type"):
    if info.get(k) != ref.get(k):
        diffs.append(f"{k}: {info.get(k)} != {ref.get(k)}")
for k, rf in ref["features"].items():
    f = info["features"].get(k)
    if f is None:
        diffs.append(f"{k} 없음")
        continue
    for kk in ("dtype", "shape", "names"):
        if f.get(kk) != rf.get(kk):
            diffs.append(f"{k}.{kk}")
    for kk, vv in (rf.get("info") or {}).items():
        if (f.get("info") or {}).get(kk) != vv:
            diffs.append(f"{k}.info.{kk}: {(f.get('info') or {}).get(kk)} != {vv}")
if set(info["features"]) != set(ref["features"]):
    diffs.append("feature 키 다름")
rep["feature_diffs"] = diffs

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ds = LeRobotDataset(f"local/{os.path.basename(root)}", root=root, video_backend="pyav")
eps = ds.meta.episodes
rows = [{k: eps[i][k] for k in ("episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index")}
        for i in range(len(eps))]
rep["episodes"] = len(rows)
rep["tasks"] = sorted({t for r in rows for t in r["tasks"]})
L = np.array([r["length"] for r in rows])
rep["length_s"] = {"min": round(L.min() / 30, 1), "median": round(float(np.median(L)) / 30, 1), "max": round(L.max() / 30, 1),
                   "over_30s": int((L > 900).sum())}
df = pd.read_parquet(os.path.join(root, "data", "chunk-000", "file-000.parquet"))
st = np.stack(df["observation.state"].to_numpy())
ac = np.stack(df["action"].to_numpy())
rep["nan"] = bool(np.isnan(st).any() or np.isnan(ac).any())
rep["state_min"] = np.round(st.min(0), 1).tolist()
rep["state_max"] = np.round(st.max(0), 1).tolist()
rep["action_min"] = np.round(ac.min(0), 1).tolist()
rep["action_max"] = np.round(ac.max(0), 1).tolist()
# 에피소드별: 시작/끝 자세, 그리퍼 최대 열림, 쥔 구간(실제 > 명령 + 1.2), 처음·끝 정지 구간
per = []
for r in rows:
    s = st[r["dataset_from_index"]:r["dataset_to_index"]]
    c = ac[r["dataset_from_index"]:r["dataset_to_index"]]
    mv = np.abs(np.diff(c[:, :5], axis=0)).max(1)
    still = mv < 0.3
    head = int(np.argmax(~still)) if (~still).any() else len(still)
    tail = int(np.argmax(~still[::-1])) if (~still).any() else len(still)
    held = (s[:, 5] - c[:, 5] >= 1.2) & (c[:, 5] > 10)
    per.append({"ep": r["episode_index"], "len": r["length"], "start_state": np.round(s[0], 0).tolist(),
                "end_state": np.round(s[-1], 0).tolist(), "grip_open_max": round(float(c[:, 5].max()), 1),
                "held_frames": int(held.sum()), "still_head": head, "still_tail": tail})
rep["start_far_from_rest"] = sum(np.abs(np.array(p["start_state"][:5]) - [0, -100, 100, 53, 0]).max() > 25 for p in per)
rep["end_far_from_rest"] = sum(np.abs(np.array(p["end_state"][:5]) - [0, -100, 100, 53, 0]).max() > 25 for p in per)
rep["no_hold_detected"] = [p["ep"] for p in per if p["held_frames"] < 10]
rep["still_head_s_median"] = round(float(np.median([p["still_head"] for p in per])) / 30, 2)
rep["still_tail_s_median"] = round(float(np.median([p["still_tail"] for p in per])) / 30, 2)
rep["still_tail_over_2s"] = [p["ep"] for p in per if p["still_tail"] > 60]
json.dump(per, open(os.path.join(out, "per_episode.json"), "w"), indent=0, default=int)

# 확인 이미지: 모든 에피소드의 마지막 front 프레임 (과일이 접시 위인지 눈으로) + 4개 에피소드 흐름
from PIL import Image, ImageDraw  # noqa: E402

W, H = 160, 120
n = len(rows)
cols = 10
grid = Image.new("RGB", (W * cols, (H + 14) * ((n + cols - 1) // cols)), "white")
d = ImageDraw.Draw(grid)
for i, r in enumerate(rows):
    x = ds[r["dataset_to_index"] - 1]
    im = (x["observation.images.front"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    gx, gy = (i % cols) * W, (i // cols) * (H + 14)
    grid.paste(Image.fromarray(im).resize((W, H)), (gx, gy + 14))
    d.text((gx + 3, gy + 1), f"ep{r['episode_index']} {r['length'] / 30:.1f}s", fill="black")
grid.save(os.path.join(out, "last_frames_front.png"))
pick = [rows[i] for i in np.linspace(0, n - 1, 4).round().astype(int)]
W2, H2 = 320, 240
sheet = Image.new("RGB", (W2 * 5, (H2 + 16) * 2 * len(pick)), "white")
d = ImageDraw.Draw(sheet)
for i, r in enumerate(pick):
    fr = np.linspace(0, r["length"] - 1, 5).round().astype(int)
    for ci, cam in enumerate(("front", "wrist")):
        y0 = (2 * i + ci) * (H2 + 16)
        for j, f in enumerate(fr):
            x = ds[r["dataset_from_index"] + int(f)]
            im = (x[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            sheet.paste(Image.fromarray(im).resize((W2, H2)), (j * W2, y0 + 16))
            d.text((j * W2 + 3, y0 + 2), f"ep{r['episode_index']} {cam} f{f}", fill="black")
sheet.save(os.path.join(out, "flow_sheet.png"))
json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, ensure_ascii=False, default=int)
print(json.dumps(rep, indent=1, ensure_ascii=False, default=int))
