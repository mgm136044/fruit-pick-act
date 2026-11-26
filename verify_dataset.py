"""생성 데이터셋 검증 (2단계). Isaac Sim 불필요, 키트 환경(lerobot 0.6.1)에서 실행.

  source ~/robot_course/activate.sh
  python verify_dataset.py --dataset ~/datasets/fruit_scripted_v1 --log logs/gen_fruit_scripted_v1.jsonl --out check_v1

1) meta/info.json features 를 기준(ref_info.json, record.py --fake-leader 로 만든 것)과 비교
2) LeRobotDataset(video_backend="pyav") 로 불러와 모든 에피소드의 처음/끝 프레임 + 한 에피소드 전체를 디코딩
3) state/action 값 범위, NaN, 과일별 에피소드·프레임 수
4) 과일별 확인 이미지: 에피소드 3개 × (첫 프레임, 쥐는 순간, 놓는 순간, 마지막) × front·wrist → PNG 한 장
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--ref", default=os.path.join(HERE, "ref_info.json"))
ap.add_argument("--log", required=True, help="gen_scripted.py 의 JSONL (쥐는/놓는 프레임 위치)")
ap.add_argument("--out", required=True)
ap.add_argument("--per-sheet", type=int, default=3)
a = ap.parse_args()

root = os.path.abspath(os.path.expanduser(a.dataset))
out = os.path.abspath(os.path.expanduser(a.out))
os.makedirs(out, exist_ok=True)
report = {"dataset": root}

# ---- 1) features 비교
info = json.load(open(os.path.join(root, "meta", "info.json")))
ref = json.load(open(a.ref))
diffs = []
for k in ("codebase_version", "fps", "robot_type", "chunks_size", "data_path", "video_path"):
    if info.get(k) != ref.get(k):
        diffs.append(f"{k}: {info.get(k)} != 기준 {ref.get(k)}")
if set(info["features"]) != set(ref["features"]):
    diffs.append(f"feature 키 다름: {sorted(set(info['features']) ^ set(ref['features']))}")
for k, rf in ref["features"].items():
    f = info["features"].get(k)
    if f is None:
        continue
    for kk in ("dtype", "shape", "names"):
        if f.get(kk) != rf.get(kk):
            diffs.append(f"{k}.{kk}: {f.get(kk)} != 기준 {rf.get(kk)}")
    for kk, vv in (rf.get("info") or {}).items():
        if (f.get("info") or {}).get(kk) != vv:
            diffs.append(f"{k}.info.{kk}: {(f.get('info') or {}).get(kk)} != 기준 {vv}")
report["feature_diffs"] = diffs
report["info"] = {k: info[k] for k in ("codebase_version", "fps", "robot_type", "total_episodes", "total_frames",
                                         "total_tasks")}
print("features 비교:", "동일" if not diffs else diffs)

# ---- 2) 불러오기 + 디코딩
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

t = time.time()
ds = LeRobotDataset(f"local/{os.path.basename(root)}", root=root, video_backend="pyav")
report["load_s"] = round(time.time() - t, 2)
eps = ds.meta.episodes
ep_rows = [{k: eps[i][k] for k in ("episode_index", "tasks", "length", "dataset_from_index", "dataset_to_index")}
           for i in range(len(eps))]
t = time.time()
bright = []
for r in ep_rows:
    for idx in (r["dataset_from_index"], r["dataset_to_index"] - 1):
        x = ds[idx]
        for cam in ("observation.images.front", "observation.images.wrist"):
            im = x[cam]
            assert tuple(im.shape) == (3, 480, 640), im.shape
            bright.append(float(im.mean()))
report["decode_first_last_s"] = round(time.time() - t, 1)
r0 = ep_rows[0]
t = time.time()
for idx in range(r0["dataset_from_index"], r0["dataset_to_index"]):
    _ = ds[idx]
report["decode_full_episode0"] = {"frames": r0["length"], "s": round(time.time() - t, 1)}
report["image_mean_min_max"] = [round(min(bright), 3), round(max(bright), 3)]
print("디코딩 OK:", report["decode_first_last_s"], "s (처음/끝),", report["decode_full_episode0"])

# ---- 3) 값 범위
import pandas as pd  # noqa: E402

df = pd.concat([pd.read_parquet(os.path.join(dp, fn)) for dp, _, fs in os.walk(os.path.join(root, "data"))
                for fn in sorted(fs) if fn.endswith(".parquet")])
st = np.stack(df["observation.state"].to_numpy())
ac = np.stack(df["action"].to_numpy())
report["nan"] = bool(np.isnan(st).any() or np.isnan(ac).any())
report["state_min"] = np.round(st.min(0), 1).tolist()
report["state_max"] = np.round(st.max(0), 1).tolist()
report["action_min"] = np.round(ac.min(0), 1).tolist()
report["action_max"] = np.round(ac.max(0), 1).tolist()
report["action_minus_state_abs_mean"] = np.round(np.abs(ac - st).mean(0), 2).tolist()
per = {}
for r in ep_rows:
    fr = r["tasks"][0].split()[2]
    p = per.setdefault(fr, {"episodes": 0, "frames": 0, "lengths": []})
    p["episodes"] += 1
    p["frames"] += r["length"]
    p["lengths"].append(r["length"])
for fr, p in per.items():
    p["sec_mean"] = round(float(np.mean(p["lengths"])) / 30, 2)
    p["sec_max"] = round(max(p["lengths"]) / 30, 2)
    del p["lengths"]
report["per_fruit"] = per
print("과일별:", per)

# ---- 4) 확인 이미지
from PIL import Image, ImageDraw  # noqa: E402

recs = {}
for line in open(a.log):
    r = json.loads(line)
    if r.get("success") and r.get("episode_index") is not None:
        recs[r["episode_index"]] = r
W, H = 320, 240
sheets = {}
for fr in sorted(per):
    rows = [r for r in ep_rows if r["tasks"][0].split()[2] == fr]
    pick = [rows[i] for i in np.linspace(0, len(rows) - 1, min(a.per_sheet, len(rows))).round().astype(int)]
    canvas = Image.new("RGB", (W * 4, (H + 18) * 2 * len(pick)), "white")
    d = ImageDraw.Draw(canvas)
    for i, r in enumerate(pick):
        kf = recs.get(r["episode_index"], {}).get("key_frames", {})
        frames = [("첫 프레임 start", 0), ("쥐기 grasp", kf.get("settle", kf.get("close", 0))),
                  ("놓기 release", kf.get("open", 0)), ("마지막 end", r["length"] - 1)]
        for ci, cam in enumerate(("front", "wrist")):
            y0 = (2 * i + ci) * (H + 18)
            for j, (lab, f) in enumerate(frames):
                x = ds[r["dataset_from_index"] + int(f)]
                im = (x[f"observation.images.{cam}"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                canvas.paste(Image.fromarray(im).resize((W, H)), (j * W, y0 + 18))
                d.text((j * W + 4, y0 + 3), f"ep{r['episode_index']} {cam} {lab.split()[-1]} f{int(f)}", fill="black")
    path = os.path.join(out, f"sheet_{fr}.png")
    canvas.save(path)
    sheets[fr] = path
report["sheets"] = sheets
json.dump(report, open(os.path.join(out, "verify_report.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(report, indent=1, ensure_ascii=False))
