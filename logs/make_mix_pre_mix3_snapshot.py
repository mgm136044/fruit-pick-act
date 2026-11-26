"""여러 데이터셋에서 에피소드를 골라 하나로 합친 학습용 데이터셋을 만든다 (원본은 읽기만, 결과는 새 폴더).

  python make_mix.py --out ~/datasets/mix_act_v2 \
      --src ~/datasets/fruit_scripted_v2:logs/gen_fruit_scripted_v2.jsonl:all:no_other_fruit \
      --src ~/datasets/team_lemon_ref::lemon: --src ~/datasets/team_strawberry_clean::strawberry:

--src 형식  경로:생성로그(없으면 빈칸):과일(all 또는 lemon,orange...):필터(no_other_fruit 등, 없으면 빈칸)[:최대개수/과일]
  - no_other_fruit : 다른 종류 방해 과일이 있는 에피소드 제외 (ACT 는 작업 문구를 입력으로 쓰지 않아 대상 과일을 구분할 수 없음)
  - default_only    : 기본 장면만
  - vary_only       : 변형 장면만 (생성 로그 필요)
과일은 작업 문구("Pick the <fruit> ...")로 판단.
"""
import argparse
import json
import os
import shutil
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--src", action="append", required=True)
a = ap.parse_args()

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

out = os.path.abspath(os.path.expanduser(a.out))
if os.path.exists(out):
    raise SystemExit(f"{out} 가 이미 있습니다")
tmp = tempfile.mkdtemp(prefix="mix_", dir=os.path.dirname(out))
parts, summary = [], []
for si, spec in enumerate(a.src):
    f = (spec.split(":") + ["", "", "", ""])[:5]
    root, log, fruits, filt, cap = os.path.expanduser(f[0]), f[1], f[2] or "all", f[3], f[4]
    ds = LeRobotDataset(f"local/{os.path.basename(root)}", root=root, video_backend="pyav")
    eps = ds.meta.episodes
    fruit_of = {int(eps[i]["episode_index"]): eps[i]["tasks"][0].split()[2] for i in range(len(eps))}
    layout = {}
    if log:
        for line in open(log):
            r = json.loads(line)
            if r.get("success") and r.get("episode_index") is not None:
                layout[r["episode_index"]] = r["layout"]
    keep, per = [], {}
    for e, fr in sorted(fruit_of.items()):
        if fruits != "all" and fr not in fruits.split(","):
            continue
        L = layout.get(e)
        if filt == "no_other_fruit" and L is not None and L["extra"]:
            continue
        if filt == "default_only" and L is not None and L["kind"] != "default":
            continue
        if filt == "vary_only" and (L is None or L["kind"] != "vary"):
            continue
        if cap and per.get(fr, 0) >= int(cap):
            continue
        per[fr] = per.get(fr, 0) + 1
        keep.append(e)
    drop = [e for e in fruit_of if e not in keep]
    summary.append({"src": root, "filter": filt or "-", "kept": len(keep), "per_fruit": per})
    if drop:
        sub = delete_episodes(ds, drop, output_dir=os.path.join(tmp, f"part{si}"), repo_id=f"local/part{si}")
    else:
        sub = ds
    parts.append(sub)
merged = merge_datasets(parts, output_repo_id=f"local/{os.path.basename(out)}", output_dir=out)
shutil.rmtree(tmp, ignore_errors=True)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("MERGED", out, merged.meta.total_episodes, "episodes", merged.meta.total_frames, "frames")
