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
parts, summary, index_rows = [], [], []
for si, spec in enumerate(a.src):
    f = (spec.split(":") + ["", "", "", ""])[:5]
    root, log, fruits, filt, cap = os.path.expanduser(f[0]), f[1], f[2] or "all", f[3], f[4]
    ds = LeRobotDataset(f"local/{os.path.basename(root)}", root=root, video_backend="pyav")
    eps = ds.meta.episodes
    fruit_of = {int(eps[i]["episode_index"]): eps[i]["tasks"][0].split()[2] for i in range(len(eps))}
    layout, recs = {}, {}
    if log:
        for line in open(log):
            r = json.loads(line)
            if (r.get("success") or r.get("result") == "saved") and r.get("episode_index") is not None:  # 생성 로그 또는 DAgger 로그
                layout[r["episode_index"]] = r.get("layout") or {"kind": "dagger", "count": r.get("count"), "extra": {},
                                                                  "robot": r.get("robot"), "cond": r.get("cond")}
                recs[r["episode_index"]] = r
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
    for e in sorted(keep):  # 합친 데이터셋의 에피소드 순서 = 소스 순서 → 원본 번호 순 (merge_datasets 가 이어 붙임)
        r = recs.get(e, {})
        index_rows.append({"src": os.path.basename(root), "src_episode": int(e), "fruit": fruit_of[e],
                           "frames_src": int(eps[e]["length"]) if "length" in eps[e] else None,
                           **{k: r.get(k) for k in ("layout", "target", "dart", "start_perturbed", "speed", "frames",
                                                    "fruit_pushed_mm", "grasp_shift_mm", "lift_dz", "final_dxy_mm")},
                           **({"dagger": {k: r.get(k) for k in ("round", "cond", "mode", "reason", "first_trigger", "t_take", "delay",
                                                                "held_at_take", "ambiguous", "plan_kind", "expert_tries", "driver")}}
                              if r.get("result") == "saved" else {})})
    if drop:
        sub = delete_episodes(ds, drop, output_dir=os.path.join(tmp, f"part{si}"), repo_id=f"local/part{si}")
    else:
        sub = ds
    parts.append(sub)
merged = merge_datasets(parts, output_repo_id=f"local/{os.path.basename(out)}", output_dir=out)
shutil.rmtree(tmp, ignore_errors=True)
# 에피소드별 장면 기록 (학습 때 --dataset.episodes 로 일부만 고를 수 있게). 합친 순서가 맞는지 길이로 확인
meps = merged.meta.episodes
for i, row in enumerate(index_rows):
    row["episode_index"] = i
    ml = int(meps[i]["length"])
    if row["frames_src"] is not None and ml != row["frames_src"]:
        raise SystemExit(f"순서 불일치: 합친 {i} 길이 {ml} != 원본 {row['src']}:{row['src_episode']} 길이 {row['frames_src']}")
with open(os.path.join(out, "scene_index.json"), "w") as fh:
    json.dump({"note": "합친 에피소드 번호 → 원본·장면 기록 (gen_scripted 로그). 사람 시연 아님", "summary": summary,
               "episodes": index_rows}, fh, ensure_ascii=False, indent=0)
print(json.dumps(summary, ensure_ascii=False, indent=1))
print("MERGED", out, merged.meta.total_episodes, "episodes", merged.meta.total_frames, "frames")
