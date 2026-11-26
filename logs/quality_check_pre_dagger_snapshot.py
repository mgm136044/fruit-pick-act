"""생성 데이터셋 품질 검사 (형식 검사 verify_dataset.py 와 별개로, '성공으로 저장됐지만 흠이 있는' 에피소드 찾기).

  source ~/robot_course/activate.sh
  python -I quality_check.py --dataset ~/datasets/fruit_mix3_A_strawberry --log logs/gen_fruit_mix3_A_strawberry.jsonl \
      --out logs/quality_mix3A_strawberry.json [--md logs/quality_mix3A_strawberry.md]

에피소드별로 재는 것 (데이터셋 parquet 의 state/action + 생성 로그):
  action 튐      팔 관절 action 의 프레임 간 최대 변화 (모터 단위). 계획은 최소 저크 곡선이라 크게 튀면 이상
  추종 오차      action[t] 와 state[t+3] 의 최대 차 (팔 관절). 크면 팔이 명령을 못 따라감(충돌·한계)
  끝 정지        마지막에 state 가 거의 안 변하는 프레임 수 (불필요한 정지 구간)
  닫힘 횟수      그리퍼 action 이 닫힘 쪽으로 내려간 횟수 (정상 2: 쥐기 + 끝에 휴식 자세로)
  쥐는 동안 이동  grasp_shift_mm (생성 로그)
  쥐는 순간 손끝 오차  tcp_err_mm.close 크기
  놓은 자리      접시 중심에서 과일까지 수평 거리 final_dxy (성공 판정 한계 100mm)
  사본 거리 차    같은 과일 여러 개일 때 가장 가까운 것과 둘째의 로봇 거리 차 (화면만으로 목표가 정해지는지)
표시(flag) 기준은 FLAGS. 표시된 에피소드 번호 목록을 JSON 에 남긴다(합칠 때 뺄 수 있게).
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--log", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--md", default=None)
a = ap.parse_args()

FLAGS = {  # 이름: (설명, 기준)
    "jump": ("팔 action 프레임 간 변화 > 10 단위 (계획 최대 약 7.5)", 10.0),
    "track": ("action 과 3프레임 뒤 state 차 > 15 단위 (빠른 이동 중 정상 약 10~12)", 15.0),
    "tail_idle": ("끝 정지 > 45 프레임(1.5초)", 45),
    "multi_close": ("닫힘 3회 이상 (정상 2회: 쥐기 + 끝 휴식 자세)", 3),
    "shift": ("쥐는 동안 과일 이동 > 10mm", 10.0),
    "tcp_close": ("쥐는 순간 손끝 오차 > 12mm", 12.0),
    "edge": ("놓은 자리가 접시 중심에서 > 70mm (한계 100)", 70.0),
    "margin": ("사본 거리 차 < 2.5cm", 0.025),
}

root = os.path.expanduser(a.dataset)
df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(root, "data", "*", "*.parquet")))])
logs = {}
for ln in open(a.log):
    r = json.loads(ln)
    if r.get("success") and r.get("episode_index") is not None:
        logs[int(r["episode_index"])] = r

rows = []
for ep, g in df.groupby("episode_index"):
    g = g.sort_values("frame_index")
    act = np.stack(g["action"].to_numpy()).astype(np.float64)
    st = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
    n = len(g)
    jump = float(np.abs(np.diff(act[:, :5], axis=0)).max()) if n > 1 else 0.0
    k = 3
    track = float(np.abs(act[:-k, :5] - st[k:, :5]).max()) if n > k else 0.0
    ds = np.abs(np.diff(st, axis=0)).max(axis=1)
    tail = 0
    for v in ds[::-1]:
        if v < 0.05:
            tail += 1
        else:
            break
    gr = act[:, 5]
    closing = np.diff(gr) < -0.3
    n_close = int(np.sum(closing[1:] & ~closing[:-1])) + int(closing[0]) if len(closing) else 0
    r = logs.get(int(ep), {})
    tcp_c = r.get("tcp_err_mm", {}).get("close") if isinstance(r.get("tcp_err_mm"), dict) else None
    fd = r.get("final_dxy_mm")
    cd = (r.get("layout") or {}).get("copies_dist") or []
    margin = (cd[1] - cd[0]) if len(cd) >= 2 else None
    row = {"ep": int(ep), "frames": n, "jump": round(jump, 2), "track": round(track, 2), "tail_idle": tail, "n_close": n_close,
           "shift": r.get("grasp_shift_mm"), "tcp_close": round(float(np.linalg.norm(tcp_c)), 1) if tcp_c else None,
           "edge": round(float(np.hypot(*fd)), 1) if fd else None, "margin": None if margin is None else round(margin, 3),
           "count": (r.get("layout") or {}).get("count"), "dart": r.get("dart"), "logged": bool(r)}
    fl = []
    if row["jump"] > FLAGS["jump"][1]:
        fl.append("jump")
    if row["track"] > FLAGS["track"][1]:
        fl.append("track")
    if row["tail_idle"] > FLAGS["tail_idle"][1]:
        fl.append("tail_idle")
    if row["n_close"] >= FLAGS["multi_close"][1]:
        fl.append("multi_close")
    if row["shift"] is not None and row["shift"] > FLAGS["shift"][1]:
        fl.append("shift")
    if row["tcp_close"] is not None and row["tcp_close"] > FLAGS["tcp_close"][1]:
        fl.append("tcp_close")
    if row["edge"] is not None and row["edge"] > FLAGS["edge"][1]:
        fl.append("edge")
    if row["margin"] is not None and row["margin"] < FLAGS["margin"][1]:
        fl.append("margin")
    row["flags"] = fl
    rows.append(row)

R = pd.DataFrame(rows)


def q(col):
    v = R[col].dropna().astype(float)
    return {"중앙": round(float(v.median()), 2), "90%": round(float(v.quantile(0.9)), 2), "99%": round(float(v.quantile(0.99)), 2),
            "최대": round(float(v.max()), 2)} if len(v) else None


summary = {"dataset": root, "episodes": len(R), "frames": int(R["frames"].sum()), "logged": int(R["logged"].sum()),
           "dist": {c: q(c) for c in ["frames", "jump", "track", "tail_idle", "n_close", "shift", "tcp_close", "edge", "margin"]},
           "flag_counts": {k: int(sum(k in f for f in R["flags"])) for k in FLAGS},
           "flagged_any": int(sum(bool(f) for f in R["flags"])),
           "flagged_episodes": {int(r.ep): r.flags for r in R.itertuples() if r.flags}}
json.dump({"summary": summary, "flags_def": {k: v[0] for k, v in FLAGS.items()}, "episodes": rows}, open(a.out, "w"),
          ensure_ascii=False, indent=1)
lines = [f"### {os.path.basename(root)}: {summary['episodes']} 에피소드 · {summary['frames']} 프레임 · 로그 연결 {summary['logged']}",
         "", "| 지표 | 중앙 | 90% | 99% | 최대 | 표시 기준 | 표시 수 |", "|---|---|---|---|---|---|---|"]
for c in ["frames", "jump", "track", "tail_idle", "n_close", "shift", "tcp_close", "edge", "margin"]:
    d = summary["dist"][c]
    key = "multi_close" if c == "n_close" else c
    fdef = FLAGS.get(key, ("", None))[0]
    cnt = summary["flag_counts"].get(key, "")
    if d:
        lines.append(f"| {c} | {d['중앙']} | {d['90%']} | {d['99%']} | {d['최대']} | {fdef} | {cnt} |")
lines.append("")
lines.append(f"표시된 에피소드 {summary['flagged_any']}개: " + ", ".join(f"{k}{v}" for k, v in list(summary["flagged_episodes"].items())[:40]))
txt = "\n".join(lines)
print(txt)
if a.md:
    open(a.md, "w", encoding="utf-8").write(txt + "\n")
