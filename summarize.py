"""생성 로그(JSONL) 집계 → 보고서용 표 (마크다운).

  python summarize.py logs/gen_fruit_scripted_v1.jsonl --dataset ~/datasets/fruit_scripted_v1 --sheets check_v1
"""
import argparse
import json
import os
import subprocess

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--dataset", required=True)
ap.add_argument("--sheets", required=True)
ap.add_argument("--gpu-log", default=None)
a = ap.parse_args()

R = [json.loads(l) for l in open(a.log)]
KO = {"Strawberry": "딸기", "Orange": "오렌지", "Lemon": "레몬", "Banana": "바나나"}
ds = os.path.expanduser(a.dataset)
print("| 과일 | 시도 | 성공 | 성공률 | 에피소드 | 프레임 | 평균 길이(초) | 데이터셋 | 확인 이미지 |")
print("|---|---|---|---|---|---|---|---|---|")
tot = [0, 0, 0]
fails = {}
for f in ("Strawberry", "Orange", "Lemon", "Banana"):
    rs = [r for r in R if r["fruit"] == f]
    if not rs:
        continue
    ok = [r for r in rs if r["success"]]
    fr = sum(r["frames"] for r in ok)
    tot[0] += len(rs)
    tot[1] += len(ok)
    tot[2] += fr
    for r in rs:
        if not r["success"]:
            fails.setdefault(f, {}).setdefault(r["fail"], 0)
            fails[f][r["fail"]] += 1
    sheet = os.path.join(os.path.abspath(a.sheets), f"sheet_{f.lower()}.png")
    print(f"| {KO[f]} | {len(rs)} | {len(ok)} | {len(ok) / len(rs):.0%} | {len(ok)} | {fr} | {np.mean([r['sec'] for r in ok]):.1f} "
          f"| `{ds}` | `{sheet}` |")
print(f"| 합계 | {tot[0]} | {tot[1]} | {tot[1] / max(tot[0], 1):.0%} | {tot[1]} | {tot[2]} | | | |")
print("\n실패 유형:", json.dumps(fails, ensure_ascii=False) if fails else "없음")
ok = [r for r in R if r["success"]]
kinds = {}
for r in ok:
    L = r["layout"]
    k = "기본 장면" if L["kind"] == "default" else "변형 장면"
    kinds[k] = kinds.get(k, 0) + 1
cnt = {}
for r in ok:
    cnt[r["layout"]["count"]] = cnt.get(r["layout"]["count"], 0) + 1
n_extra = sum(bool(r["layout"]["extra"]) for r in ok)
n_robot = sum(any(abs(v) > 0 for v in r["layout"]["robot"]) for r in ok)
print(f"\n장면: {kinds}, 같은 과일 개수 {dict(sorted(cnt.items()))}, 방해 과일 있음 {n_extra}, 로봇 이동 {n_robot}, "
      f"시작 자세 변화 {sum(r['start_perturbed'] for r in ok)}")
wall = [r["wall_s"] for r in R]
print(f"시도당 실제 시간: 평균 {np.mean(wall):.1f}s (최소 {min(wall):.0f}, 최대 {max(wall):.0f}), 총 {sum(wall) / 60:.1f}분")
sec = [r["sec"] for r in ok]
print(f"에피소드 길이: 평균 {np.mean(sec):.1f}s, 최대 {max(sec):.1f}s (30초 제한)")
push = [r.get("fruit_pushed_mm", 0) or 0 for r in ok]
print(f"내려갈 때 과일 밀림 > 5mm: {sum(p > 5 for p in push)}/{len(push)}")
dxy = np.array([r["final_dxy_mm"] for r in ok])
print(f"놓인 위치(접시 중심 기준) |dxy| 평균 {np.linalg.norm(dxy, axis=1).mean():.0f}mm, 최대 {np.linalg.norm(dxy, axis=1).max():.0f}mm (성공 기준 x·y 각 100mm)")
print("디스크:", subprocess.run(["du", "-sh", ds], capture_output=True, text=True).stdout.strip())
if a.gpu_log and os.path.exists(a.gpu_log):
    used = [int(l.split()[1].rstrip(",")) for l in open(a.gpu_log) if len(l.split()) > 2]
    print(f"GPU 메모리 최고 {max(used)} MiB / 6144 MiB")
