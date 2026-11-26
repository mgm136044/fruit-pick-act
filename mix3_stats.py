"""mix3 생성 로그(JSONL) → 구성 통계표 (목표 비율과 비교) + 잡기 품질 지표. 표준 라이브러리만.

  python3 -I mix3_stats.py logs/gen_fruit_mix3_A_*.jsonl            # 저장된 것 기준
  python3 -I mix3_stats.py logs/gen_fruit_mix3_trial_*.jsonl --all  # 시험: 시도 전체(성공/실패) 기준
  python3 -I mix3_stats.py ... --ref logs/gen_fruit_scripted_v3.jsonl  # v3 와 잡기 지표 비교
"""
import argparse
import collections
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("logs", nargs="+")
ap.add_argument("--all", action="store_true", help="실패 시도도 구성 통계에 포함")
ap.add_argument("--ref", default=None, help="비교할 로그 (예: v3)")
ap.add_argument("--md", default=None, help="마크다운 표로 저장")
a = ap.parse_args()

KO = {"Strawberry": "딸기", "Orange": "오렌지", "Lemon": "레몬", "Banana": "바나나"}
TARGET = [  # (이름, 판정 함수, 목표 범위 %)
    ("기본 자리 (±3cm, 로봇·접시 기본)", lambda r: r["layout"]["kind"] == "default", (10, 15)),
    ("같은 과일 2~4개", lambda r: (r["layout"].get("placed") or r["layout"]["count"]) > 1, (45, 55)),
    ("  └ 4개", lambda r: (r["layout"].get("placed") or r["layout"]["count"]) == 4, (12, 18)),
    ("과일 1개 ±8cm + 회전", lambda r: r["layout"].get("single_mid"), (15, 25)),
    ("과일 1개 조리대 전체", lambda r: r["layout"].get("single_random"), (15, 25)),
    ("다른 종류 과일", lambda r: bool(r["layout"]["extra"]), (0, 0)),
    ("로봇 이동", lambda r: any(abs(v) > 1e-6 for v in r["layout"]["robot"]), (35, 45)),
    ("접시 이동 ±6cm", lambda r: r["layout"].get("plate") is not None, (25, 35)),
    ("조명 변화", lambda r: r["layout"].get("light") is not None, (25, 35)),
    ("DART", lambda r: r.get("dart"), (25, 35)),
    ("목표가 로봇에서 0.30m 넘게", lambda r: (r["layout"].get("copies_dist") or [0])[0] >= 0.30, (15, 100)),
]


def load(paths):
    rows = []
    for p in paths:
        for line in open(p):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def q(vals, p):
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return None
    return round(vals[min(len(vals) - 1, int(p * (len(vals) - 1) + 0.5))], 1)


rows = load(a.logs)
ok = [r for r in rows if r.get("success")]
base = rows if a.all else ok
out = []
out.append(f"시도 {len(rows)} · 성공 {len(ok)} ({100 * len(ok) / max(1, len(rows)):.0f}%) · 구성 통계 기준: {'시도 전체' if a.all else '저장된 것'} {len(base)}")
per = collections.defaultdict(lambda: [0, 0, collections.Counter()])
for r in rows:
    k = KO.get(r["fruit"], r["fruit"])
    per[k][0] += 1
    per[k][1] += bool(r.get("success"))
    per[k][2][r.get("fail") or "성공"] += 1
out.append("\n| 과일 | 시도 | 성공 | 성공률 | 결과 |\n|---|---|---|---|---|")
for k, (n, s, c) in per.items():
    out.append(f"| {k} | {n} | {s} | {100 * s / max(1, n):.0f}% | {dict(c)} |")
out.append("\n| 장면 특징 | 실제 | 목표 | |\n|---|---|---|---|")
for name, fn, (lo, hi) in TARGET:
    v = 100 * sum(1 for r in base if fn(r)) / max(1, len(base))
    flag = "" if lo - 5 <= v <= hi + 5 else " ⚠"
    out.append(f"| {name} | {v:.0f}% | {lo}~{hi}% |{flag} |")
cnt = collections.Counter((r["layout"].get("placed") or r["layout"]["count"]) for r in base)
gaps = [(r["layout"]["copies_dist"][1] - r["layout"]["copies_dist"][0]) * 100 for r in base
        if r["layout"].get("copies_dist") and len(r["layout"]["copies_dist"]) > 1]
out.append(f"\n같은 과일 개수 분포: {dict(sorted(cnt.items()))} · 가장 가까운 것과 둘째 거리 차(cm) 최소 {min(gaps):.1f} · 중앙 {st.median(gaps):.1f}" if gaps else "")
secs = [r["sec"] for r in ok]
if secs:
    out.append(f"에피소드 길이(저장): 평균 {st.mean(secs):.1f}초 · 최대 {max(secs):.1f}초 · 총 {sum(r['frames'] for r in ok):,} 프레임")


def grasp_metrics(rs, label):
    g = [r.get("grasp_shift_mm") for r in rs]
    pu = [r.get("fruit_pushed_mm") for r in rs]
    lifted = [r for r in rs if r.get("lift_dz") is not None]
    return (f"| {label} | {len(rs)} | {q(pu, .5)} / {q(pu, .9)} | "
            f"{q(g, .5)} / {q(g, .9)} | {sum(1 for v in g if v is not None and v > 10)} | "
            f"{sum(1 for r in lifted if r['lift_dz'] < 0.03)} |")


out.append("\n| 잡기 지표 | 시도 | 하강 끝 밀림 mm (중앙/90%) | 쥐는 동안 이동 mm (중앙/90%) | 쥐는 동안 10mm 넘게 | 못 들어올림 |\n|---|---|---|---|---|---|")
out.append(grasp_metrics(rows, "이 로그"))
if a.ref:
    out.append(grasp_metrics(load([a.ref]), "비교: " + a.ref.split("/")[-1]))
txt = "\n".join(out)
print(txt)
if a.md:
    open(a.md, "w").write(txt + "\n")
