"""3단계-1: HF LightwheelAI/leisaac-pick-orange (사람 텔레오퍼레이션, 오렌지 3개 연속) 에서
쥐는 시점 3개·놓는 시점 3개를 찾고, 그 시점 관절값의 순기구학으로 오렌지·접시 위치(로봇 베이스 기준)를 복원한다.

  python orange_recover.py   →  orange_segments.json

- 관절값: LeRobot 단위 → rad (수업 키트 so101.motor_to_rad, LeIsaac 과 같은 변환)
- 그리퍼 패턴 (리서치 요약 2번): 열림 ≈40~47, 쥐는 동안 명령 ≈25~30, 실제 ≈29~33 (오렌지에 막혀 명령보다 덜 닫힘)
  쥐기 = 명령이 열림에서 '쥔 값' 쪽으로 내려가 12프레임 이상 머무름, 놓기 = 그 뒤 다시 열림 쪽으로 올라감
- 오렌지 중심 = 쥔 순간 두 손가락 사이 (그리퍼 각도에서 벌림 폭 → 고정 손가락에서 폭/2), 깊이는 TCP 기본값
- 접시 = 놓는 순간 TCP 수평 위치 (오렌지가 그 아래로 떨어짐)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.environ.get("ROBOT_KIT", os.path.expanduser("~/student_kit")), "isaac"))
from course_env.so101 import motor_to_rad  # noqa: E402
from kin import SO101Kin  # noqa: E402

# 그리퍼 각도(도) → TCP 깊이(-0.085) 에서 손가락 벌림 폭 (explore 메시 분석, -0.08/-0.09 평균)
GAP_TABLE = [(-10, 8.2), (0, 21.7), (10, 32.0), (14, 36.2), (18, 40.4), (22, 45.4), (26, 49.9), (30, 54.3), (34, 59.1),
             (38, 63.3), (42, 69.2)]
FIXED_INNER_X, TCP_DEPTH = -0.0105, -0.085


def gap_at(qdeg):
    a = np.array(GAP_TABLE)
    return float(np.interp(qdeg, a[:, 0], a[:, 1])) / 1000


def events(g_cmd, g_state):
    """(grasp 시작, 쥔 순간, release 시작) 목록. g_cmd = 그리퍼 명령(LeRobot 단위)."""
    hi = np.percentile(g_cmd[g_cmd > 20], 90) if (g_cmd > 20).any() else 40
    out, t, n = [], 0, len(g_cmd)
    while t < n:
        # 열린 상태 (hi-5 이상) 에서 hi-8 아래로 내려가는 지점
        if g_cmd[t] < hi - 8 and t > 0 and g_cmd[t - 1] >= hi - 8:
            t0 = t
            u = t
            while u < n and g_cmd[u] < hi - 6:
                u += 1
            if u - t0 >= 12 and g_cmd[t0:u].min() > 15:  # 쥐는 구간 (휴식 자세로 접는 마지막은 제외: 15 미만까지 닫힘)
                seg = g_state[t0:u]
                # 쥔 순간: 실제 값이 멈춘 첫 프레임 (5프레임 변화 < 0.3)
                held = t0 + next((k for k in range(len(seg) - 5) if abs(seg[k + 5] - seg[k]) < 0.3), len(seg) // 2)
                out.append({"close_start": int(t0), "held": int(held), "release": int(u)})
            t = u
        t += 1
    return out


def events_blocked(g_cmd, g_state, min_len=15, gap_fill=8, margin=1.2):
    """쥔 구간 = 실제 그리퍼가 명령보다 margin 이상 덜 닫힘(과일에 막힘)이 min_len 프레임 이상.
    짧은 끊김(gap_fill 이하)은 메움. 돌려주는 값: [{close_start, held, release}]"""
    h = (g_state - g_cmd >= margin) & (g_cmd > 10)
    idx = np.flatnonzero(h)
    runs = []
    if len(idx):
        s0 = p0 = idx[0]
        for i in idx[1:]:
            if i - p0 > gap_fill:
                runs.append((s0, p0))
                s0 = i
            p0 = i
        runs.append((s0, p0))
    out = []
    for a, b in runs:
        if b - a + 1 >= min_len:
            # 닫기 시작 = 구간 앞에서 명령이 내려가기 시작한 곳 (최대 30프레임 앞까지)
            c = a
            while c > max(0, a - 30) and g_cmd[c - 1] > g_cmd[c] + 0.05:
                c -= 1
            out.append({"close_start": int(c), "held": int(min(a + 3, b)), "release": int(b + 1)})
    return out


def main():
    kin = SO101Kin()
    fs = sorted(glob.glob(os.path.join(HERE, "hf_orange", "data", "chunk-000", "*.parquet")))
    res = []
    for f in fs:
        d = pd.read_parquet(f)
        st = np.stack(d["observation.state"].to_numpy()).astype(np.float64)
        ac = np.stack(d["action"].to_numpy()).astype(np.float64)
        ev = events_blocked(ac[:, 5], st[:, 5])
        ep = int(d["episode_index"].iloc[0])
        segs = []
        for e in ev:
            q = motor_to_rad(st[e["held"]])
            qdeg_g = float(np.degrees(q[5]))
            w = gap_at(qdeg_g)
            kin.tcp = np.array([FIXED_INNER_X + w / 2, 0.0, TCP_DEPTH])
            p_or, a_dir, _ = kin.tcp_frame(q[:5])
            qr = motor_to_rad(st[e["release"]])
            p_rel, _, _ = kin.tcp_frame(qr[:5])
            segs.append({**e, "orange_b": np.round(p_or, 4).tolist(), "release_b": np.round(p_rel, 4).tolist(),
                         "grip_state_held": round(float(st[e["held"], 5]), 1),
                         "grip_cmd_held": round(float(ac[e["held"], 5]), 1), "width_est_mm": round(w * 1000, 1),
                         "tilt_deg": round(kin.tilt_deg(q[:5]), 1)})
        # 실제로 옮긴 것만: 쥔 위치와 놓은 위치가 5cm 이상 떨어지고, 쥔 높이가 조리대 근처
        segs = [s_ for s_ in segs if np.hypot(*(np.array(s_["release_b"][:2]) - s_["orange_b"][:2])) > 0.05
                and s_["orange_b"][2] < 0.10]
        res.append({"episode": ep, "frames": len(d), "n_events": len(segs), "segments": segs})
    with open(os.path.join(HERE, "orange_segments.json"), "w") as f:
        json.dump(res, f, indent=1)
    n3 = sum(r["n_events"] == 3 for r in res)
    print(f"에피소드 {len(res)}개, 쥐기 3번 검출 {n3}개, 검출 수 분포 {np.bincount([r['n_events'] for r in res]).tolist()}")
    allseg = [s for r in res for s in r["segments"]]
    O = np.array([s["orange_b"] for s in allseg])
    R = np.array([s["release_b"] for s in allseg])
    print("오렌지 위치(베이스) 평균", O.mean(0).round(3), "표준편차", O.std(0).round(3), "z 범위", O[:, 2].min().round(3), O[:, 2].max().round(3))
    print("놓는 위치(베이스) 평균", R.mean(0).round(3), "표준편차", R.std(0).round(3))
    print("쥔 실제 그리퍼", np.percentile([s["grip_state_held"] for s in allseg], [10, 50, 90]),
          "명령", np.percentile([s["grip_cmd_held"] for s in allseg], [10, 50, 90]),
          "추정 폭 mm", np.percentile([s["width_est_mm"] for s in allseg], [10, 50, 90]))


if __name__ == "__main__":
    main()
