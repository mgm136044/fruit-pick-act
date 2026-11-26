"""과일별 HTML 보고서 생성 (로컬 파일). 다시 실행하면 최신 로그·학습·평가 결과로 갱신된다.

  source ~/robot_course/activate.sh
  python make_report.py            # → report_html/index.html, strawberry.html, orange.html, lemon.html, banana.html
  python make_report.py --no-gif   # GIF 다시 만들지 않음 (이미 있으면 재사용)
"""
import argparse
import glob
import html
import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "report_html")
ASSETS = os.path.join(OUT, "assets")
HOME = os.path.expanduser("~")
ap = argparse.ArgumentParser()
ap.add_argument("--no-gif", action="store_true")
ap.add_argument("--train-run", default="act_scripted_v12,act_scripted_v3mix,act_mix2_100k_color", help="비교할 학습 이름들 (쉼표). 없는 것은 건너뜀")
a = ap.parse_args()
os.makedirs(ASSETS, exist_ok=True)

FRUITS = ["Strawberry", "Orange", "Lemon", "Banana"]
KO = {"Strawberry": "딸기", "Orange": "오렌지", "Lemon": "레몬", "Banana": "바나나"}
EN = {f: f.lower() for f in FRUITS}
RUNS = [("pilot", "파일럿", "logs/gen_pilot_v1.jsonl", f"{HERE}/pilot_v1"),
        ("v1", "본 생성 v1", "logs/gen_fruit_scripted_v1.jsonl", f"{HOME}/datasets/fruit_scripted_v1"),
        ("v2", "개선 v2", "logs/gen_fruit_scripted_v2.jsonl", f"{HOME}/datasets/fruit_scripted_v2"),
        ("v3", "다양화 v3", "logs/gen_fruit_scripted_v3.jsonl", f"{HOME}/datasets/fruit_scripted_v3")]
DATA = ("v1", "v2", "v3")  # 저장 데이터셋 단계 (파일럿 제외)


def load_jsonl(p):
    p = os.path.join(HERE, p)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


LOGS = {k: load_jsonl(p) for k, _, p, _ in RUNS}

# ---------------------------------------------------------------- 과일별 설명 (생성 방법·과정)
GEOM = {
    "Strawberry": ("높이 46mm, 수평 단면 54.6×41.9mm (메시 주축), 30g",
                   "v1: 둥근 과일로 보고 손목 기본 방향으로 쥠 (폭 45mm 가정, 쥐기 25). "
                   "v2: 메시 주축을 재 보니 길쭉해서 짧은 축(41.9mm)을 가로질러 쥐도록 손목을 돌림 (쥐기 23).",
                   "열기 42 / 쥐기 v1 25 → v2 23"),
    "Orange": ("54×56×54mm 구형, 120g",
               "방향에 상관없이 손목 기본 방향으로 쥠. 내려갈 때 벌린 손가락 자리에 다른 과일이 있으면(v2) 손목을 ±45°·90° 돌려 피함.",
               "열기 48 / 쥐기 33"),
    "Lemon": ("높이 45mm, 수평 단면 58.5×45.6mm (대각선 방향으로 길쭉), 80g",
              "v1: AABB(51×50mm)만 보고 둥근 과일로 처리 → 긴 축으로 쥔 경우 손가락에 끼거나 미끄러짐. "
              "v2: 짧은 축(45.6mm)을 가로질러 쥐고, 놓을 때 그리퍼를 접근 때보다 6 더 엶.",
              "열기 46 / 쥐기 v1 29 → v2 25.5 · 놓기 v1 43 → v2 52"),
    "Banana": ("93×152×31mm 휜 모양, 120g, 단면 폭 32~34mm",
               "긴 축에 수직으로 손목을 돌려 짧은 쪽을 가로질러 쥠. v1: 가운데(y=0) 단면을 잡았는데 그곳 중심선이 12.3° 틀어져 있어 "
               "고정 손가락이 바나나 위에 얹히는 실패가 남. v2: 중심선이 곧은 y=-20mm 지점(폭 31.6mm)을 잡고 고정 손가락 여유 5→7mm.",
               "열기 36 / 쥐기 16"),
}
COMMON_METHOD = [
    ("환경", "수업 키트 그대로 (Isaac Sim 6.2 · Isaac Lab 3.0, PickFruitEnvCfg, 기본 렌더링, 창 없이). 키트 원본은 고치지 않고 import."),
    ("기록 형식", "record.py 와 같음: 30fps, front·wrist 640×480, state = 실제 관절, action = 그 스텝에 보낸 관절 목표 (LeRobot 단위), task = \"Pick the … and put it on the plate.\""),
    ("정책", "상태머신: 휴식 자세 → 과일 위 7cm → 잠깐 멈춤 → 수직 하강 → 쥐기 → 10cm 들기 → 접시 위 → 내려놓기 → 놓기 → 위로 → 휴식 자세에 닿으면 종료"),
    ("관절 목표", "로봇 USD 에서 뽑은 순기구학 + 감쇠 최소제곱 IK, 중력 처짐 보정(PhysX 중력 토크 ÷ 관절 강성) + 적분 보정"),
    ("다양성 (v1·v2)", "기본 장면 약 절반 + 변형 장면(같은 과일 2~3개, 다른 과일 1~2종, 로봇 ±10cm·±30°). 시작 자세 25% 변화, 접근·잡기·놓기 위치와 속도 ±10%"),
    ("다양성 (v3)", "1차 정책이 기본 궤적을 외운 것을 보고 바꿈: 모든 장면에서 과일을 조리대에 무작위로 놓음(기본 자리 0%), 같은 과일 2~3개 68%, "
                   "조명 세기 ×0.5~1.5·색 ×0.85~1.15 절반, DART(과일 위로 가는 동안 실행 명령에만 흔들림, 기록은 깨끗한 명령 → 바로잡는 동작) 절반, 로봇 이동 43%, 시작 자세 31%"),
    ("저장", "과일이 접시 위(키트 성공 판정)에 있고 팔이 휴식 자세로 돌아온 것만 저장, 30초 안"),
]

# ---------------------------------------------------------------- SVG 차트
CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--ring:rgba(11,11,11,.10);--warnbg:#fff4e5;--warnink:#7a4a00}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--ring:rgba(255,255,255,.10);--warnbg:#3a2a10;--warnink:#fab219}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--ring:rgba(255,255,255,.10);--warnbg:#3a2a10;--warnink:#fab219}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:8px 0 4px}h2{font-size:19px;margin:36px 0 10px}h3{font-size:16px;margin:20px 0 8px}
p,li{color:var(--ink)}.sub{color:var(--ink2)}nav{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 16px}nav a{color:var(--ink);text-decoration:none;padding:4px 12px;border:1px solid var(--ring);border-radius:999px;background:var(--surface)}
nav a.on{font-weight:600;border-color:var(--ink2)}.notice{background:var(--warnbg);color:var(--warnink);padding:10px 14px;border-radius:8px;margin:12px 0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}
.tile .l{color:var(--ink2);font-size:13px}.tile .v{font-size:26px;font-weight:600}.tile .d{color:var(--muted);font-size:12px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:14px 16px;margin:12px 0}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
svg{max-width:100%;height:auto;display:block}svg text{fill:var(--ink2);font:12px system-ui,sans-serif}svg .val{fill:var(--ink);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{border-bottom:1px solid var(--grid);padding:5px 8px;text-align:left}
th{color:var(--ink2);font-weight:600}details{margin-top:6px}summary{cursor:pointer;color:var(--ink2);font-size:13px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--ink2);margin:4px 0}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;vertical-align:-1px}
img.gif{width:100%;border-radius:8px;border:1px solid var(--ring)}code{font-size:13px}.kv td:first-child{color:var(--ink2);white-space:nowrap;width:110px}
.overflow{overflow-x:auto}
"""


def esc(s):
    return html.escape(str(s))


def table(headers, rows):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="overflow"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def bar_chart(labels, values, title, unit="%", ymax=100, series_var="--s1", sub=None, tips=None):
    """단일 계열 세로 막대 (값은 막대 끝에)."""
    W, H, L, B, T = 520, 240, 44, 34, 16
    n = len(labels)
    pw, ph = W - L - 12, H - B - T
    bw = min(24, pw / max(n, 1) * 0.5)
    g = [f'<svg width="{W}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for t in range(0, ymax + 1, ymax // 4):
        y = T + ph - ph * t / ymax
        g.append(f'<line x1="{L}" x2="{W - 12}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"/>'
                 f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{t}{unit}</text>')
    for i, (lab, v) in enumerate(zip(labels, values)):
        cx = L + pw * (i + 0.5) / n
        if v is None:
            g.append(f'<text x="{cx:.1f}" y="{T + ph - 6}" text-anchor="middle">진행 중</text>')
        else:
            h = ph * min(v, ymax) / ymax
            y = T + ph - h
            r = min(4, h)
            tip = tips[i] if tips else f"{lab}: {v:.0f}{unit}"
            g.append(f'<g><title>{esc(tip)}</title><path d="M{cx - bw / 2:.1f},{T + ph} V{y + r:.1f} Q{cx - bw / 2:.1f},{y:.1f} {cx - bw / 2 + r:.1f},{y:.1f} '
                     f'H{cx + bw / 2 - r:.1f} Q{cx + bw / 2:.1f},{y:.1f} {cx + bw / 2:.1f},{y + r:.1f} V{T + ph} Z" fill="var({series_var})"/>'
                     f'<rect x="{cx - pw / n / 2:.1f}" y="{T}" width="{pw / n:.1f}" height="{ph}" fill="transparent"/></g>'
                     f'<text class="val" x="{cx:.1f}" y="{y - 6:.1f}" text-anchor="middle">{v:.0f}{unit}</text>')
        g.append(f'<text x="{cx:.1f}" y="{H - 12}" text-anchor="middle">{esc(lab)}</text>')
    g.append(f'<line x1="{L}" x2="{W - 12}" y1="{T + ph}" y2="{T + ph}" stroke="var(--axis)"/></svg>')
    return "".join(g)


def grouped_bar(groups, series, values, title):
    """여러 계열 묶음 막대. values[s][g] (None = 아직 없음)."""
    W, H, L, B, T = 640, 272, 44, 46, 18
    pw, ph = W - L - 12, H - B - T
    ng, ns = len(groups), len(series)
    bw = min(24, pw / ng / (ns + 1.5))
    g = [f'<svg width="{W}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for t in range(0, 101, 25):
        y = T + ph - ph * t / 100
        g.append(f'<line x1="{L}" x2="{W - 12}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/><text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{t}%</text>')
    for gi, gname in enumerate(groups):
        c0 = L + pw * (gi + 0.5) / ng - (ns * bw + (ns - 1) * 2) / 2
        for si, (sname, var) in enumerate(series):
            v = values[si][gi]
            x = c0 + si * (bw + 2)
            if v is None:
                continue
            h = ph * v / 100
            y = T + ph - h
            r = min(4, h)
            g.append(f'<g><title>{esc(gname.replace("|", " "))} · {esc(sname)}: {v:.0f}%</title><path d="M{x:.1f},{T + ph} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} '
                     f'Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{T + ph} Z" fill="var({var})"/></g>')
        cx = L + pw * (gi + 0.5) / ng
        parts = gname.split("|")
        g.append(f'<text x="{cx:.1f}" y="{H - 26}" text-anchor="middle">' + "".join(
            f'<tspan x="{cx:.1f}" dy="{0 if k == 0 else 14}">{esc(t)}</tspan>' for k, t in enumerate(parts)) + "</text>")
    g.append(f'<line x1="{L}" x2="{W - 12}" y1="{T + ph}" y2="{T + ph}" stroke="var(--axis)"/></svg>')
    leg = '<div class="legend">' + "".join(f'<span><i style="background:var({v})"></i>{esc(s)}</span>' for s, v in series) + "</div>"
    return leg + "".join(g)


def scatter_xy(points, plate, title):
    """조리대 위 과일 처음 위치 (위에서 본 그림, 정면 카메라 방향에 맞춰 x 를 뒤집음). points: (x, y, ok, tip)."""
    W, H, P = 520, 360, 30
    x0, x1, y0, y1 = 1.92, 2.54, -0.50, -0.22
    sx = lambda x: P + (x1 - x) / (x1 - x0) * (W - 2 * P)  # noqa: E731 (정면 카메라: 장면 +x 가 화면 왼쪽)
    sy = lambda y: P + (y - y0) / (y1 - y0) * (H - 2 * P)  # noqa: E731 (장면 +y(로봇 쪽)가 화면 위)
    g = [f'<svg width="{W}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">',
         f'<rect x="{P}" y="{P}" width="{W - 2 * P}" height="{H - 2 * P}" fill="none" stroke="var(--grid)"/>']
    # 조리대 배치 영역 (record.py AREA)
    g.append(f'<rect x="{sx(2.30):.1f}" y="{sy(-0.48):.1f}" width="{sx(1.95) - sx(2.30):.1f}" height="{sy(-0.26) - sy(-0.48):.1f}" fill="none" stroke="var(--axis)" stroke-dasharray="0" rx="4"/>'
             f'<text x="{sx(2.30) + 4:.1f}" y="{sy(-0.48) + 14:.1f}">변형 장면 배치 영역</text>')
    if plate:
        r = 0.101 / (x1 - x0) * (W - 2 * P)
        g.append(f'<circle cx="{sx(plate[0]):.1f}" cy="{sy(plate[1]):.1f}" r="{r:.1f}" fill="none" stroke="var(--muted)" stroke-width="1.5"/>'
                 f'<text x="{sx(plate[0]):.1f}" y="{sy(plate[1]) + 4:.1f}" text-anchor="middle">접시(기준)</text>')
    g.append(f'<text x="{W / 2}" y="{P - 10}" text-anchor="middle">↑ 로봇 쪽 (정면 카메라에서 본 방향)</text>')
    for x, y, ok, tip in points:
        g.append(f'<g><title>{esc(tip)}</title><circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="5" fill="var({"--s1" if ok else "--s2"})" '
                 f'stroke="var(--surface)" stroke-width="2"/></g>')
    g.append("</svg>")
    n_ok = sum(p[2] for p in points)
    leg = (f'<div class="legend"><span><i style="background:var(--s1)"></i>성공 {n_ok}</span>'
           f'<span><i style="background:var(--s2)"></i>실패 {len(points) - n_ok}</span></div>')
    return leg + "".join(g)


def hist(values, title, bins, unit="초"):
    cnt, edges = np.histogram(values, bins=bins)
    labels = [f"{edges[i]:.1f}" for i in range(len(cnt))]
    W, H, L, B, T = 520, 220, 40, 34, 14
    pw, ph = W - L - 12, H - B - T
    m = max(cnt.max(), 1)
    top = int(math.ceil(m / 5) * 5)
    n = len(cnt)
    bw = pw / n - 2
    g = [f'<svg width="{W}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for t in range(0, top + 1, max(1, top // 5)):
        y = T + ph - ph * t / top
        g.append(f'<line x1="{L}" x2="{W - 12}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/><text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{t}</text>')
    for i, c in enumerate(cnt):
        x = L + i * (pw / n) + 1
        h = ph * c / top
        y = T + ph - h
        r = min(4, h)
        if c:
            g.append(f'<g><title>{edges[i]:.1f}~{edges[i + 1]:.1f}{unit}: {c}개</title><path d="M{x:.1f},{T + ph} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} '
                     f'H{x + bw - r:.1f} Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{T + ph} Z" fill="var(--s1)"/></g>')
        if i % max(1, n // 8) == 0:
            g.append(f'<text x="{x:.1f}" y="{H - 14}">{labels[i]}{unit}</text>')
    g.append(f'<line x1="{L}" x2="{W - 12}" y1="{T + ph}" y2="{T + ph}" stroke="var(--axis)"/></svg>')
    return "".join(g)


def line_chart(xs, ys, title, xlab="스텝", ylab="loss", xmax=None):
    W, H, L, B, T = 640, 240, 52, 34, 14
    pw, ph = W - L - 16, H - B - T
    xm, ym = max(xmax or 0, max(xs)), max(ys) * 1.05
    sx = lambda x: L + pw * x / xm  # noqa: E731
    sy = lambda y: T + ph - ph * y / ym  # noqa: E731
    g = [f'<svg width="{W}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">']
    for k in range(5):
        y = ym * k / 4
        g.append(f'<line x1="{L}" x2="{W - 16}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="var(--grid)"/><text x="{L - 6}" y="{sy(y) + 4:.1f}" text-anchor="end">{y:.2f}</text>')
    for k in range(5):
        x = xm * k / 4
        g.append(f'<text x="{sx(x):.1f}" y="{H - 12}" text-anchor="middle">{f"{x / 1000:g}K" if xm >= 4000 else f"{x:.0f}"}</text>')
    d = "M" + " L".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
    g.append(f'<path d="{d}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    g.append(f'<circle cx="{sx(xs[-1]):.1f}" cy="{sy(ys[-1]):.1f}" r="4" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/>'
             f'<text class="val" x="{sx(xs[-1]) + (8 if sx(xs[-1]) < L + 80 else -6):.1f}" y="{sy(ys[-1]) - 8:.1f}" '
             f'text-anchor="{"start" if sx(xs[-1]) < L + 80 else "end"}">{ys[-1]:.3f}</text>')
    for x, y in zip(xs[::max(1, len(xs) // 60)], ys[::max(1, len(xs) // 60)]):
        g.append(f'<g><title>{x:,} {xlab}: {ylab} {y:.4f}</title><circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="6" fill="transparent"/></g>')
    g.append(f'<line x1="{L}" x2="{W - 16}" y1="{T + ph}" y2="{T + ph}" stroke="var(--axis)"/></svg>')
    return "".join(g)


# ---------------------------------------------------------------- GIF
def make_gif(root, ep, path, stride=4, w=192):
    if a.no_gif and os.path.exists(path):
        return True
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from PIL import Image
        ds = LEROBOT_CACHE.get(root)
        if ds is None:
            ds = LEROBOT_CACHE[root] = LeRobotDataset(f"local/{os.path.basename(root)}", root=root, video_backend="pyav")
        e = ds.meta.episodes[int(ep)]
        frames = []
        h = int(w * 3 / 4)
        for i in range(e["dataset_from_index"], e["dataset_to_index"], stride):
            x = ds[i]
            ims = [Image.fromarray((x[f"observation.images.{c}"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)).resize((w, h))
                   for c in ("front", "wrist")]
            fr = Image.new("RGB", (w * 2 + 4, h), (250, 250, 248))
            fr.paste(ims[0], (0, 0))
            fr.paste(ims[1], (w + 4, 0))
            frames.append(fr.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 * stride / 30), loop=0, optimize=True)
        return True
    except Exception as ex:  # noqa: BLE001
        print("GIF 실패", root, ep, ex)
        return False


LEROBOT_CACHE = {}

# ---------------------------------------------------------------- 학습·평가 결과 (있으면, 여러 학습 비교)
TRAIN_RUNS = [n for n in a.train_run.split(",") if n and os.path.isdir(os.path.join(HERE, "train_runs", n))]
RUN_KO = {"act_scripted_v12": "1차 학습 (v1+v2, 기본 장면 73%)", "act_scripted_v3mix": "2차 학습 (v3 위주, 기본 자리 19%)",
          "act_mix2_100k_color": "학교 GPU 10만 스텝 (2차와 같은 데이터, 배치 16, 색 증강)"}
SCENE_SHORT = {"rp": "run_policy|기본 24", "default": "기본", "count": "같은 과일|2~3개", "count4": "같은 과일|4개", "robot": "로봇|이동",
               "light": "조명", "mixed": "혼합", "distract": "다른 종류|(참고)"}
SCENE_KO = {"default": "기본", "count": "같은 과일 2~3개", "count4": "같은 과일 4개 (서버형)", "robot": "로봇 이동", "light": "조명",
            "mixed": "혼합", "distract": "다른 종류 과일 (서버엔 없음, 참고)"}
VAR_KO = {"n100": "기본 실행 (n_action_steps 100)", "c4": "같은 과일 4개 장면 (assets_c4)", "n50": "n_action_steps 50",
          "n20": "n_action_steps 20", "te01": "temporal ensemble 0.01"}


def parse_train(name):
    d = os.path.join(HERE, "train_runs", name)
    log = os.path.join(d, "train.log")
    if not os.path.exists(log):
        return None
    xs, ys, total = [], [], None
    for line in open(log, errors="ignore").read().replace("\r", "\n").split("\n"):
        if line.startswith("===== "):  # train.sh 가 실행마다 남기는 시작 줄 → 마지막 실행만
            xs, ys, total = [], [], None
        # tqdm 의 정확한 스텝 수 (lerobot 의 step:4K 는 반올림 표시) + 같은 줄의 전체 loss
        m = re.search(r" (\d+)/(\d+) \[.*?step:\S+ .*? loss:([\d.]+)", line)
        if m:
            xs.append(int(m.group(1)))
            ys.append(float(m.group(3)))
            total = int(m.group(2))
    meta = {}
    p = os.path.join(d, "train_meta.json")
    if os.path.exists(p):
        meta = json.load(open(p))
    return {"xs": xs, "ys": ys, "meta": meta, "total": total}


def parse_evals(name):
    rp = None
    p = os.path.join(HERE, "logs", f"eval_run_policy_{name}.log")
    if os.path.exists(p):
        eps = re.findall(r"에피소드 \d+ \((\S+)\): (성공|실패) \((\d+) 스텝", open(p, errors="ignore").read())
        rp = eps or None
    px = {}
    for p in sorted(glob.glob(os.path.join(HERE, "logs", f"eval_proxy_{name}_*.json"))):
        px[os.path.basename(p)[len(f"eval_proxy_{name}_"):-5]] = json.load(open(p))
    return rp, px


TR = {}
for _n in TRAIN_RUNS:
    _rp, _px = parse_evals(_n)
    TR[_n] = {"train": parse_train(_n), "rp": _rp, "px": _px}


# ---------------------------------------------------------------- 공통 통계
def stats(run, fruit):
    rs = [r for r in LOGS[run] if r["fruit"] == fruit]
    if not rs:
        return None
    ok = [r for r in rs if r["success"]]
    fails = {}
    for r in rs:
        if not r["success"]:
            fails[r["fail"]] = fails.get(r["fail"], 0) + 1
    return {"n": len(rs), "ok": len(ok), "rate": 100 * len(ok) / len(rs), "frames": sum(r["frames"] for r in ok),
            "sec": np.mean([r["sec"] for r in ok]) if ok else 0, "fails": fails, "rs": rs, "okr": ok,
            "default": sum(r["layout"]["kind"] == "default" and not r["layout"].get("single_random") for r in ok),
            "vary": sum(not (r["layout"]["kind"] == "default" and not r["layout"].get("single_random")) for r in ok)}


def inline_assets(htm):
    """assets/*.gif 를 base64 로 넣어 HTML 파일 하나로 열리게 (로컬 보존용)."""
    import base64

    def rep(m):
        fp = os.path.join(OUT, m.group(1))
        if not os.path.exists(fp):
            return m.group(0)
        return 'src="data:image/gif;base64,' + base64.b64encode(open(fp, "rb").read()).decode() + '"'
    return re.sub(r'src="(assets/[^"]+\.gif)"', rep, htm)


def page(title, body, active):
    nav = '<nav><a href="index.html" class="{}">전체 요약</a>'.format("on" if active == "index" else "") + "".join(
        f'<a href="{EN[f]}.html" class="{"on" if active == f else ""}">{KO[f]}</a>' for f in FRUITS) + "</nav>"
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body><main>{nav}{body}</main></body></html>')


NOTICE = ('<div class="notice"><b>사람 시연 아님.</b> 이 데이터는 스크립트 정책(상태머신)이 시뮬레이터에서 자동으로 만든 것이다. '
          '자동 생성 데이터가 과제 규칙상 허용됨은 사용자가 확인했다(2026-11-25). 모델 제출은 하지 않았다.</div>')


def small_policy_gif(src, run, max_s=15, step=2):
    """평가 GIF(10fps, 최대 30초)를 보고서용으로 줄임: 앞 max_s 초, 5fps."""
    from PIL import Image, ImageSequence
    dst_dir = os.path.join(ASSETS, f"policy_{run}_small")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    if not os.path.exists(dst):
        im = Image.open(src)
        fr = [f.copy() for f in ImageSequence.Iterator(im)][: max_s * 10: step]
        fr[0].save(dst, save_all=True, append_images=fr[1:], duration=100 * step, loop=0, optimize=True)
    return os.path.relpath(dst, OUT)


def policy_gallery(run, fruit=None, tags=("n100", "c4"), scenes=None):
    """eval_proxy 가 저장한 정책 롤아웃 GIF (장면별)."""
    eps = []
    for t in tags:
        rep = TR[run]["px"].get(t)
        if rep:
            eps += [e for e in rep["episodes"] if e.get("gif") and os.path.exists(e["gif"] if os.path.isabs(e["gif"]) else os.path.join(HERE, e["gif"]))]
    if scenes:
        eps = [e for e in eps if e["scene"] in scenes]
    if fruit is not None:
        eps = [e for e in eps if e["fruit"] == fruit]
    if not eps:
        return ""
    cards = []
    for e in eps:
        src = e["gif"] if os.path.isabs(e["gif"]) else os.path.join(HERE, e["gif"])
        rel = small_policy_gif(src, run)
        res = "성공" if e["success"] else f"실패 · {e.get('why', '')}"
        cards.append(f'<div class="card"><h3>{esc(SCENE_KO.get(e["scene"], e["scene"]))} · {esc(KO[e["fruit"]])}</h3><img class="gif" src="{rel}" alt="정책 롤아웃">'
                     f'<p class="sub">{esc(res)} · {e["steps"]}스텝({e["steps"] / 30:.1f}초) · 최고 들림 {e.get("max_lift_cm")}cm · 앞 15초만</p></div>')
    return (f"<h3>{esc(RUN_KO.get(run, run))} 정책이 실제로 움직이는 모습 (평가 롤아웃, 정면 | 손목)</h3><div class='grid2'>" + "".join(cards) + "</div>")


def _rate(rep, scene, fruit):
    v = rep["scenes"].get(scene)
    if v is None:
        return None
    if fruit is None:
        return 100 * v["success"]
    pf = v["per_fruit"].get(KO[fruit])
    return None if pf is None else 100 * pf


def train_section(fruit=None):
    runs = [n for n in TRAIN_RUNS if TR[n]["train"] and TR[n]["train"]["xs"]]  # loss 기록이 있는 학습
    ev = [n for n in TRAIN_RUNS if TR[n]["rp"] or TR[n]["px"].get("n100")]  # 평가가 있는 학습 (학교 GPU 처럼 loss 기록이 없어도)
    if not runs and not ev:
        return ('<div class="card"><p>학습 진행 예정. 이 보고서는 <code>python make_report.py</code> 로 다시 만들면 결과가 채워진다.</p></div>')
    s = ""
    # 1) 학습끼리 비교 (평가가 있는 것만)
    if ev:
        rows_def = [("기본 환경 (키트 run_policy, 24)", "rp", None)]
        scenes = []
        for n in ev:
            for t in ("n100", "c4"):
                for sc in (TR[n]["px"].get(t) or {"scenes": {}})["scenes"]:
                    if (t, sc) not in scenes:
                        scenes.append((t, sc))
        rows_def += [(SCENE_KO.get(sc, sc), t, sc) for t, sc in scenes]
        vals = {}
        for n in ev:
            col = []
            for label, t, sc in rows_def:
                if t == "rp":
                    eps = [e for e in (TR[n]["rp"] or []) if fruit is None or e[0] == KO[fruit]]
                    col.append((100 * sum(e[1] == "성공" for e in eps) / len(eps), f"{sum(e[1] == '성공' for e in eps)}/{len(eps)}") if eps else None)
                else:
                    rep = TR[n]["px"].get(t)
                    r = None if rep is None else _rate(rep, sc, fruit)
                    if r is None:
                        col.append(None)
                    else:
                        k_ = [e for e in rep["episodes"] if e["scene"] == sc and (fruit is None or e["fruit"] == fruit)]
                        col.append((r, f"{sum(e['success'] for e in k_)}/{len(k_)}"))
            vals[n] = col
        head = ["평가 장면"] + [RUN_KO.get(n, n) for n in ev]
        rows = [(label,) + tuple("—" if vals[n][i] is None else f"{vals[n][i][0]:.0f}% ({vals[n][i][1]})" for n in ev) for i, (label, *_) in enumerate(rows_def)]
        score_row = ("서버형 점수 (distract 제외 장면 평균 ×100)",) + tuple(
            (f'{TR[n]["px"]["n100"]["score_mean_of_scenes_x100"]}' if fruit is None and TR[n]["px"].get("n100") else "—") for n in ev)
        if fruit is None:
            rows.append(score_row)
        groups = [SCENE_SHORT.get("rp" if t == "rp" else sc, label) for label, t, sc in rows_def]
        series = [(RUN_KO.get(n, n), v) for n, v in zip(ev, ("--s1", "--s2", "--s3", "--s4"))]
        gv = [[None if x is None else x[0] for x in vals[n]] for n in ev]
        s += (f'<div class="card"><h3>학습별 평가 성공률 비교{"" if fruit is None else " · " + KO[fruit]}</h3>'
              + grouped_bar(groups, series, gv, "학습별 평가 성공률") + table(head, rows)
              + '<p class="sub">에피소드 수가 장면당 8~24개라 한두 개 차이는 우연일 수 있다. 다른 종류 과일 장면은 서버 평가에 없어 점수에서 뺐다.</p></div>')
        # 실패 이유 (전체 페이지만)
        if fruit is None:
            wr = []
            for n in ev:
                rep = TR[n]["px"].get("n100")
                if rep:
                    wr.append((RUN_KO.get(n, n), ", ".join(f"{k} {v}" for k, v in rep.get("why", {}).items())))
            if wr:
                s += '<div class="card"><h3>변형 장면 평가의 결과 이유 (에피소드 수)</h3>' + table(["학습", "이유별 수"], wr) + "</div>"
        # 실행 설정 비교 (n50·n20·te01)
        vr = []
        for n in ev:
            has_var = any(TR[n]["px"].get(x) for x in ("n50", "n20", "te01"))  # 변형이 있을 때만 기본 실행도 함께 보여 줌
            for t in ("n100", "n50", "n20", "te01"):
                rep = TR[n]["px"].get(t)
                if rep and has_var:
                    vr.append((RUN_KO.get(n, n), VAR_KO.get(t, t)) + tuple(
                        "—" if _rate(rep, sc, fruit) is None else f"{_rate(rep, sc, fruit):.0f}%" for sc in ("default", "count")))
        if vr:
            s += ('<div class="card"><h3>실행 설정만 바꿔 비교 (재학습 없음)</h3>' + table(["학습", "실행 설정", "기본", "같은 과일 2~3개"], vr)
                  + '<p class="sub">n_action_steps 를 줄이면 동작 묶음을 더 자주 다시 예측한다. 나은 설정은 체크포인트 config.json 에 적어 내면 서버도 같게 실행된다.</p></div>')
    # 2) 학습별 설정과 loss
    for n in runs:
        T = TR[n]["train"]
        m = T["meta"]
        s += (f'<details {"open" if n == runs[-1] else ""}><summary><b>{esc(RUN_KO.get(n, n))}</b> 설정과 loss</summary>'
              '<div class="card"><table class="kv">' + "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in m.items()) + "</table></div>"
              f'<div class="card"><h3>학습 loss ({esc(n)})</h3>{line_chart(T["xs"], T["ys"], "학습 loss", xmax=T["total"])}<p class="sub">{T["xs"][-1]:,} / {T["total"] or T["xs"][-1]:,} 스텝{" (학습 중)" if T["total"] and T["xs"][-1] < T["total"] - 200 else ""}</p></div></details>')
    # 3) 롤아웃 GIF: 최신 학습 먼저
    for n in reversed(ev):
        g = policy_gallery(n, fruit, scenes=None if fruit is not None else ("default", "count", "count4"))
        if g:
            s += g if n == ev[-1] else f"<details><summary>{esc(RUN_KO.get(n, n))} 롤아웃 보기</summary>{g}</details>"
    return s


# ---------------------------------------------------------------- 과일 페이지
plate_ref = (2.41531, -0.33843)
summary_rows = []
for fruit in FRUITS:
    st = {k: stats(k, fruit) for k, *_ in RUNS}
    body = [f"<h1>{KO[fruit]} ({fruit}) 시연 자동 생성 보고서</h1><p class='sub'>로봇제어인공지능 텀프로젝트 · SO-101 과일 집기 · 생성 {esc(__import__('time').strftime('%Y-%m-%d %H:%M'))}</p>", NOTICE]
    # 타일
    tiles = []
    for k, name, *_ in RUNS:
        s = st[k]
        if s:
            tiles.append(f'<div class="tile"><div class="l">{name} 성공률</div><div class="v">{s["rate"]:.0f}%</div><div class="d">{s["ok"]}/{s["n"]} 시도</div></div>')
    tot_ep = sum(st[k]["ok"] for k in DATA if st[k])
    tot_fr = sum(st[k]["frames"] for k in DATA if st[k])
    tiles.append(f'<div class="tile"><div class="l">저장 에피소드 (v1+v2+v3)</div><div class="v">{tot_ep}</div><div class="d">{tot_fr:,} 프레임</div></div>')
    body.append(f'<div class="tiles">{"".join(tiles)}</div>')
    # GIF
    body.append("<h2>시연 영상 (정면 · 손목 카메라, 실제 속도)</h2><div class='grid2'>")
    picks = (("default", "기본 장면", ("v1", "v2"), lambda L: L["kind"] == "default"),
             ("vary", "변형 장면", ("v1", "v2"), lambda L: L["kind"] == "vary"),
             ("v3count3", "v3 · 같은 과일 3개 (무작위 배치)", ("v3",), lambda L: L["count"] == 3 and not L["extra"]),
             ("v3dart", "v3 · 흔들림 넣고 바로잡기(DART) + 조명 변화", ("v3",), None))
    for kind, label, srcs, cond in picks:
        rec = None
        for k in srcs:
            for r in (st[k]["okr"] if st[k] else []):
                ok_ = (r.get("dart") and r["layout"].get("light")) if cond is None else cond(r["layout"])
                if ok_ and r.get("episode_index") is not None:
                    rec = (k, r)
                    break
            if rec:
                break
        if rec:
            k, r = rec
            root = dict((x[0], x[3]) for x in RUNS)[k]
            fn = f"{EN[fruit]}_{kind}.gif"
            if make_gif(root, r["episode_index"], os.path.join(ASSETS, fn)):
                L = r["layout"]
                desc = "기준 자리 ±3cm" if kind == "default" else (f"같은 과일 {L['count']}개 · 다른 과일 {L['extra'] or '없음'} · 로봇 {L['robot']}"
                                                                 + (f" · 조명 {L['light']}" if L.get("light") else "") + (" · DART" if r.get("dart") else ""))
                body.append(f'<div class="card"><h3>{label}</h3><img class="gif" src="assets/{fn}" alt="{KO[fruit]} {label} 시연">'
                            f'<p class="sub">{k} 에피소드 {r["episode_index"]} · {r["sec"]}초 · {esc(desc)}</p></div>')
    body.append("</div>")
    # 생성 방법
    gm = GEOM[fruit]
    body.append("<h2>데이터셋 생성 방법</h2><div class='card'><table class='kv'>"
                + f"<tr><td>과일 메시</td><td>{esc(gm[0])}</td></tr><tr><td>잡는 방법</td><td>{esc(gm[1])}</td></tr><tr><td>그리퍼 명령</td><td>{esc(gm[2])} (LeRobot 단위, 0 닫힘 ~ 100 열림)</td></tr>"
                + "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in COMMON_METHOD) + "</table></div>")
    # 결과
    body.append("<h2>성공 확률</h2><div class='grid2'><div class='card'><h3>단계별 생성 성공률</h3>")
    labels = [n for k, n, *_ in RUNS]
    vals = [st[k]["rate"] if st[k] else None for k, *_ in RUNS]
    tips = [f"{n}: {st[k]['ok']}/{st[k]['n']} = {st[k]['rate']:.0f}%" if st[k] else f"{n}: 아직 없음" for k, n, *_ in RUNS]
    body.append(bar_chart(labels, vals, f"{KO[fruit]} 단계별 성공률", tips=tips))
    rate_table = (table(["단계", "시도", "성공", "성공률", "프레임", "평균 길이", "기본/변형 장면", "실패 유형"],
                      [(n, st[k]["n"], st[k]["ok"], f"{st[k]['rate']:.0f}%", f"{st[k]['frames']:,}", f"{st[k]['sec']:.1f}초",
                        f"{st[k]['default']}/{st[k]['vary']}", ", ".join(f"{a_} {b_}" for a_, b_ in st[k]["fails"].items()) or "없음")
                       for k, n, *_ in RUNS if st[k]]))
    body.append("</div><div class='card'><h3>과일 처음 위치와 결과 (모든 단계)</h3>")
    pts = []
    for k, n, *_ in RUNS:
        for r in (st[k]["rs"] if st[k] else []):
            pts.append((r["fruit_xy_init"][0], r["fruit_xy_init"][1], r["success"],
                        f"{n} 시도 {r['attempt']}: {'성공' if r['success'] else '실패(' + str(r['fail']) + ')'} · {r['layout']['kind']} · 로봇 {r['layout']['robot']}"))
    body.append(scatter_xy(pts, plate_ref, f"{KO[fruit]} 처음 위치"))
    body.append("<p class='sub'>로봇을 옮기거나 돌린 시도도 장면 좌표로 그렸다. 가운데 작은 덩어리가 기본 장면(기준 자리 ±3cm)이다.</p></div></div>")
    body.append(f"<div class='card'><h3>단계별 결과 표</h3>{rate_table}</div>")
    lens = [r["sec"] for k in DATA if st[k] for r in st[k]["okr"]]
    if lens:
        body.append(f"<div class='card'><h3>저장 에피소드 길이 분포 (v1+v2+v3, {len(lens)}개)</h3>{hist(lens, '길이', bins=np.arange(7.5, 12.01, 0.5))}"
                    f"<p class='sub'>평균 {np.mean(lens):.1f}초, 최대 {max(lens):.1f}초 (평가 제한 30초). 사람 시연(팀원 데이터)은 중앙값 6~8초로 더 빠르다.</p></div>")
    # 과정
    body.append("<h2>과정</h2><div class='card'><ol>")
    for k, n, *_ in RUNS:
        s = st[k]
        if s:
            body.append(f"<li><b>{n}</b>: {s['ok']}/{s['n']} 성공" + (f" (실패 {', '.join(f'{a_} {b_}' for a_, b_ in s['fails'].items())})" if s["fails"] else "") + "</li>")
    body.append("</ol>")
    fails = [(k, r) for k in DATA if st[k] for r in st[k]["rs"] if not r["success"]]
    if fails:
        body.append("<details><summary>실패 시도 자세히</summary>" + table(
            ["단계/시도", "유형", "장면", "로봇", "내려갈 때 밀림(mm)", "들어 올린 높이(cm)"],
            [(f"{k} #{r['attempt']}", r["fail"],
              f"{r['layout']['kind']} count={r['layout']['count']} extra={r['layout']['extra']}", r["layout"]["robot"],
              r.get("fruit_pushed_mm"), round(r["lift_dz"] * 100, 1)) for k, r in fails]) + "</details>")
    body.append("</div>")
    # 팀원 데이터 (레몬·딸기)
    if fruit in ("Lemon", "Strawberry"):
        rep = os.path.join(HERE, "team_check", fruit.lower(), "report.json")
        if os.path.exists(rep):
            t = json.load(open(rep))
            body.append("<h2>팀원 사람 시연 데이터 (참고)</h2><div class='card'>" + table(
                ["항목", "값"], [("경로", t["dataset"]), ("에피소드", t["episodes"]), ("작업 문구", t["tasks"][0]),
                               ("길이", f"{t['length_s']['min']}~{t['length_s']['max']}초 (중앙 {t['length_s']['median']}초)"),
                               ("형식 차이 (기준 대비)", "없음" if not t["feature_diffs"] else t["feature_diffs"]),
                               ("불량", "ep55 (시작부터 딸기가 접시 위, 1.8초 정지) → 제외 복사본 ~/datasets/team_strawberry_clean" if fruit == "Strawberry" else "없음 (60개 모두 끝 프레임에서 접시 위)")])
                + "<p class='sub'>장면 구성: 0~19번 과일 1개, 20~39번 같은 과일 2개, 40~59번 3개. 다른 종류 방해 과일·로봇 이동은 없다.</p></div>")
    body.append("<h2>학습 과정과 결과</h2>" + train_section(fruit))
    with open(os.path.join(OUT, f"{EN[fruit]}.html"), "w") as f:
        f.write(inline_assets(page(f"{KO[fruit]} 생성 보고서", "".join(body), fruit)))
    summary_rows.append((fruit, st))

# ---------------------------------------------------------------- 요약 페이지
body = ["<h1>과일 집기 시연 자동 생성 · 전체 요약</h1><p class='sub'>사람 녹화 없이 수업 키트 환경에서 record.py 와 같은 형식의 데이터셋을 만들고, 그 데이터로 학습한 정책을 평가한다.</p>", NOTICE]
tot = {k: [0, 0] for k, *_ in RUNS}
for fruit, st in summary_rows:
    for k, *_ in RUNS:
        if st[k]:
            tot[k][0] += st[k]["n"]
            tot[k][1] += st[k]["ok"]
tiles = "".join(f'<div class="tile"><div class="l">{n} 성공률</div><div class="v">{100 * tot[k][1] / tot[k][0]:.0f}%</div><div class="d">{tot[k][1]}/{tot[k][0]} 시도</div></div>'
                for k, n, *_ in RUNS if tot[k][0])
body.append(f'<div class="tiles">{tiles}</div>')
series = [(n, v) for (k, n, *_), v in zip(RUNS, ("--s1", "--s2", "--s3", "--s4"))]
vals = [[st[k]["rate"] if st[k] else None for _, st in summary_rows] for k, *_ in RUNS]
body.append("<h2>과일별 생성 성공률</h2><div class='card'>" + grouped_bar([KO[f] for f, _ in summary_rows], series, vals, "과일별 성공률") + table(
    ["과일"] + [n for k, n, *_ in RUNS], [(KO[f],) + tuple(f"{st[k]['ok']}/{st[k]['n']} ({st[k]['rate']:.0f}%)" if st[k] else "진행 중" for k, *_ in RUNS)
                                         for f, st in summary_rows]) + "</div>")
body.append("<h2>과일별 보고서</h2><div class='grid2'>" + "".join(
    f'<div class="card"><h3><a href="{EN[f]}.html">{KO[f]}</a></h3><img class="gif" src="assets/{EN[f]}_default.gif" alt="{KO[f]} 시연"></div>' for f in FRUITS) + "</div>")
body.append("<h2>생성 방법 (공통)</h2><div class='card'><table class='kv'>" + "".join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in COMMON_METHOD) + "</table></div>")
body.append("<h2>학습 과정과 결과</h2>" + train_section(None))
with open(os.path.join(OUT, "index.html"), "w") as f:
    f.write(inline_assets(page("과일 집기 시연 생성 요약", "".join(body), "index")))
print("보고서:", os.path.join(OUT, "index.html"))
