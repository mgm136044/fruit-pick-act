"""reports_w1/*.md (단계 보고서) → reports_w1/index.html 한 파일 (로컬 보존용, 공개하지 않음).

  source ~/robot_course/activate.sh && python -I ~/fruit_datagen/reports_w1/build_html.py
"""
import glob
import html
import os
import re
import time

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(f for f in glob.glob(os.path.join(HERE, "*.md")))
CSS = """
:root{--bg:#fbfbf9;--fg:#1d1d1b;--mut:#5f5e5a;--line:#e2e0d8;--card:#ffffff;--acc:#2f6fb0;--code:#f2f1ec}
@media (prefers-color-scheme: dark){:root{--bg:#18181a;--fg:#e9e7e1;--mut:#a3a19a;--line:#34343a;--card:#202024;--acc:#7fb0e6;--code:#26262b}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.65 -apple-system,"Noto Sans KR","Apple SD Gothic Neo",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 80px}nav{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;margin-bottom:18px;z-index:2}
nav a{color:var(--acc);text-decoration:none;margin-right:14px;white-space:nowrap;font-size:14px}nav{overflow-x:auto}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 22px 18px;margin:0 0 22px}
h1{font-size:22px;margin:18px 0 8px}h2{font-size:17px;margin:22px 0 8px;border-top:1px solid var(--line);padding-top:14px}h3{font-size:15px}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px;display:block;overflow-x:auto}th,td{border:1px solid var(--line);padding:5px 8px;vertical-align:top;text-align:left}
th{background:var(--code)}code{background:var(--code);padding:1px 4px;border-radius:4px;font-size:12.5px}pre{background:var(--code);padding:10px 12px;border-radius:8px;overflow-x:auto}pre code{padding:0}
blockquote{color:var(--mut);border-left:3px solid var(--line);margin:8px 0;padding:2px 12px}.sub{color:var(--mut);font-size:13px}
"""
def indent4(txt):
    """Python-Markdown 은 하위 목록을 4칸 들여쓰기로만 인식한다. 보고서는 2칸으로 쓰므로 코드 블록 밖의 앞 공백을 2배로."""
    out, fence = [], False
    for ln in txt.split("\n"):
        if ln.lstrip().startswith("```"):
            fence = not fence
        if not fence and ln.startswith(" ") and not ln.lstrip().startswith("```"):
            n = len(ln) - len(ln.lstrip(" "))
            ln = " " * (n * 2) + ln.lstrip(" ")
        out.append(ln)
    return "\n".join(out)


secs, nav = [], []
for i, f in enumerate(files):
    txt = indent4(open(f, encoding="utf-8").read())
    title = re.search(r"^# (.+)$", txt, re.M)
    title = title.group(1) if title else os.path.basename(f)
    body = markdown.markdown(txt, extensions=["tables", "fenced_code", "sane_lists"])
    secs.append(f'<section id="s{i}">{body}</section>')
    nav.append(f'<a href="#s{i}">{html.escape(title.split("·")[0].strip())}</a>')
page = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>1주차 단계 보고서</title><style>{CSS}</style></head><body><main>'
        f'<h1>1주차 마감 단계 보고서 (과일 집기 ACT)</h1><p class="sub">로컬 보존용 · 사람 시연 아님(스크립트 자동 시연) · 생성 {time.strftime("%Y-%m-%d %H:%M")} · '
        f'전략: <code>STRATEGY_W1.md</code> · 진행: <code>ROADMAP.md</code> 2부</p>'
        f'<nav>{"".join(nav)}</nav>{"".join(secs)}</main></body></html>')
out = os.path.join(HERE, "index.html")
open(out, "w", encoding="utf-8").write(page)
print("보고서:", out, f"({len(files)}개 단계)")
