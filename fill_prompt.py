#!/usr/bin/env python3
"""프롬프트 초안의 자리표시({EP} 등)를 실제 값으로 채워 outbox 에 쓴다.

  python3 fill_prompt.py prompts/draft_C_mix3A_chunk100.md prompts/outbox/20261126_..md \
      EP="551 162000" SIZE=123 SHA=abcd...
채우지 못한 자리표시가 남으면 파일을 쓰지 않고 실패한다.
"""
import re
import sys

src, out, *pairs = sys.argv[1:]
text = open(src, encoding="utf-8").read()
for p in pairs:
    k, v = p.split("=", 1)
    text = text.replace("{" + k + "}", v)
left = sorted(set(re.findall(r"\{(EP2?|SIZE2?|SHA2?)\}", text)))
if left:
    sys.exit(f"남은 자리표시: {left}")
open(out, "w", encoding="utf-8").write(text)
print("작성:", out)
