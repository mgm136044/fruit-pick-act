#!/usr/bin/env python3
"""밤새 로컬 평가 대기열: 학교에서 HF 비공개 저장소에 올라오는 제출 후보 zip 을 받아, 기준 모델과 같은 56장면으로 평가하고 짝 비교한다.

  source ~/robot_course/activate.sh && nohup python -I eval_queue.py --until "2026-11-27 12:00" > logs/eval_queue.out 2>&1 &

- 후보는 CANDS 순서(우선순위)대로, 올라와 있는 것부터 하나씩 한다. 다 하면 10분마다 다시 확인한다.
- 받은 zip: 저장소 README 의 sha256 과 대조(README 에 있으면) → 파일이 zip 맨 위에만 있는지 확인 → te_models/<이름>/ 에 풀기.
- 평가: P0-c(기준 te0.01 60.0)와 같은 명령 — eval_proxy 6장면×8(점수는 distract 뺀 40) + 같은 과일 4개 16(assets_c4).
  zip 의 config 를 그대로 쓴다(te 판이면 te).
- GPU 가 다른 일(스윕·생성·과제5 등)에 쓰이고 있으면 끝날 때까지 기다린다(Isaac 인스턴스는 하나만).
- 결과: logs/eval_queue.jsonl 한 줄씩 + progress.log 요약(기준과 짝 비교: 이김·짐·부호 검정 p).
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
BASE = ("logs/eval_proxy_act_mix2_100k_color_te01.json", "logs/eval_proxy_act_mix2_100k_color_c4te01.json")  # 서버 54.2 모델(te0.01)
CANDS = [  # (이름, 저장소, zip) — 위에서부터 우선
    ("act_mix2_200k_color_te", "mgm136044/act_mix2_200k_color", "submission_act_mix2_200k_color_te.zip"),
    ("act_mix3A_100k_te", "mgm136044/act_mix3", "submission_act_mix3A_100k_te.zip"),
    ("act_mix3A_R1a_170k_te", "mgm136044/act_mix3", "submission_act_mix3A_R1a_170k_te.zip"),
    ("act_mix2_c50_100k_te", "mgm136044/act_mix2_c50", "submission_act_mix2_c50_100k_te.zip"),
    ("act_mix3A_c50_100k_te", "mgm136044/act_mix3", "submission_act_mix3A_c50_100k_te.zip"),
    ("act_mix3A_c50_R1a_170k_te", "mgm136044/act_mix3", "submission_act_mix3A_c50_R1a_170k_te.zip"),
    ("act_mix2_150k_color_te", "mgm136044/act_mix2_200k_color", "submission_act_mix2_150k_color_te.zip"),
]
GPU_JOBS = re.compile(r"eval_sweep\.py|eval_proxy\.py|dagger_collect\.py|gen_scripted\.py|run_policy\.py")  # + nvidia-smi 계산 프로세스


def log(msg, progress=False):
    line = f"[{time.strftime('%F %H:%M')}] {msg}"
    print(line, flush=True)
    if progress:
        open("progress.log", "a", encoding="utf-8").write(line + "\n")


def gpu_busy():
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=30).stdout
    except Exception:  # noqa: BLE001
        q = ""
    apps = [l for l in q.splitlines() if l.strip() and "gnome-remote-desktop" not in l]
    me = str(os.getpid())
    ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    jobs = [l for l in ps.splitlines() if GPU_JOBS.search(l) and "eval_queue" not in l and not l.strip().startswith(me + " ")]
    return apps or jobs


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(api, name, repo, zname):
    from huggingface_hub import hf_hub_download
    dl_dir = os.path.join("downloads", name)
    os.makedirs(dl_dir, exist_ok=True)
    zp = hf_hub_download(repo, zname, local_dir=dl_dir)
    h = sha256(zp)
    note = "README 에 sha 없음"
    try:
        readme = open(hf_hub_download(repo, "README.md", local_dir=dl_dir), encoding="utf-8").read()
        for line in readme.splitlines():
            if zname in line:
                m = re.search(r"[0-9a-f]{64}", line)
                if m:
                    if m.group(0) != h:
                        raise RuntimeError(f"sha256 불일치: README {m.group(0)[:16]}… / 받은 것 {h[:16]}…")
                    note = "README sha 일치"
                    break
    except RuntimeError:
        raise
    except Exception:  # noqa: BLE001
        pass
    dst = os.path.join("te_models", name)
    mark = os.path.join(dst, ".zip_sha256")
    if os.path.exists(dst):
        if os.path.exists(mark) and open(mark).read().strip() == h:
            cfg = json.load(open(os.path.join(dst, "config.json")))
            return h, note + ", 이미 풀어 둠", {k: cfg.get(k) for k in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff")}
        raise RuntimeError(f"{dst} 가 이미 있음(다른 zip, 지우지 않음)")
    with zipfile.ZipFile(zp) as zf:
        names = zf.namelist()
        if any("/" in n or n.startswith(".") for n in names):
            raise RuntimeError(f"zip 안에 폴더·숨김 파일: {names}")
        for need in ("config.json", "model.safetensors"):
            if need not in names:
                raise RuntimeError(f"zip 에 {need} 없음")
        os.makedirs(dst)
        zf.extractall(dst)
    open(mark, "w").write(h)
    cfg = json.load(open(os.path.join(dst, "config.json")))
    return h, note, {k: cfg.get(k) for k in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff")}


def run_eval(name):
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")
    ck = os.path.join("te_models", name)
    o1, o2 = f"logs/eval_proxy_{name}_q.json", f"logs/eval_proxy_{name}_qc4.json"
    with open(f"logs/eval_proxy_{name}_q.log", "w") as f:
        subprocess.run([sys.executable, "eval_proxy.py", "--checkpoint", ck, "--per-scene", "8",
                        "--gif-dir", f"report_html/assets/policy_{name}_q", "--gif-per-scene", "2",
                        "--out", o1, "--viz", "none"], stdout=f, stderr=subprocess.STDOUT, env=env)
    with open(f"logs/eval_proxy_{name}_qc4.log", "w") as f:
        subprocess.run([sys.executable, "eval_proxy.py", "--checkpoint", ck, "--scenes", "count4", "--per-scene", "16",
                        "--gif-dir", f"report_html/assets/policy_{name}_qc4", "--gif-per-scene", "2",
                        "--out", o2, "--viz", "none"], stdout=f, stderr=subprocess.STDOUT,
                       env=dict(env, COURSE_ASSETS=os.path.join(HERE, "assets_c4")))
    return o1, o2


def scored(paths):
    eps = []
    for p in paths:
        eps += [e for e in json.load(open(p))["episodes"] if e["scene"] != "distract"]
    return eps


def compare(o1, o2):
    a, b = scored([o1, o2]), scored(BASE)
    if len(a) != len(b) or any((x["scene"], x["fruit"]) != (y["scene"], y["fruit"]) for x, y in zip(a, b)):
        return {"error": f"장면 순서가 기준과 다름 ({len(a)} vs {len(b)})"}
    win = sum(x["success"] and not y["success"] for x, y in zip(a, b))
    lose = sum(y["success"] and not x["success"] for x, y in zip(a, b))
    n = win + lose
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(win, lose) + 1)) / 2 ** n) if n else 1.0
    by = {}
    for x in a:
        s = by.setdefault(x["scene"], [0, 0]); s[0] += bool(x["success"]); s[1] += 1
    why = {}
    for x in a:
        why[x.get("why")] = why.get(x.get("why"), 0) + 1
    return {"succ": sum(bool(x["success"]) for x in a), "n": len(a), "base_succ": sum(bool(y["success"]) for y in b),
            "win": win, "lose": lose, "sign_p": round(p, 3), "by_scene": by, "why": why,
            "late_succ_gt300": sum(bool(x["success"]) and x["steps"] > 300 for x in a),
            "score40": json.load(open(o1))["score_mean_of_scenes_x100"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", default="2026-11-27 12:00")
    ap.add_argument("--wait-sweep", default="logs/run_sweep_after_R1a.log", help="이 로그에 SWEEP_REST_DONE 이 나오거나 스윕 연결이 끝날 때까지 기다림")
    a = ap.parse_args()
    until = dt.datetime.strptime(a.until, "%Y-%m-%d %H:%M")
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import HfApi
    api = HfApi()
    done = set()
    if os.path.exists("logs/eval_queue.jsonl"):
        done = {json.loads(l)["name"] for l in open("logs/eval_queue.jsonl") if l.strip()}
    log(f"평가 대기열 시작 · {a.until} 까지 · 이미 한 것 {sorted(done)}", progress=True)
    while subprocess.run(["pgrep", "-f", "^bash run_sweep_after_R1a.sh"], capture_output=True).returncode == 0:
        time.sleep(60)
    last_idle = 0
    while dt.datetime.now() < until:
        todo = [c for c in CANDS if c[0] not in done]
        if not todo:
            log("후보 모두 평가함 → 끝", progress=True)
            break
        picked = None
        for name, repo, zname in todo:
            try:
                if zname in api.list_repo_files(repo):
                    picked = (name, repo, zname)
                    break
            except Exception as e:  # noqa: BLE001
                log(f"{repo} 확인 실패: {type(e).__name__}")
        if not picked:
            if time.time() - last_idle > 3600:
                log(f"올라온 후보 없음 · 남은 {[c[0] for c in todo]}")
                last_idle = time.time()
            time.sleep(600)
            continue
        name, repo, zname = picked
        t_wait = time.time()
        while (busy := gpu_busy()):
            if time.time() - t_wait > 1800:
                log(f"GPU 사용 중이라 기다림(30분+): {busy[:3]}")
                t_wait = time.time()
            time.sleep(60)
        t0 = time.time()
        try:
            h, note, cfg = fetch(api, name, repo, zname)
        except Exception as e:  # noqa: BLE001
            log(f"✘ {name}: 받기 실패 → 건너뜀: {e}", progress=True)
            done.add(name)
            open("logs/eval_queue.jsonl", "a").write(json.dumps({"name": name, "error": str(e)}, ensure_ascii=False) + "\n")
            continue
        log(f"{name}: 받음 {repo}/{zname} sha {h[:16]}… ({note}) config {cfg} → 평가 시작(56)", progress=True)
        o1, o2 = run_eval(name)
        try:
            r = compare(o1, o2)
        except Exception as e:  # noqa: BLE001
            r = {"error": f"결과 읽기 실패: {e}"}
        r.update({"name": name, "repo": repo, "zip": zname, "sha256": h, "config": cfg, "minutes": round((time.time() - t0) / 60)})
        open("logs/eval_queue.jsonl", "a").write(json.dumps(r, ensure_ascii=False) + "\n")
        done.add(name)
        if "error" in r:
            log(f"✘ {name}: {r['error']}", progress=True)
        else:
            log(f"{name}: 56장면 {r['succ']}/{r['n']} (기준 te0.01 {r['base_succ']}/{r['n']}) · 이김 {r['win']} 짐 {r['lose']} p={r['sign_p']} · "
                f"40장면 점수 {r['score40']} · 300스텝 뒤 성공 {r['late_succ_gt300']} · {r['by_scene']}", progress=True)
    print("EVAL_QUEUE_END")


if __name__ == "__main__":
    main()
