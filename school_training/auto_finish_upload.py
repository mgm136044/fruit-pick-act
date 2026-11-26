#!/usr/bin/env python3
"""학교 GPU 세션용: 학습이 끝나면 사람이나 에이전트가 없어도 아래를 차례로 한다.
  저장 지점 → 제출 zip(config 그대로 / te) → 형식 검사(서버 실행 방식과 같게) → sha256 → HF 비공개 저장소 업로드 → 서버 값 대조

학습 폴더(train.sh 가 있는 곳)에서 학습용 python 으로 띄운다(lerobot 0.6.1 이 깔린 것, 기본 ~/robot_train_env/bin/python):

  nohup ~/robot_train_env/bin/python auto_finish_upload.py --name act_mix3A_100k --repo mgm136044/act_mix3 \
      --zips 100000:plain:submission_act_mix3A_100k,100000:te:submission_act_mix3A_100k_te,80000:te:submission_act_mix3A_80k_te,60000:te:submission_act_mix3A_60k_te \
      --note "데이터 fruit_mix3_A (551ep, 스크립트 자동 시연, 사람 시연 아님)" > auto_finish_act_mix3A_100k.log 2>&1 &

- 시작하자마자 HF 로그인·저장소(비공개)를 확인한다. 안 되면 바로 끝낸다(종료 코드 2) → 로그를 보고 사람이 로그인을 고친다.
- 학습 프로세스(`--output_dir=<폴더>/outputs/<이름>`)가 끝나고 가장 큰 스텝의 저장 지점(training_state 포함)이 생기면 진행한다.
  학습 프로세스가 없는데 저장 지점도 없으면 5분 뒤 끝낸다(종료 코드 3, 학습이 멈춘 것).
- README.md 는 저장소에 이미 있으면 받아서 이 학습의 절을 덧붙인다(다른 학습의 내용을 지우지 않는다).
- te = 사본 config.json 의 n_action_steps=1, temporal_ensemble_coeff=0.01 두 값만 바꾼 것. 원본 저장 지점은 건드리지 않는다.
- 토큰은 어디에도 쓰지 않는다(HF_TOKEN 환경 변수나 huggingface-cli login 캐시를 huggingface_hub 가 알아서 읽음).
- --dry-run: 업로드만 빼고 전부 한다(README 는 받아서 합친 결과를 로컬에만 쓴다).
마지막 줄이 AUTO_UPLOAD_DONE 이면 성공이다.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile

os.environ.pop("HF_HUB_OFFLINE", None)  # train.sh 는 오프라인으로 돌지만 업로드는 온라인이어야 함

REQUIRED = ["config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"]
IMAGE_KEYS = {"observation.images.front": "front", "observation.images.wrist": "wrist"}
STATE_KEY = "observation.state"


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def training_pids(out):
    """이 학습 폴더를 쓰는 학습 프로세스(lerobot_train · train_act.py · train.sh). 이 스크립트 자신은 뺀다."""
    me, pids = os.getpid(), []
    name = os.path.basename(out)
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "auto_finish_upload" in cmd:
            continue
        if f"--output_dir={out}" in cmd or (("train_act.py" in cmd or "train.sh" in cmd) and f"--name {name}" in cmd):
            pids.append(int(d))
    return pids


def ckpt_dir(out, step):
    return os.path.join(out, "checkpoints", f"{step:06d}")


def ckpt_ready(out, step):
    d = ckpt_dir(out, step)
    return os.path.isdir(os.path.join(d, "training_state")) and all(
        os.path.isfile(os.path.join(d, "pretrained_model", f)) for f in REQUIRED)


def make_zip(src, dst_zip, mode):
    """src(pretrained_model) → 임시 사본(te 면 config 두 값만 바꿈) → 파일을 zip 맨 위에."""
    tmp = tempfile.mkdtemp(prefix="afu_")
    try:
        for f in sorted(os.listdir(src)):
            if os.path.isfile(os.path.join(src, f)):
                shutil.copy2(os.path.join(src, f), os.path.join(tmp, f))
        cfgp = os.path.join(tmp, "config.json")
        cfg = json.load(open(cfgp))
        if mode == "te":
            cfg["n_action_steps"], cfg["temporal_ensemble_coeff"] = 1, 0.01
            json.dump(cfg, open(cfgp, "w"), indent=4)
        elif mode != "plain":
            raise ValueError(f"모드는 plain 또는 te: {mode}")
        with zipfile.ZipFile(dst_zip + ".part", "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(os.listdir(tmp)):
                zf.write(os.path.join(tmp, f), f)
        os.replace(dst_zip + ".part", dst_zip)
        return {k: cfg.get(k) for k in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff")}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_zip(zpath, device):
    """서버 실행기(lerobot_policy.py)와 같은 방식으로 불러 act 를 2에피소드 × 120번 부른다."""
    import numpy as np
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    tmp = tempfile.mkdtemp(prefix="afu_chk_")
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            size = sum(i.file_size for i in zf.infolist())
            if any("/" in n for n in names):
                raise RuntimeError(f"zip 안에 폴더가 있음: {names}")
            missing = [f for f in REQUIRED if f not in names]
            if missing:
                raise RuntimeError(f"zip 에 {missing} 없음")
            if size > 3000e6:
                raise RuntimeError(f"압축 해제 {size / 1e6:.0f}MB > 3000MB")
            zf.extractall(tmp)
        cfg = json.load(open(os.path.join(tmp, "config.json")))
        if cfg.get("type") != "act":
            raise RuntimeError(f"type={cfg.get('type')}")
        policy = ACTPolicy.from_pretrained(tmp)
        policy.config.device = device
        policy.to(device).eval()
        pre, post = make_pre_post_processors(policy.config, pretrained_path=tmp,
                                             preprocessor_overrides={"device_processor": {"device": device}})
        keys = {k: v for k, v in IMAGE_KEYS.items() if k in policy.config.input_features}
        rng = np.random.default_rng(0)
        ms = []
        for _ in range(2):
            policy.reset(); pre.reset(); post.reset()
            for _ in range(120):
                obs = {"front": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
                       "wrist": rng.integers(0, 256, (480, 640, 3), dtype=np.uint8),
                       "state": (rng.normal(size=6) * 30).astype(np.float32)}
                t0 = time.perf_counter()
                with torch.no_grad():
                    q = getattr(policy, "_action_queue", None)
                    if q and getattr(policy.config, "temporal_ensemble_coeff", None) is None:
                        a = post(q.popleft())
                    else:
                        batch = {k: torch.from_numpy(obs[v]).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                                 for k, v in keys.items()}
                        if STATE_KEY in policy.config.input_features:
                            batch[STATE_KEY] = torch.from_numpy(obs["state"]).unsqueeze(0)
                        batch["task"] = ["Pick the banana and put it on the plate."]
                        a = post(policy.select_action(pre(batch)))
                ms.append((time.perf_counter() - t0) * 1000)
                a = a.squeeze(0).float().cpu().numpy().reshape(-1)
                if a.shape != (6,) or not np.all(np.isfinite(a)):
                    raise RuntimeError(f"action 이 숫자 6개가 아니거나 NaN: {a}")
        return f"OK (act 240번, 평균 {np.mean(ms):.1f}ms, 최대 {np.max(ms):.0f}ms, {device}, 압축 해제 {size / 1e6:.0f}MB)"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="train.sh --name 과 같은 학습 이름 (outputs/<이름>)")
    ap.add_argument("--repo", required=True, help="올릴 HF 모델 저장소 (비공개로만)")
    ap.add_argument("--zips", required=True, help="스텝:plain|te:zip이름 을 쉼표로")
    ap.add_argument("--note", default="", help="README 에 적을 데이터 설명")
    ap.add_argument("--root", default=os.getcwd(), help="학습 폴더 (기본: 지금 폴더)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-hours", type=float, default=40)
    ap.add_argument("--device", default=None, help="형식 검사 장치 (기본: cuda 있으면 cuda)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = os.path.abspath(a.root)
    out = os.path.join(root, "outputs", a.name)
    specs = []
    for s in a.zips.split(","):
        step, mode, zname = s.strip().split(":")
        specs.append((int(step), mode, zname if zname.endswith(".zip") else zname + ".zip"))
    final = max(s[0] for s in specs)
    log(f"시작: 학습 {out} · 기다릴 저장 지점 {final} · zip {[s[2] for s in specs]} · 저장소 {a.repo}"
        f"{' · DRY-RUN(업로드 안 함)' if a.dry_run else ''}")

    # 1) HF 확인 (먼저: 몇 시간 기다린 뒤 로그인 문제로 실패하지 않게)
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    api = HfApi()
    try:
        who = api.whoami()["name"]
    except Exception as e:  # noqa: BLE001
        log(f"✘ HF 로그인 안 됨 ({type(e).__name__}). 사람이 이 셸에서 HF_TOKEN 을 넣거나 huggingface-cli login 후 다시 띄울 것")
        sys.exit(2)
    try:
        info = api.repo_info(a.repo)
        if not info.private:
            log(f"✘ {a.repo} 가 공개 저장소다. 올리지 않는다")
            sys.exit(2)
        log(f"HF 확인: 사용자 {who}, {a.repo} 있음(비공개)")
    except Exception:  # noqa: BLE001
        if a.dry_run:
            log(f"HF 확인: 사용자 {who}, {a.repo} 없음 → 실제 실행이면 비공개로 만든다")
        else:
            api.create_repo(a.repo, private=True, exist_ok=True)
            if not api.repo_info(a.repo).private:
                log(f"✘ {a.repo} 가 비공개가 아니다. 올리지 않는다")
                sys.exit(2)
            log(f"HF 확인: 사용자 {who}, {a.repo} 비공개로 새로 만듦")

    # 2) 학습 끝 기다리기
    t_start, misses, last_note = time.time(), 0, 0
    pids = training_pids(out)
    log(f"학습 프로세스 감지: {pids if pids else '없음'} · 최종 저장 지점 {'있음' if ckpt_ready(out, final) else '아직 없음'}")
    while True:
        pids, ready = training_pids(out), ckpt_ready(out, final)
        if not pids and ready:
            break
        if not pids and not ready:
            misses += 1
            if misses >= 5:
                log(f"✘ 학습 프로세스가 없고 {final} 저장 지점도 없다 → 학습이 멈춘 것 같다. "
                    f"train.log 확인 후 같은 train.sh 명령으로 이어 학습하고 이 스크립트를 다시 띄울 것")
                sys.exit(3)
        else:
            misses = 0
        if time.time() - t_start > a.max_hours * 3600:
            log("✘ 최대 대기 시간 초과")
            sys.exit(3)
        if time.time() - last_note > 1800:
            try:
                step = json.load(open(os.path.join(out, "checkpoints", "last", "training_state", "training_step.json")))["step"]
            except (OSError, ValueError, KeyError):
                step = "?"
            log(f"기다리는 중: 학습 pid {pids}, 마지막 저장 {step}")
            last_note = time.time()
        time.sleep(a.poll)
    log("학습 끝 확인 → 60초 뒤 zip 만들기")
    time.sleep(60 if not a.dry_run else 0)

    # 3) zip · 형식 검사 · sha256
    import torch
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows, files = [], []
    for step, mode, zname in specs:
        src = os.path.join(ckpt_dir(out, step), "pretrained_model")
        dst = os.path.join(root, zname)
        if not os.path.isdir(src):
            log(f"✘ {zname}: 저장 지점 {step} 없음 → 건너뜀")
            continue
        if os.path.exists(dst):
            log(f"{zname}: 이미 있음 → 다시 만들지 않고 검사만")
            c = json.load(zipfile.ZipFile(dst).open("config.json"))
            c = {k: c.get(k) for k in ("chunk_size", "n_action_steps", "temporal_ensemble_coeff")}
        else:
            c = make_zip(src, dst, mode)
        try:
            chk = check_zip(dst, device)
        except Exception as e:  # noqa: BLE001
            log(f"✘ {zname}: 형식 검사 실패 → 올리지 않음: {e}")
            continue
        h, sz = sha256(dst), os.path.getsize(dst)
        log(f"{zname}: {sz} bytes · sha256 {h} · {c} · 검사 {chk}")
        rows.append((zname, step, mode, c, sz, h, chk))
        files.append(dst)
    if not files:
        log("✘ 올릴 zip 이 없다")
        sys.exit(4)

    # 4) README (있으면 받아서 덧붙임)
    try:
        tc = json.load(open(os.path.join(ckpt_dir(out, final), "pretrained_model", "train_config.json")))
        it = (tc.get("dataset") or {}).get("image_transforms") or {}
        setting = (f"배치 {tc.get('batch_size')}, seed {tc.get('seed')}, 스텝 {tc.get('steps')}, "
                   f"chunk_size {(tc.get('policy') or {}).get('chunk_size')}, "
                   f"이미지 증강 {'켬' if it.get('enable') else '끔'} {sorted((it.get('tfs') or {}).keys())}, "
                   f"lr {(tc.get('optimizer') or {}).get('lr')}")
    except (OSError, ValueError) as e:
        setting = f"(train_config.json 읽기 실패: {e})"
    try:
        old = open(hf_hub_download(a.repo, "README.md", force_download=True), encoding="utf-8").read()
    except Exception:  # noqa: BLE001
        old = f"# {a.repo.split('/')[-1]} (비공개 · 스크립트 자동 시연으로 학습, 사람 시연 아님)\n\n평가 서버 제출 후보 zip 모음. te = config 사본의 n_action_steps=1, temporal_ensemble_coeff=0.01.\n"
    sec = [f"\n## {a.name} (자동 업로드 {time.strftime('%F %H:%M')})\n", f"- 데이터: {a.note}" if a.note else "",
           f"- 학습 설정: {setting}", "", "| zip | 저장 지점 | 실행 설정 | 크기(bytes) | sha256 | 형식 검사 |", "|---|---|---|---|---|---|"]
    for zname, step, mode, c, sz, h, chk in rows:
        sec.append(f"| `{zname}` | {step} | {mode} (chunk {c['chunk_size']}, n {c['n_action_steps']}, te {c['temporal_ensemble_coeff']}) | {sz} | `{h}` | {chk.split(' (')[0]} |")
    readme = os.path.join(root, f"README_{a.name}_upload.md")
    open(readme, "w", encoding="utf-8").write(old.rstrip("\n") + "\n" + "\n".join(x for x in sec if x is not None) + "\n")
    log(f"README 작성: {readme}")

    # 5) 업로드 + 서버 값 대조
    if a.dry_run:
        log("DRY-RUN: 업로드하지 않음")
        print("AUTO_UPLOAD_DONE (dry-run)")
        return
    ops = [CommitOperationAdd(path_in_repo=os.path.basename(f), path_or_fileobj=f) for f in files]
    ops.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme))
    for attempt in range(3):
        try:
            api.create_commit(a.repo, operations=ops, commit_message=f"{a.name}: zip {len(files)}개 자동 업로드")
            break
        except Exception as e:  # noqa: BLE001
            log(f"업로드 실패 {attempt + 1}/3: {type(e).__name__}: {str(e)[:200]}")
            if attempt == 2:
                sys.exit(5)
            time.sleep(120)
    info = api.repo_info(a.repo, files_metadata=True)
    remote = {s.rfilename: (s.size, getattr(s.lfs, "sha256", None) if s.lfs else None) for s in info.siblings}
    bad = [r[0] for r in rows if remote.get(r[0]) != (r[4], r[5])]
    if bad or not info.private:
        log(f"✘ 서버 값이 다름 {bad} 또는 비공개 아님(private={info.private})")
        sys.exit(6)
    log(f"업로드 확인: {a.repo} (비공개) · zip {len(rows)}개 크기·sha256 일치")
    print("AUTO_UPLOAD_DONE")


if __name__ == "__main__":
    main()
