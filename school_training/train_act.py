"""학습 서버에서 ACT 학습 (train.sh 가 부름). 데이터셋 zip 압축 풀기 → lerobot-train → 제출용 zip.

  bash train.sh                              # 이 폴더의 데이터셋 zip 으로 기본 설정 학습
  bash train.sh --steps 30000 --batch-size 16 --augment
  bash train.sh --data fruit_v2.zip --name act_v2

- 이 폴더 안에 모두 만든다: data/<데이터셋>/ (압축 푼 것), outputs/<이름>/ (학습 결과), train.log
- 중간에 끊겨도 같은 명령을 다시 실행하면 마지막 저장 지점(save-freq 스텝마다)부터 이어서 학습
- 끝나면 submission_<이름>.zip (pretrained_model 폴더 안의 파일들) → 내려받아 형식 검사 후 제출
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REQUIRED_FILES = ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")
IMAGE_KEYS = ("observation.images.front", "observation.images.wrist")


def find_zip(arg: str | None) -> str:
    if arg:
        p = os.path.abspath(arg)
        if not os.path.isfile(p):
            sys.exit(f"✘ {arg} 가 없습니다")
        return p
    zips = [f for f in sorted(os.listdir(HERE)) if f.endswith(".zip") and not f.startswith("submission_")]
    if len(zips) != 1:
        sys.exit("✘ 데이터셋 zip 을 정하세요: --data <파일.zip>" + (f" (후보: {', '.join(zips)})" if zips else
                                                                 " (이 폴더에 zip 이 없습니다)"))
    return os.path.join(HERE, zips[0])


def extract(zpath: str) -> str:
    """data/<이름>/ 에 압축 풀기 (같은 zip 이면 건너뜀). 돌려주는 값: 데이터셋 폴더."""
    with zipfile.ZipFile(zpath) as zf:
        names = zf.namelist()
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1 or not any(n.endswith("/meta/info.json") for n in names):
            sys.exit("✘ 데이터셋 zip 형식이 아닙니다 (record.py / pack_dataset.py 로 만든 zip 을 쓰세요)")
        name = tops.pop()
        dst = os.path.join(HERE, "data", name)
        stamp = f"{os.path.getsize(zpath)}:{int(os.path.getmtime(zpath))}"
        mark = os.path.join(dst, ".unpacked")
        if os.path.isfile(mark) and open(mark).read() == stamp:
            print(f"데이터셋: {dst} (이미 압축 풂)")
            return dst
        if os.path.exists(dst):
            shutil.rmtree(dst)
        print(f"압축 푸는 중: {os.path.basename(zpath)} → {dst}", flush=True)
        root = os.path.realpath(os.path.join(HERE, "data"))
        for n in names:  # zip 안의 경로가 data/ 밖을 가리키지 않는지
            if not os.path.realpath(os.path.join(root, n)).startswith(root + os.sep):
                sys.exit(f"✘ 잘못된 경로가 들어 있는 zip: {n}")
        zf.extractall(root)
    with open(mark, "w") as f:
        f.write(stamp)
    return dst


def check_dataset(root: str) -> dict:
    with open(os.path.join(root, "meta", "info.json")) as f:
        info = json.load(f)
    feats = info.get("features", {})
    problems = []
    for k in ("observation.state", "action"):
        if list(feats.get(k, {}).get("shape", [])) != [6]:
            problems.append(f"{k} 크기가 6 이 아님")
    for k in IMAGE_KEYS:
        if k in feats and list(feats[k]["shape"][:2]) != [480, 640]:
            problems.append(f"{k} 크기가 480x640 이 아님")
    if not any(k in feats for k in IMAGE_KEYS):
        problems.append("카메라 영상이 없음")
    if problems:
        sys.exit("✘ 과제 형식과 다른 데이터셋: " + ", ".join(problems))
    print(f"  에피소드 {info['total_episodes']}개, 프레임 {info['total_frames']}개 ({info['total_frames'] / info['fps'] / 60:.1f}분)")
    return info


def usable_cpus() -> int:
    """이 프로세스가 실제로 쓸 수 있는 CPU 수 (서버·컨테이너가 제한한 만큼)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:  # cgroup v2 CPU 제한 (도커 등)
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(1, int(int(quota) / int(period))))
    except (OSError, ValueError):
        pass
    return n


def shm_gb() -> float:
    try:
        st = os.statvfs("/dev/shm")
        return st.f_bavail * st.f_frsize / 1e9
    except OSError:
        return 0.0


def pick_workers(asked: int | None) -> int:
    """데이터 읽기 프로세스 수. 지정 안 하면 CPU 수와 공유 메모리(/dev/shm)를 보고 정함."""
    cpus, shm = usable_cpus(), shm_gb()
    if asked is not None:
        n = asked
    else:
        n = max(0, min(4, cpus - 1))
        if n and shm < 2:
            print(f"  공유 메모리(/dev/shm)가 {shm:.1f}GB 로 작아 데이터 읽기 프로세스를 쓰지 않습니다 (--workers 0)")
            n = 0
    print(f"  CPU {cpus}개 사용 가능, 공유 메모리 {shm:.1f}GB → 데이터 읽기 프로세스 {n}개")
    return n


def explain_exit(code: int) -> str:
    if code in (-9, 137):
        return ("메모리(RAM)가 부족해 강제로 종료된 것 같습니다 → --batch-size 4 --workers 0 으로 다시 해 보세요 "
                "(같은 서버의 다른 작업이 메모리를 많이 쓰는지도 확인: free -g)")
    if code in (-7, 135):
        return "공유 메모리(/dev/shm) 부족으로 보입니다 → --workers 0 으로 다시 해 보세요"
    return "위 오류를 확인하세요 (맨 아래의 'leaked semaphore' 경고는 종료 뒤 정리 메시지라 원인이 아닙니다)"


def last_step(out: str) -> int:
    p = os.path.join(out, "checkpoints", "last", "training_state", "training_step.json")
    try:
        with open(p) as f:
            return int(json.load(f)["step"])
    except (OSError, ValueError, KeyError):
        return 0


def pack_submission(out: str, name: str) -> str:
    model = os.path.join(out, "checkpoints", "last", "pretrained_model")
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(model, f))]
    if missing:
        sys.exit(f"✘ 체크포인트에 {', '.join(missing)} 가 없습니다 ({model})")
    dst = os.path.join(HERE, f"submission_{name}.zip")
    with zipfile.ZipFile(dst + ".part", "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(model)):
            p = os.path.join(model, f)
            if os.path.isfile(p):
                zf.write(p, f)
    os.replace(dst + ".part", dst)
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="데이터셋 zip (기본: 이 폴더의 zip 하나)")
    ap.add_argument("--name", help="학습 이름 (기본: act_<데이터셋>)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=5000, help="이 스텝마다 저장 (끊겼을 때 여기서부터 이어서)")
    ap.add_argument("--augment", action="store_true", help="이미지 증강 (밝기·대비 등을 무작위로 바꿔 학습)")
    ap.add_argument("--workers", type=int, default=None, help="데이터 읽기 프로세스 수 (기본: CPU 수를 보고 자동, 최대 4)")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--restart", action="store_true", help="이어서 하지 않고 처음부터 (기존 outputs/<이름> 삭제)")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="lerobot-train 에 그대로 넘길 옵션 (맨 뒤에, 예: --extra --policy.chunk_size=50)")
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("✘ GPU(CUDA)를 쓸 수 없습니다. nvidia-smi 로 GPU 와 드라이버를 확인하세요.")
    print(f"GPU: {torch.cuda.get_device_name(0)} (보이는 GPU {torch.cuda.device_count()}개, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '전체')})")

    workers = pick_workers(a.workers)
    zpath = find_zip(a.data)
    root = extract(zpath)
    check_dataset(root)
    dname = os.path.basename(root)
    name = a.name or f"act_{dname}"
    out = os.path.join(HERE, "outputs", name)
    if a.restart and os.path.exists(out):
        shutil.rmtree(out)

    done = last_step(out)
    if done >= a.steps:
        print(f"이미 {done} 스텝까지 학습했습니다 (outputs/{name}). 다시 하려면 --restart 또는 --name 을 바꾸세요.")
    else:
        if done:
            print(f"이어서 학습: {done} → {a.steps} 스텝 (outputs/{name})")
            cmd = ["--config_path=" + os.path.join(out, "checkpoints", "last", "pretrained_model", "train_config.json"),
                   "--resume=true", f"--steps={a.steps}", f"--output_dir={out}", f"--dataset.root={root}",
                   f"--num_workers={workers}"]
        else:
            if os.path.exists(out):  # 저장 지점 없이 끊긴 이전 시도
                shutil.rmtree(out)
            print(f"학습 시작: {a.steps} 스텝, 배치 {a.batch_size} → outputs/{name}")
            cmd = ["--policy.type=act", f"--dataset.repo_id=local/{dname}", f"--dataset.root={root}",
                   "--dataset.video_backend=pyav", f"--output_dir={out}", f"--job_name={name}",
                   f"--steps={a.steps}", f"--batch_size={a.batch_size}", f"--save_freq={min(a.save_freq, a.steps)}",
                   f"--num_workers={workers}", f"--seed={a.seed}", "--policy.device=cuda",
                   "--policy.push_to_hub=false", "--wandb.enable=false"]
            if a.augment:
                cmd.append("--dataset.image_transforms.enable=true")
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", *cmd, *a.extra]
        print("$ " + " ".join(cmd), flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"✘ 학습이 중간에 멈췄습니다 (종료 코드 {r.returncode}). {explain_exit(r.returncode)}\n"
                     f"  같은 명령을 다시 실행하면 마지막 저장 지점({last_step(out)} 스텝)부터 이어서 합니다.")
    dst = pack_submission(out, name)
    print(f"\n완료: {dst} ({os.path.getsize(dst) / 1e6:.0f}MB)\n"
          f"  이 파일을 내 컴퓨터로 내려받아 형식 검사(check_submission.py) 후 평가 서버에 제출하세요.")


if __name__ == "__main__":
    main()
