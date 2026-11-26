#!/usr/bin/env bash
# 학습 서버에서 ACT 학습: 이 폴더(데이터셋 zip + train.sh + train_act.py)에서
#
#   bash train.sh                         # 기본 (2만 스텝, 배치 8)
#   bash train.sh --steps 30000 --augment # 옵션은 train_act.py 로 그대로 (bash train.sh --help)
#   CUDA_VISIBLE_DEVICES=1 bash train.sh  # GPU 고르기 (여러 장일 때)
#
# 처음 한 번은 학습 환경(Python + LeRobot 0.6.1 + PyTorch)을 만든다 (5~10분, 약 6GB). 위치: $TRAIN_ENV (기본 ~/robot_train_env)
# 이미 lerobot 0.6.1 이 깔린 Python 이 있으면 TRAIN_PYTHON=/경로/python 으로 지정.
# 로그는 train.log 에도 남는다. 터미널을 닫아도 계속하려면:  nohup bash train.sh > /dev/null 2>&1 &   (진행: tail -f train.log)
set -euo pipefail
cd "$(dirname "$0")"
LEROBOT_VERSION=0.6.1
ENV="${TRAIN_ENV:-$HOME/robot_train_env}"

# uv 설치: curl → wget → python3 순서로 있는 것을 씀 (새로 깐 Ubuntu 에는 curl 이 없음)
install_uv() {
  if command -v curl >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v python3 >/dev/null; then
    echo "curl/wget 이 없어 python3 로 uv 를 받습니다"
    python3 - <<'PYEOF'
import io, os, platform, tarfile, urllib.request
arch = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[platform.machine().lower()]
url = f"https://github.com/astral-sh/uv/releases/latest/download/uv-{arch}-unknown-linux-gnu.tar.gz"
data = urllib.request.urlopen(url, timeout=120).read()
dst = os.path.expanduser("~/.local/bin"); os.makedirs(dst, exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data)) as tf:
    for m in tf.getmembers():
        name = os.path.basename(m.name)
        if m.isfile() and name in ("uv", "uvx"):
            with open(os.path.join(dst, name), "wb") as f:
                f.write(tf.extractfile(m).read())
            os.chmod(os.path.join(dst, name), 0o755)
PYEOF
  else
    echo "실패: curl, wget, python3 가 모두 없습니다 → sudo apt install curl 후 다시 실행하세요" >&2; exit 1
  fi
  command -v uv >/dev/null || { echo "실패: uv 설치가 안 됐습니다 (인터넷 연결 확인, 또는 sudo apt install curl 후 다시 실행)" >&2; exit 1; }
}
has_lerobot() { "$1" -c "import importlib.metadata as m, sys; sys.exit(m.version('lerobot') != '$LEROBOT_VERSION')" 2>/dev/null; }

if [ -n "${TRAIN_PYTHON:-}" ]; then
  PY="$TRAIN_PYTHON"
  has_lerobot "$PY" || { echo "✘ $PY 에 lerobot $LEROBOT_VERSION 이 없습니다"; exit 1; }
elif [ -x "$ENV/bin/python" ] && has_lerobot "$ENV/bin/python"; then
  PY="$ENV/bin/python"
else
  echo "학습 환경 만들기: $ENV (처음 한 번, 5~10분)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv || echo "! nvidia-smi 가 없습니다 (GPU 확인 필요)"
  if ! command -v uv >/dev/null; then
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null || install_uv
  fi
  [ -x "$ENV/bin/python" ] || uv venv --python 3.12 "$ENV"
  # 영상은 pyav 로 읽는다 (torchcodec 은 서버마다 FFmpeg/CUDA 문제를 일으켜 뺌). torch 는 드라이버에 맞춰 자동 선택
  uv pip install --python "$ENV/bin/python" "lerobot[dataset,training]==$LEROBOT_VERSION" --torch-backend auto
  uv pip uninstall --python "$ENV/bin/python" torchcodec >/dev/null 2>&1 || true
  PY="$ENV/bin/python"
fi

export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1   # 로컬 데이터셋만 씀 (Hugging Face 접속 안 함)
echo "===== $(date '+%F %T') 학습 시작 · $PY" >> train.log
"$PY" train_act.py "$@" 2>&1 | tee -a train.log
exit "${PIPESTATUS[0]}"
