#!/usr/bin/env bash
# 로봇 제어 AI 실습 환경 설치 (Ubuntu 22.04 / 24.04, NVIDIA RTX GPU). root 권한 필요 없음.
#   Isaac Sim 6.2 + Isaac Lab 3.0 + LeRobot 0.6.1 (리더 암, 데이터셋, ACT 학습)을 한 환경에 설치
#
#   bash setup/install.sh                       # ~/robot_course 에 설치
#   COURSE_DIR=/큰디스크/robot_course bash setup/install.sh
#
# 다시 실행하면 끝난 단계는 건너뜀.
set -euo pipefail

KIT="${KIT_DIR:-$HOME/student_kit}"   # [수정] 복사본 위치와 무관하게 키트 폴더를 가리키게
DIR="${COURSE_DIR:-$HOME/robot_course}"
ENV="$DIR/env"
PY="$ENV/bin/python"
OVERRIDES="https://raw.githubusercontent.com/isaac-sim/IsaacLab/v3.0.0/tools/wheel_builder/uv-overrides.txt"
MIN_DRIVER="595.58"

step() { echo; echo "=== $* ==="; }
fail() { echo "실패: $*" >&2; exit 1; }
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
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# 컴퓨터에 원래 있던 Python 설정이 섞이지 않게
unset PYTHONPATH PYTHONHOME PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null

step "1/5 컴퓨터 확인"
command -v nvidia-smi >/dev/null || fail "nvidia-smi 가 없습니다. NVIDIA 드라이버를 먼저 설치하세요."
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
echo "GPU: $GPU, 드라이버 $DRV"
version_ge "$DRV" "$MIN_DRIVER" || fail "드라이버 $DRV < $MIN_DRIVER. Isaac Sim 6.2 는 $MIN_DRIVER 이상이 필요합니다 (sudo ubuntu-drivers install 또는 NVIDIA 홈페이지)."
GLIBC=$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}')   # [수정] head 가 파이프를 먼저 닫으면 pipefail 로 141 종료
version_ge "$GLIBC" "2.35" || fail "glibc $GLIBC < 2.35 (Ubuntu 22.04 이상 필요)"
mkdir -p "$DIR"
FREE=$(df -BG --output=avail "$DIR" | tail -1 | tr -dc 0-9)
(( FREE >= 40 )) || fail "$DIR 디스크 여유 ${FREE}GB < 40GB. COURSE_DIR 로 다른 디스크를 지정하세요."
if ! ldconfig -p | grep "libGLU.so.1" >/dev/null; then   # [수정] grep -q 조기 종료로 인한 SIGPIPE 방지
  echo "주의: libGLU 가 없습니다 → 설치 후 실행이 안 되면:  sudo apt install libglu1-mesa"
fi

step "2/5 uv (파이썬 설치 도구)"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || install_uv
uv --version
export UV_CACHE_DIR="$DIR/uv-cache"   # 받는 파일은 설치 폴더 안에 (끝나면 지움)

step "3/5 Isaac Sim 6.2 + Isaac Lab 3.0 (약 20GB 다운로드, 20~40분)"
[ -x "$PY" ] || uv venv --python 3.12 "$ENV"
uv pip install --python "$PY" "isaaclab[isaacsim]==3.0.0" --overrides "$OVERRIDES" \
  --extra-index-url https://pypi.nvidia.com --torch-backend cu130 --index-strategy unsafe-best-match

step "4/5 LeRobot 0.6.1 (리더 암, 데이터셋, ACT 학습)"
uv pip install --python "$PY" "lerobot[feetech,dataset,training]==0.6.1" --torch-backend cu130
# 영상은 pyav 로 읽는다 (--dataset.video_backend=pyav). torchcodec 은 컴퓨터마다 FFmpeg/CUDA 문제를 일으켜 뺌
uv pip uninstall --python "$PY" torchcodec >/dev/null 2>&1 || true

step "5/5 확인"
"$PY" - <<'PYEOF'
import importlib.metadata as m
import torch
x = torch.randn(256, 256, device="cuda"); (x @ x).sum().item()
print("OK: isaacsim", m.version("isaacsim"), "· isaaclab", m.version("isaaclab"), "· lerobot", m.version("lerobot"),
      "· torch", torch.__version__, "·", torch.cuda.get_device_name(0))
PYEOF

cat > "$DIR/activate.sh" <<EOF
# 매번:  source $DIR/activate.sh
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
source "$ENV/bin/activate"
export ROBOT_KIT="$KIT"
EOF
rm -rf "$UV_CACHE_DIR"
echo
echo "설치 완료. 다음 순서:"
echo "  source $DIR/activate.sh"
echo "  python \$ROBOT_KIT/isaac/scripts/check_env.py --out check_out     # 첫 실행은 10분 이상 걸림 (셰이더 준비)"
echo "  (처음 실행할 때 NVIDIA Omniverse 라이선스(EULA) 동의를 물어보면 Yes 를 입력하세요)"
