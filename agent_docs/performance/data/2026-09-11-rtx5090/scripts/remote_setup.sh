#!/bin/bash
# Remote setup for the libuipc 5090 benchmark instance (Ubuntu 24.04, CUDA 12.8.1 devel image)
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "== host facts"; nvidia-smi --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current --format=csv; nproc; free -g | head -2; df -h /work / | tail -2; nvcc --version | tail -2; lsb_release -ds; python3 --version
apt-get update -qq >/dev/null 2>&1 || true
apt-get install -y -qq git curl zip unzip tar pkg-config build-essential ninja-build cmake python3 python3-venv python3-pip python3-dev rsync autoconf automake libtool bison flex >/dev/null 2>&1
cmake --version | head -1; g++ --version | head -1; ninja --version
mkdir -p /work && cd /work
# source trees from bundles
for t in head base; do
  if [ ! -d /work/libuipc-$t ]; then
    git clone -q -b bench/$t /work/ship/libuipc.bundle /work/libuipc-$t 2>/dev/null || { git clone -q /work/ship/libuipc.bundle /work/libuipc-$t; git -C /work/libuipc-$t checkout -q refs/bench/$t 2>/dev/null || true; }
  fi
done
git -C /work/libuipc-head checkout -q df57bfb8 && git -C /work/libuipc-base checkout -q e1eed4b9
for t in head base; do
  R=/work/libuipc-$t
  git -C $R clone -q /work/ship/samples.bundle $R/libuipc-samples 2>/dev/null || true
  git -C $R/libuipc-samples checkout -q 4fb26b7
  echo "$t: $(git -C $R rev-parse HEAD) samples $(git -C $R/libuipc-samples rev-parse HEAD)"
done
# venvs with the prebuilt wheels (cp312)
for v in head base; do
  W=$( [ $v = head ] && echo perf2f47bf50 || echo dahle1eed4b9 )
  python3 -m venv /work/venv-wheel-$v
  /work/venv-wheel-$v/bin/pip install -q --upgrade pip
  /work/venv-wheel-$v/bin/pip install -q numpy "/work/ship/pyuipc-0.9.0+$W-py3-none-any.whl"
  /work/venv-wheel-$v/bin/python -c "import uipc; print('$v wheel', uipc.__version__)"
done
echo SETUP_OK
