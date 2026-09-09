#!/bin/bash
# Smoke test: run libuipc-samples example 34_cloth_stack for 20 frames with the from-source pyuipc.
source /workspace/deps/libuipc-src/env-src.sh
cd /workspace/archive/libuipc-samples/examples/34_cloth_stack && PS_SHIM_FRAMES=${1:-20} exec $UIPC_PY main.py
