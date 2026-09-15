#!/bin/bash
pgrep -af "sanitizers/matrix|compute-sanitizer --tool|TreeLauncherSubreaper|Release/bin/uipc_test_|main.py --headless" | grep -v "ps_san" | cut -c1-110
echo "gpu apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"
