#!/bin/bash
# Kill every sanitizer matrix driver and sanitized process by PID (patterns live here, not in the caller's command line).
for pat in "bash /workspace/output/round6/sanitizers/matrix" "compute-sanitizer --tool" "TreeLauncherSubreaper" "Release/bin/uipc_test_" "main.py --headless"; do
  for p in $(pgrep -f "$pat"); do [ "$p" != "$$" ] && [ "$p" != "$PPID" ] && kill "$p" 2>/dev/null; done
done
sleep 3
for pat in "compute-sanitizer --tool" "TreeLauncherSubreaper" "Release/bin/uipc_test_" "main.py --headless"; do
  for p in $(pgrep -f "$pat"); do [ "$p" != "$$" ] && [ "$p" != "$PPID" ] && kill -9 "$p" 2>/dev/null; done
done
for i in $(seq 1 30); do sleep 2; [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && [ -z "$(pgrep -f 'compute-sanitizer --tool')" ] && { echo "GPU idle after $((i*2)) s"; exit 0; }; done
echo "GPU NOT idle"; exit 1
