"""Per-function SASS comparison between the branch base and HEAD.

nvcc encodes the absolute source path and the temp-file PID into every
anonymous-namespace symbol, so the *names* always differ between two compiles
from two directories; the normalisation below removes exactly that and nothing
else. Functions that exist on only one side are listed separately (s10 adds one
verify-only kernel to abd_linear_subsystem.cu, behind UIPC_ABD_GH_PREPASS_VERIFY)."""
import re, sys, collections, os

def norm_name(n):
    n = re.sub(r'_GLOBAL__N__[0-9a-f]+_', '_GLOBAL__N__X_', n)
    n = re.sub(r'__nv_static_[0-9]+__[0-9a-f]+_', '__nv_static_X__X_', n)
    n = re.sub(r'_cu_[0-9a-f]+_[0-9]+', '_cu_X_X', n)
    return n

def parse(path):
    fns = collections.OrderedDict()
    cur = None
    for line in open(path):
        m = re.search(r'Function : (\S+)', line)
        if m:
            cur = norm_name(m.group(1))
            fns[cur] = []
            continue
        if cur is not None:
            # keep only the instruction text, drop addresses and encodings
            m2 = re.match(r'\s+/\*[0-9a-f]+\*/\s+(.*?);', line)
            if m2:
                fns[cur].append(m2.group(1).strip())
    return fns

base, head = sys.argv[1], sys.argv[2]
b, h = parse(base), parse(head)
common = [k for k in h if k in b]
only_h = [k for k in h if k not in b]
only_b = [k for k in b if k not in h]
bad = [k for k in common if b[k] != h[k]]
ni = sum(len(h[k]) for k in common)
tag = os.path.basename(head).split('.')[0]
print(f"{tag:26s} common functions {len(common):3d} ({ni:6d} instructions)  "
      f"differing {len(bad)}  base-only {len(only_b)}  head-only {len(only_h)}")
for k in bad[:5]:
    print("   DIFFERS:", k[:110])
for k in only_h:
    print("   head-only (new):", k[:110])
for k in only_b:
    print("   base-only (removed):", k[:110])
sys.exit(1 if bad or only_b else 0)
