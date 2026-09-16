# s17 process note: a transient slow-box window invalidated the first crease-press A/B

The first crease-press ab.py (n=10/arm, label `s17-pcgpoll-cp`, ~17:20) read
meanFrameMs **391.5 (old, UIPC_PCG_POLL=0) vs 398.4 (new)** — BOTH arms ~65 %
above the round's ~230-250 ms class, with per-frame Newton/PCG counts
frame-by-frame IDENTICAL to s10's drift runs (7/960 vs 7/955, 5/180 vs 5/180,
...), i.e. same trajectory, +65 % wall, in the arm that runs the old code
path. mas-bunny measured minutes earlier in the same window was normal
(61.59 ms vs the historical ~61.5).

Diagnosis sequence:
1. GPU healthy (boost 1935-1950 MHz at 99-100 % util under load, 47-62 °C,
   172 W; 32 cores, load 2; the background gallery/render processes idle).
2. Rebuilt the head binary (git checkout da9bd2d0 of the two files, fresh .o):
   **245.4 / 246.5 ms** — normal speed, same count class (PCG 125-128k).
3. Re-applied the s17 patch, rebuilt, interleaved single runs:
   POLL=0 223.2/239.6, POLL=1 209.8/267.5 — normal class again.

Conclusion: a transient box state (~20 min, cause not identified — no
contending GPU process, no clock throttle, no CPU starvation visible) slowed
CREASE-PRESS-sized work by ~65 % in both arms; the first A/B was discarded
and rerun (`ab_cp2`). Lesson reaffirmed: the ab.py old arm is itself the
box-health canary — when the OLD arm's absolute level departs the round's
class, the whole A/B is void, not just noisy; compare against a known-good
reference (s10's drift jsons) before reading any delta. Second lesson, paid
for once: running the correctness gate WHILE an A/B is in the background
contends the GPU — the polluted A/B was killed and rerun; gates and
benchmarks never overlap again in this round.
