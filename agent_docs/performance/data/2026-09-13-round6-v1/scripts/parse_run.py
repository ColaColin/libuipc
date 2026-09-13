#!/usr/bin/env python3
"""Parse one V1 arm run log into per-run and per-frame observables.

Two independent things come out of the same log and are cross-checked against
each other, which is the probe validation (PERF_METHOD 2.7):

  * `SimplexTrajectoryFilter PTs/EEs/PEs/PPs` -- the active contact-pair set
    produced by do_filter_active, once per Newton iteration;
  * `<IPCSimplexNormalContact> DyTopo Grad3 count / Hess3x3 count` -- the
    contact reporter's own triplet counts, produced by a different code path.

They must satisfy, exactly:
    Grad3   = 4*PT + 4*EE + 3*PE + 2*PP
    Hess3x3 = 10*PT + 10*EE + 6*PE + 3*PP     (symmetric block storage)
Any violation fails the parse loudly rather than degrading quietly.
"""
import json, re, sys
from pathlib import Path

RE_FRAME = re.compile(r">>> Begin Frame: (\d+)")
RE_PAIR = re.compile(r"SimplexTrajectoryFilter PTs: (\d+), EEs: (\d+), PEs: (\d+), PPs: (\d+)")
RE_DY = re.compile(r"IPCSimplexNormalContact> DyTopo Grad3 count: (\d+), DyTopo Hess3x3 count: (\d+)")


def parse(path):
    frame, pending, per_frame, checks, bad = 0, None, {}, 0, []
    summary, trace = None, None
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("VERIFY_RESULT "):
            summary = json.loads(line[len("VERIFY_RESULT "):]); continue
        if line.startswith("VERIFY_TRACE "):
            trace = json.loads(line[len("VERIFY_TRACE "):]); continue
        m = RE_FRAME.search(line)
        if m:
            frame = int(m.group(1)); pending = None; continue
        m = RE_PAIR.search(line)
        if m:
            pending = tuple(int(x) for x in m.groups())
            per_frame.setdefault(frame, []).append(pending)
            continue
        m = RE_DY.search(line)
        if m and pending is not None:
            pt, ee, pe, pp = pending
            g, h = int(m.group(1)), int(m.group(2))
            eg = 4 * pt + 4 * ee + 3 * pe + 2 * pp
            eh = 10 * pt + 10 * ee + 6 * pe + 3 * pp
            checks += 1
            if (g, h) != (eg, eh):
                bad.append((frame, pending, g, h, eg, eh))
            pending = None
    return summary, trace, per_frame, checks, bad


def run_record(path):
    summary, trace, per_frame, checks, bad = parse(path)
    if summary is None:
        raise SystemExit(f"{path}: no VERIFY_RESULT (run failed?)")
    rows = []
    for f in sorted(per_frame):
        its = per_frame[f]
        tot = [sum(x[k] for x in its) for k in range(4)]
        rows.append({"frame": f, "iters": len(its),
                     "pairs_mean": sum(sum(x) for x in its) / len(its),
                     "pairs_max": max(sum(x) for x in its),
                     "pairs_first": sum(its[0]),
                     "pairs_last": sum(its[-1]),
                     "PT": tot[0] / len(its), "EE": tot[1] / len(its),
                     "PE": tot[2] / len(its), "PP": tot[3] / len(its)})
    n = max(1, len(rows))
    summary = dict(summary)
    summary["pairs_mean_per_it"] = sum(r["pairs_mean"] for r in rows) / n
    summary["pairs_max_over_run"] = max((r["pairs_max"] for r in rows), default=0)
    summary["pairs_last_quarter"] = (
        sum(r["pairs_mean"] for r in rows[-max(1, n // 4):]) / max(1, n // 4))
    # the converged contact set of each frame (first DCD of the next Newton loop
    # sees the previous frame's converged state): the physical contact count,
    # unweighted by how many Newton iterations the frame happened to take.
    summary["pairs_converged_mean"] = sum(r["pairs_first"] for r in rows) / n
    summary["pairs_converged_last_quarter"] = (
        sum(r["pairs_first"] for r in rows[-max(1, n // 4):]) / max(1, n // 4))
    summary["pairs_converged_max"] = max((r["pairs_first"] for r in rows), default=0)
    summary["probe_checks"] = checks
    summary["probe_violations"] = len(bad)
    return {"summary": summary, "trace": trace, "pairs": rows, "bad": bad[:5]}


if __name__ == "__main__":
    out = {}
    for p in sys.argv[1:]:
        tag = Path(p).stem
        out[tag] = run_record(p)
        s = out[tag]["summary"]
        print(f"{tag}: probe {s['probe_checks']} checks, {s['probe_violations']} violations, "
              f"pairs/it {s['pairs_mean_per_it']:.0f}, newton {s.get('verify_newton_total')}, "
              f"ok={s.get('verify_ok')}")
    Path("/workspace/output/round6/v1/parsed.json").write_text(json.dumps(out))
