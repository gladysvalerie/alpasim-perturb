"""Braking analysis for AlpaSim lead-vehicle runs (read-only).

For every rollout under the given run folder(s) this prints, measured from the
moment the lead vehicle starts braking (--brake-at):
  - when the reasoning text first mentions slowing / braking / stopping,
  - when 1.5's *planned* path first decelerates harder than 1 m/s^2, and its peak,
  - when the *car* (ego) actually decelerates harder than 1 m/s^2, and its peak,
  - the collision time (from metrics.parquet) and how often the text changed.
Then averages per lead braking strength (variant names like a5_g20 -> -5 m/s^2),
and optional step-by-step timelines (--show).

Nothing is written; it only reads rollout.asl and metrics.parquet.

Usage (from the repo root, after `source /workspace/alpasim_env.sh`):
  uv run python analysis/braking_timeline.py /workspace/alpasim_runs/real9 \
      --show a3_g20/796572e4 --show a5_g30/63d84706 2>&1 | grep -v "cubic spline"
  # zoom into the test9 creep, 0.25 s steps from 7 to 12 s:
  uv run python analysis/braking_timeline.py /workspace/alpasim_runs/test9 \
      --show all --from 7 --to 12 --step 0.25 2>&1 | grep -v "cubic spline"

Caveat: the text categories are keyword matches. "since it is slowing ahead"
counts as SLOW but describes the lead, and "stopped ahead" counts as STOP.
"""
import argparse
import asyncio
import glob
import os
import pickle
import re

import numpy as np
import polars as pl
from alpasim_utils.logs import async_read_pb_log


def cat(text):
    s = (text or "").lower()
    if "stop" in s:
        return "STOP"
    if re.search(r"brak|slow|decel|yield", s):
        return "SLOW"
    if re.search(r"keep|distance|follow|maintain", s):
        return "KEEP"
    return "OTHER"


def plan_profile(ts, xy, horizon=1.0):
    """(planned decel m/s2 with + = slowing, planned distance m) over the first `horizon` s of one plan."""
    dt = np.diff(ts)
    ok = dt > 1e-6
    if ok.sum() < 3:
        return np.nan, np.nan
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    v = seg[ok] / dt[ok]
    tm = ((ts[:-1] + ts[1:]) / 2)[ok] - ts[0]
    if tm[-1] < 0.1 + horizon:
        return np.nan, np.nan
    decel = (np.interp(0.1, tm, v) - np.interp(0.1 + horizon, tm, v)) / horizon
    dist = np.interp(horizon, ts - ts[0], np.concatenate([[0.0], np.cumsum(seg)]))
    return decel, dist


async def load(asl):
    ego, plans = {}, []
    async for m in async_read_pb_log(asl):
        k = m.WhichOneof("log_entry")
        if k == "actor_poses":
            for a in m.actor_poses.actor_poses:
                if a.actor_id == "EGO":
                    ego[m.actor_poses.timestamp_us] = (a.actor_pose.vec.x, a.actor_pose.vec.y)
        elif k == "driver_return" and len(m.driver_return.trajectory.poses) > 3:
            ps = m.driver_return.trajectory.poses
            ts = np.array([p.timestamp_us for p in ps], float)
            xy = np.array([(p.pose.vec.x, p.pose.vec.y) for p in ps])
            try:
                r = pickle.loads(m.driver_return.debug_info.unstructured_debug_info).get("reasoning_text")
            except Exception:
                r = None
            decel, dist = plan_profile(ts / 1e6, xy)
            plans.append((ts[0], decel, dist, "" if r is None else str(r)))
    return ego, plans


def analyse(asl, brake_at):
    d = os.path.dirname(asl)
    ego, plans = asyncio.run(load(asl))
    ts = np.array(sorted(ego))
    t0 = ts[0]
    t = (ts - t0) / 1e6
    e = np.array([ego[x] for x in ts])
    v = np.linalg.norm(np.diff(e, axis=0), axis=1) / np.diff(t)
    v = np.convolve(v, np.ones(3) / 3, mode="same")
    tv = t[1:]
    ex = (v[:-5] - v[5:]) / 0.5  # executed decel over 0.5 s, + = slowing
    tx = tv[:-5]
    hit = None
    mp = glob.glob(d + "/metrics.parquet")
    if mp:
        df = pl.read_parquet(mp[0]).filter(pl.col("valid"))
        t0m = df["timestamps_us"].min()
        h = df.filter((pl.col("name") == "collision_any") & (pl.col("values") > 0))
        if len(h):
            hit = (h["timestamps_us"].min() - t0m) / 1e6
    end = hit if hit else float(t[-1])
    P = [((a - t0) / 1e6, dc, ds, r) for a, dc, ds, r in plans]
    w = [p for p in P if brake_at <= p[0] < end]
    first = lambda xs: xs[0] if xs else None
    win = (tx >= brake_at) & (tx < end)
    return dict(
        hit=hit,
        end=end,
        word=first([p[0] for p in w if cat(p[3]) in ("SLOW", "STOP")]),
        p_on=first([p[0] for p in w if p[1] > 1.0]),
        p_pk=np.nanmax([p[1] for p in w]) if w else np.nan,
        x_on=first([a for a, x in zip(tx[win], ex[win]) if x > 1.0]),
        x_pk=ex[win].max() if win.any() else np.nan,
        changes=sum(1 for i in range(1, len(w)) if w[i][3] != w[i - 1][3]),
        P=P,
        tv=tv,
        v=v,
    )


def timeline(name, R, t_from, t_to, step):
    print(f"\n--- {name}: time | ego speed | plan: decel / distance over next 1 s | reasoning")
    for tt in np.arange(t_from, t_to + 1e-6, step):
        p = min(R["P"], key=lambda q: abs(q[0] - tt))
        sp = np.interp(tt, R["tv"], R["v"])
        print(f"  {tt:5.2f}s | {sp * 100:6.1f} cm/s | {p[1]:+5.2f} m/s2 / {p[2]:5.2f} m | {cat(p[3]):5} {p[3][:80]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run folder(s), e.g. /workspace/alpasim_runs/real9")
    ap.add_argument("--brake-at", type=float, default=3.0, help="s when the lead starts braking (default 3.0)")
    ap.add_argument("--show", action="append", default=[],
                    help="timeline for <variant>/<run-id prefix>, or 'all'; repeatable")
    ap.add_argument("--from", dest="t_from", type=float, default=2.5, help="timeline start, s (default 2.5)")
    ap.add_argument("--to", dest="t_to", type=float, default=None,
                    help="timeline end, s (default: crash + 2 s, at most 12 s)")
    ap.add_argument("--step", type=float, default=0.5, help="timeline step, s (default 0.5)")
    args = ap.parse_args()
    rel = lambda x: f"+{x - args.brake_at:3.1f}s" if x is not None else "never "

    rows = []
    print("variant  run      | crash | 1st slow/stop word | plan decel>1 | plan peak | car decel>1 | car peak | text changes")
    print(f"                   (times are after the lead starts braking at {args.brake_at:.1f} s)")
    for run in args.runs:
        run = run.rstrip("/")
        for asl in sorted(glob.glob(f"{run}/**/rollouts/*/*/rollout.asl", recursive=True)):
            var = os.path.relpath(asl.split("/rollouts/")[0], run)
            var = os.path.basename(run) if var == "." else var
            rid = os.path.basename(os.path.dirname(asl))[:8]
            R = analyse(asl, args.brake_at)
            rows.append((var, R))
            crash = f"{R['hit']:4.1f}s" if R["hit"] else "none "
            print(f"{var:<8} {rid} | {crash} | {rel(R['word']):>18} | {rel(R['p_on']):>12} | "
                  f"{R['p_pk']:4.1f} m/s2 | {rel(R['x_on']):>11} | {R['x_pk']:4.1f} m/s2 | {R['changes']}")
            if "all" in args.show or any(f"{var}/{rid}".startswith(s) for s in args.show):
                t_to = args.t_to if args.t_to is not None else min(R["end"] + 2.0, 12.0)
                timeline(f"{var} {rid}", R, args.t_from, t_to, args.step)
                print()

    groups = sorted({m.group(1) for v, _ in rows if (m := re.match(r"a(\d+)_", v))}, key=int)
    if groups:
        print("\nAverages per lead braking strength (all gaps):")
    for a in groups:
        G = [R for v, R in rows if v.startswith(f"a{a}_")]
        lag = lambda k: np.mean([R[k] - args.brake_at for R in G if R[k] is not None]) if any(R[k] is not None for R in G) else float("nan")
        print(f"  lead -{a} m/s2: plan peak {np.nanmean([R['p_pk'] for R in G]):4.1f} | "
              f"car peak {np.nanmean([R['x_pk'] for R in G]):4.1f} m/s2 | word after {lag('word'):3.1f}s | "
              f"plan>1 after {lag('p_on'):3.1f}s | car>1 after {lag('x_on'):3.1f}s | "
              f"crashes {sum(1 for R in G if R['hit'])}/{len(G)}")


if __name__ == "__main__":
    main()
