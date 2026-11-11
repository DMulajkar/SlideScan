"""SlideScan Phase 2: grade a method's candidates against the board, per run and across runs.

  python score.py runs/* [--method grazing] [--gate] [--radius 10]
      -> methods/<name>/results.json in each run that has methods/<name>/candidates.json,
         then a condition-by-height table of k/n counts (hits / covered target-repeats), never bare percentages

candidates.json (every method writes it, board mm):
  {"candidates": [{"x_mm", "y_mm", "score", "est_height_mm" (or null), "image"}],
   "covered_targets": [target ids the method looked at], "covered_area_cm2": board area it looked at}
A candidate within --radius mm of a target hits it; each target is hit at most once (repeats from other images
are duplicates, not alarms). Near a decoy (id D..) = decoy alarm; near nothing = background alarm.
Outcome per target: hit, miss (covered, no hit) or not covered. Qualifying = measured_mm >= 3.2.
A condition is meta.json minus date, repeat, operator and notes; runs that share it are repeats.
"""
import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

QUAL_MM = 3.2
# ponytail: starting guess, same as walkaround's --min-height. Nothing is required below QUAL_MM, but a hit on a
# flush or 1 mm bolt means the method flags a bump at every bolt without telling 0 from 5 mm, which Gate A cannot see.
LOW_MM = 2.0
# ponytail: starting guess. A 1/4 in bolt is 6.35 mm wide and mapped bolt heads land 1-6 mm off on the
# practice runs; tune on real data with each hit's dist_mm in results.json.
RADIUS_MM = 10.0
NOT_CONDITION = {"date", "repeat", "operator", "notes"}


def per_100cm2(n, area):
    return round(100 * n / area, 3) if area else None


def score(run, method, radius=RADIUS_MM):
    """Grade one run's methods/<method>/candidates.json; writes and returns results.json."""
    run = Path(run)
    board = json.loads((run / "targets.json").read_text())["targets"]
    c = json.loads((run / "methods" / method / "candidates.json").read_text())
    hits, decoy, bg = {}, Counter(), 0
    for k in c["candidates"]:
        t = min(board, key=lambda t: math.dist((t["x_mm"], t["y_mm"]), (k["x_mm"], k["y_mm"])))
        d = math.dist((t["x_mm"], t["y_mm"]), (k["x_mm"], k["y_mm"]))
        if d > radius:
            bg += 1
        elif t["id"].startswith("D"):
            decoy[t["id"]] += 1
        elif t["id"] not in hits or d < hits[t["id"]]["dist_mm"]:  # keep the closest; the rest are duplicates
            hits[t["id"]] = {"dist_mm": round(d, 2), "est_height_mm": k.get("est_height_mm")}
    covered, area = set(c["covered_targets"]), c["covered_area_cm2"]
    targets = {}
    for t in board:
        if t["id"].startswith("D"):
            continue
        outcome = "hit" if t["id"] in hits else "miss" if t["id"] in covered else "not covered"
        targets[t["id"]] = {"height_mm": t["measured_mm"], "qualifying": t["measured_mm"] >= QUAL_MM,
                            "outcome": outcome, **hits.get(t["id"], {})}
    res = {"method": method, "radius_mm": radius, "targets": targets,
           "decoy_alarms": dict(decoy), "background_alarms": bg, "covered_area_cm2": area,
           "decoy_alarms_per_100cm2": per_100cm2(sum(decoy.values()), area),
           "background_alarms_per_100cm2": per_100cm2(bg, area)}
    (run / "methods" / method / "results.json").write_text(json.dumps(res, indent=2))
    return res


def tally(results):
    """{height: [hits, covered, not covered]} over every target in every repeat."""
    t = defaultdict(lambda: [0, 0, 0])
    for r in results:
        for v in r["targets"].values():
            cell = t[v["height_mm"]]
            cell[0] += v["outcome"] == "hit"
            cell[1] += v["outcome"] != "not covered"
            cell[2] += v["outcome"] == "not covered"
    return dict(t)


def smallest(cells):
    """Lowest height at and above which every covered target was hit in every repeat (None if the tallest missed).
    Monotone on purpose: a lucky 1 mm hit must not undercut a 4 mm miss."""
    best = None
    for h in sorted((h for h in cells if cells[h][1]), reverse=True):
        if cells[h][0] < cells[h][1]:
            break
        best = h
    return best


def low(cells):
    """(hits, covered target-repeats) on targets under LOW_MM. Reported next to Gate A, not part of it."""
    q = [v for h, v in cells.items() if h < LOW_MM]
    return sum(v[0] for v in q), sum(v[1] for v in q)


def gate(cells):
    """Gate A over one condition: (hits, n) on qualifying covered target-repeats, qualifying not covered."""
    q = [v for h, v in cells.items() if h >= QUAL_MM]
    return sum(v[0] for v in q), sum(v[1] for v in q), sum(v[2] for v in q)


def report(runs, method, radius=RADIUS_MM, show_gate=False):
    groups = defaultdict(list)  # condition -> results of its repeats
    for run in runs:
        if (run / "methods" / method / "candidates.json").exists():
            meta = json.loads((run / "meta.json").read_text())
            key = tuple(sorted((k, str(v)) for k, v in meta.items() if k not in NOT_CONDITION))
            groups[key].append(score(run, method, radius))
    n_runs = sum(map(len, groups.values()))
    print(f"\n== method {method}: {n_runs} runs scored, {len(runs) - n_runs} without methods/{method}/candidates.json")
    if not groups:
        return groups
    conds = [dict(k) for k in groups]
    vary = sorted(k for k in set().union(*conds) if len({c.get(k) for c in conds}) > 1)
    same = [f"{k}={v}" for k, v in sorted(conds[0].items()) if k not in vary]
    print("same in all runs: " + ", ".join(same))
    labels = [", ".join(f"{k}={c.get(k)}" for k in vary) or "all runs" for c in conds]
    cells = [tally(r) for r in groups.values()]
    heights = sorted({h for c in cells for h in c})
    w = max(map(len, labels + ["condition"]))
    fmt = lambda h: f"{h:g}"
    print(f"hits/covered per measured height (mm), over targets x repeats; radius {radius:g} mm")
    print(f"{'condition':<{w}}  reps  " + "".join(f"{fmt(h):>7}" for h in heights)
          + "  smallest reliable  decoy/bg alarms  per 100 cm2")
    for label, res, c in zip(labels, groups.values(), cells):
        s = smallest(c)
        dec, bg = sum(sum(r["decoy_alarms"].values()) for r in res), sum(r["background_alarms"] for r in res)
        area = sum(r["covered_area_cm2"] or 0 for r in res)
        rate = f"{per_100cm2(dec, area):g}/{per_100cm2(bg, area):g}" if area else "n/a"
        print(f"{label:<{w}}  {len(res):>4}  "
              + "".join(f"{f'{c[h][0]}/{c[h][1]}' if h in c and c[h][1] else '-':>7}" for h in heights)
              + f"  {(fmt(s) + ' mm') if s is not None else 'none':>17}  {f'{dec}/{bg}':>15}  {rate:>11}")
    if show_gate:
        print(f"\nGate A: zero misses on qualifying targets (measured >= {QUAL_MM:g} mm) that were covered, in every repeat")
        print(f"{'condition':<{w}}  reps  hits/n  not covered  hits <{LOW_MM:g} mm  result  95% bound on miss rate")
        for label, res, c in zip(labels, groups.values(), cells):
            k, n, nc = gate(c)
            ok = n and k == n
            # rule of three: zero misses in n independent trials -> 95% upper bound about 3/n (rough below n = 30)
            bound = f"<= 3/{n} = {3 / n:.2f} (rule of three)" if ok else "-"
            lk, ln = low(c)
            print(f"{label:<{w}}  {len(res):>4}  {f'{k}/{n}':>6}  {nc:>11}  {f'{lk}/{ln}':>{9 + len(f'{LOW_MM:g}')}}  "
                  f"{'PASS' if ok else 'FAIL':>6}  {bound}")
            if lk:
                print(f"{'':<{w}}  note: {lk} of {ln} covered targets under {LOW_MM:g} mm were flagged, so a PASS here "
                      f"does not show the method separates a flush bolt from a real protrusion")
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run folders; globs like runs/* are expanded here (PowerShell doesn't)")
    p.add_argument("--method", help="methods/<name> to score; default: every method found")
    p.add_argument("--gate", action="store_true", help="also print the Gate A check")
    p.add_argument("--radius", type=float, default=RADIUS_MM, help="match radius in mm")
    args = p.parse_args()
    runs = [Path(r) for a in args.runs for r in (sorted(glob.glob(a)) or [a]) if Path(r).is_dir()]
    methods = [args.method] if args.method else sorted({m.parent.name for r in runs
                                                         for m in r.glob("methods/*/candidates.json")})
    if not methods:
        raise SystemExit("no methods/<name>/candidates.json in any run")
    for m in methods:
        report(runs, m, args.radius, args.gate)


if __name__ == "__main__":
    main()
