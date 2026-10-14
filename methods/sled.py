"""SlideScan Phase 2, method 7: pulled go/no-go sled -> methods/sled/candidates.json. No images.

  python methods/sled.py <run>

Two files typed into the run folder, board mm as in targets.json:
  sled_lanes.csv  pull,x0_mm,y0_mm,x1_mm,y1_mm       the rectangle the bridge swept on each pull
  sled.csv        target_id,pull,caught,x_mm,y_mm    one row per pull for every target in that pull's lane,
                                                     caught 1 or 0 (x_mm, y_mm blank); a catch on nothing:
                                                     target_id blank, caught 1, where it stopped in x_mm, y_mm
Coverage and covered_area_cm2 come from the lanes (area summed per pull, as false catches are counted per pull).
Rows and lanes must agree both ways: a missing pass row would turn a miss into "not covered".
A target is a candidate only if every pull over it caught it: score.py counts one hit per target per run, so a catch
on one pull must not hide a pass on another. Repeats are separate runs, as for every method. Each catch on a decoy
(id D..) or on nothing is its own candidate, i.e. one alarm. score 1, est_height_mm and image null: go/no-go only.
"""
import argparse
import csv
import json
import sys
from pathlib import Path


def read(f):
    if not f.exists():
        sys.exit(f"{f.parent}: no {f.name}; see python methods/sled.py -h")
    with open(f, newline="", encoding="utf-8-sig") as fh:  # utf-8-sig: Excel's CSV starts with a BOM
        return [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(fh, skipinitialspace=True)]


def cand(x, y):
    return {"x_mm": x, "y_mm": y, "score": 1.0, "est_height_mm": None, "image": None}


def run(path):
    path = Path(path)
    board = {t["id"]: (t["x_mm"], t["y_mm"]) for t in json.loads((path / "targets.json").read_text())["targets"]}
    rows = read(path / "sled_lanes.csv")
    try:
        lanes = {r["pull"]: [float(r[k]) for k in ("x0_mm", "y0_mm", "x1_mm", "y1_mm")] for r in rows}
    except (KeyError, ValueError):
        sys.exit("sled_lanes.csv needs columns pull,x0_mm,y0_mm,x1_mm,y1_mm with numbers")
    if len(lanes) != len(rows):
        sys.exit("sled_lanes.csv lists a pull twice")
    inside = lambda r, x, y: min(r[0], r[2]) <= x <= max(r[0], r[2]) and min(r[1], r[3]) <= y <= max(r[1], r[3])
    want = {(p, t) for p, r in lanes.items() for t, xy in board.items() if inside(r, *xy)}

    caught, cands = {}, []
    for i, r in enumerate(read(path / "sled.csv"), 2):  # line 1 is the header
        p, t, c = r.get("pull", ""), r.get("target_id", ""), r.get("caught", "")
        if p not in lanes or c not in ("0", "1"):
            sys.exit(f"sled.csv line {i}: pull must be one in sled_lanes.csv, and caught 1 or 0")
        if not t:
            try:
                xy = float(r.get("x_mm")), float(r.get("y_mm"))
            except (TypeError, ValueError):
                xy = None
            if c != "1" or xy is None:
                sys.exit(f"sled.csv line {i}: a row with no target_id is a catch on nothing: "
                         "caught 1, and where it stopped in x_mm, y_mm")
            cands.append(cand(*xy))
        elif (p, t) not in want or (p, t) in caught:
            sys.exit(f"sled.csv line {i}: {t} is not in pull {p}'s lane in sled_lanes.csv, or is listed twice")
        else:
            caught[p, t] = c == "1"
    if want - caught.keys():
        sys.exit("sled.csv has no row for " + ", ".join(f"{t} on pull {p}" for p, t in sorted(want - caught.keys()))
                 + " (in that pull's lane; write caught 0 for a pass)")
    n_false = len(cands)

    tally = {}
    for t, xy in board.items():
        n = [caught[p, t] for p in lanes if (p, t) in caught]
        if n:
            tally[t] = f"{sum(n)}/{len(n)}"
            if t.startswith("D"):
                cands += [cand(*xy) for _ in range(sum(n))]
            elif all(n):
                cands.append(cand(*xy))

    area = round(sum(abs((r[2] - r[0]) * (r[3] - r[1])) for r in lanes.values()) / 100, 1)
    out = path / "methods" / "sled"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(
        {"candidates": cands, "covered_targets": sorted(tally), "covered_area_cm2": area}, indent=2))
    print(f"{len(lanes)} pulls, {area:g} cm2 swept; {len(tally)} of {len(board)} targets in a lane; "
          f"{n_false} catches on no target; {len(cands)} candidates")
    print("caught/pulls over it: " + ", ".join(f"{t} {v}" for t, v in tally.items())
          + " (a target counts only if caught on every pull)")
    return cands


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    run(p.parse_args().run)


if __name__ == "__main__":
    main()
