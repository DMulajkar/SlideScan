"""python test_sled.py [keep_dir] -- methods/sled.py on a synthetic sled run (the sled isn't built yet), then score.py
end to end. Lean-board targets plus one decoy, 3 lanes pulled up the board, lane 1 pulled twice.
Pass a folder to keep the synthetic run there."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import score as sc
from methods import sled as sl

ROOT = Path(__file__).resolve().parent
W, H, BRIDGE = 333.38, 349.25, 60  # lean board mm; bridge (lane) width mm
TARGETS = {"T01": (288.93, 34.24, 0), "T02": (165.1, 34.24, 1), "T04": (44.63, 34.24, 3),
           "T05": (44.63, 152.98, 3.2), "T06": (165.1, 152.98, 4), "T07": (288.93, 152.98, 5),
           "T08": (288.93, 276.22, 10), "D01": (288.93, 215.0, 0)}  # D01: a decoy the lean board doesn't have yet
LANES = {"1": 44.63, "2": 165.1, "3": 288.93, "4": 44.63}  # pull: lane center x; pull 4 repeats lane 1
BOARD = {"targets": [{"id": t, "measured_mm": h, "x_mm": x, "y_mm": y} for t, (x, y, h) in TARGETS.items()]}


def rows(change=None):
    """A 3.2 mm bridge catches h >= 3.2. Then: T04 (3.0) snags once on pull 4, D01 snags, a seam catches on pull 2."""
    caught = {(p, t): h >= 3.2 for p, xc in LANES.items() for t, (x, _, h) in TARGETS.items() if abs(x - xc) <= BRIDGE / 2}
    caught.update({("4", "T04"): True, ("3", "D01"): True, **(change or {})})
    return [f"{t},{p},{int(c)},," for (p, t), c in caught.items() if c is not None] + [",2,1,165.1,250"]


def write(run, lines):
    (run / "sled.csv").write_text("target_id, pull, caught, x_mm, y_mm\n" + "\n".join(lines) + "\n")


def refusal(run, lines):
    write(run, lines)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            sl.run(run)
    except SystemExit as e:
        return str(e)
    raise AssertionError(f"accepted: {lines}")


with tempfile.TemporaryDirectory() as tmp:
    run = Path(sys.argv[1] if sys.argv[1:] else tmp) / "synthetic-sled"
    run.mkdir(parents=True)
    (run / "targets.json").write_text(json.dumps(BOARD))
    (run / "meta.json").write_text(json.dumps({"date": "2026-10-14", "device": "sled", "method": "sled",
                                               "surface": "matte-flat", "repeat": 1, "operator": "synthetic"}))
    (run / "sled_lanes.csv").write_text("pull,x0_mm,y0_mm,x1_mm,y1_mm\n" + "".join(
        f"{p},{xc - BRIDGE / 2:g},0,{xc + BRIDGE / 2:g},{H}\n" for p, xc in LANES.items()))

    # T05 passed on the repeat pull: the catch on pull 1 must not hide it
    write(run, rows({("4", "T05"): False}))
    with contextlib.redirect_stdout(io.StringIO()):
        sl.run(run)
    assert sc.gate(sc.tally([sc.score(run, "sled")])) == (3, 4, 0)

    # rows and lanes must agree: a forgotten pass row would make a miss "not covered"
    assert "no row for T05 on pull 4" in refusal(run, rows({("4", "T05"): None}))
    assert "T07 is not in pull 1's lane" in refusal(run, rows() + ["T07,1,1,,"])
    assert "where it stopped" in refusal(run, rows() + [",1,1,,"])

    write(run, rows())  # last, so a kept run is the good one
    out = subprocess.run([sys.executable, str(ROOT / "methods" / "sled.py"), str(run)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    c = json.loads((run / "methods" / "sled" / "candidates.json").read_text())
    assert sorted(c["covered_targets"]) == sorted(TARGETS) and c["covered_area_cm2"] == 4 * BRIDGE * H / 100, c
    at = sorted((k["x_mm"], k["y_mm"]) for k in c["candidates"])
    # T05-T08 caught on every pull; T04 only 1/2 so no candidate; D01 and the seam: one candidate each
    assert at == sorted([TARGETS[t][:2] for t in ("T05", "T06", "T07", "T08", "D01")] + [(165.1, 250)]), at
    assert "T04 1/2" in out.stdout and "T05 2/2" in out.stdout, out.stdout

    out = subprocess.run([sys.executable, str(ROOT / "score.py"), str(run), "--method", "sled", "--gate"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    gate = out.stdout.split("Gate A")[1]
    assert "1 runs scored" in out.stdout and "4/4" in gate and "PASS" in gate, out.stdout
    res = json.loads((run / "methods" / "sled" / "results.json").read_text())
    assert {t: v["outcome"] for t, v in res["targets"].items()} == {
        "T01": "miss", "T02": "miss", "T04": "miss", "T05": "hit", "T06": "hit", "T07": "hit", "T08": "hit"}, res
    assert res["decoy_alarms"] == {"D01": 1} and res["background_alarms"] == 1, res
print("ok")
