"""python test_score.py -- score.py on hand-written candidates for a known board, so every count is known."""
import contextlib
import io
import json
import tempfile
from pathlib import Path

import score as sc

BOARD = {"targets": [
    {"id": "T01", "measured_mm": 0, "x_mm": 50, "y_mm": 50},
    {"id": "T04", "measured_mm": 3.0, "x_mm": 150, "y_mm": 50},
    {"id": "T05", "measured_mm": 3.2, "x_mm": 50, "y_mm": 150},
    {"id": "T08", "measured_mm": 10, "x_mm": 150, "y_mm": 150},
    {"id": "D01", "measured_mm": 0, "x_mm": 250, "y_mm": 50},
]}
META = {"date": "2026-10-14", "method": "grazing-light", "light_level": "bright", "repeat": 1, "operator": "x", "notes": ""}
ALL = ["T01", "T04", "T05", "T08", "D01"]


def cand(x, y):
    return {"x_mm": x, "y_mm": y, "score": 1.0, "est_height_mm": None, "image": "frames/a.jpg"}


RUNS = {  # name: (meta changes, covered, area cm2, candidates)
    "r1": ({}, ALL, 100, [cand(53, 150),                    # T05 hit, 3 mm off
                          cand(151, 151), cand(154, 150),   # T08 twice: closest kept, the other is a duplicate
                          cand(252, 50),                    # decoy alarm
                          cand(300, 300),                   # background alarm
                          cand(160.5, 50)]),                # 10.5 mm from T04: outside R, so a background alarm
    "r2": ({"repeat": 2}, ["T04", "T05", "T08"], 50, [cand(150, 59.9), cand(50, 150), cand(150, 150)]),
    "r3": ({"light_level": "dim"}, ALL, 100, [cand(150, 150)]),  # only T08 found
    "r4": ({}, ALL, 100, None),  # no candidates.json for this method: skipped
}

with tempfile.TemporaryDirectory() as tmp:
    runs = []
    for name, (change, covered, area, cands) in RUNS.items():
        run = Path(tmp) / name
        (run / "methods" / "test").mkdir(parents=True)
        (run / "targets.json").write_text(json.dumps(BOARD))
        (run / "meta.json").write_text(json.dumps(dict(META, **change)))
        if cands is not None:
            (run / "methods" / "test" / "candidates.json").write_text(json.dumps(
                {"candidates": cands, "covered_targets": covered, "covered_area_cm2": area}))
        runs.append(run)

    r1 = sc.score(runs[0], "test")
    assert {k: v["outcome"] for k, v in r1["targets"].items()} == \
        {"T01": "miss", "T04": "miss", "T05": "hit", "T08": "hit"}, r1["targets"]
    assert r1["targets"]["T05"]["dist_mm"] == 3.0 and r1["targets"]["T08"]["dist_mm"] == 1.41
    assert r1["targets"]["T05"]["qualifying"] and not r1["targets"]["T04"]["qualifying"]
    assert r1["decoy_alarms"] == {"D01": 1} and r1["background_alarms"] == 2
    assert r1["decoy_alarms_per_100cm2"] == 1.0 and r1["background_alarms_per_100cm2"] == 2.0
    assert json.loads((runs[0] / "methods" / "test" / "results.json").read_text()) == r1

    wide = sc.score(runs[0], "test", radius=11)  # the calibration knob: T04's candidate now lands
    assert wide["targets"]["T04"]["outcome"] == "hit" and wide["background_alarms"] == 1

    r2 = sc.score(runs[1], "test")
    assert r2["targets"]["T01"]["outcome"] == "not covered" and r2["targets"]["T04"]["outcome"] == "hit"

    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        groups = sc.report(runs, "test", show_gate=True)
    text = out.getvalue()
    bright, dim = (sc.tally(groups[k]) for k in sorted(groups, key=lambda k: dict(k)["light_level"] != "bright"))
    # condition A (bright, 2 repeats): 0 mm 0/1 (not covered in r2), 3 mm 1/2, 3.2 mm 2/2, 10 mm 2/2
    assert bright == {0: [0, 1, 1], 3.0: [1, 2, 0], 3.2: [2, 2, 0], 10: [2, 2, 0]}, bright
    assert dim == {0: [0, 1, 0], 3.0: [0, 1, 0], 3.2: [0, 1, 0], 10: [1, 1, 0]}, dim
    assert sc.smallest(bright) == 3.2 and sc.smallest(dim) == 10
    assert sc.gate(bright) == (4, 4, 0) and sc.gate(dim) == (1, 2, 0)

    assert "3 runs scored, 1 without" in text, text
    assert "same in all runs: method=grazing-light" in text, text
    row = next(l for l in text.splitlines() if l.startswith("light_level=bright") and "mm" in l)
    assert row.split()[1:7] == ["2", "0/1", "1/2", "2/2", "2/2", "3.2"], row  # reps, then k/n per height
    assert "1/2" in row.split("mm")[1], row  # 1 decoy / 2 background alarms
    gate_rows = text.split("Gate A")[1]
    assert "4/4" in gate_rows and "PASS" in gate_rows and "3/4 = 0.75" in gate_rows, gate_rows
    assert "1/2" in gate_rows and "FAIL" in gate_rows, gate_rows
    assert "%" not in text.replace("95% bound", ""), "k/n counts only, never bare percentages"

# low-height hits: reported next to Gate A, and the note appears only when one happened
assert sc.low({0: [1, 1, 0], 1: [0, 1, 0], 3.2: [2, 2, 0]}) == (1, 2) and sc.low({3.2: [1, 1, 0]}) == (0, 0)
with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp) / "r5"
    (run / "methods" / "test").mkdir(parents=True)
    (run / "targets.json").write_text(json.dumps(BOARD))
    (run / "meta.json").write_text(json.dumps(META))
    (run / "methods" / "test" / "candidates.json").write_text(json.dumps(
        {"candidates": [cand(50, 50), cand(50, 150)], "covered_targets": ALL, "covered_area_cm2": 100}))  # flush T01 and T05
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sc.report([run], "test", show_gate=True)
    g = out.getvalue().split("Gate A")[1]
    assert "hits <2 mm" in g and "1/1" in g and "PASS" in g and "does not show the method separates" in g, g
assert "does not show the method separates" not in text, "no low hits in the main fixture, so no note"

# --min-est: candidates below the cut vanish before scoring; ones with no estimate are kept; results.json records it
with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp) / "r6"
    (run / "methods" / "test").mkdir(parents=True)
    (run / "targets.json").write_text(json.dumps(BOARD))
    (run / "meta.json").write_text(json.dumps(META))
    est = lambda x, y, h: dict(cand(x, y), est_height_mm=h)
    (run / "methods" / "test" / "candidates.json").write_text(json.dumps({
        "candidates": [est(50, 50, 3.5),            # flush T01, reads 3.5
                       est(150, 50, 4.8),           # T04, reads 4.8
                       est(50, 150, 6.0),           # T05, reads 6.0
                       est(300, 300, 4.0),          # background, reads 4.0
                       cand(150, 150)],             # T08, no estimate: always kept
        "covered_targets": ALL, "covered_area_cm2": 100}))
    full = sc.score(run, "test")
    assert {k: v["outcome"] for k, v in full["targets"].items()} == {"T01": "hit", "T04": "hit", "T05": "hit", "T08": "hit"}
    assert full["background_alarms"] == 1 and full["min_est_mm"] is None
    cut = sc.score(run, "test", min_est=5.0)
    assert {k: v["outcome"] for k, v in cut["targets"].items()} == {"T01": "miss", "T04": "miss", "T05": "hit", "T08": "hit"}
    assert cut["background_alarms"] == 0 and cut["min_est_mm"] == 5.0
    assert json.loads((run / "methods" / "test" / "results.json").read_text()) == cut
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        sc.report([run], "test", show_gate=True, min_est=5.0)
    assert "candidates below 5 mm estimated height dropped" in out.getvalue(), out.getvalue()

# monotone: a lucky 1 mm hit does not undercut a 4 mm miss; a missed tallest height means none
assert sc.smallest({1: [1, 1, 0], 4: [0, 1, 0], 10: [1, 1, 0]}) == 10
assert sc.smallest({3.2: [1, 1, 0], 10: [0, 1, 0]}) is None
assert sc.gate({3.2: [0, 0, 2]}) == (0, 0, 2)  # never covered: n = 0, so it can't pass
print("ok")
