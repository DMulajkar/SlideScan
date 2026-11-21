# Contributing

SlideScan is a research prototype: it flags spots that might stick out of a surface so an inspector can check them with
the real gauge. It does not pass or fail anything. Read "What has been shown" in `README.md` before building on it.

## Run the tests
```
pip install -e .
for t in test_profile test_score test_slidescan test_geometry test_grazing test_sled; do python $t.py; done
```
Each prints `ok`. Every module ships a self-test on synthetic data; a new one should too, and say what it did not test.
`test_walkaround.py`, `test_replay.py` and `test_colmap.py` need `pip install -e ".[sfm]"` and take minutes.

## Add a piece of equipment
Equipment is a data file. You need the rule from the standard, written down with where it came from.
1. **Gauges.** If the standard uses a test shape that isn't in `gauges/`, add `gauges/<name>.json`: sizes as printed
   (the unit in a `unit` key), and a `source` naming the figure and edition. Don't enter a number you haven't seen.
2. **Profile.** Copy `equipment/slide.json` to `equipment/<name>.json` and set:
   `check` (a hazard check from `checks.py`), `gauges`, `qualifying_mm` (the height a required target must reach),
   `low_mm`, `match_radius_mm`, `flag_min_est_mm`, and the `notes` saying where each number came from.
   `calibration` holds what was fitted on which hardware; don't reuse the slide's.
3. **Check it.** `python test_profile.py` validates every profile (required keys, known check, gauge files exist).
   `python score.py runs/* --equipment <name> --gate` then scores with your numbers.
4. **Validate on a board.** A profile is a claim until a board of known bumps has been scanned for it. Say in the pull
   request what you scanned. Only a flat board has been validated so far: the walkaround assumes one plane per patch,
   so curved surfaces, tubes and moving parts need a new surface model first.

## Add a method (a way of finding bumps)
Write `methods/<name>.py` with `run(run_dir, ...)` that writes `methods/<name>/candidates.json` as in `SCHEMA.md`, plus a
`test_<name>.py` on synthetic data. Add the name to its check in `checks.py`.

## Add a hazard check (a new kind of hazard, such as gaps)
Add a key to `CHECKS` in `checks.py` with its methods, then a profile that names it. Entanglement gaps (a 0.12 in feeler
gauge) are the obvious next one and have no method yet.

## Licence
MIT (see `LICENSE`). `yolo/` depends on Ultralytics, which is AGPL-3.0: see `yolo/README.md` before using or shipping it.
