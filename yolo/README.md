# yolo/ - stage-1 bolt detector (feasibility test, 2026-11-11)

Finds the bolts at low resolution so the slow close height check only has to run at those spots (Phase 3 of the slide-protrusion roadmap).
One class (bolt). Leave-one-run-out: train on three walkaround runs, test on the fourth.

```
labels.py          box projection from board positions + camera poses, YOLO format, matching (pure; test_labels.py)
make_dataset.py    auto-labels every registered frame of runs A, B2, tex1, tex2; saves each at 1920/960/480 px wide on a 1920x1088 canvas
train_eval.py      trains YOLO11n on three runs, tests the fourth, writes results/fold<N>.json
run_make.sh, run_fold.sh, run_folds123.sh   WSL wrappers
results/           fold0-3.json (recall by pixel size, confidence and target)
qa_boxes.jpg       sample frames with their boxes: look at it before trusting a run's labels
```
Run: `python yolo/make_dataset.py` (WSL, `~/vslam/.venv`), copy `yolo/data` to `~/yolo/data`, then `python yolo/train_eval.py --fold N --data ~/yolo/data` in `~/yolo/.venv` (ultralytics, CUDA torch). `python yolo/test_labels.py` is the self-test.

What it showed and what it did not: see the wiki pages "Stage-1 Bolt Detector (YOLO) - Feasibility Test" and "SlideScan - Command Reference".
Short version: pooled over the four held-out runs a proposal landed on the bolt in 637 to 649 of 652 at confidence 0.05 across the three pixel sizes;
run A (the one held-out run that differs in texture and orientation) 128 to 140 of 142. Likely inflated for the other three (same board, bolts and the
handwritten numbers beside each bolt as the training runs). The pixel sizes are a downscale proxy for 50 cm, 1 m and 2 m, not real far shots.
Ultralytics YOLO is AGPL-3.0: the licence question for Alan is open.
`data/` (111 MB), logs and `*.pt` weights are git-ignored and stay on the PC.
