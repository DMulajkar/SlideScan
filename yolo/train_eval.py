"""Stage-1 bolt detector, leave-one-run-out: train on three runs, test on the fourth.

  python yolo/train_eval.py --fold N [--epochs 25] [--model yolo11n.pt] [--data DIR]    (WSL, ~/yolo/.venv, GPU)

Folds are by run (A, B2, tex1, tex2), never by frame: neighbouring frames of one video are near duplicates, so a frame split
would only measure memory. Each fold trains on all three pixel sizes (50 cm, 1 m and 2 m proxies, see make_dataset.py) and
tests the held-out run at each. Training uses the last epoch (val=False), not a 'best' epoch picked on the test run.
'Found' = the centre of a proposal lies inside the true bolt box (a stage-1 proposal only has to land on the bolt); IoU >= 0.5
is reported too. Reported at confidence 0.05 (the permissive setting stage 1 needs), 0.25 and 0.5, per pixel size and per
target (T05-T08 are the 3.2 mm and taller bolts). False proposals are per image. Writes yolo/results/fold<N>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import labels as lb  # noqa: E402

RUNS = ["10-29_a", "11-11_b2", "11-11_tex1", "11-11_tex2"]
WIDTHS = (1920, 960, 480)
CONFS = (0.05, 0.25, 0.5)
IMGSZ = lb.CANVAS[0]


def lists(data):
    """Image paths per (run, width), rebased onto `data` so a copy of the dataset on another disk works."""
    def rebase(ln):
        p = Path(ln)
        return str(data / "images" / p.parts[-2] / p.name)
    return {(r, w): [rebase(ln) for ln in (data / "lists" / f"{r}_w{w}.txt").read_text().split("\n") if ln]
            for r in RUNS for w in WIDTHS}


def prepare(data, fold):
    L = lists(data)
    test = RUNS[fold]
    train = [p for r in RUNS if r != test for w in WIDTHS for p in L[(r, w)]]
    (data / "folds").mkdir(exist_ok=True)
    (data / "folds" / f"fold{fold}_train.txt").write_text("\n".join(train) + "\n")
    (data / "folds" / f"fold{fold}_val.txt").write_text("\n".join(L[(test, 960)]) + "\n")   # unused for selection, YOLO wants one
    yaml = data / "folds" / f"fold{fold}.yaml"
    yaml.write_text(f"path: {data}\ntrain: {data / 'folds' / f'fold{fold}_train.txt'}\n"
                    f"val: {data / 'folds' / f'fold{fold}_val.txt'}\nnames:\n  0: bolt\n")
    return yaml, L, test


def read_gt(img_path):
    lab = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    rows = [tuple(map(float, ln.split()[1:])) for ln in lab.read_text().split("\n") if ln.strip()]
    return [lb.from_yolo(r) for r in rows]


def evaluate(model, data, L, test, manifest):
    tgt = {m["image"]: m["targets"] for m in manifest}
    res = {}
    for w in WIDTHS:
        paths = L[(test, w)]
        pred = {}
        for i in range(0, len(paths), 16):
            for p, r in zip(paths[i:i + 16], model.predict(paths[i:i + 16], imgsz=IMGSZ, conf=0.05, iou=0.5, max_det=100,
                                                         verbose=False, device=0)):
                b, c = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                pred[p] = [(*bb, cc) for bb, cc in zip(b.tolist(), c.tolist())]
        for conf in CONFS:
            tot = found = found_iou = false = props = 0
            per = {}
            for p in paths:
                gts = read_gt(p)
                ids = tgt["/".join(Path(p).parts[-2:])]
                pr = [q for q in pred[p] if q[4] >= conf]
                f_c, fp = lb.match(pr, gts, lb.centre_in)
                f_i, _ = lb.match(pr, gts, lambda a, b: lb.iou(a, b) >= 0.5)
                tot += len(gts)
                found += len(f_c)
                found_iou += len(f_i)
                false += fp
                props += len(pr)
                for j, tid in enumerate(ids):
                    d = per.setdefault(tid, [0, 0])
                    d[0] += j in f_c
                    d[1] += 1
            res[f"w{w}|conf{conf}"] = {"images": len(paths), "gt": tot, "found_centre": found, "found_iou50": found_iou,
                                       "false_per_image": round(false / len(paths), 2), "proposals_per_image": round(props / len(paths), 2),
                                       "per_target": {k: f"{v[0]}/{v[1]}" for k, v in sorted(per.items())}}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fold", type=int, required=True, help="index of the held-out run in " + str(RUNS))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    from ultralytics import YOLO
    data = Path(a.data)
    yaml, L, test = prepare(data, a.fold)
    out = Path.home() / "yolo" / "runs"
    weights = out / f"fold{a.fold}" / "weights" / "last.pt"
    if not a.eval_only:
        YOLO(a.model).train(data=str(yaml), imgsz=IMGSZ, epochs=a.epochs, batch=a.batch, device=0, workers=4, val=False,
                            project=str(out), name=f"fold{a.fold}", exist_ok=True, scale=0.25, flipud=0.5, fliplr=0.5,
                            close_mosaic=3, plots=False, seed=0, patience=1000)
    manifest = json.loads((data / "manifest.json").read_text())
    res = evaluate(YOLO(str(weights)), data, L, test, manifest)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"fold{a.fold}.json").write_text(json.dumps({"held_out": test, "epochs": a.epochs, "model": a.model, "results": res}, indent=1))
    print(f"\nfold {a.fold}: held-out run {test}")
    print(f"{'size':>6} {'conf':>5} {'found(centre)':>14} {'found(IoU.5)':>13} {'false/img':>10} {'props/img':>10}  per target")
    for k, v in res.items():
        w, c = k.split("|")
        print(f"{w:>6} {c[4:]:>5} {v['found_centre']:>6}/{v['gt']:<7} {v['found_iou50']:>6}/{v['gt']:<6} {v['false_per_image']:>10} {v['proposals_per_image']:>10}  "
              + " ".join(f"{t}:{x}" for t, x in v["per_target"].items()))


if __name__ == "__main__":
    main()
