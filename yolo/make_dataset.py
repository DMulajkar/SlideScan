"""Build the stage-1 bolt dataset from walkaround runs: every registered frame gets a box per bolt, computed from the
board layout (targets.json) and that frame's pose, then the frame is saved at three pixel sizes.

  python yolo/make_dataset.py [--stride 3] [--out yolo/data] [runs ...]      (WSL venv with pycolmap, run from SlideScan/)

Distance proxy: a frame shrunk to W px wide has the bolt the same pixel size as a 4K camera at 25 * 3840 / W cm
(1920 px ~ 50 cm, 960 ~ 1 m, 480 ~ 2 m). It keeps the same view, so it tests pixel size only, not real optics, lighting
or the wider field of view of a real far shot. Each shrunk frame is pasted, centred, on a 1920 x 1088 grey canvas.

Output: data/images/<run>_w<W>/<frame>.jpg, data/labels/<run>_w<W>/<frame>.txt (one class, 0 = bolt),
data/lists/<run>_w<W>.txt (image paths), data/manifest.json, data/qa_boxes.jpg (sample frames with their boxes).
A box is only as good as the board layout (ruler-placed, a few mm) and the tag-fit pose (about 1.5 mm); look at
qa_boxes.jpg before trusting a run. Hands and shadows covering a bolt are not detected, so a few boxes are on a hidden bolt.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "methods"))
import labels as lb          # noqa: E402
import replay as rp          # noqa: E402  (tag fit and poses)
import walkaround as wk      # noqa: E402  (camera model)

RUNS = ["2026-10-29_walkaround-a", "2026-11-11_walkaround-b2", "2026-11-11_walkaround-tex1", "2026-11-11_walkaround-tex2"]
WIDTHS = (1920, 960, 480)
SIZE_MM = 9.0                # bolt head across the flats, about; boxes are drawn from a column this wide


def head_height(t):
    return max(float(t["measured_mm"]), 1.5)     # a flush bolt still has a head


def run_boxes(run, stride):
    """[(frame name, [(target id, full-res pixel box), ...]), ...] for every stride-th registered frame."""
    g = rp.load_geometry(run, points=False)
    out = []
    for n in g["names"][::stride]:
        K, dist = wk.cv_camera(g["cams"][g["cam_of"][n]], 1.0)
        cam = g["cams"][g["cam_of"][n]]
        Rb, tb = g["pose_b"][n]
        boxes = []
        for t in g["targets"]:
            b = lb.project_box(K, dist, Rb, tb, (t["x"], t["y"]), head_height(t), half=SIZE_MM / 2,
                               img_wh=(cam["width"], cam["height"]))
            if b is not None:
                boxes.append((t["id"], b))
        out.append((n, boxes, g["img_dir"]))
    return out


def write_run(run, items, out, widths=WIDTHS):
    name = Path(run).name.replace("2026-", "").replace("_walkaround-", "_")
    man = []
    for W in widths:
        tag = f"{name}_w{W}"
        (out / "images" / tag).mkdir(parents=True, exist_ok=True)
        (out / "labels" / tag).mkdir(parents=True, exist_ok=True)
        paths = []
        for n, boxes, img_dir in items:
            img = cv2.imread(str(img_dir / n))
            if img.shape[0] > img.shape[1]:                      # portrait frame (run A): turn it to landscape
                boxes = [(i, lb.rotate_box_cw(b, img.shape[0])) for i, b in boxes]
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h0, w0 = img.shape[:2]
            scale = W / w0
            small = cv2.resize(img, (W, round(h0 * scale)), interpolation=cv2.INTER_AREA)
            canvas = np.full((lb.CANVAS[1], lb.CANVAS[0], 3), 114, np.uint8)
            ox, oy = lb.paste_offset((w0, h0), scale)
            canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
            stem = Path(n).stem
            p = out / "images" / tag / f"{stem}.jpg"
            cv2.imwrite(str(p), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            lines = [f"0 {c[0]:.6f} {c[1]:.6f} {c[2]:.6f} {c[3]:.6f}"
                     for c in (lb.to_yolo(b, scale, (ox, oy)) for _, b in boxes)]
            (out / "labels" / tag / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            paths.append(str(p))
            man.append({"run": name, "width": W, "image": f"{tag}/{stem}.jpg", "targets": [i for i, _ in boxes]})
        (out / "lists").mkdir(exist_ok=True)
        (out / "lists" / f"{tag}.txt").write_text("\n".join(paths) + "\n")
    return man


def qa_sheet(out, run_items, k=3):
    """Sample frames with their boxes and target ids, one row per run, so wrong boxes are visible."""
    rows = []
    for run, items in run_items:
        tiles = []
        for n, boxes, img_dir in [items[i] for i in np.linspace(0, len(items) - 1, k).round().astype(int)]:
            img = cv2.imread(str(img_dir / n), cv2.IMREAD_REDUCED_COLOR_4)
            for tid, b in boxes:
                x0, y0, x1, y1 = [int(v / 4) for v in b]
                cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), 2)
                cv2.putText(img, tid, (x0, max(14, y0 - 4)), 0, 0.6, (0, 255, 255), 2)
            if img.shape[0] > img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            tiles.append(cv2.resize(img, (640, 360)))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out / "qa_boxes.jpg"), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folder names under runs/ (default: the four walkaround runs)")
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth registered frame")
    ap.add_argument("--out", default=str(HERE / "data"))
    a = ap.parse_args()
    out = Path(a.out)
    runs_dir = HERE.parent / "runs"
    manifest, run_items = [], []
    for r in (a.runs or RUNS):
        items = run_boxes(runs_dir / r, a.stride)
        n_box = sum(len(b) for _, b, _ in items)
        print(f"{r}: {len(items)} frames, {n_box} boxes ({n_box / max(1, len(items)):.1f} per frame)", flush=True)
        manifest += write_run(r, items, out)
        run_items.append((r, items))
    qa_sheet(out, run_items)
    (out / "manifest.json").write_text(json.dumps(manifest))
    print(f"wrote {len(manifest)} images under {out}")


if __name__ == "__main__":
    main()
