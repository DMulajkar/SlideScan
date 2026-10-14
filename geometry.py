"""SlideScan Phase 2: markers -> board-mm <-> pixel mapping, target coverage, rectified crops.

  python geometry.py <run> [--min-markers 2]  -> geometry.json (per image: homography, markers used, residuals,
                                                 targets in view) + geometry_preview.jpg (board rectified, targets circled)

Board coordinates are mm as in targets.json: origin bottom-left, x right, y up. H maps board mm -> image pixels.
H covers the board plane only: a tall bolt's tip shows parallax (about height x off-axis distance / standoff).
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

# ponytail: starting guess. Real 4-marker residuals were 0.5-1.4 mm (ruler-measured positions);
# a marker glued rotated 90 deg shows about 27+ mm. Tune on more runs.
MAX_ERR_MM = 2.0


def corners_mm(x, y, s):
    """A marker's 4 corners in board mm, in ArUco order (TL, TR, BR, BL).
    Markers are glued with their label edge toward -y, so the marker's top faces +y."""
    h = s / 2
    return np.array([[x - h, y + h], [x + h, y + h], [x + h, y - h], [x - h, y - h]], np.float64)


def to_pixel(H, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float64).reshape(-1, 1, 2), H).reshape(-1, 2)


def to_board(H, pts):
    return to_pixel(np.linalg.inv(H), pts)


def fit(det, markers, s, min_markers=2):
    """det = {id: 4 pixel corners}, markers = {id: (x_mm, y_mm)}, s = marker side mm.
    Returns (H or None, ids used, {id: worst corner residual in board mm})."""
    det = {int(k): np.asarray(v, np.float64) for k, v in det.items()}
    ids = sorted(i for i in det if i in markers)
    if len(ids) < min_markers:
        return None, ids, {}
    known = {i: corners_mm(*markers[i], s) for i in ids}
    # ponytail: plain least squares, no RANSAC. Outliers are whole markers and 2-4 markers can't outvote one;
    # a bad marker shows up in the residuals and gets fixed in board.json.
    H, _ = cv2.findHomography(np.vstack([known[i] for i in ids]), np.vstack([det[i] for i in ids]), 0)
    if H is None:
        return None, ids, {}
    return H, ids, {i: float(np.linalg.norm(to_board(H, det[i]) - known[i], axis=1).max()) for i in ids}


def in_view(H, size, x, y, margin_mm=10):
    """True if the +-margin_mm square around board point (x, y) lands fully inside a (w, h) image."""
    m = margin_mm
    p = to_pixel(H, [[x - m, y - m], [x + m, y - m], [x + m, y + m], [x - m, y + m]])
    return bool(((p >= 0) & (p <= np.subtract(size, 1))).all())


def crop(img, H, x, y, size_mm, px_per_mm):
    """Board-aligned square of size_mm centered on board point (x, y), at px_per_mm, +y up.
    Every crop has the same mm per pixel, so a 3 mm bolt is the same size in every image."""
    n, k = round(size_mm * px_per_mm), 1 / px_per_mm
    # crop pixel (u, v) -> board mm; pixel centers sit half a pixel in, and v grows downward so y flips
    A = np.array([[k, 0, x - size_mm / 2 + k / 2], [0, -k, y + size_mm / 2 - k / 2], [0, 0, 1]])
    return cv2.warpPerspective(img, H @ A, (n, n), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)


def preview(img, H, board, s, label, ppm=4):
    """Everything rectified, each target's measured position circled: bolts inside circles = mapping agrees."""
    xy = np.array([(e["x_mm"], e["y_mm"]) for e in board["markers"] + board["targets"]])
    (cx, cy), size = (xy.min(0) + xy.max(0)) / 2, float((xy.max(0) - xy.min(0)).max()) + 60
    out = crop(img, H, cx, cy, size, ppm)

    def px(x, y):  # board mm -> preview pixel
        return round((x - cx + size / 2) * ppm - 0.5), round((cy + size / 2 - y) * ppm - 0.5)

    red, blue = (0, 0, 255), (255, 0, 0)
    for m in board["markers"]:
        cv2.polylines(out, [np.array([px(*c) for c in corners_mm(m["x_mm"], m["y_mm"], s)])], True, blue, 2)
    for t in board["targets"]:
        u, v = px(t["x_mm"], t["y_mm"])
        cv2.circle(out, (u, v), 5 * ppm, red, 2)  # 5 mm radius; a 1/4 in shaft is 3.2 mm radius
        cv2.drawMarker(out, (u, v), red, cv2.MARKER_CROSS, 2 * ppm, 1)
        cv2.putText(out, f"{t['id']} {t.get('measured_mm')} mm", (u + 6 * ppm, v - 2 * ppm),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, red, 2)
    cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, red, 2)
    return out


def run(path, min_markers=2):
    path = Path(path)
    board = json.loads((path / "targets.json").read_text())
    s = board.get("marker_mm")
    vals = [s] + [e.get(k) for e in board["markers"] + board["targets"] for k in ("x_mm", "y_mm")]
    if not all(isinstance(v, (int, float)) for v in vals):
        sys.exit(f"{path}: marker_mm or a marker/target position is missing; run: python slidescan.py check {path}")
    if not (path / "detections.json").exists():
        sys.exit(f"{path}: no detections.json; run: python slidescan.py frames {path}")
    markers = {m["id"]: (m["x_mm"], m["y_mm"]) for m in board["markers"]}
    dets = json.loads((path / "detections.json").read_text())["images"]

    out = {}
    for name, det in dets.items():
        H, ids, err = fit(det, markers, s, min_markers)
        rec = {"size": None, "H": None, "markers": ids, "err_mm": {str(i): round(e, 2) for i, e in err.items()},
               "covered": []}
        if H is not None:
            h, w = cv2.imread(str(path / name)).shape[:2]  # imread, not a header read: it applies EXIF rotation like detect did
            rec.update(size=[w, h], H=H.tolist(),
                       covered=[t["id"] for t in board["targets"] if in_view(H, (w, h), t["x_mm"], t["y_mm"])])
        out[name] = rec
    (path / "geometry.json").write_text(json.dumps({"min_markers": min_markers, "max_err_mm": MAX_ERR_MM, "images": out}))

    mapped = {k: v for k, v in out.items() if v["H"]}
    print(f"{len(mapped)} of {len(out)} images mapped (at least {min_markers} markers)")
    for i in sorted(markers):
        e = [v["err_mm"][str(i)] for v in mapped.values() if str(i) in v["err_mm"]]
        if e:
            med = statistics.median(e)
            flag = f"  <- over {MAX_ERR_MM} mm: misplaced, rotated or mis-measured? check board.json" if med > MAX_ERR_MM else ""
            print(f"marker {i}: median residual {med:.2f} mm, worst {max(e):.2f} mm, in {len(e)} images{flag}")
    seen = {t for v in mapped.values() for t in v["covered"]}
    never = [t["id"] for t in board["targets"] if t["id"] not in seen]
    print(f"targets covered: {len(seen)} of {len(board['targets'])}" + (f"; never fully in view: {never}" if never else ""))

    if mapped:
        best = max(mapped, key=lambda k: (len(mapped[k]["markers"]), len(mapped[k]["covered"]),
                                          -max(mapped[k]["err_mm"].values())))
        H = np.array(mapped[best]["H"])
        label = f"{best}  markers {mapped[best]['markers']}"
        cv2.imwrite(str(path / "geometry_preview.jpg"), preview(cv2.imread(str(path / best)), H, board, s, label))
        print(f"preview: {path / 'geometry_preview.jpg'} (from {best})")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--min-markers", type=int, default=2,
                   help="fewest markers for a mapping; 1 marker extrapolated up to 28 mm off on run B")
    args = p.parse_args()
    run(args.run, args.min_markers)


if __name__ == "__main__":
    main()
