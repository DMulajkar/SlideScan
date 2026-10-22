"""SlideScan Phase 2, method 3: free COLMAP reconstruction of a board sweep -> methods/colmap/{candidates,report}.json

  python methods/colmap.py <run> [--h-min 2]

Needs `python slidescan.py frames <run>` first (frames/ and detections.json) and `pip install pycolmap`. The frames go
through pycolmap (SIFT, exhaustive matching, incremental mapping). The result has arbitrary scale and pose, so the
markers' triangulated corners are fitted to their ruler-measured board positions (similarity transform): every point
is then in board mm, z = height above the marker plane. It answers four things, each printed and saved in report.json:
  does it reconstruct   frames registered, points, reprojection error
  accuracy floor        marker corners after the fit: how far from where the ruler put them (xy and z, mm)
  plane roughness       z of reconstructed points away from targets and markers (a bump must stand out of this)
  per target            points within 6 mm of each bolt and their height, next to measured_mm
Blind candidates: 3+ points within 6 mm of each other, median height at least --h-min mm above the plane, one per
10 mm; score and est_height_mm = that median. Per target, max z is outlier-prone; median z is the better height.
"""
import argparse
import json
import math
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import pycolmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # geometry.py lives one folder up
import geometry as g  # noqa: E402

H_MIN_MM = 2.0     # ponytail: starting guess; set from the plane roughness this prints once a real run exists
NEAR_MM = 6        # points this close to a bolt's center (xy) count as on it; head radius is about 3-5 mm
MIN_PTS = 3        # points a candidate needs
MATCH_MM = 10      # merge radius for candidates (= score.py radius)
CELL_MM = 10       # coverage grid


def reconstruct(run, work):
    """The biggest reconstruction pycolmap finds in run/frames, or None."""
    shutil.rmtree(work, ignore_errors=True)  # extract_features appends to an existing database
    (work / "sparse").mkdir(parents=True)
    db, imgs = work / "db.db", Path(run) / "frames"
    pycolmap.extract_features(db, imgs, camera_mode=pycolmap.CameraMode.SINGLE)  # one phone, one lens
    pycolmap.match_exhaustive(db)  # ponytail: O(n^2) pairs; fine for ~100 frames, use match_sequential beyond that
    recs = pycolmap.incremental_mapping(db, imgs, work / "sparse")
    return max(recs.values(), key=lambda r: r.num_reg_images(), default=None)


def pose(im):
    """3x4 camera-from-world matrix of a registered image."""
    return np.asarray(im.cam_from_world().matrix())


def corner_points(rec, dets, markers, s):
    """({(marker id, corner index): reconstruction xyz}, matching board mm xy), DLT over every registered image that saw it."""
    rows = {}
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        P = pose(im)
        for mid, c in dets.get(f"frames/{im.name}", {}).items():
            if int(mid) not in markers:
                continue
            for k, uv in enumerate(c):  # +0.5: OpenCV pixel centers are integers, COLMAP's are half-integers
                x, y = im.camera.cam_from_img(np.asarray(uv, float) + 0.5)
                rows.setdefault((int(mid), k), []).extend([x * P[2] - P[0], y * P[2] - P[1]])
    X, B = [], []
    for (mid, k), r in sorted(rows.items()):
        if len(r) < 4:  # one view cannot triangulate
            continue
        v = np.linalg.svd(np.array(r))[2][-1]
        X.append(v[:3] / v[3])
        B.append(g.corners_mm(*markers[mid], s)[k])
    return np.array(X), np.array(B)


def fit_board(X, B, cams):
    """Similarity from reconstruction coordinates to board mm (x, y, z up from the marker plane), and its inverse.
    X (n,3) marker corners, B (n,2) their board mm, cams (m,3) camera centers (+z faces them, so no mirror image)."""
    c = X.mean(0)
    n = np.linalg.svd(X - c)[2][2]
    if (cams.mean(0) - c) @ n < 0:
        n = -n
    e1 = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 else [0, 1, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)  # (e1, e2, n) right-handed, like (board x, y, up)
    P = np.c_[(X - c) @ e1, (X - c) @ e2]
    mp, mb = P.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((B - mb).T @ (P - mp))
    D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R, scale = U @ D @ Vt, np.trace(np.diag(S) @ D) / ((P - mp) ** 2).sum()

    def to_board(Y):
        Q = np.atleast_2d(Y) - c
        return np.c_[scale * (np.c_[Q @ e1, Q @ e2] - mp) @ R.T + mb, scale * Q @ n]

    def to_world(Bxyz):
        Bxyz = np.atleast_2d(Bxyz)
        ab = ((Bxyz[:, :2] - mb) / scale) @ R + mp
        return c + np.outer(ab[:, 0], e1) + np.outer(ab[:, 1], e2) + np.outer(Bxyz[:, 2] / scale, n)

    return to_board, to_world


def visible(rec, to_world, pts_xy, margin_px=20):
    """How many registered images see each board point (x, y, 0); pinhole approximation, ignores lens distortion."""
    W = to_world(np.c_[pts_xy, np.zeros(len(pts_xy))])
    count = np.zeros(len(W), int)
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        Xc = W @ pose(im)[:, :3].T + pose(im)[:, 3]
        uv = (Xc[:, :2] / np.maximum(Xc[:, 2:], 1e-9)) @ np.diag([im.camera.focal_length_x, im.camera.focal_length_y]) \
            + [im.camera.principal_point_x, im.camera.principal_point_y]
        count += (Xc[:, 2] > 0) & (uv >= margin_px).all(1) & (uv[:, 0] <= im.camera.width - margin_px) \
            & (uv[:, 1] <= im.camera.height - margin_px)
    return count


def analyse(rec, board, dets, h_min=H_MIN_MM):
    """(candidates.json dict, report dict) from a reconstruction and the run's board + marker detections."""
    s, bw, bh = board["marker_mm"], *board["board_size_mm"]
    markers = {m["id"]: (m["x_mm"], m["y_mm"]) for m in board["markers"]}
    ids = rec.reg_image_ids()
    cams = np.array([rec.images[i].projection_center() for i in ids])
    X, B = corner_points(rec, dets, markers, s)
    if len(X) < 8:
        sys.exit(f"only {len(X)} marker corners triangulated; need at least 2 markers seen from 2 registered images")
    to_board, to_world = fit_board(X, B, cams)
    res = to_board(X) - np.c_[B, np.zeros(len(B))]

    pts = np.array([p.xyz for p in rec.points3D.values() if p.track.length() >= 3]).reshape(-1, 3)
    Y = to_board(pts) if len(pts) else np.empty((0, 3))
    Y = Y[(Y[:, 0] > -10) & (Y[:, 0] < bw + 10) & (Y[:, 1] > -10) & (Y[:, 1] < bh + 10) & (abs(Y[:, 2]) < 60)]
    half = s / 2 + 3
    on_marker = np.zeros(len(Y), bool)
    for mx, my in markers.values():
        on_marker |= (abs(Y[:, 0] - mx) < half) & (abs(Y[:, 1] - my) < half)
    tgt = np.array([(t["x_mm"], t["y_mm"]) for t in board["targets"]])
    near = (np.linalg.norm(Y[:, None, :2] - tgt[None], axis=2) < 3 * NEAR_MM).any(1) if len(Y) else on_marker
    bg = Y[~on_marker & ~near, 2]
    bg_med = float(np.median(bg)) if len(bg) else None
    bg_sigma = float(1.4826 * np.median(abs(bg - bg_med))) if len(bg) else None

    targets = {}
    for t in board["targets"]:
        d = np.linalg.norm(Y[:, :2] - [t["x_mm"], t["y_mm"]], axis=1) if len(Y) else np.empty(0)
        z = Y[d < NEAR_MM, 2]
        targets[t["id"]] = {"measured_mm": t["measured_mm"], "points": len(z),
                            "max_z_mm": round(float(z.max()), 2) if len(z) else None,
                            "median_z_mm": round(float(np.median(z)), 2) if len(z) else None}

    # a candidate needs MIN_PTS points above h_min/2 within NEAR_MM: one stray sparse point is an outlier, not a bump
    hi = Y[~on_marker & (Y[:, 2] >= h_min / 2)]
    cand = []
    for p in hi[np.argsort(-hi[:, 2])]:
        if p[2] < h_min:
            break
        grp = hi[np.linalg.norm(hi[:, :2] - p[:2], axis=1) < NEAR_MM]
        z, (x, y) = float(np.median(grp[:, 2])), grp[:, :2].mean(0)
        if len(grp) >= MIN_PTS and z >= h_min and all(math.dist((x, y), (k["x_mm"], k["y_mm"])) > MATCH_MM for k in cand):
            cand.append({"x_mm": round(float(x), 2), "y_mm": round(float(y), 2), "score": round(z, 2),
                         "est_height_mm": round(z, 2), "image": "colmap"})
    gx, gy = np.meshgrid(np.arange(CELL_MM / 2, bw, CELL_MM), np.arange(CELL_MM / 2, bh, CELL_MM))
    seen_cells = visible(rec, to_world, np.c_[gx.ravel(), gy.ravel()]) >= 2
    seen_t = visible(rec, to_world, tgt) >= 2
    area = round(float(seen_cells.sum()) * CELL_MM ** 2 / 100, 1)
    cands = {"candidates": cand, "covered_targets": [t["id"] for t, v in zip(board["targets"], seen_t) if v],
             "covered_area_cm2": area}
    mean_err = rec.compute_mean_reprojection_error()
    report = {"images_registered": len(ids), "images_total": None, "points": rec.num_points3D(),
              "mean_reprojection_px": round(float(mean_err), 3),
              "marker_corners": len(X), "marker_corner_rms_xy_mm": round(float(np.sqrt((res[:, :2] ** 2).sum(1).mean())), 3),
              "marker_corner_max_xy_mm": round(float(np.linalg.norm(res[:, :2], axis=1).max()), 3),
              "marker_corner_rms_z_mm": round(float(np.sqrt((res[:, 2] ** 2).mean())), 3),
              "plane_points": len(bg), "plane_median_z_mm": bg_med and round(bg_med, 3),
              "plane_sigma_z_mm": bg_sigma and round(bg_sigma, 3), "targets": targets, "h_min_mm": h_min}
    return cands, report


def run(path, h_min=H_MIN_MM):
    path = Path(path)
    if not (path / "detections.json").exists():
        sys.exit(f"{path}: no detections.json; run: python slidescan.py frames {path}")
    board = json.loads((path / "targets.json").read_text())
    dets = json.loads((path / "detections.json").read_text())["images"]
    out = path / "methods" / "colmap"
    out.mkdir(parents=True, exist_ok=True)
    total = len(list((path / "frames").glob("*.jpg")))
    rec = reconstruct(path, out / "work")
    if rec is None:
        print(f"{path}: COLMAP could not reconstruct anything from {total} frames (too few matching features?)")
        sys.exit(1)
    cands, rep = analyse(rec, board, dets, h_min)
    rep["images_total"] = total
    (out / "candidates.json").write_text(json.dumps(cands, indent=2))
    (out / "report.json").write_text(json.dumps(rep, indent=2))

    print(f"registered {rep['images_registered']} of {total} frames; {rep['points']} points; "
          f"mean reprojection error {rep['mean_reprojection_px']} px")
    print(f"marker corners ({rep['marker_corners']}) vs ruler positions after the fit: rms {rep['marker_corner_rms_xy_mm']} mm "
          f"in plane (worst {rep['marker_corner_max_xy_mm']}), {rep['marker_corner_rms_z_mm']} mm out of plane")
    print(f"plane roughness: {rep['plane_points']} points, median z {rep['plane_median_z_mm']} mm, sigma {rep['plane_sigma_z_mm']} mm")
    print("per target (points within 6 mm):  id  measured  points  max z  median z")
    for i, v in rep["targets"].items():
        print(f"{'':34}{i:>4}  {v['measured_mm']:>8g}  {v['points']:>6}  {v['max_z_mm']!s:>5}  {v['median_z_mm']!s:>8}")
    print(f"{len(cands['candidates'])} candidates at {h_min:g} mm or more; covered targets {len(cands['covered_targets'])} "
          f"of {len(board['targets'])}, {cands['covered_area_cm2']} cm2 of board seen from 2+ frames")
    return cands, rep


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--h-min", type=float, default=H_MIN_MM, help="lowest height that counts as a candidate, mm")
    args = p.parse_args()
    run(args.run, args.h_min)


if __name__ == "__main__":
    main()
