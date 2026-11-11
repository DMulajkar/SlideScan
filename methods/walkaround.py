"""walkaround.py - step 2 of the walk-around method: poses + one tag -> bolt heights.

Input : runs/<run>/methods/walkaround/sfm.json  (from walkaround_sfm.py), the frames,
        the run's targets.json, and board.json (marker layout).
Output: runs/<run>/methods/walkaround/
          scale.json       how the tag fixed the metric scale, and whether it drifted
          heights.json     targeted: height at every known target, with error vs the calipers
          candidates.json  blind: bumps found anywhere on the board (graded by score.py)
          heightmap.png    height-above-local-board map with the targets circled
          heightmap.npz    the raw maps, for plots later
          diagnostics.json settings and score statistics

    python methods/walkaround.py runs/<run> [--mode both|targets|blind] [--board board.json]

How it measures. SfM gives every frame's pose in an arbitrary scale. The printed tag,
seen in a few frames, fixes metric scale and the board plane once (a similarity fit of its
triangulated corners to their known board positions). The tag does not have to be in
every frame. Heights then come from a plane sweep: for a patch of the board at a
hypothesised height h, project it into every frame and compare what the frames see; the
height where the views agree best is the surface height. A bolt head has its own height
and the board around it is read the same way, so a bolt's height is reported
relative to the local board. A scale error scales heights only proportionally.

Conventions (from the Phase 1 plan): board millimetres, +z toward the camera, markers glued
square to the board with their top edge toward +y, so a marker's corners are
TL(cx-s/2, cy+s/2), TR(cx+s/2, cy+s/2), BR(cx+s/2, cy-s/2), BL(cx-s/2, cy-s/2).
Nothing in raw/ is touched. Markers are never detector input: they are blanked in blind mode.
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

try:
    from scipy import ndimage as ndi
except ImportError:  # scipy ships with the anaconda base env used for SlideScan
    ndi = None

OUT_SUB = os.path.join("methods", "walkaround")


# ----------------------------------------------------------------------------- io
def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_board(run_dir, board=None):
    """board.json: --board, the run folder, SlideScan/ (two levels up), or the cwd."""
    cands = [board] if board else [
        os.path.join(run_dir, "board.json"),
        os.path.join(run_dir, "..", "..", "board.json"),
        "board.json",
    ]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("board.json not found; looked in: " + ", ".join(c for c in cands if c)
                     + ". Pass --board PATH.")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _xy(item):
    """Board position of a marker or target: measured x_mm/y_mm first, then the planned one."""
    if _num(item.get("x_mm")) and _num(item.get("y_mm")):
        return float(item["x_mm"]), float(item["y_mm"])
    for k in ("planned_xy_mm", "xy_mm", "center_mm"):
        v = item.get(k)
        if isinstance(v, (list, tuple)) and len(v) >= 2 and _num(v[0]) and _num(v[1]):
            return float(v[0]), float(v[1])
    return None


def load_markers(board, marker_mm=None):
    """-> ({marker_id: (cx, cy)}, size_mm). Tolerates list or dict layouts in board.json."""
    items = board.get("markers", [])
    if isinstance(items, dict):
        items = [dict(v, id=int(k)) if isinstance(v, dict) and "id" not in v else v
                 for k, v in items.items()]
    size = marker_mm or board.get("marker_mm") or board.get("marker_size_mm") or 40.0
    out = {}
    for m in items:
        if not isinstance(m, dict) or "id" not in m:
            continue
        xy = _xy(m)
        if xy is not None:
            out[int(m["id"])] = xy
    if not out:
        keys = sorted(board.keys())
        raise SystemExit("no markers with positions found in board.json (top-level keys: "
                         f"{keys}). Expected a 'markers' list of {{id, x_mm, y_mm}}.")
    return out, float(size)


def load_targets(run_dir):
    p = os.path.join(run_dir, "targets.json")
    if not os.path.isfile(p):
        return []
    d = load_json(p)
    items = d.get("targets", d) if isinstance(d, dict) else d
    if isinstance(items, dict):
        items = [dict(v, id=k) if isinstance(v, dict) else v for k, v in items.items()]
    out = []
    for t in items:
        if not isinstance(t, dict):
            continue
        tid = t.get("id", t.get("target_id", t.get("name")))
        xy = _xy(t)
        if tid is None or xy is None:
            continue
        meas = t.get("measured_mm")
        out.append({"id": str(tid), "x": xy[0], "y": xy[1],
                    "measured_mm": float(meas) if _num(meas) else None,
                    "decoy": str(tid).startswith("D")})
    return out


# -------------------------------------------------------------------------- camera
def cv_camera(cam, scale=1.0):
    """COLMAP camera -> OpenCV (K, dist) at an image scale. COLMAP puts the first pixel centre
    at 0.5, OpenCV at 0; that half pixel matters at the sub-pixel accuracy this needs."""
    m, p = cam["model"], cam["params"]
    if m == "SIMPLE_PINHOLE":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = [0, 0, 0, 0]
    elif m == "PINHOLE":
        fx, fy, cx, cy = p[:4]; d = [0, 0, 0, 0]
    elif m == "SIMPLE_RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = [p[3], 0, 0, 0]
    elif m == "RADIAL":
        fx = fy = p[0]; cx, cy = p[1], p[2]; d = [p[3], p[4], 0, 0]
    elif m == "OPENCV":
        fx, fy, cx, cy = p[:4]; d = [p[4], p[5], p[6], p[7]]
    else:
        raise SystemExit(f"camera model {m} not supported; rerun walkaround_sfm.py with --camera-model SIMPLE_RADIAL")
    K = np.array([[fx * scale, 0, cx * scale - 0.5], [0, fy * scale, cy * scale - 0.5], [0, 0, 1]], float)
    return K, np.array(d, float)


class View:
    """One frame prepared for the plane sweep: filtered grey image plus its pose in board mm."""

    def __init__(self, name, img, K, dist, R, t):
        self.name, self.img, self.K, self.dist = name, img, K, dist
        self.R, self.t = np.asarray(R, float), np.asarray(t, float).reshape(3)
        self.rvec = cv2.Rodrigues(self.R)[0]
        self.h, self.w = img.shape[:2]


def prep_image(gray, hp_sigma=6.0):
    """Band-pass the grey image: removes lighting gradients, keeps texture. float32, mean ~0."""
    g = gray.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 0.8) - cv2.GaussianBlur(g, (0, 0), hp_sigma)


def read_gray(path, max_width):
    """-> (grey uint8, scale). EXIF rotation is ignored because COLMAP ignores it too."""
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
    if g is None:
        raise SystemExit(f"cannot read {path}")
    scale = 1.0
    if max_width and g.shape[1] > max_width:
        scale = max_width / g.shape[1]
        g = cv2.resize(g, (max_width, int(round(g.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    return g, scale


# ----------------------------------------------------------------------------- tag
def make_detector(dict_name="DICT_4X4_50"):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(d, p)


def detect_tags(detector, gray):
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).astype(float) for i, c in zip(ids.ravel(), corners)}


def marker_corners_board(cx, cy, s):
    h = s / 2.0
    return np.array([[cx - h, cy + h, 0], [cx + h, cy + h, 0], [cx + h, cy - h, 0], [cx - h, cy - h, 0]], float)


def triangulate(Ps, xs):
    """DLT from normalised image points. Ps: (3,4) cam_from_world each; xs: (x, y) each."""
    A = []
    for P, (x, y) in zip(Ps, xs):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def umeyama(src, dst):
    """Least-squares similarity dst ~ s R src + T (proper rotation). -> s, R, T, rms."""
    n = len(src)
    ms, md = src.mean(0), dst.mean(0)
    X, Y = src - ms, dst - md
    U, D, Vt = np.linalg.svd(Y.T @ X / n)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / ((X ** 2).sum() / n))
    T = md - s * R @ ms
    res = dst - (s * (R @ src.T).T + T)
    return s, R, T, float(np.sqrt((res ** 2).sum(1).mean()))


def fit_similarity(obs, poses, cams, markers, msize, names, min_angle_deg=1.5):
    """Triangulate every tag corner seen in >=2 of `names`, then fit them to board mm.

    obs[name] = {marker_id: (4,2) corners in OpenCV pixels}; poses[name] = (R, t) cam_from_world
    in the SfM frame; cams[name] = (K, dist). Returns a dict or None if too few corners."""
    pts_src, pts_dst, info = [], [], []
    for mid, (cx, cy) in markers.items():
        board_c = marker_corners_board(cx, cy, msize)
        for ci in range(4):
            Ps, xs, cen = [], [], []
            for n in names:
                c = obs.get(n, {}).get(mid)
                if c is None:
                    continue
                K, dist = cams[n]
                x = cv2.undistortPoints(c[ci].reshape(1, 1, 2), K, dist).reshape(2)
                R, t = poses[n]
                Ps.append(np.hstack([R, t.reshape(3, 1)]))
                xs.append(x)
                cen.append(-R.T @ t)
            if len(Ps) < 2:
                continue
            X = triangulate(Ps, xs)
            rays = [(X - c) / np.linalg.norm(X - c) for c in cen]
            ang = max(math.degrees(math.acos(np.clip(a @ b, -1, 1))) for a in rays for b in rays)
            if ang < min_angle_deg:
                info.append({"marker": mid, "corner": ci, "dropped": f"ray angle {ang:.2f} deg < {min_angle_deg}"})
                continue
            pts_src.append(X)
            pts_dst.append(board_c[ci])
            info.append({"marker": mid, "corner": ci, "views": len(Ps), "ray_angle_deg": round(ang, 2)})
    if len(pts_src) < 4:
        return {"ok": False, "n_points": len(pts_src), "info": info}
    src, dst = np.array(pts_src), np.array(pts_dst)
    s, R, T, rms = umeyama(src, dst)
    fitted = s * (R @ src.T).T + T
    return {"ok": True, "scale_mm_per_unit": s, "R": R, "T": T, "rms_mm": rms,
            "z_rms_mm": float(np.sqrt((fitted[:, 2] ** 2).mean())),
            "n_points": len(src), "info": info}


# ------------------------------------------------------------------- plane sweep
def sample_views(views, P):
    """Intensity of each 3D board point (mm) in each view -> (V, M) float32, NaN when not visible."""
    out = np.full((len(views), len(P)), np.nan, np.float32)
    P = np.ascontiguousarray(P, float)
    for i, v in enumerate(views):
        pix = cv2.projectPoints(P.reshape(-1, 1, 3), v.rvec, v.t, v.K, v.dist)[0].reshape(-1, 2)
        zc = P @ v.R[2] + v.t[2]
        ok = (zc > 1e-3) & (pix[:, 0] >= 1) & (pix[:, 0] <= v.w - 2) & (pix[:, 1] >= 1) & (pix[:, 1] <= v.h - 2)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            continue
        m = idx.size
        rows = -(-m // 2048)  # remap wants 2-D maps; keep every side well under its size limit
        mx = np.zeros(rows * 2048, np.float32)
        my = np.zeros(rows * 2048, np.float32)
        mx[:m] = pix[idx, 0]
        my[:m] = pix[idx, 1]
        r = cv2.remap(v.img, mx.reshape(rows, 2048), my.reshape(rows, 2048), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)
        out[i, idx] = r.reshape(-1)[:m]
    return out


def patch_offsets(patch_mm, patch_n):
    g = np.linspace(-patch_mm / 2.0, patch_mm / 2.0, patch_n)
    ox, oy = np.meshgrid(g, g)
    return np.stack([ox.ravel(), oy.ravel()], 1)


def sweep(views, xy, heights, offs, frac=0.7, min_views=4, flat_std=1.0, budget=12_000_000):
    """Photo-consistency of a horizontal patch at every (x, y) and every hypothesised height.

    For each hypothesis the patch is sampled in every view and normalised; a view's agreement
    is its correlation with the mean of the other views, and the score is the mean of the best
    `frac` of those, so views where something occludes the patch do not wreck it.
    Returns (h_best, score_best, n_views_best, n_seen_at_zero), each of length len(xy)."""
    N, H, S, V = len(xy), len(heights), len(offs), len(views)
    h_best = np.full(N, np.nan)
    s_best = np.full(N, -1.0)
    n_best = np.zeros(N, int)
    n_seen = np.zeros(N, int)
    i0 = int(np.argmin(np.abs(heights)))
    step = max(1, int(budget // max(1, H * S * V)))
    for a in range(0, N, step):
        b = min(N, a + step)
        n = b - a
        P = np.empty((n, H, S, 3))
        P[..., 0] = xy[a:b, 0][:, None, None] + offs[None, None, :, 0]
        P[..., 1] = xy[a:b, 1][:, None, None] + offs[None, None, :, 1]
        P[..., 2] = heights[None, :, None]
        smp = sample_views(views, P.reshape(-1, 3)).reshape(V, n, H, S)
        smp = np.moveaxis(smp, 0, 2)                                   # (n, H, V, S)
        vis = ~np.isnan(smp).any(-1)                                   # patch fully in view
        X = np.nan_to_num(smp)
        X -= X.mean(-1, keepdims=True)
        X *= vis[..., None]
        nrm = np.linalg.norm(X, axis=-1)
        use = vis & (nrm > flat_std * math.sqrt(S))
        U = np.where(use[..., None], X / np.maximum(nrm, 1e-9)[..., None], 0.0).astype(np.float32)
        n_use = use.sum(-1)
        T = U.sum(2)[:, :, None, :] - U                                # leave-one-out template
        tn = np.linalg.norm(T, axis=-1)
        corr = np.where(use & (tn > 1e-6), (U * T).sum(-1) / np.maximum(tn, 1e-9), -np.inf)
        srt = -np.sort(-corr, axis=-1)
        k = np.clip(np.ceil(frac * n_use).astype(int), 1, V)
        cs = np.cumsum(np.where(np.isfinite(srt), srt, 0.0), axis=-1)
        sc = np.take_along_axis(cs, (k - 1)[..., None], -1)[..., 0] / k
        sc = np.where(n_use >= min_views, sc, -1.0)
        bi = sc.argmax(1)
        rng = np.arange(n)
        sb = sc[rng, bi]
        hb = heights[bi].astype(float)
        inner = (bi > 0) & (bi < H - 1)
        lo = sc[rng, np.clip(bi - 1, 0, H - 1)]
        hi = sc[rng, np.clip(bi + 1, 0, H - 1)]
        den = lo - 2 * sb + hi
        ok = inner & (lo > -1) & (hi > -1) & (den < -1e-9)
        d = np.where(ok, 0.5 * (lo - hi) / np.where(ok, den, 1.0), 0.0)
        dh = heights[1] - heights[0] if H > 1 else 0.0
        hb = hb + np.clip(d, -0.5, 0.5) * dh
        h_best[a:b] = np.where(sb > -1, hb, np.nan)
        s_best[a:b] = sb
        n_best[a:b] = n_use[rng, bi]
        n_seen[a:b] = vis[:, i0, :].sum(-1)
    return h_best, s_best, n_best, n_seen


def height_map(views, xs, ys, heights, offs, skip=None, **kw):
    """Sweep over a rectangular grid of cells. skip: bool (len(ys), len(xs)) cells to leave NaN."""
    X, Y = np.meshgrid(xs, ys)
    flat = np.ones(X.size, bool) if skip is None else ~skip.ravel()
    idx = np.flatnonzero(flat)
    xy = np.stack([X.ravel()[idx], Y.ravel()[idx]], 1)
    h, s, n, seen = sweep(views, xy, heights, offs, **kw)
    out = []
    for arr, fill in ((h, np.nan), (s, -1.0), (n, 0), (seen, 0)):
        full = np.full(X.size, fill, float)
        full[idx] = arr
        out.append(full.reshape(X.shape))
    return tuple(out)  # h, score, n_views, n_seen


def seen_count(views, xy, z=2.0):
    """How many views have the point (x, y, z) in front of the camera and inside the image."""
    P = np.column_stack([xy, np.full(len(xy), z)])
    cnt = np.zeros(len(xy), int)
    for v in views:
        pix = cv2.projectPoints(P.reshape(-1, 1, 3), v.rvec, v.t, v.K, v.dist)[0].reshape(-1, 2)
        zc = P @ v.R[2] + v.t[2]
        cnt += ((zc > 1e-3) & (pix[:, 0] >= 8) & (pix[:, 0] <= v.w - 9)
                & (pix[:, 1] >= 8) & (pix[:, 1] <= v.h - 9)).astype(int)
    return cnt


# --------------------------------------------------------------------- estimators
def median3(a, fill):
    return ndi.median_filter(np.where(np.isfinite(a), a, fill), size=3, mode="nearest")


def estimate_target(xs, ys, h, s, tx, ty, min_score, search_r, bg_r):
    """Bolt height = (max of the 3x3-median-smoothed height map within search_r of the target)
    minus the median height of the cells between bg_r and the window edge (the local board)."""
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X - tx, Y - ty)
    valid = np.isfinite(h) & (s >= min_score)
    bg = valid & (r > bg_r)
    if bg.sum() >= 15:
        ref, ref_from = float(np.median(h[bg])), "local_median"
    else:
        ref, ref_from = 0.0, "tag_plane"
    near = valid & (r <= search_r)
    if not near.any():
        return {"est_height_mm": None, "ref_height_mm": ref, "ref_from": ref_from, "n_valid_cells": int(valid.sum())}
    hs = median3(h, ref)
    cand = np.where(near, hs, -np.inf)
    j = np.unravel_index(int(np.argmax(cand)), cand.shape)
    return {"est_height_mm": float(hs[j] - ref), "x_hat_mm": float(X[j]), "y_hat_mm": float(Y[j]),
            "ref_height_mm": ref, "ref_from": ref_from, "peak_score": float(s[j]),
            "n_valid_cells": int(valid.sum())}


def blind_candidates(xs, ys, h, s, skip, grid_mm, min_height, min_score, ref_win_mm, min_cells):
    valid = np.isfinite(h) & (s >= min_score)
    glob = float(np.median(h[valid])) if valid.any() else 0.0
    hh = np.where(valid, h, glob)
    win = max(3, int(round(ref_win_mm / grid_mm)) | 1)
    ref = ndi.median_filter(hh, size=win, mode="nearest")
    # no smoothing here: a bolt head is only 2-3 cells wide at this grid, and a 3x3 median erases it.
    # Isolated single-cell spikes are dropped by the min_cells test below instead.
    rel = np.where(valid, h - ref, 0.0)
    mask = valid & (rel >= min_height)
    if skip is not None:
        mask &= ~skip
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    cands = []
    for k in range(1, n + 1):
        cells = lab == k
        if cells.sum() < min_cells:
            continue
        w = rel[cells]
        yy, xx = np.nonzero(cells)
        wsum = w.sum()
        cands.append({"x_mm": float(np.sum(xs[xx] * w) / wsum), "y_mm": float(np.sum(ys[yy] * w) / wsum),
                      "score": float(s[cells].mean()), "est_height_mm": float(w.max()), "image": None})
    cands.sort(key=lambda c: -c["est_height_mm"])
    return cands, rel


# ------------------------------------------------------------------------ outputs
def draw_map(path, xs, ys, rel, targets, est, markers, msize, vmax=8.0):
    px = max(2, int(round(1000 / max(1, len(xs)))))
    img8 = np.clip(np.nan_to_num(rel, nan=0.0) / vmax * 255, 0, 255).astype(np.uint8)
    col = cv2.applyColorMap(img8, cv2.COLORMAP_TURBO)
    col[~np.isfinite(rel)] = (40, 40, 40)
    col = cv2.resize(col, (len(xs) * px, len(ys) * px), interpolation=cv2.INTER_NEAREST)
    col = col[::-1].copy()                                   # +y up, as on the board
    H = col.shape[0]

    def to_px(x, y):
        return int(round((x - xs[0]) / (xs[1] - xs[0]) * px + px / 2)), \
               H - int(round((y - ys[0]) / (ys[1] - ys[0]) * px + px / 2))

    for mid, (cx, cy) in markers.items():
        a, b = to_px(cx - msize / 2, cy + msize / 2), to_px(cx + msize / 2, cy - msize / 2)
        cv2.rectangle(col, a, b, (255, 255, 255), 1)
    for t in targets:
        c = to_px(t["x"], t["y"])
        cv2.circle(col, c, int(7 / (xs[1] - xs[0]) * px), (255, 255, 255), 1)
        e = est.get(t["id"])
        lab = t["id"] + (f" {e:.1f}" if e is not None else " --")
        if t["measured_mm"] is not None:
            lab += f" ({t['measured_mm']:.1f})"
        cv2.putText(col, lab, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, col)


# -------------------------------------------------------------------------- driver
def run(run_dir, board=None, mode="both", max_views=30, max_width=2400, hp_sigma=6.0,
        h_min=-3.0, h_max=16.0, step_t=0.25, step_b=0.5, grid_mm=3.0, roi_mm=14.0, roi_step=1.0,
        patch_mm=3.0, patch_n=5, frac=0.7, min_score=0.4, min_views=4, cov_views=6,
        min_height=2.0, ref_win_mm=40.0, search_r=7.0, bg_r=9.0, marker_mm=None, marker_margin=3.0,
        dict_name="DICT_4X4_50", max_tag_images=None, max_tag_rms=3.0, min_tag_angle=1.5,
        region=None, margin_mm=25.0, flat_std=1.0, verbose=True, views_override=None):
    say = print if verbose else (lambda *a, **k: None)
    if ndi is None:
        raise SystemExit("scipy is required (pip install scipy)")
    t0 = time.time()
    out_dir = os.path.join(run_dir, OUT_SUB)
    sfm = load_json(os.path.join(out_dir, "sfm.json"))
    markers, msize = load_markers(load_json(find_board(run_dir, board)), marker_mm)
    targets = load_targets(run_dir)
    img_dir = sfm["image_dir"] if os.path.isabs(sfm["image_dir"]) else os.path.join(run_dir, sfm["image_dir"])

    # --- 1. tag: fix metric scale and the board plane --------------------------------
    det = make_detector(dict_name)
    poses_sfm, cams, obs, scales = {}, {}, {}, {}
    for im in sfm["images"]:
        M = np.array(im["cam_from_world"], float)
        poses_sfm[im["name"]] = (M[:, :3], M[:, 3])
    names = sorted(poses_sfm)
    cam_of = {i["name"]: i["camera_id"] for i in sfm["images"]}
    for n in names:
        g, sc = read_gray(os.path.join(img_dir, n), max_width)
        cam = sfm["cameras"][cam_of[n]]
        if abs(g.shape[1] / sc - cam["width"]) > 1 or abs(g.shape[0] / sc - cam["height"]) > 1.5:
            raise SystemExit(f"{n} is {g.shape[1] / sc:.0f}x{g.shape[0] / sc:.0f} but COLMAP's camera is "
                             f"{cam['width']}x{cam['height']}: the frames changed or were rotated since sfm.json was written")
        cams[n] = cv_camera(cam, sc)
        scales[n] = sc
        d = {i: c for i, c in detect_tags(det, g).items() if i in markers}
        if d:
            obs[n] = d
    tagged = sorted(obs)
    say(f"tag seen in {len(tagged)} of {len(names)} registered frames")
    if max_tag_images and len(tagged) > max_tag_images:
        keep = np.linspace(0, len(tagged) - 1, max_tag_images).round().astype(int)
        tagged = [tagged[i] for i in sorted(set(keep))]
        obs = {n: obs[n] for n in tagged}
        say(f"using the tag from only {len(tagged)} frames (--max-tag-images)")
    fit = fit_similarity(obs, poses_sfm, cams, markers, msize, tagged, min_tag_angle)
    if not fit["ok"]:
        raise SystemExit(
            f"the tag fixed only {fit['n_points']} corners (need 4). It must be visible, sharp, in at least "
            "two frames taken from clearly different positions (rays >1.5 degrees apart). Film the tag "
            "at the start and again at the end of the sweep.")
    s, R, T = fit["scale_mm_per_unit"], fit["R"], fit["T"]
    say(f"tag fit: {fit['n_points']} corners, rms {fit['rms_mm']:.2f} mm, scale {s:.4f} mm per SfM unit")
    if fit["rms_mm"] > max_tag_rms:
        raise SystemExit(
            f"the tag does not fit the board layout: rms {fit['rms_mm']:.2f} mm (limit {max_tag_rms}). "
            "Either a marker is glued rotated or off its board.json position, or board.json's y axis is "
            "not 'toward the marker's top edge' (this code assumes it is). Check the marker positions first.")
    centers = np.array([s * (R @ (-poses_sfm[n][0].T @ poses_sfm[n][1])) + T for n in names])
    if np.median(centers[:, 2]) < 0:
        raise SystemExit("the cameras came out below the board: the board frame is mirrored relative to the "
                         "marker corner order assumed here. See the note at the top of walkaround.py.")

    # does the scale agree between the start and the end of the sweep?
    drift = None
    if len(tagged) >= 4:
        half = len(tagged) // 2
        fa = fit_similarity(obs, poses_sfm, cams, markers, msize, tagged[:half], min_tag_angle)
        fb = fit_similarity(obs, poses_sfm, cams, markers, msize, tagged[half:], min_tag_angle)
        if fa["ok"] and fb["ok"]:
            drift = {"scale_first_half": fa["scale_mm_per_unit"], "scale_second_half": fb["scale_mm_per_unit"],
                     "difference_pct": 100.0 * (fb["scale_mm_per_unit"] / fa["scale_mm_per_unit"] - 1.0),
                     "rms_first_mm": fa["rms_mm"], "rms_second_mm": fb["rms_mm"]}
            say(f"scale first half vs second half of the tagged frames: {drift['difference_pct']:+.2f} %")
    scale_report = {"scale_mm_per_unit": s, "rms_mm": fit["rms_mm"], "z_rms_mm": fit["z_rms_mm"],
                    "n_corners": fit["n_points"], "tagged_frames": len(tagged), "registered_frames": len(names),
                    "drift": drift, "corners": fit["info"]}
    with open(os.path.join(out_dir, "scale.json"), "w") as f:
        json.dump(scale_report, f, indent=1)

    # --- 2. views in board millimetres ------------------------------------------------
    if views_override is not None:
        views = views_override
    else:
        pick = np.unique(np.linspace(0, len(names) - 1, min(max_views, len(names))).round().astype(int))
        views = []
        for i in pick:
            n = names[i]
            g, sc = read_gray(os.path.join(img_dir, n), max_width)
            Rc, tc = poses_sfm[n]
            Rb = Rc @ R.T
            tb = s * tc - Rb @ T
            K, dist = cams[n]
            views.append(View(n, prep_image(g, hp_sigma), K, dist, Rb, tb))
    say(f"{len(views)} views prepared ({time.time() - t0:.0f} s)")

    offs = patch_offsets(patch_mm, patch_n)
    kw = dict(frac=frac, min_views=min_views, flat_std=flat_std)
    result = {"targets": [], "candidates": None}
    est_for_map = {}

    # --- 3. targeted: height at every known target ----------------------------------
    if mode in ("both", "targets") and targets:
        heights = np.arange(h_min, h_max + 1e-9, step_t)
        for t in targets:
            xs = np.arange(t["x"] - roi_mm, t["x"] + roi_mm + 1e-9, roi_step)
            ys = np.arange(t["y"] - roi_mm, t["y"] + roi_mm + 1e-9, roi_step)
            h, sc, nv, seen = height_map(views, xs, ys, heights, offs, **kw)
            e = estimate_target(xs, ys, h, sc, t["x"], t["y"], min_score, search_r, bg_r)
            ns = int(seen[len(ys) // 2, len(xs) // 2])
            rec = {"id": t["id"], "x_mm": t["x"], "y_mm": t["y"], "measured_mm": t["measured_mm"],
                   "covered": bool(ns >= cov_views), "n_views_seen": ns, **e}
            if rec["est_height_mm"] is not None and t["measured_mm"] is not None:
                rec["error_mm"] = rec["est_height_mm"] - t["measured_mm"]
            result["targets"].append(rec)
            est_for_map[t["id"]] = rec["est_height_mm"]
        say(f"targeted done ({time.time() - t0:.0f} s)")

    # --- 4. blind: bumps anywhere ------------------------------------------------------
    rel = xs_b = ys_b = None
    if mode in ("both", "blind"):
        if region:
            x0, y0, x1, y1 = region
        else:
            allx = [p[0] for p in markers.values()] + [t["x"] for t in targets]
            ally = [p[1] for p in markers.values()] + [t["y"] for t in targets]
            x0, x1 = min(allx) - margin_mm, max(allx) + margin_mm
            y0, y1 = min(ally) - margin_mm, max(ally) + margin_mm
        xs_b = np.arange(x0, x1 + 1e-9, grid_mm)
        ys_b = np.arange(y0, y1 + 1e-9, grid_mm)
        Xg, Yg = np.meshgrid(xs_b, ys_b)
        pad = msize / 2 + marker_margin
        skip = np.zeros(Xg.shape, bool)
        for cx, cy in markers.values():
            skip |= (np.abs(Xg - cx) <= pad) & (np.abs(Yg - cy) <= pad)
        seen_g = seen_count(views, np.stack([Xg.ravel(), Yg.ravel()], 1)).reshape(Xg.shape)
        active = (seen_g >= cov_views) & ~skip
        heights_b = np.arange(h_min, h_max + 1e-9, step_b)
        h, sc, nv, seen = height_map(views, xs_b, ys_b, heights_b, offs, skip=~active, **kw)
        cands, rel = blind_candidates(xs_b, ys_b, h, sc, skip, grid_mm, min_height, min_score, ref_win_mm, 2)
        cov_ids = []
        for t in targets:
            d = np.hypot(Xg - t["x"], Yg - t["y"]) <= 5.0
            if d.any() and np.median(seen_g[d]) >= cov_views:
                cov_ids.append(t["id"])
        result["candidates"] = {"candidates": cands, "covered_targets": cov_ids,
                                "covered_area_cm2": float(active.sum() * grid_mm ** 2 / 100.0)}
        np.savez_compressed(os.path.join(out_dir, "heightmap.npz"), xs=xs_b, ys=ys_b, h=h, score=sc,
                            n_views=nv, rel=rel, active=active)
        say(f"blind done: {len(cands)} candidate(s), {active.sum()} cells scored ({time.time() - t0:.0f} s)")
        with open(os.path.join(out_dir, "candidates.json"), "w") as f:
            json.dump(result["candidates"], f, indent=1)
        draw_map(os.path.join(out_dir, "heightmap.png"), xs_b, ys_b, np.where(active, rel, np.nan),
                 targets, est_for_map, markers, msize)

    # --- 5. report ---------------------------------------------------------------------
    if result["targets"]:
        errs = [r["error_mm"] for r in result["targets"] if "error_mm" in r and r["covered"]]
        summary = {"n_targets": len(result["targets"]), "n_with_estimate": sum(r["est_height_mm"] is not None for r in result["targets"]),
                   "mean_error_mm": float(np.mean(errs)) if errs else None,
                   "mean_abs_error_mm": float(np.mean(np.abs(errs))) if errs else None,
                   "max_abs_error_mm": float(np.max(np.abs(errs))) if errs else None}
        with open(os.path.join(out_dir, "heights.json"), "w") as f:
            json.dump({"summary": summary, "targets": result["targets"]}, f, indent=1)
        say(f"\n{'target':8}{'measured':>10}{'estimate':>10}{'error':>8}{'score':>7}{'views':>7}  note")
        for r in result["targets"]:
            e = r["est_height_mm"]
            say(f"{r['id']:8}{'' if r['measured_mm'] is None else format(r['measured_mm'], '.2f'):>10}"
                f"{'--' if e is None else format(e, '.2f'):>10}"
                f"{'' if 'error_mm' not in r else format(r['error_mm'], '+.2f'):>8}"
                f"{'' if 'peak_score' not in r else format(r['peak_score'], '.2f'):>7}{r['n_views_seen']:>7}  "
                f"{'' if r['covered'] else 'NOT COVERED ':}{'' if r['ref_from'] == 'local_median' else 'no local board (plain?) '}")
        if summary["mean_abs_error_mm"] is not None:
            say(f"mean abs error {summary['mean_abs_error_mm']:.2f} mm, bias {summary['mean_error_mm']:+.2f} mm")
    with open(os.path.join(out_dir, "diagnostics.json"), "w") as f:
        json.dump({"views": [v.name for v in views], "seconds": time.time() - t0,
                   "settings": {"max_views": max_views, "max_width": max_width, "hp_sigma": hp_sigma,
                                "heights_mm": [h_min, h_max], "patch_mm": patch_mm, "patch_n": patch_n,
                                "frac": frac, "min_score": min_score, "min_views": min_views,
                                "grid_mm": grid_mm, "min_height_mm": min_height}}, f, indent=1)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--board", help="board.json (default: run folder, SlideScan/, cwd)")
    ap.add_argument("--mode", choices=["both", "targets", "blind"], default="both")
    ap.add_argument("--max-views", type=int, default=30, help="frames used in the sweep (evenly spread)")
    ap.add_argument("--max-width", type=int, default=2400, help="downscale frames wider than this")
    ap.add_argument("--min-score", type=float, default=0.4, help="photo-consistency needed to trust a cell")
    ap.add_argument("--min-height", type=float, default=2.0, help="blind: smallest bump to report, mm")
    ap.add_argument("--grid-mm", type=float, default=3.0, help="blind: cell size, mm")
    ap.add_argument("--patch-mm", type=float, default=3.0)
    ap.add_argument("--max-tag-images", type=int, help="use the tag from at most N frames (tag-once test)")
    ap.add_argument("--marker-mm", type=float, help="printed marker side (default board.json, else 40)")
    ap.add_argument("--dict", dest="dict_name", default="DICT_4X4_50")
    ap.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="blind: board region, mm")
    ap.add_argument("--h-min", type=float, default=-3.0, help="lowest height swept, mm")
    ap.add_argument("--h-max", type=float, default=16.0,
                    help="highest height swept, mm. Heights read high, so a 10 mm bolt can come out near 14: "
                         "keep this well above the tallest target (run B2 missed its 10 mm bolt at 12)")
    a = ap.parse_args(argv)
    run(a.run, board=a.board, mode=a.mode, max_views=a.max_views, max_width=a.max_width,
        min_score=a.min_score, min_height=a.min_height, grid_mm=a.grid_mm, patch_mm=a.patch_mm,
        max_tag_images=a.max_tag_images, marker_mm=a.marker_mm, dict_name=a.dict_name, region=a.region,
        h_min=a.h_min, h_max=a.h_max)


if __name__ == "__main__":
    main()
