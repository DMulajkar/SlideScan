"""SlideScan: replay video of a walkaround run. Recording on the left, the reconstruction growing on the right.

  python replay.py runs/<run> [--out FILE.mp4] [--tail 10] [--every 1]

Run in the WSL venv (needs pycolmap, opencv, numpy), after steps 1 and 2 of WALKAROUND_README.md.
Build-up: left = the recording and a top-down map of the camera path; right = COLMAP points appear when a frame
first sees them, with the board, markers, bolts and the path. Last --tail seconds: a top-down colour map (frames
projected onto the board plane, median of overhead views) and a slowly turning 3D colour point map whose height comes
from methods/walkaround/heightmap.npz (smoothed, x4). An illustration, not a measurement: the last frames print the
run's mean absolute height error from heights.json. --every N keeps every Nth frame (quick preview).
Also writes panel_topdown_map.png next to the video. If ffmpeg is on PATH the video is re-encoded to H.264.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "methods"))
import walkaround as wk  # reuse: tag detection, similarity fit, camera model, board loaders

VIDEO_FPS = 30.0
SIZE, LW = (1280, 720), 480                      # canvas, width of the left panel
BG, TXT, MUTED, ACC, PATH, BOARD = (20, 17, 14), (232, 228, 222), (150, 146, 140), (70, 190, 255), (60, 140, 255), (72, 112, 150)


# ------------------------------------------------------------------ geometry helpers
def view_matrix(az_deg, el_deg):
    """Board mm -> (x, y-up, depth) for a viewer at azimuth az and elevation el (90 = straight down)."""
    a, e = math.radians(az_deg), math.radians(el_deg)
    Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    Tl = np.array([[1, 0, 0], [0, math.sin(e), math.cos(e)], [0, math.cos(e), -math.sin(e)]])
    return Tl @ Rz


class Proj:
    """Fixed projection of board mm into the pixel rectangle (x0, y0, w, h), scaled to fit `fit_pts`."""

    def __init__(self, M, fit_pts, rect):
        q = np.asarray(fit_pts, float) @ M.T
        lo, hi = q[:, :2].min(0), q[:, :2].max(0)
        self.M, self.rect, self.mid = M, rect, (lo + hi) / 2
        self.sc = min(rect[2] / max(hi[0] - lo[0], 1e-6), rect[3] / max(hi[1] - lo[1], 1e-6)) * 0.94

    def __call__(self, P):
        q = np.atleast_2d(np.asarray(P, float)) @ self.M.T
        x0, y0, w, h = self.rect
        return x0 + w / 2 + self.sc * (q[:, 0] - self.mid[0]), y0 + h / 2 - self.sc * (q[:, 1] - self.mid[1]), q[:, 2]


def topdown(views, size_mm, ppm=2, max_deg=40, max_views=40, band=64):
    """Colour map of the board plane. views: (get_img, K, Rb, tb) with get_img() an undistorted BGR frame and
    Rb, tb the pose in board mm (x_cam = Rb X + tb). Uses views looking within max_deg of straight down, takes the
    per-cell median. Returns (BGR uint8 H x W x 3, covered bool H x W, number of views used); row 0 is y = max."""
    W, H = int(round(size_mm[0] * ppm)), int(round(size_mm[1] * ppm))
    G = np.array([[1 / ppm, 0, 0], [0, -1 / ppm, (H - 1) / ppm], [0, 0, 1]])  # grid pixel -> board mm
    down = [v for v in views if -v[2][2, 2] > math.cos(math.radians(max_deg))]
    if not down:
        return np.zeros((H, W, 3), np.uint8), np.zeros((H, W), bool), 0
    pick = [down[i] for i in np.unique(np.linspace(0, len(down) - 1, min(max_views, len(down))).round().astype(int))]
    stack = np.full((len(pick), H, W, 3), np.nan, np.float32)
    for i, (get_img, K, Rb, tb) in enumerate(pick):
        img = get_img()
        M = K @ np.column_stack([Rb[:, 0], Rb[:, 1], tb]) @ G                    # grid pixel -> source pixel
        fl = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        warped = cv2.warpPerspective(img, M, (W, H), flags=fl).astype(np.float32)
        ok = cv2.warpPerspective(np.ones(img.shape[:2], np.uint8), M, (W, H), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        ok = cv2.erode(ok.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0       # drop the blurry frame edge
        stack[i][ok] = warped[ok]
    out = np.zeros((H, W, 3), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # cells no view covers are all-NaN on purpose
        for r in range(0, H, band):
            out[r:r + band] = np.nan_to_num(np.nanmedian(stack[:, r:r + band], axis=0))
    cov = ~np.isnan(stack).all(axis=(0, 3))
    return out.astype(np.uint8), cov, len(pick)


def look_down_pose(C, target, up=(0.0, 1.0, 0.0)):
    """Board-frame pose (Rb, tb) of an OpenCV camera at C looking at target, board +y up in the image."""
    C, z = np.asarray(C, float), np.asarray(target, float) - np.asarray(C, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    Rb = np.stack([x, np.cross(z, x), z])
    return Rb, -Rb @ C


# ------------------------------------------------------------------ loading
def frame_no(name):
    return int(Path(name).stem.rsplit("_", 1)[1])


def load_geometry(run, points=True):
    """Poses in board mm (tag fit, as walkaround.py) and, if `points`, the COLMAP sparse points (needs pycolmap)."""
    out = run / "methods" / "walkaround"
    sfm = json.loads((out / "sfm.json").read_text())
    markers, msize = wk.load_markers(wk.load_json(wk.find_board(str(run))), None)
    targets = wk.load_targets(str(run))
    img_dir = Path(sfm["image_dir"]) if os.path.isabs(sfm["image_dir"]) else run / sfm["image_dir"]
    det = wk.make_detector()
    poses, cams, obs = {}, {}, {}
    for im in sfm["images"]:
        M = np.array(im["cam_from_world"], float)
        poses[im["name"]] = (M[:, :3], M[:, 3])
    names = sorted(poses)
    cam_of = {i["name"]: i["camera_id"] for i in sfm["images"]}
    for n in names:                                  # same tag fit as walkaround.py, so the frames line up exactly
        g, sc = wk.read_gray(str(img_dir / n), 2400)
        cams[n] = wk.cv_camera(sfm["cameras"][cam_of[n]], sc)
        d = {i: c for i, c in wk.detect_tags(det, g).items() if i in markers}
        if d:
            obs[n] = d
    fit = wk.fit_similarity(obs, poses, cams, markers, msize, sorted(obs), 1.5)
    if not fit["ok"]:
        raise SystemExit("the tag fit failed; run walkaround.py first and check its messages")
    s, R, T = fit["scale_mm_per_unit"], fit["R"], fit["T"]
    pose_b = {}
    for n in names:
        Rc, tc = poses[n]
        Rb = Rc @ R.T
        pose_b[n] = (Rb, s * tc - Rb @ T)
    centers = np.array([-pose_b[n][0].T @ pose_b[n][1] for n in names])
    axes = np.array([pose_b[n][0][2] for n in names])
    if points:
        import pycolmap
        sparse = out / "colmap" / "sparse"
        recs = [pycolmap.Reconstruction(str(d)) for d in sorted(sparse.iterdir()) if d.is_dir()]
        rec = max(recs, key=lambda r: r.num_reg_images())
        id2f = {iid: frame_no(im.name) for iid, im in rec.images.items()}
        X, C, F = [], [], []
        for p in rec.points3D.values():
            X.append(p.xyz)
            C.append(np.asarray(p.color)[::-1])
            F.append(min(id2f[e.image_id] for e in p.track.elements))
        X = s * (np.array(X) @ R.T) + T
    else:
        X, C, F = np.zeros((0, 3)), np.zeros((0, 3)), []
    box = (X[:, 0] > -80) & (X[:, 0] < 420) & (X[:, 1] > -80) & (X[:, 1] < 450) & (X[:, 2] > -60) & (X[:, 2] < 400)
    order = np.argsort(np.array(F)[box], kind="stable")
    brd = json.loads((run / "targets.json").read_text()).get("board_size_mm") or [333.4, 349.3]
    return dict(names=names, f=np.array([frame_no(n) for n in names]), centers=centers, axes=axes, pose_b=pose_b,
                cams=sfm["cameras"], cam_of=cam_of, pts=X[box][order], col=np.array(C)[box][order].astype(np.uint8),
                pt_f=np.array(F)[box][order], markers=markers, msize=msize, targets=targets, board=brd,
                img_dir=img_dir, fit=fit, n_cams=len(names), out=out)


def smooth_height(run, XY, sigma=1.3, lo=-3.0, hi=12.0):
    """Height above the local board (mm) at board points XY from heightmap.npz: low-score cells 0, smoothed."""
    z = np.load(run / "methods" / "walkaround" / "heightmap.npz")
    rel = np.where(z["active"] & (z["score"] >= 0.4), z["rel"], 0.0)
    rel = cv2.GaussianBlur(np.clip(rel, lo, hi).astype(np.float32), (0, 0), sigma)
    xs, ys = z["xs"], z["ys"]
    mx = ((XY[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(np.float32)
    my = ((XY[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(np.float32)
    return cv2.remap(rel, mx.reshape(1, -1), my.reshape(1, -1), cv2.INTER_LINEAR, borderValue=0).ravel()


# ------------------------------------------------------------------ drawing
def put(img, text, org, scale=0.5, color=TXT, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def poly(img, uv, color, fill=False):
    pts = np.round(np.column_stack(uv[:2])).astype(np.int32)
    (cv2.fillPoly if fill else cv2.polylines)(img, [pts], *( [color] if fill else [True, color, 1, cv2.LINE_AA] ))


def splat(img, u, v, col, r=1):
    """Paint points far to near (caller sorts) as (2r+1)-pixel squares."""
    h, w = img.shape[:2]
    u, v = np.round(u).astype(int), np.round(v).astype(int)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            img[vv[ok], uu[ok]] = col[ok]


def board_layer(img, proj, g, flat=False):
    bw, bh = g["board"]
    poly(img, proj(np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]])), BOARD, fill=True)
    for mid, (cx, cy) in g["markers"].items():
        poly(img, proj(wk.marker_corners_board(cx, cy, g["msize"])), (35, 30, 28), fill=True)
    for t in g["targets"]:
        u, v, _ = proj(np.array([[t["x"], t["y"], 0.0]]))
        cv2.circle(img, (int(u[0]), int(v[0])), 4, (200, 220, 240), 1, cv2.LINE_AA)
        if flat:
            put(img, t["id"], (int(u[0]) + 6, int(v[0]) - 5), 0.35, MUTED)
    poly(img, proj(np.array([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]])), (110, 160, 205))


def fmt_t(f):
    return f"{f / VIDEO_FPS:4.1f} s"


# ------------------------------------------------------------------ height surface stage
def surface_stage(vw, run, g, hs, left, hold_s, fps, ex_max=4.0):
    """The raw plane-sweep height map as a turbo-coloured 3D mesh: it rises out of the board, then turns.
    Each bolt is labelled 'id caliper->scan' (mm). Cells the sweep did not score are dark and flat."""
    z = np.load(run / "methods" / "walkaround" / "heightmap.npz")
    xs, ys, act = z["xs"], z["ys"], z["active"]
    rel = np.clip(z["rel"], -3.0, 12.0)
    ny, nx = rel.shape
    X, Y = np.meshgrid(xs, ys)
    lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(256, 3)
    vcol = lut[(np.clip((rel + 1) / 9, 0, 1) * 255).astype(int)].astype(int)
    vcol[~act] = (22, 22, 22)
    qcol = ((vcol[:-1, :-1] + vcol[1:, :-1] + vcol[1:, 1:] + vcol[:-1, 1:]) // 4).reshape(-1, 3)
    Zf = np.where(act, rel, 0.0).ravel()
    bw, bh = g["board"]
    box = np.array([[x, y, zz] for x in (0, bw) for y in (0, bh) for zz in (0, 12 * ex_max)], float)
    rect = (LW, 70, SIZE[0] - LW, SIZE[1] - 130)
    sc = min(Proj(view_matrix(a, 34), box, rect).sc for a in np.linspace(-35, 25, 13))
    est = {t["id"]: t for t in json.loads((g["out"] / "heights.json").read_text())["targets"]}
    idx = np.arange(ny * nx).reshape(ny, nx)
    i00, i10, i11, i01 = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    total, n_rise = int(hold_s * fps), min(15, int(hold_s * fps) // 3)
    missed = hs["mean_abs_error_mm"] > 1.0
    for i in range(total):
        ex = ex_max * min(1.0, (i + 1) / n_rise)
        az = -35 + 60 * max(0, i - n_rise) / max(1, total - n_rise)
        p = Proj(view_matrix(az, 34), box, rect)
        p.sc = sc
        u, v, d = p(np.column_stack([X.ravel(), Y.ravel(), ex * Zf]))
        quad = np.stack([np.column_stack([u[k], v[k]]) for k in (i00, i10, i11, i01)], 1).round().astype(np.int32)
        order = np.argsort(-(d[i00] + d[i10] + d[i11] + d[i01]))              # far to near
        c = np.full((SIZE[1], SIZE[0], 3), BG, np.uint8)
        c[:, :LW] = left
        for q in order:
            cv2.fillConvexPoly(c, quad[q], tuple(int(x) for x in qcol[q]))
        for mid, (cx, cy) in g["markers"].items():
            poly(c, p(wk.marker_corners_board(cx, cy, g["msize"])), (235, 235, 235))
        for t in g["targets"]:
            e = est.get(t["id"])
            if e and e.get("est_height_mm") is not None:
                uu, vv, _ = p(np.array([[t["x"], t["y"], ex * max(e["est_height_mm"], t["measured_mm"], 1.0) + 4]]))
                put(c, f"{t['id']}  {t['measured_mm']:.1f}->{e['est_height_mm']:.1f}", (int(uu[0]) - 30, int(vv[0]) - 6), 0.4)
        put(c, "RECONSTRUCTION", (LW + 16, 38), 0.65)
        put(c, f"Plane-sweep height map, {ex_max:g}x exaggerated, label = caliper->scan (mm)", (LW + 16, 58), 0.42, MUTED)
        put(c, ("Run missed its target: " if missed else "") + f"mean error {hs['mean_abs_error_mm']:.2f} mm, "
               f"bias {hs['mean_error_mm']:+.2f} mm", (LW + 16, SIZE[1] - 18), 0.5, PATH)
        vw.write(c)


# ------------------------------------------------------------------ render
def render(run, out, tail_s=10.0, every=1, fps=10, ppm=2, hold_s=8.0):
    t0 = time.time()
    g = load_geometry(run)
    print(f"loaded: {g['n_cams']} cameras, {len(g['pts'])} points ({time.time() - t0:.0f} s)")
    frames = sorted((run / "frames").glob("*.jpg"))[::max(1, every)]
    bw, bh = g["board"]
    box = np.array([[x, y, z] for x in (0, bw) for y in (0, bh) for z in (0, 220)], float)
    pr = Proj(view_matrix(-35, 38), box, (LW, 60, SIZE[0] - LW, SIZE[1] - 110))
    pu, pv, _ = pr(g["pts"])
    right = np.full((SIZE[1], SIZE[0], 3), BG, np.uint8)
    board_layer(right, pr, g)
    pm = Proj(view_matrix(0, 90), np.array([[0, 0, 0], [bw, bh, 0]]), (20, 365, LW - 40, SIZE[1] - 375))
    left = np.full((SIZE[1], SIZE[0], 3), BG, np.uint8)
    board_layer(left, pm, g, flat=True)
    cu, cv_, _ = pm(g["centers"])
    ru, rv, _ = pr(g["centers"])
    tip_u, tip_v, _ = pr(g["centers"] + 60 * g["axes"])
    vw_path = str(out) + ".raw.mp4"
    vw = cv2.VideoWriter(vw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, SIZE)
    for fp in frames:
        f = frame_no(fp.name)
        k = int(np.searchsorted(g["f"], f, side="right"))                 # cameras placed so far
        n = int(np.searchsorted(g["pt_f"], f, side="right"))              # points seen so far
        c = np.full((SIZE[1], SIZE[0], 3), BG, np.uint8)
        c[:, LW:] = right[:, LW:]
        c[:, :LW] = left[:, :LW]
        splat(c, pu[:n], pv[:n], np.maximum(g["col"][:n], 150), 1)
        if k:
            cv2.polylines(c, [np.round(np.column_stack([ru[:k], rv[:k]])).astype(np.int32)], False, PATH, 1, cv2.LINE_AA)
            cv2.line(c, (int(ru[k - 1]), int(rv[k - 1])), (int(tip_u[k - 1]), int(tip_v[k - 1])), ACC, 1, cv2.LINE_AA)
            cv2.circle(c, (int(ru[k - 1]), int(rv[k - 1])), 5, ACC, -1, cv2.LINE_AA)
            cv2.polylines(c, [np.round(np.column_stack([cu[:k], cv_[:k]])).astype(np.int32)], False, PATH, 1, cv2.LINE_AA)
            cv2.circle(c, (int(cu[k - 1]), int(cv_[k - 1])), 5, ACC, -1, cv2.LINE_AA)
        img = cv2.imread(str(fp), cv2.IMREAD_REDUCED_COLOR_4)
        c[60:60 + 270, :LW] = cv2.resize(img, (LW, 270), interpolation=cv2.INTER_AREA)
        put(c, "RECORDING", (14, 38), 0.65)
        cv2.circle(c, (140, 32), 5, (60, 60, 230), -1, cv2.LINE_AA)
        put(c, fmt_t(f), (LW - 90, 38), 0.6, ACC)
        put(c, "CAMERA PATH, TOP-DOWN (board mm)", (14, 350), 0.42, MUTED)
        put(c, "RECONSTRUCTION", (LW + 16, 38), 0.65)
        put(c, "COLMAP: points appear when a frame first sees them", (LW + 16, 58), 0.42, MUTED)
        put(c, f"3D points {n:,}    cameras placed {k}/{g['n_cams']}", (LW + 16, SIZE[1] - 18), 0.5, ACC)
        vw.write(c)
    print(f"build-up: {len(frames)} frames ({time.time() - t0:.0f} s)")
    hs = json.loads((g["out"] / "heights.json").read_text())["summary"]
    if hold_s > 0:
        surface_stage(vw, run, g, hs, c[:, :LW].copy(), hold_s, fps)
        print(f"height surface: {int(hold_s * fps)} frames ({time.time() - t0:.0f} s)")

    # --- tail: top-down colour map, then a turning 3D colour point map -------------------
    views = []
    for n in g["names"]:
        cam = g["cams"][g["cam_of"][n]]
        K, dist = wk.cv_camera(cam, 0.5)

        def get(n=n, K=K, dist=dist):
            im = cv2.imread(str(g["img_dir"] / n), cv2.IMREAD_REDUCED_COLOR_2)
            return cv2.undistort(im, K, dist)
        views.append((get, K, g["pose_b"][n][0], g["pose_b"][n][1]))
    cmap, cov, nv = topdown(views, g["board"], ppm=ppm)
    cmap[~cov] = BOARD
    cv2.imwrite(str(out.with_name("panel_topdown_map.png")), cmap)
    print(f"top-down map from {nv} overhead views ({time.time() - t0:.0f} s)")
    step = 2 * ppm                                                           # one point per 2 mm
    gv, gu = np.mgrid[0:cmap.shape[0]:step, 0:cmap.shape[1]:step]
    keep = cov[gv, gu]
    gx_mm, gy_mm = gu / ppm, (cmap.shape[0] - 1 - gv) / ppm
    zg = 4.0 * smooth_height(run, np.column_stack([gx_mm.ravel(), gy_mm.ravel()])).reshape(gv.shape)
    dzdy, dzdx = np.gradient(zg, -2.0, 2.0)                                  # rows run toward -y, spacing 2 mm
    lx, ly, lz = 0.5, 0.5, 0.7                                               # light from the upper left, fairly high
    shade = ((-dzdx * lx - dzdy * ly + lz) / np.sqrt(1 + dzdx ** 2 + dzdy ** 2) / lz).clip(0.5, 1.6)
    P = np.column_stack([gx_mm[keep], gy_mm[keep], zg[keep]])
    col = np.clip(cmap[gv[keep], gu[keep]] * shade[keep][:, None], 0, 255).astype(np.uint8)
    box3 =np.array([[x, y, z] for x in (0, bw) for y in (0, bh) for z in (0, 50)], float)
    rect3 = (LW, 70, SIZE[0] - LW, SIZE[1] - 130)
    pr3 = Proj(view_matrix(-35, 34), box3, rect3)
    pr3.sc = min(Proj(view_matrix(az, 34), box3, rect3).sc for az in np.linspace(-35, 25, 13))
    mh, mw = cmap.shape[:2]
    sc2 = min((LW - 30) / mw, (SIZE[1] - 190) / mh)
    thumb = cv2.resize(cmap, (int(mw * sc2), int(mh * sc2)), interpolation=cv2.INTER_AREA)
    cands = json.loads((g["out"] / "candidates.json").read_text())["candidates"]
    for k in cands:                                   # rings on the photo where the sweep flags a bump, with its height
        ux, vy = int(k["x_mm"] * ppm * sc2), int((mh - 1 - k["y_mm"] * ppm) * sc2)
        cv2.circle(thumb, (ux, vy), 11, ACC, 1, cv2.LINE_AA)
        put(thumb, f"{k['est_height_mm']:.1f}", (ux + 13, vy + 4), 0.42, ACC)
    n_tail = int(tail_s * fps)
    for i in range(n_tail):
        az = -35 + 60 * min(1.0, i / max(1, n_tail - 12))
        pa = Proj(view_matrix(az, 34), box3, pr3.rect)               # re-centred for this angle, scale kept fixed
        pa.sc = pr3.sc
        u, v, d = pa(P)
        o = np.argsort(-d)                                                   # far to near
        c = np.full((SIZE[1], SIZE[0], 3), BG, np.uint8)
        splat(c, u[o], v[o], col[o], 2)
        c[110:110 + thumb.shape[0], 15:15 + thumb.shape[1]] = thumb
        bar = int(100 * ppm * sc2)
        cv2.line(c, (15, 120 + thumb.shape[0] + 12), (15 + bar, 120 + thumb.shape[0] + 12), ACC, 3)
        put(c, "100 mm", (15 + bar + 8, 120 + thumb.shape[0] + 17), 0.45, ACC)
        put(c, "rings: places the sweep flags as a bump (2 mm or more),", (14, 120 + thumb.shape[0] + 42), 0.38, ACC)
        put(c, "label = its estimated height above the local board, mm", (14, 120 + thumb.shape[0] + 60), 0.38, ACC)
        put(c, "TOP-DOWN COLOUR MAP", (14, 38), 0.65)
        put(c, f"recording frames projected onto the board plane (median of {nv} overhead views)", (14, 70), 0.36, MUTED)
        put(c, "PANEL COLOUR POINT MAP (3D)", (LW + 16, 38), 0.65)
        put(c, f"{len(P):,} points, coloured from the recording; height from the plane sweep, smoothed, x4, shaded",
            (LW + 16, 58), 0.42, MUTED)
        put(c, f"indicative, not measured: mean height error {hs['mean_abs_error_mm']:.1f} mm on the bolts, "
               f"bias {hs['mean_error_mm']:+.1f} mm", (LW + 16, SIZE[1] - 18), 0.45, ACC)
        vw.write(c)
    vw.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", vw_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", str(out)], check=True)
        os.remove(vw_path)
    else:
        os.replace(vw_path, out)
    print(f"wrote {out} ({time.time() - t0:.0f} s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out")
    ap.add_argument("--tail", type=float, default=10.0, help="seconds of colour-map ending")
    ap.add_argument("--every", type=int, default=1, help="keep every Nth extracted frame (preview)")
    ap.add_argument("--hold", type=float, default=8.0, help="seconds of raw height-surface view (0 = skip)")
    a = ap.parse_args()
    run = Path(a.run)
    render(run, Path(a.out) if a.out else run / "reconstruction_replay.mp4", a.tail, a.every, hold_s=a.hold)


if __name__ == "__main__":
    main()
