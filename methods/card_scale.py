"""Marker-free scale from a bank card lying flat on the board (card test, step 1).

COLMAP's camera poses are in an arbitrary unit. A card of known size (ISO ID-1, 85.60 x 53.98 mm; caliper the real one and
pass --card-mm) seen in several frames from different positions gives the scale: its four corners are found in each frame,
triangulated with COLMAP's poses, and the corner distances in SfM units against the known sides give mm per unit.
The printed markers, if they are in the footage, only score this (scale.json).

  python methods/card_scale.py runs/<run> [--frames 60] [--card-mm 85.60 53.98] [--corners corners.json]
  python methods/card_scale.py --selftest                       (synthetic card only)

Reads runs/<run>/methods/walkaround/sfm.json and the frames it points to (or runs/<run>/frames); writes card_scale.json beside it.
--corners: {"frame name": [[x, y] x4]} gives the card corners by hand for frames where detection fails (pixel coordinates).
Nothing existing is changed. Written 2026-11-13; detection is tuned on a synthetic card only, so expect to adjust it on real footage.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

CARD_MM = (85.60, 53.98)


# ------------------------------------------------------------------ geometry helpers
def order_corners(q):
    """Four points -> (4,2) in a fixed rotational order around their centre (start point arbitrary)."""
    q = np.asarray(q, float).reshape(4, 2)
    c = q.mean(0)
    return q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]


def quad_area(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def to_normalized(pts, cam):
    """Pixel points -> normalised image coordinates (lens distortion removed). cam: COLMAP model name + params."""
    p = cam["params"]
    if cam["model"] == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        K, dist = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.array([k, 0, 0, 0.0])
    elif cam["model"] == "SIMPLE_PINHOLE":
        f, cx, cy = p
        K, dist = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]), np.zeros(4)
    else:
        raise ValueError(f"camera model {cam['model']} not handled (SIMPLE_RADIAL and SIMPLE_PINHOLE are)")
    return cv2.undistortPoints(np.asarray(pts, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2), float(f)


def triangulate(obs):
    """Point closest to all rays. obs: list of (R, t, xn) with cam_from_world R, t and a normalised image point xn."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for R, t, x in obs:
        d = R.T @ np.array([x[0], x[1], 1.0])
        d /= np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ (-R.T @ t)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def reproj_px(X, R, t, xn, f):
    c = R @ X + t
    if c[2] <= 0:
        return 1e9
    return float(f * np.linalg.norm(c[:2] / c[2] - xn))


# ------------------------------------------------------------------ the scale from corners (pure; tested by --selftest)
def scale_from_corners(corners_px, poses, cams, card_mm=CARD_MM, ref_tries=5, max_err_px=3.0):
    """corners_px {frame: (4,2) pixel corners, any start point, same rotational order}; poses {frame: (R, t)} cam_from_world
    in SfM units; cams {frame: camera dict}. Returns mm per SfM unit and checks:
      aspect_error_pct (long side / short side against the card's), planarity_pct (4th corner off the plane of three),
      reproj_px, half_diff_pct (scale from the first half of the frames vs the second), n_frames, corners_units (4,3)."""
    names = [n for n in corners_px if n in poses]
    if len(names) < 2:
        raise ValueError("need the card in at least 2 frames taken from different positions")
    xn, fpx, area = {}, {}, {}
    for n in names:
        q = order_corners(corners_px[n])
        xn[n], fpx[n] = to_normalized(q, cams[n])
        area[n] = quad_area(q)
    best = None
    for ref in sorted(names, key=lambda n: -area[n])[:ref_tries]:
        lab = {ref: xn[ref]}
        for n in names:
            if n == ref:
                continue
            errs = []
            for s in range(4):                                  # which corner of this frame is the reference's corner 0
                q = np.roll(xn[n], s, axis=0)
                e = 0.0
                for i in range(4):
                    X = triangulate([(*poses[ref], xn[ref][i]), (*poses[n], q[i])])
                    e += reproj_px(X, *poses[ref], xn[ref][i], fpx[ref]) + reproj_px(X, *poses[n], q[i], fpx[n])
                errs.append(e / 8)
            s = int(np.argmin(errs))
            if errs[s] < 4 * max_err_px:
                lab[n] = np.roll(xn[n], s, axis=0)
        for _ in range(2):                                       # triangulate, drop frames that do not agree, repeat
            if len(lab) < 2:
                break
            X = np.array([triangulate([(*poses[n], lab[n][i]) for n in lab]) for i in range(4)])
            per = {n: np.mean([reproj_px(X[i], *poses[n], lab[n][i], fpx[n]) for i in range(4)]) for n in lab}
            keep = {n: lab[n] for n in lab if per[n] <= max(max_err_px, 2 * np.median(list(per.values())))}
            if len(keep) == len(lab):
                break
            lab = keep
        if len(lab) >= 2:
            X = np.array([triangulate([(*poses[n], lab[n][i]) for n in lab]) for i in range(4)])
            rms = float(np.sqrt(np.mean([reproj_px(X[i], *poses[n], lab[n][i], fpx[n]) ** 2 for n in lab for i in range(4)])))
            if best is None or (len(lab), -rms) > (len(best[0]), -best[2]):
                best = (lab, X, rms)
    if best is None:
        raise ValueError("no consistent set of frames: the card was not found at the same place in two frames")
    lab, X, rms = best
    L, S = max(card_mm), min(card_mm)

    def measure(X):
        sd = np.array([np.linalg.norm(X[i] - X[(i + 1) % 4]) for i in range(4)])
        return (2 * L + 2 * S) / sd.sum(), sd

    scale, sd = measure(X)
    a, b = (sd[0] + sd[2]) / 2, (sd[1] + sd[3]) / 2
    long_u, short_u = max(a, b), min(a, b)
    n3 = np.cross(X[1] - X[0], X[2] - X[0])
    n3 /= np.linalg.norm(n3)
    planarity = 100 * abs(np.dot(X[3] - X[0], n3)) / sd.mean()
    half = None
    ns = sorted(lab)
    if len(ns) >= 4:
        h = len(ns) // 2
        sc = []
        for grp in (ns[:h], ns[h:]):
            Xh = np.array([triangulate([(*poses[n], lab[n][i]) for n in grp]) for i in range(4)])
            sc.append(measure(Xh)[0])
        half = 100 * (sc[1] / sc[0] - 1)
    return {"scale_mm_per_unit": float(scale), "aspect_error_pct": float(100 * ((long_u / short_u) / (L / S) - 1)),
            "planarity_pct": float(planarity), "reproj_px": rms, "half_diff_pct": half, "n_frames": len(lab),
            "scale_from_long_sides": float(L / long_u), "scale_from_short_sides": float(S / short_u),
            "corners_units": X.tolist(), "frames_used": ns}


# ------------------------------------------------------------------ finding the card in an image (tuned on synthetic images only)
def _rough_quad(gray_small, lo=0.0015, hi=0.5, aspect_range=(1.3, 2.1)):
    """Best four-sided convex outline with a card-like aspect, in small-image pixels, or None."""
    h, w = gray_small.shape
    g = cv2.GaussianBlur(gray_small, (5, 5), 0)
    e = cv2.dilate(cv2.Canny(g, 25, 80), np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best, best_score = None, -1.0
    for c in cnts:
        hull = cv2.convexHull(c)
        a = cv2.contourArea(hull)
        if not lo * w * h < a < hi * w * h:
            continue
        ap = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
        if len(ap) != 4:
            continue
        q = order_corners(ap.reshape(4, 2))
        if quad_area(q) < 0.85 * a:
            continue
        s = np.linalg.norm(q - np.roll(q, -1, axis=0), axis=1)
        asp = max((s[0] + s[2]) / (s[1] + s[3]), (s[1] + s[3]) / (s[0] + s[2]))
        if not aspect_range[0] <= asp <= aspect_range[1]:
            continue
        score = a / (1.0 + 4 * abs(asp - 1.586))
        if score > best_score:
            best, best_score = q, score
    return best


def _refine_quad(gray, q, half_width=None):
    """Fit a straight line to each side from the strongest edge across it, intersect neighbours: corners to a fraction of a pixel,
    and the rounded card corners do not matter because only the middle of each side is used."""
    H, W = gray.shape
    hw = int(half_width or max(5, 0.006 * W))
    g = cv2.GaussianBlur(gray, (0, 0), 1.0).astype(np.float32)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([-d[1], d[0]])
        ts = np.linspace(0.2, 0.8, 40)
        offs = np.arange(-hw, hw + 1, 0.5)
        P = a[None, None, :] + ts[:, None, None] * (b - a)[None, None, :] + offs[None, :, None] * nrm[None, None, :]
        prof = cv2.remap(g, P[..., 0].astype(np.float32), P[..., 1].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        grad = np.abs(np.gradient(prof, axis=1))
        j = grad.argmax(1)
        pts = []
        for k in range(len(ts)):
            jj = int(np.clip(j[k], 1, grad.shape[1] - 2))
            y0, y1, y2 = grad[k, jj - 1], grad[k, jj], grad[k, jj + 1]
            den = y0 - 2 * y1 + y2
            sub = 0.5 * (y0 - y2) / den if abs(den) > 1e-9 else 0.0
            pts.append(P[k, jj] + sub * 0.5 * nrm)
        vx, vy, x0, y0 = cv2.fitLine(np.array(pts, np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0], float), np.array([vx, vy], float)))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]                       # corner i = side i-1 meets side i
        M = np.array([d1, -d2]).T
        if abs(np.linalg.det(M)) < 1e-6:
            return None
        s = np.linalg.solve(M, p2 - p1)
        out.append(p1 + s[0] * d1)
    out = np.array(out)
    if np.abs(out - q).max() > 4 * hw:
        return None
    return out


def detect_card(img, work_width=1600):
    """BGR or grey image -> (4,2) card corners in pixels (rotational order, arbitrary start) or None."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    sc = min(1.0, work_width / gray.shape[1])
    small = cv2.resize(gray, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else gray
    q = _rough_quad(small)
    if q is None:
        return None
    r = _refine_quad(gray, q / sc)
    return None if r is None else order_corners(r)


# ------------------------------------------------------------------ run folder driver
def run(run_dir, n_frames=60, card_mm=CARD_MM, corners_file=None):
    w = os.path.join(run_dir, "methods", "walkaround")
    sfm = json.load(open(os.path.join(w, "sfm.json")))
    img_dir = sfm["image_dir"] if os.path.isdir(sfm["image_dir"]) else os.path.join(run_dir, "frames")
    imgs = sorted(sfm["images"], key=lambda i: i["name"])
    pick = np.unique(np.linspace(0, len(imgs) - 1, min(n_frames, len(imgs))).round().astype(int))
    manual = json.load(open(corners_file)) if corners_file else {}
    corners, poses, cams = {}, {}, {}
    for i in pick:
        im = imgs[i]
        n = im["name"]
        if n in manual:
            c = np.array(manual[n], float)
        else:
            c = detect_card(cv2.imread(os.path.join(img_dir, n)))
        if c is None:
            continue
        M = np.array(im["cam_from_world"], float)
        corners[n], poses[n], cams[n] = c, (M[:, :3], M[:, 3]), sfm["cameras"][im["camera_id"]]
    res = scale_from_corners(corners, poses, cams, card_mm)
    res.update({"frames_tried": int(len(pick)), "frames_with_a_card": len(corners), "card_mm": list(card_mm)})
    sp = os.path.join(w, "scale.json")
    if os.path.exists(sp):
        marker = json.load(open(sp))["scale_mm_per_unit"]
        res["marker_scale_mm_per_unit"] = marker
        res["error_vs_marker_pct"] = 100 * (res["scale_mm_per_unit"] / marker - 1)
    json.dump(res, open(os.path.join(w, "card_scale.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------ self-test (a rendered card on a rendered board)
def _look_at(pos, target, roll_deg=0.0):
    z = target - pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0.0, 1.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    a = np.radians(roll_deg)
    x, y = np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y
    R = np.array([x, y, z])                                   # world -> camera
    return R, -R @ pos


def _rounded_rect(cx, cy, w, h, r, n=12):
    pts = []
    for (ox, oy, a0) in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
        for a in np.linspace(a0, a0 + 90, n):
            pts.append((cx + ox + r * np.cos(np.radians(a)), cy + oy + r * np.sin(np.radians(a))))
    return np.array(pts)


def _scene(rng, k_true=47.0, n_views=8):
    ppm, size = 8, 260                                         # texture pixels per mm, board size in mm
    S = size * ppm
    noise = cv2.resize(rng.normal(0, 1, (S // 10, S // 10)).astype(np.float32), (S, S), interpolation=cv2.INTER_CUBIC)
    tex = np.clip(np.dstack([95 + 16 * noise, 135 + 16 * noise, 175 + 16 * noise]), 0, 255).astype(np.uint8)   # brown, like cardboard
    card = (_rounded_rect(130, 130, *CARD_MM, 3.18) * ppm).astype(np.int32)
    cv2.fillPoly(tex, [card], (160, 70, 40))
    cv2.fillPoly(tex, [(_rounded_rect(60, 60, 38.3, 38.3, 0.1) * ppm).astype(np.int32)], (10, 10, 10))   # a marker-like square
    W, H, f = 1920, 1080, 1500.0
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    views = []
    for i in range(n_views):
        ang = 2 * np.pi * i / n_views
        pos = np.array([130 + 110 * np.cos(ang), 130 + 80 * np.sin(ang), 240 + 20 * np.sin(2 * ang)])
        roll = [0, 90, 0, 180, 0, 0, -90, 0][i % 8]
        R, t = _look_at(pos, np.array([130 + 8 * np.cos(3 * ang), 130 + 5 * np.sin(3 * ang), 0.0]), roll)
        Hm = K @ np.c_[R[:, 0], R[:, 1], t] @ np.diag([1 / ppm, 1 / ppm, 1])
        img = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        img = np.clip(img.astype(np.float32) + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
        views.append((f"v{i}", img, (R, t / k_true)))          # poses in SfM units: 1 unit = k_true mm
    cam = {"model": "SIMPLE_RADIAL", "params": [f, W / 2, H / 2, 0.0]}
    return views, cam


def selftest():
    rng = np.random.default_rng(5)
    k_true = 47.0
    views, cam = _scene(rng, k_true)
    corners, poses, cams = {}, {}, {}
    for n, img, pose in views:
        c = detect_card(img)
        assert c is not None, f"card not found in {n}"
        corners[n], poses[n], cams[n] = c, pose, cam
    r = scale_from_corners(corners, poses, cams)
    assert abs(r["scale_mm_per_unit"] / k_true - 1) < 0.01, r
    assert abs(r["aspect_error_pct"]) < 1.5 and r["planarity_pct"] < 1.0 and r["n_frames"] >= 6, r
    # a frame whose corners are wrong (the marker-like square, or a random quad) is dropped, the scale holds
    bad = dict(corners)
    bad["v3"] = np.array([[200, 200], [500, 220], [480, 400], [210, 380]], float)
    r2 = scale_from_corners(bad, poses, cams)
    assert abs(r2["scale_mm_per_unit"] / k_true - 1) < 0.01 and "v3" not in r2["frames_used"], r2
    # an image with no card gives None, not a wrong quad
    empty = cv2.GaussianBlur(rng.integers(60, 190, (1080, 1920, 3)).astype(np.uint8), (0, 0), 6)
    assert detect_card(empty) is None
    # one frame is not enough
    try:
        scale_from_corners({"v0": corners["v0"]}, poses, cams)
        raise AssertionError("one frame was accepted")
    except ValueError:
        pass
    # a card of another size changes the scale in proportion (a wrong --card-mm would show up as a wrong scale)
    r3 = scale_from_corners(corners, poses, cams, card_mm=(85.60 * 1.03, 53.98 * 1.03))
    assert abs(r3["scale_mm_per_unit"] / (k_true * 1.03) - 1) < 0.01, r3
    print(f"selftest ok: card found in all {len(views)} rendered views (two rolled 90/180 degrees, a marker-like square ignored); "
          f"scale {r['scale_mm_per_unit']:.2f} vs {k_true} ({100 * (r['scale_mm_per_unit'] / k_true - 1):+.2f} %), aspect {r['aspect_error_pct']:+.2f} %, "
          f"reprojection {r['reproj_px']:.2f} px; a wrong frame dropped; one frame refused")
    print("NOT tested: a real card (glare, dark or similar colours, bending, motion blur), lens distortion, real COLMAP poses, other phones")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--card-mm", type=float, nargs=2, default=list(CARD_MM), metavar=("LONG", "SHORT"))
    ap.add_argument("--corners")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.run:
        selftest()
        sys.exit(0)
    r = run(a.run, a.frames, tuple(a.card_mm), a.corners)
    print(f"card scale {r['scale_mm_per_unit']:.3f} mm/unit from {r['n_frames']} of {r['frames_with_a_card']} frames with a card "
          f"({r['frames_tried']} tried); aspect {r['aspect_error_pct']:+.2f} %, planarity {r['planarity_pct']:.2f} %, reprojection {r['reproj_px']:.2f} px"
          + (f"; marker {r['marker_scale_mm_per_unit']:.3f} -> error {r['error_vs_marker_pct']:+.2f} %" if "error_vs_marker_pct" in r else ""))
