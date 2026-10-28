"""Self-test for methods/walkaround.py and methods/walkaround_sfm.py, on a synthetic board.

    python test_walkaround.py            # everything (the SfM part needs pycolmap)
    python test_walkaround.py --quick    # skip COLMAP: plane sweep with the true camera poses only

The scene is a textured 340 x 350 mm board with four ArUco markers and seven raised bolt
tops (flat discs at known heights), photographed from a ring of camera positions. Nothing
here touches a real run folder.
"""
import json
import os
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "methods"))
import walkaround as wa  # noqa: E402

PPM = 6.0                    # texture pixels per mm
BOARD_W, BOARD_H = 340.0, 350.0
IMG_W, IMG_H, FOCAL = 1600, 1200, 1400.0
MARKER_MM = 40.0
MARKERS = {0: (110.0, 110.0), 1: (230.0, 110.0), 2: (110.0, 240.0), 3: (230.0, 240.0)}
BOLT_R = 4.5
BOLTS = [  # id, x, y, true height mm
    ("T01", 145.0, 155.0, 1.0), ("T02", 170.0, 155.0, 2.0), ("T03", 195.0, 155.0, 3.0),
    ("T04", 145.0, 195.0, 3.2), ("T05", 170.0, 195.0, 4.0), ("T06", 195.0, 195.0, 5.0),
    ("T07", 170.0, 175.0, 10.0),
]


def noise_texture(h, w, seed):
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    for sigma, amp in ((1.0, 1.0), (2.5, 1.2), (6.0, 1.0), (14.0, 0.6)):
        n = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
        acc += amp * n / n.std()
    acc = (acc - acc.mean()) / acc.std()
    return np.clip(128 + 45 * acc, 0, 255).astype(np.uint8)


def build_scene():
    tex = noise_texture(int(BOARD_H * PPM), int(BOARD_W * PPM), 1)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for mid, (cx, cy) in MARKERS.items():
        side = int(MARKER_MM * PPM)
        m = cv2.aruco.generateImageMarker(d, mid, side)
        q = int(6 * PPM)  # white quiet zone
        sq = np.full((side + 2 * q, side + 2 * q), 255, np.uint8)
        sq[q:q + side, q:q + side] = m
        u0 = int(round((cx - MARKER_MM / 2) * PPM)) - q
        v0 = int(round((BOARD_H - cy - MARKER_MM / 2) * PPM)) - q   # texture row 0 is the top (+y)
        tex[v0:v0 + sq.shape[0], u0:u0 + sq.shape[1]] = sq
    bolt_tex = [noise_texture(64, 64, 100 + i) for i in range(len(BOLTS))]
    return tex, bolt_tex


def pose_look_at(C, target, up=(0.0, 1.0, 0.0)):
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])           # rows are the camera axes in world coordinates (y down)
    return R, -R @ C


def render(R, t, tex, bolt_tex, seed):
    C = -R.T @ t
    u, v = np.meshgrid(np.arange(IMG_W), np.arange(IMG_H))
    cx, cy = (IMG_W - 1) / 2.0, (IMG_H - 1) / 2.0
    d_cam = np.stack([(u - cx) / FOCAL, (v - cy) / FOCAL, np.ones_like(u, float)], -1).reshape(-1, 3)
    d = d_cam @ R                      # camera -> world directions (R^T applied to rows)
    t_cur = (0.0 - C[2]) / d[:, 2]
    p = C + t_cur[:, None] * d
    mx = (p[:, 0] * PPM).astype(np.float32).reshape(IMG_H, IMG_W)
    my = ((BOARD_H - p[:, 1]) * PPM).astype(np.float32).reshape(IMG_H, IMG_W)
    img = cv2.remap(tex, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    for (bid, bx, by, bh), bt in zip(BOLTS, bolt_tex):
        th = (bh - C[2]) / d[:, 2]
        ph = C + th[:, None] * d
        inside = ((ph[:, 0] - bx) ** 2 + (ph[:, 1] - by) ** 2 < BOLT_R ** 2) & (th > 0) & (th < t_cur)
        inside = inside.reshape(IMG_H, IMG_W)
        lu = ((ph[:, 0] - bx + BOLT_R) / (2 * BOLT_R) * 63).astype(np.float32).reshape(IMG_H, IMG_W)
        lv = ((ph[:, 1] - by + BOLT_R) / (2 * BOLT_R) * 63).astype(np.float32).reshape(IMG_H, IMG_W)
        samp = cv2.remap(bt, lu, lv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        img = np.where(inside, samp, img)
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(img, (0, 0), 0.7) + rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def camera_ring(n=24, radius=60.0, z0=250.0, centre=(170.0, 175.0)):
    poses = []
    for i in range(n):
        a = 2 * np.pi * i / n
        z = z0 + 25.0 * np.sin(2 * a)
        C = np.array([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), z])
        poses.append(pose_look_at(C, (centre[0] + 5 * np.cos(3 * a), centre[1] + 5 * np.sin(3 * a), 0.0)))
    return poses


def make_run(root, poses, tex, bolt_tex):
    run = os.path.join(root, "run")
    os.makedirs(os.path.join(run, "frames"))
    for i, (R, t) in enumerate(poses):
        cv2.imwrite(os.path.join(run, "frames", f"f{i:03d}.png"), render(R, t, tex, bolt_tex, i))
    board = {"marker_mm": MARKER_MM,
             "markers": [{"id": k, "x_mm": v[0], "y_mm": v[1]} for k, v in MARKERS.items()]}
    targets = {"targets": [{"id": b[0], "x_mm": b[1], "y_mm": b[2], "measured_mm": b[3]} for b in BOLTS]}
    json.dump(board, open(os.path.join(run, "board.json"), "w"))
    json.dump(targets, open(os.path.join(run, "targets.json"), "w"))
    return run


# ------------------------------------------------------------------------- tests
def test_geometry_helpers():
    rng = np.random.default_rng(0)
    src = rng.uniform(-50, 50, (12, 3))
    Rt = cv2.Rodrigues(np.array([0.3, -0.4, 0.8]))[0]
    dst = 2.5 * (Rt @ src.T).T + np.array([10.0, -20.0, 5.0])
    s, R, T, rms = wa.umeyama(src, dst)
    assert abs(s - 2.5) < 1e-9 and rms < 1e-9 and np.allclose(R, Rt)
    # triangulation from three cameras
    X = np.array([5.0, -3.0, 40.0])
    Ps, xs = [], []
    for ang in (-0.2, 0.0, 0.25):
        Rc = cv2.Rodrigues(np.array([0.0, ang, 0.0]))[0]
        tc = np.array([ang * 30, 1.0, 0.0])
        Ps.append(np.hstack([Rc, tc.reshape(3, 1)]))
        xc = Rc @ X + tc
        xs.append(xc[:2] / xc[2])
    assert np.allclose(wa.triangulate(Ps, xs), X, atol=1e-8)
    print("ok geometry helpers")


def test_parsers():
    m, size = wa.load_markers({"marker_mm": 40, "markers": [{"id": 0, "x_mm": None, "planned_xy_mm": [10, 20]},
                                                           {"id": 1, "x_mm": 1.5, "y_mm": 2.5}]})
    assert m == {0: (10.0, 20.0), 1: (1.5, 2.5)} and size == 40.0
    try:
        wa.load_markers({"markers": []})
    except SystemExit as e:
        assert "no markers" in str(e)
    else:
        raise AssertionError("empty marker list not refused")
    print("ok parsers")


def views_from_truth(poses, tex, bolt_tex):
    K = np.array([[FOCAL, 0, (IMG_W - 1) / 2.0], [0, FOCAL, (IMG_H - 1) / 2.0], [0, 0, 1]])
    out = []
    for i, (R, t) in enumerate(poses):
        g = render(R, t, tex, bolt_tex, i)
        out.append(wa.View(f"f{i:03d}", wa.prep_image(g, 6.0), K, np.zeros(4), R, t))
    return out


def test_engine_known_poses(poses, tex, bolt_tex):
    views = views_from_truth(poses[::2], tex, bolt_tex)   # 12 views, true poses, no SfM
    offs = wa.patch_offsets(3.0, 5)
    heights = np.arange(-3.0, 12.0 + 1e-9, 0.25)
    worst = 0.0
    for bid, bx, by, bh in BOLTS:
        xs = np.arange(bx - 14, bx + 14 + 1e-9, 1.0)
        ys = np.arange(by - 14, by + 14 + 1e-9, 1.0)
        h, s, n, seen = wa.height_map(views, xs, ys, heights, offs, frac=0.7, min_views=4)
        e = wa.estimate_target(xs, ys, h, s, bx, by, 0.4, 7.0, 9.0)
        err = e["est_height_mm"] - bh
        worst = max(worst, abs(err))
        print(f"  {bid} true {bh:5.2f}  est {e['est_height_mm']:6.2f}  err {err:+.2f}  score {e['peak_score']:.2f}  ref {e['ref_height_mm']:+.2f}")
    assert worst < 0.5, f"engine error {worst:.2f} mm with the true poses"
    print(f"ok plane sweep with true poses (worst error {worst:.2f} mm)")


def check_targets(run, res, label):
    sc = json.load(open(os.path.join(run, "methods", "walkaround", "scale.json")))
    errs = {r["id"]: r["error_mm"] for r in res["targets"] if "error_mm" in r}
    worst = max(abs(v) for v in errs.values())
    assert len(errs) == len(BOLTS), f"only {len(errs)} of {len(BOLTS)} bolts got an estimate"
    assert worst < 0.8, f"{label}: error {worst:.2f} mm"
    print(f"ok {label} (worst error {worst:.2f} mm, tag rms {sc['rms_mm']:.2f} mm)")


def test_end_to_end(root, poses, tex, bolt_tex):
    """COLMAP once, then the measuring step twice: tag in every frame, and tag from only 6 frames."""
    import walkaround_sfm as sfm
    run = make_run(root, poses, tex, bolt_tex)
    sfm.run(run, matcher="exhaustive", max_image_size=1600, max_features=4096,
            focal_35mm=FOCAL / IMG_W * 36.0)       # the true focal length, as the iPhone guess would be
    res = wa.run(run, mode="both", max_views=16, grid_mm=4.0, min_height=2.0, verbose=True)
    check_targets(run, res, "end to end")
    # blind mode: every bolt of 3 mm or more must be found, and nothing else may be reported
    cands = res["candidates"]["candidates"]
    found = sorted(round(c["est_height_mm"], 1) for c in cands)
    print("blind candidates (est height mm):", found)
    for bid, bx, by, bh in BOLTS:
        if bh >= 2.0:
            near = [c for c in cands if np.hypot(c["x_mm"] - bx, c["y_mm"] - by) < 8.0]
            assert near, f"blind mode missed {bid} ({bh} mm)"
    stray = [c for c in cands if min(np.hypot(c["x_mm"] - b[1], c["y_mm"] - b[2]) for b in BOLTS) >= 8.0]
    assert not stray, f"{len(stray)} blind candidate(s) away from every bolt: {stray}"
    res2 = wa.run(run, mode="targets", max_views=16, max_tag_images=6, verbose=False)
    check_targets(run, res2, "tag used from only 6 frames")


def main():
    quick = "--quick" in sys.argv
    test_geometry_helpers()
    test_parsers()
    tex, bolt_tex = build_scene()
    poses = camera_ring()
    test_engine_known_poses(poses, tex, bolt_tex)
    if quick:
        return
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        print("skip end-to-end: pycolmap not installed")
        return
    with tempfile.TemporaryDirectory() as root:
        test_end_to_end(root, poses, tex, bolt_tex)


if __name__ == "__main__":
    main()
