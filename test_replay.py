"""python test_replay.py -- replay.py's geometry on synthetic data: no run folder, no pycolmap, no video needed."""
import cv2
import numpy as np

import replay as rp

# 1. view_matrix / Proj: straight down puts +x right and +y up; a point 10 mm higher moves up the screen when tilted
M = rp.view_matrix(0, 90)
pr = rp.Proj(M, np.array([[0, 0, 0], [100, 100, 0]]), (0, 0, 200, 200))
u, v, d = pr(np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 10]]))
assert u[1] > u[0] and v[2] < v[0], "x right, y up on screen"
assert abs(u[3] - u[0]) < 1e-6 and d[3] < d[0], "straight down: height changes depth only, nearer = smaller depth"
pt = rp.Proj(rp.view_matrix(0, 35), np.array([[0, 0, 0], [100, 100, 40]]), (0, 0, 200, 200))
assert pt(np.array([[50, 50, 30]]))[1][0] < pt(np.array([[50, 50, 0]]))[1][0], "higher point is higher on screen"

# 2. topdown: a known texture on the board plane, seen by three tilted synthetic cameras, must come back out
rng = np.random.default_rng(0)
size_mm, ppx = (120.0, 100.0), 4                               # board mm, texture px per mm
tex = cv2.GaussianBlur(rng.integers(0, 255, (int(size_mm[1] * ppx), int(size_mm[0] * ppx), 3)).astype(np.uint8), (0, 0), 6)
tex = cv2.normalize(tex, None, 40, 215, cv2.NORM_MINMAX)
Gt = np.array([[1 / ppx, 0, 0], [0, -1 / ppx, (tex.shape[0] - 1) / ppx], [0, 0, 1]])  # texture px -> board mm
K = np.array([[900.0, 0, 480], [0, 900.0, 270], [0, 0, 1]])
views = []
for C in ([60, 50, 160], [45, 40, 150], [75, 60, 155]):
    Rb, tb = rp.look_down_pose(C, [60, 50, 0])
    Hb = K @ np.column_stack([Rb[:, 0], Rb[:, 1], tb])
    img = cv2.warpPerspective(tex, Hb @ Gt, (960, 540), flags=cv2.INTER_LINEAR)
    views.append((lambda img=img: img, K, Rb, tb))
views.append((lambda: np.zeros((540, 960, 3), np.uint8), K, np.diag([1.0, -1, -1]) @ np.eye(3)[[0, 2, 1]], np.zeros(3)))  # sideways: must be skipped
cmap, cov, n = rp.topdown(views, size_mm, ppm=2, max_deg=40)
assert n == 3, n
ref = cv2.resize(tex, (cmap.shape[1], cmap.shape[0]), interpolation=cv2.INTER_AREA)
inner = cv2.erode(cov.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
assert inner.mean() > 0.5, "the cameras should cover most of the board"
err = np.abs(cmap.astype(float) - ref.astype(float))[inner].mean()
assert err < 6, f"recovered colour map differs from the texture by {err:.1f} grey levels"

# 3. splat paints far to near: the nearer point (drawn last) wins where two overlap
img = np.zeros((10, 10, 3), np.uint8)
rp.splat(img, np.array([5.0, 5.0]), np.array([5.0, 5.0]), np.array([[255, 0, 0], [0, 255, 0]], np.uint8), 1)
assert tuple(img[5, 5]) == (0, 255, 0)

# 4. surface_stage: a synthetic 9 mm bump on a flat map must come out warm (red) on a cool floor, one frame per tick
import json
import tempfile
from pathlib import Path


class Rec:
    def __init__(self):
        self.frames = []

    def write(self, f):
        self.frames.append(f.copy())


with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    out = run / "methods" / "walkaround"
    out.mkdir(parents=True)
    xs, ys = np.arange(0, 100, 5.0), np.arange(0, 80, 5.0)
    rel = np.zeros((len(ys), len(xs)))
    rel[8, 10] = 9.0                                              # bump at x = 50, y = 40
    np.savez(out / "heightmap.npz", xs=xs, ys=ys, rel=rel, active=np.ones(rel.shape, bool))
    (out / "heights.json").write_text(json.dumps({"targets": [{"id": "T01", "est_height_mm": 9.0}]}))
    g = {"board": (100.0, 80.0), "markers": {0: (20.0, 20.0)}, "msize": 10.0, "out": out,
         "targets": [{"id": "T01", "x": 50.0, "y": 40.0, "measured_mm": 8.0}]}
    rec = Rec()
    rp.surface_stage(rec, run, g, {"mean_abs_error_mm": 0.5, "mean_error_mm": 0.1}, np.zeros((720, rp.LW, 3), np.uint8), 2.0, 10)
    assert len(rec.frames) == 20 and rec.frames[0].shape == (720, 1280, 3)
    right = rec.frames[-1][:, rp.LW:].astype(int)
    warm = (right[..., 2] > 150) & (right[..., 0] < 90)           # BGR: strong red, little blue
    assert warm.sum() > 20, "the 9 mm bump should show as a red peak"
    assert (right[..., 0] > right[..., 2]).sum() > 500, "the flat floor should be cool (blue)"
print("ok")
