"""python test_colmap.py [keep_dir] -- methods/colmap.py on a synthetic sweep (no camera needed).
A richly textured board with four markers and bolt heads at real heights, rendered from 24 poses with true parallax
(each head is its own plane, at its height). This is the BEST case: perfect texture on every head, no blur, no glare,
no noise in the poses. It proves the plumbing and sets the ceiling; a real run can only do worse.
Pass a folder to keep the synthetic run there."""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import slidescan as s
from methods import colmap

PPM, W, H, SIDE, HEAD = 4, 333.38, 349.25, 38.3, 6  # texture px/mm; board mm; marker side; head radius mm
MARKERS = {0: (108.16, 95.25), 1: (228.6, 94.08), 2: (112.0, 222.25), 3: (222.25, 215.9)}
TARGETS = {"T01": (288.93, 34.24, 0), "T02": (165.1, 34.24, 1), "T05": (44.63, 152.98, 3.2),
           "T06": (165.1, 152.98, 4), "T07": (288.93, 152.98, 5), "T08": (288.93, 276.22, 10)}
SIZE, F = (1280, 960), 1000.0
K = np.array([[F, 0, SIZE[0] / 2], [0, F, SIZE[1] / 2], [0, 0, 1]])
TW, TH = round(W * PPM), round(H * PPM)
TEX = np.array([[1 / PPM, 0, 0.5 / PPM], [0, -1 / PPM, (TH - 0.5) / PPM], [0, 0, 1]])  # texture px -> board mm


def texture(rng, shape):
    """Contrasty multi-scale blobs: plenty of SIFT features everywhere."""
    t = sum(cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), sg) * sg for sg in (1.5, 3, 6, 12))
    return np.clip(128 + 60 * t / t.std(), 0, 255)


def camera(C, look=(W / 2, H / 2, 0)):
    """R, t of a camera at C looking at `look`, image up = board +y (OpenCV: x right, y down, z forward)."""
    z = np.subtract(look, C) / np.linalg.norm(np.subtract(look, C))
    x = np.cross(z, [0, 1, 0]) if abs(z[1]) < 0.99 else np.cross(z, [1, 0, 0])
    x /= np.linalg.norm(x)
    R = np.array([x, np.cross(z, x), z])
    return R, -R @ C


def make(run, board):
    (run / "frames").mkdir(parents=True)
    rng = np.random.default_rng(1)
    base, head = texture(rng, (TH, TW)), texture(rng, (TH, TW))
    n = round(SIDE * PPM)
    for i, (x, y) in MARKERS.items():
        u, v, q = round(x * PPM - n / 2), round((H - y) * PPM - n / 2), 8 * PPM  # white quiet zone
        base[v - q:v + n + q, u - q:u + n + q] = 255
        base[v:v + n, u:u + n] = np.where(cv2.aruco.generateImageMarker(s.DICT, i, n) > 0, 235, 20)
    yy, xx = np.mgrid[:TH, :TW]
    bx, by = (xx + 0.5) / PPM, H - (yy + 0.5) / PPM
    dets = {}
    for k in range(24):
        a = 2 * np.pi * k / 24
        C = np.array([W / 2 + 130 * np.cos(a) + 40 * np.cos(3 * a), H / 2 + 130 * np.sin(a), 420 + 30 * np.sin(2 * a)])
        R, t = camera(C)
        flags = cv2.INTER_LINEAR  # the matrices map texture px -> image px (forward)
        img = cv2.warpPerspective(base, K @ np.c_[R[:, 0], R[:, 1], t] @ TEX, SIZE, flags=flags)
        for x, y, h in TARGETS.values():  # a head is a disk in the plane z = h: its own homography
            if not h:
                continue
            alpha = ((np.hypot(bx - x, by - y) <= HEAD) * 255).astype(np.float32)
            Hh = K @ np.c_[R[:, 0], R[:, 1], R[:, 2] * h + t] @ TEX
            m = cv2.warpPerspective(alpha, Hh, SIZE, flags=flags) / 255
            img = img * (1 - m) + cv2.warpPerspective(head, Hh, SIZE, flags=flags) * m
        img = np.clip(img + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
        name = f"sweep_{k:06d}.jpg"
        cv2.imwrite(str(run / "frames" / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        dets[f"frames/{name}"] = s.detect(img)
    (run / "detections.json").write_text(json.dumps({"every": 1, "images": dets}))
    (run / "targets.json").write_text(json.dumps(board))
    return dets


BOARD = {"board_size_mm": [W, H], "marker_mm": SIDE,
         "markers": [{"id": i, "x_mm": x, "y_mm": y} for i, (x, y) in MARKERS.items()],
         "targets": [{"id": t, "measured_mm": h, "x_mm": x, "y_mm": y} for t, (x, y, h) in TARGETS.items()]}

with tempfile.TemporaryDirectory() as tmp:
    run = Path(sys.argv[1] if sys.argv[1:] else tmp) / "synthetic-colmap"
    dets = make(run, BOARD)
    assert all(len(d) == 4 for d in dets.values()), [len(d) for d in dets.values()]  # all 4 markers in every frame

    cands, rep = colmap.run(run, h_min=2)
    assert rep["images_registered"] >= 20 and rep["mean_reprojection_px"] < 1.5, rep
    # the fit pins scale and pose to the ruler positions: marker corners come back within 0.5 mm, flat
    assert rep["marker_corner_rms_xy_mm"] < 0.5 and rep["marker_corner_rms_z_mm"] < 0.5, rep
    assert abs(rep["plane_median_z_mm"]) < 0.5 and rep["plane_sigma_z_mm"] < 0.5, rep
    # real parallax: the 10 mm head must come back near 10 mm, the 4 mm near 4 mm
    for t in ("T05", "T06", "T07", "T08"):
        v = rep["targets"][t]
        assert v["points"] and abs(v["median_z_mm"] - v["measured_mm"]) < 0.5, (t, v)
    assert rep["targets"]["T01"]["max_z_mm"] is None or rep["targets"]["T01"]["max_z_mm"] < 1, rep["targets"]["T01"]
    assert len(cands["covered_targets"]) == len(TARGETS) and cands["covered_area_cm2"] > 800, cands
    # blind: the four heads at 3.2 mm and up, nothing else (T02 at 1 mm is under h_min, T01 is flush)
    near = sorted(min((np.hypot(c["x_mm"] - x, c["y_mm"] - y), t) for t, (x, y, _) in TARGETS.items())[::-1]
                  for c in cands["candidates"])
    assert [t for t, _ in near] == ["T05", "T06", "T07", "T08"] and max(d for _, d in near) < 5, near
print("ok")
