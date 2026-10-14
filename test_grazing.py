"""python test_grazing.py [keep_dir] -- methods/grazing.py on a synthetic grazing-light run (no camera needed).
Bolt shadows of known length drawn on the lean board, seen from two tripod positions, then score.py end to end.
Pass a folder to keep the synthetic run there."""
import contextlib
import io
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import geometry as g
import slidescan as s
from methods import grazing as gz

ROOT = Path(__file__).resolve().parent
PPM, W, H, SIDE, HEAD = 5, 333.38, 349.25, 38.3, 5.5  # drawing px/mm; lean board mm; marker side; head radius
MARKERS = {0: (108.16, 95.25), 1: (228.6, 94.08), 2: (112.0, 222.25), 3: (222.25, 215.9)}
TARGETS = {"T01": (288.93, 34.24, 0), "T02": (165.1, 34.24, 1), "T05": (44.63, 152.98, 3.2),
           "T06": (165.1, 152.98, 4), "T08": (288.93, 276.22, 10)}
REF = (166.69, 174.63)
LIGHT = {"light_azimuth_deg": 0, "light_distance_mm": 600, "light_elevation_deg": 15}  # off the +x edge
LXY, LZ = np.add(REF, (600, 0)), 600 * math.tan(math.radians(15))
B = np.array([[PPM, 0, -0.5], [0, -PPM, H * PPM - 0.5], [0, 0, 1]])  # board mm -> drawing px (pixel centers)
CW, CH = round(W * PPM), round(H * PPM)
SIZE = (1400, 1500)
POSES = [cv2.getPerspectiveTransform(np.float32([[0, 0], [CW, 0], [CW, CH], [0, CH]]), np.float32(q)) for q in (
    [[120, 80], [1300, 140], [1250, 1420], [60, 1350]],    # p01: whole board
    [[400, 100], [2000, 160], [1950, 1400], [360, 1380]])]  # p02: left ~60% only (T01, T08 out of view)


def true_shadow(x, y, h):
    """Point light: a head edge at height h, d mm from the light horizontally, casts d*h/(z-h) beyond itself."""
    u = np.subtract((x, y), LXY) / math.dist((x, y), LXY)
    return u, (math.dist((x, y), LXY) + HEAD) * h / (LZ - h)


def make(run, board):
    """off = ambient x albedo; on = off + falling-off grazing light x albedo, except in the drawn shadows."""
    (run / "raw").mkdir(parents=True)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:CH, :CW]
    bx, by = (xx + 0.5) / PPM, H - (yy + 0.5) / PPM  # drawing px -> board mm
    albedo = 0.6 + 0.05 * cv2.GaussianBlur(rng.standard_normal((CH, CW)).astype(np.float32), (0, 0), 3) * 3
    n = round(SIDE * PPM)
    for i, (x, y) in MARKERS.items():
        u, v = round((x - SIDE / 2) * PPM), round((H - y - SIDE / 2) * PPM)
        albedo[v:v + n, u:u + n] = np.where(cv2.aruco.generateImageMarker(s.DICT, i, n) > 0, 0.9, 0.05)
    shade = np.zeros((CH, CW), bool)
    for x, y, h in TARGETS.values():
        head = np.hypot(bx - x, by - y) <= HEAD
        if h:
            u, L = true_shadow(x, y, h)
            t = np.clip((bx - x) * u[0] + (by - y) * u[1], 0, L)  # stadium: the head disk swept L along u
            shade |= (np.hypot(bx - x - t * u[0], by - y - t * u[1]) <= HEAD) & ~head
        albedo[head] = 0.75
    r = np.sqrt((bx - LXY[0]) ** 2 + (by - LXY[1]) ** 2 + LZ ** 2)
    lit = 150 * (r.min() / r) ** 3  # irradiance ~ z / r^3: about 5x brighter at the near edge than the far one
    off = 80 * albedo
    on = off + lit * albedo * ~shade
    geo = {}
    for k, P in enumerate(POSES, 1):
        for kind, img in (("off", off), ("on", on)):
            photo = cv2.warpPerspective(img + rng.normal(0, 2, img.shape), P, SIZE)
            cv2.imwrite(str(run / "raw" / f"p{k:02d}_{kind}.jpg"), np.clip(photo, 0, 255).astype(np.uint8),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            geo[f"raw/p{k:02d}_{kind}.jpg"] = {"size": None, "H": None, "markers": [], "err_mm": {}, "covered": []}
        Hk = P @ B  # the pose photo itself isn't needed: geometry.json carries its mapping
        geo[f"raw/p{k:02d}_pose.jpg"] = {"size": list(SIZE), "H": Hk.tolist(), "markers": [0, 1, 2, 3], "err_mm": {},
                                         "covered": [t for t, (x, y, _) in TARGETS.items() if g.in_view(Hk, SIZE, x, y)]}
    (run / "geometry.json").write_text(json.dumps({"images": geo}))
    (run / "targets.json").write_text(json.dumps(board))
    (run / "meta.json").write_text(json.dumps(dict(s.META, date="2026-10-14", device="synthetic", method="grazing-light",
                                                   surface="matte-flat", light_level="bright", repeat=1, **LIGHT)))
    return geo


def nearest(c):
    return min((math.dist((c["x_mm"], c["y_mm"]), xy[:2]), t) for t, xy in TARGETS.items())


BOARD = {"board_size_mm": [W, H], "light_ref_mm": list(REF), "marker_mm": SIDE,
         "markers": [{"id": i, "x_mm": x, "y_mm": y} for i, (x, y) in MARKERS.items()],
         "targets": [{"id": t, "measured_mm": h, "x_mm": x, "y_mm": y} for t, (x, y, h) in TARGETS.items()]}

with tempfile.TemporaryDirectory() as tmp:
    run = Path(sys.argv[1] if sys.argv[1:] else tmp) / "synthetic-grazing"
    geo = make(run, dict(BOARD, markers=[]))
    assert geo["raw/p02_pose.jpg"]["covered"] == ["T02", "T05", "T06"], geo["raw/p02_pose.jpg"]["covered"]

    # markers left out of targets.json -> nothing blanked: their black squares read as shadows (Design Rule 7)
    with contextlib.redirect_stdout(io.StringIO()):
        kept, _ = gz.run(run)
    assert all(any(math.dist((k["x_mm"], k["y_mm"]), m) < SIDE for k in kept) for m in MARKERS.values()), kept

    (run / "targets.json").write_text(json.dumps(BOARD))
    out = subprocess.run([sys.executable, str(ROOT / "methods" / "grazing.py"), str(run)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    c = json.loads((run / "methods" / "grazing" / "candidates.json").read_text())
    heights = json.loads((run / "methods" / "grazing" / "heights.json").read_text())["targets"]

    # blind: one candidate per shadow >= 5 mm, near its bolt, both positions merged; T02's 3.9 mm shadow is under
    # L_MIN, T01 is flush, and the blanked markers give none
    near = sorted(nearest(k)[::-1] for k in c["candidates"])
    assert [t for t, _ in near] == ["T05", "T06", "T08"] and max(d for _, d in near) < 5, near
    assert "5 before merging" in out.stdout, out.stdout  # T05, T06 seen from both positions
    assert sorted(c["covered_targets"]) == sorted(TARGETS)
    area = (W * H - len(MARKERS) * (SIDE + 2 * gz.MARKER_PAD_MM) ** 2) / 100  # p01 sees the whole board
    assert abs(c["covered_area_cm2"] - area) < 5, (c["covered_area_cm2"], area)

    # targeted: every drawn shadow measured within 1 mm; heights back from L*tan(e)
    for t, (x, y, h) in TARGETS.items():
        L = true_shadow(x, y, h)[1] if h else 0
        assert abs(heights[t]["shadow_mm"] - L) < 1, (t, heights[t], L)
        assert abs(heights[t]["est_height_mm"] - h) < 0.3, (t, heights[t])
    assert heights["T05"]["images"] == 2 and heights["T08"]["images"] == 1

    out = subprocess.run([sys.executable, str(ROOT / "score.py"), str(run), "--method", "grazing", "--gate"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    gate = out.stdout.split("Gate A")[1]
    assert "1 runs scored" in out.stdout and "3/3" in gate and "PASS" in gate, out.stdout
    res = json.loads((run / "methods" / "grazing" / "results.json").read_text())
    assert res["background_alarms"] == 0 and res["targets"]["T08"]["est_height_mm"] > 9.5, res
print("ok")
