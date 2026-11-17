"""Scale from one measured length (the "measure once" idea, 2026-11-17).

Pick two points that are easy to see (two bolt heads, two tick marks on a tape measure lying in the scene), measure the
straight distance between them with a tape or calipers, and tap each point in a few frames taken from clearly different
positions. Each point is triangulated with COLMAP's poses; the measured length over their distance in SfM units is the
scale in mm per unit, and COLMAP's camera path carries that one scale through the whole video. The printed markers, if
they are in the footage, only score it (scale.json).

  python methods/length_scale.py runs/<run> --taps taps.json --mm 343.97
  python methods/length_scale.py --selftest                     (synthetic cameras only)

taps.json: {"A": {"frame.jpg": [x, y], ...}, "B": {"frame.jpg": [x, y], ...}} in full-size frame pixels, OpenCV
convention (the first pixel's centre is 0,0). Each point needs at least 2 frames; 4 or more from around the orbit is better.
Reads runs/<run>/methods/walkaround/sfm.json; writes length_scale.json beside it. Nothing existing is changed.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np

from walkaround import cv_camera, triangulate


def locate(taps, poses, cams, max_err_px=12.0, min_angle_deg=2.0):
    """One tapped point -> its 3D position in SfM units. taps {frame: (x, y)}; poses {frame: (R, t)} cam_from_world;
    cams {frame: (K, dist)} from cv_camera. Drops the worst tap while it is far off the others (and 3+ remain)."""
    names = [n for n in taps if n in poses]
    if len(names) < 2:
        raise ValueError(f"a point needs taps in at least 2 registered frames, got {len(names)}")
    xn = {n: cv2.undistortPoints(np.array(taps[n], float).reshape(1, 1, 2), *cams[n]).reshape(2) for n in names}
    while True:
        X = triangulate([np.hstack([poses[n][0], poses[n][1].reshape(3, 1)]) for n in names], [xn[n] for n in names])
        err = {}
        for n in names:
            R, t = poses[n]
            K, d = cams[n]
            p = cv2.projectPoints(X.reshape(1, 3), cv2.Rodrigues(R)[0], t, K, d)[0].reshape(2)
            err[n] = float(np.linalg.norm(p - np.asarray(taps[n], float)))
        worst = max(err, key=err.get)
        if len(names) >= 3 and err[worst] > max(max_err_px, 3 * np.median(list(err.values()))):
            names.remove(worst)
            continue
        break
    cen = [-poses[n][0].T @ poses[n][1] for n in names]
    rays = [(X - c) / np.linalg.norm(X - c) for c in cen]
    ang = max(math.degrees(math.acos(np.clip(a @ b, -1, 1))) for a in rays for b in rays)
    if ang < min_angle_deg:
        raise ValueError(f"the frames see this point from nearly the same place (widest ray angle {ang:.2f} deg); "
                         "tap it in frames from different positions")
    return {"X": X, "frames_used": names, "dropped": sorted(set(taps) & set(poses) - set(names)),
            "reproj_px": float(np.sqrt(np.mean([err[n] ** 2 for n in names]))), "max_ray_angle_deg": ang}


def scale_from_taps(taps, length_mm, poses, cams):
    a, b = locate(taps["A"], poses, cams), locate(taps["B"], poses, cams)
    d = float(np.linalg.norm(a["X"] - b["X"]))
    out = {"scale_mm_per_unit": length_mm / d, "length_mm": length_mm, "length_units": d}
    for k, r in (("A", a), ("B", b)):
        out[k] = {"frames_used": r["frames_used"], "dropped": r["dropped"], "reproj_px": r["reproj_px"],
                  "max_ray_angle_deg": r["max_ray_angle_deg"], "X_units": r["X"].tolist()}
    return out


def load_sfm(run_dir):
    sfm = json.load(open(os.path.join(run_dir, "methods", "walkaround", "sfm.json")))
    poses, cams = {}, {}
    for im in sfm["images"]:
        M = np.array(im["cam_from_world"], float)
        poses[im["name"]] = (M[:, :3], M[:, 3])
        cams[im["name"]] = cv_camera(sfm["cameras"][str(im["camera_id"])])
    return poses, cams


def run(run_dir, taps_file, length_mm):
    poses, cams = load_sfm(run_dir)
    res = scale_from_taps(json.load(open(taps_file)), length_mm, poses, cams)
    w = os.path.join(run_dir, "methods", "walkaround")
    sp = os.path.join(w, "scale.json")
    if os.path.exists(sp):
        marker = json.load(open(sp))["scale_mm_per_unit"]
        res["marker_scale_mm_per_unit"] = marker
        res["error_vs_marker_pct"] = 100 * (res["scale_mm_per_unit"] / marker - 1)
    res["taps_file"] = os.path.abspath(taps_file)
    json.dump(res, open(os.path.join(w, "length_scale.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------ self-test (synthetic cameras around a board)
def selftest():
    rng = np.random.default_rng(3)
    k_true = 47.0                                             # 1 SfM unit = 47 mm
    cam = {"model": "SIMPLE_RADIAL", "params": [3500.0, 1920.0, 1080.0, 0.02]}
    K, dist = cv_camera(cam)
    A_mm, B_mm = np.array([40.0, 30.0, 3.0]), np.array([290.0, 275.0, 10.0])   # like T04 and T08 on the tex board
    poses, cams, taps = {}, {}, {"A": {}, "B": {}}
    for i in range(16):
        ang = 2 * np.pi * i / 16
        pos = np.array([165 + 160 * np.cos(ang), 150 + 140 * np.sin(ang), 260.0])
        z = np.array([165.0, 150.0, 0.0]) - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, [0.0, 1.0, 0.0])
        x /= np.linalg.norm(x)
        R = np.array([x, np.cross(z, x), z])
        t = -R @ pos
        n = f"f{i:02d}.jpg"
        poses[n], cams[n] = (R, t / k_true), (K, dist)       # poses in SfM units
        for k, P in (("A", A_mm), ("B", B_mm)):
            if i % 2 == (k == "B"):                          # each point tapped in 8 frames, different ones
                p = cv2.projectPoints(P.reshape(1, 3), cv2.Rodrigues(R)[0], t, K, dist)[0].reshape(2)
                taps[k][n] = (p + rng.normal(0, 3.0, 2)).tolist()   # a 3 px tap error
    L = float(np.linalg.norm(A_mm - B_mm))
    r = scale_from_taps(taps, L, poses, cams)
    assert abs(r["scale_mm_per_unit"] / k_true - 1) < 0.005, r
    assert len(r["A"]["frames_used"]) == 8 and r["A"]["reproj_px"] < 6, r
    # a tap on the wrong thing in one frame is dropped and the scale holds
    bad = {"A": dict(taps["A"]), "B": dict(taps["B"])}
    bad["A"]["f04.jpg"] = [bad["A"]["f04.jpg"][0] + 80, bad["A"]["f04.jpg"][1] - 60]
    r2 = scale_from_taps(bad, L, poses, cams)
    assert r2["A"]["dropped"] == ["f04.jpg"] and abs(r2["scale_mm_per_unit"] / k_true - 1) < 0.005, r2
    # a wrong typed length moves the scale in proportion
    assert abs(scale_from_taps(taps, 1.03 * L, poses, cams)["scale_mm_per_unit"] / (1.03 * k_true) - 1) < 0.005
    # one frame, or two frames from the same place, is refused
    for one in ({"f00.jpg": taps["A"]["f00.jpg"]},
                {"f00.jpg": taps["A"]["f00.jpg"], "dup.jpg": taps["A"]["f00.jpg"]}):
        p2, c2 = dict(poses), dict(cams)
        p2["dup.jpg"], c2["dup.jpg"] = (poses["f00.jpg"][0], poses["f00.jpg"][1] + 1e-4), (K, dist)
        try:
            locate(one, p2, c2)
            raise AssertionError("a point seen from one place was accepted")
        except ValueError:
            pass
    print(f"selftest ok: scale {r['scale_mm_per_unit']:.3f} vs {k_true} ({100 * (r['scale_mm_per_unit'] / k_true - 1):+.2f} %) "
          f"from a {L:.1f} mm length, 3 px tap noise, 8 frames per point (reprojection {r['A']['reproj_px']:.1f} / "
          f"{r['B']['reproj_px']:.1f} px); a wrong tap dropped; a wrong length scales in proportion; one viewpoint refused")
    print("NOT tested here: real taps on real frames, real COLMAP poses (run it on a run folder for that), a tape measured along a curve")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--taps")
    ap.add_argument("--mm", type=float, help="the measured straight distance between point A and point B, in mm")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.run:
        selftest()
        sys.exit(0)
    if not (a.taps and a.mm):
        ap.error("--taps and --mm are both needed with a run folder")
    r = run(a.run, a.taps, a.mm)
    print(f"length scale {r['scale_mm_per_unit']:.3f} mm/unit from {a.mm} mm = {r['length_units']:.4f} units; "
          f"A: {len(r['A']['frames_used'])} frames, {r['A']['reproj_px']:.1f} px, {r['A']['max_ray_angle_deg']:.0f} deg; "
          f"B: {len(r['B']['frames_used'])} frames, {r['B']['reproj_px']:.1f} px, {r['B']['max_ray_angle_deg']:.0f} deg"
          + (f"; marker {r['marker_scale_mm_per_unit']:.3f} -> error {r['error_vs_marker_pct']:+.2f} %" if "error_vs_marker_pct" in r else ""))
