"""Marker-free scale from a visual-inertial path (phone motion sensors), step 2 of the sensor-route test.

COLMAP gives camera centres in an arbitrary unit. An app with visual-inertial tracking (for example Spectacular Rec
processed with `sai-cli process`, or ARKit/ARCore) gives camera positions in metres on the same recording. If the
two paths are matched by time, the similarity transform between them has a scale: mm per SfM unit. That scale
replaces the printed marker's (scale.json) in walkaround.py. The marker, if present in the footage, only scores it.

  from sensor_scale import scale_from_paths
  r = scale_from_paths(t_colmap_s, centres_units, t_vio_s, positions_m)      # seconds, (N,3) units, seconds, (M,3) metres
  r["scale_mm_per_unit"], r["offset_s"], r["rms_mm"], r["drift_pct"]

  python methods/sensor_scale.py --selftest        (synthetic paths only)

NOT written yet: the reader for Spectacular Rec's recording layout (needs a real recording), and the option that
lets walkaround.py take this scale instead of the marker's. Nothing existing is changed.
"""
import argparse

import numpy as np


def umeyama(X, Y):
    """Similarity fit Y ~ s R X + T (least squares). X, Y are (n,3). Returns s, R, T."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    var_x = (Xc ** 2).sum() / len(X)
    U, S, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / var_x)
    return s, R, my - s * R @ mx


def _trimmed_fit(X, Y, keep=0.8, iters=4):
    """Fit, drop the worst (1-keep) of the points by residual, refit. -> s, R, T, rms of the kept points, kept mask."""
    mask = np.ones(len(X), bool)
    for _ in range(iters):
        s, R, T = umeyama(X[mask], Y[mask])
        res = np.linalg.norm(s * X @ R.T + T - Y, axis=1)
        cut = np.quantile(res, keep)
        new = res <= cut
        if new.sum() < 10 or (new == mask).all():
            break
        mask = new
    s, R, T = umeyama(X[mask], Y[mask])
    res = np.linalg.norm(s * X[mask] @ R.T + T - Y[mask], axis=1)
    return s, R, T, float(np.sqrt((res ** 2).mean())), mask


def _interp(t, P, tq):
    return np.c_[[np.interp(tq, t, P[:, k]) for k in range(3)]].T


def scale_from_paths(t_colmap, C_units, t_vio, P_m, max_offset=0.2, step=0.01, keep=0.8, min_extent_mm=50.0):
    """Mm per SfM unit from matching a COLMAP camera path with a metric visual-inertial path.

    t_colmap (N,) seconds, C_units (N,3) camera centres in SfM units; t_vio (M,) seconds, P_m (M,3) metres.
    The VIO path is interpolated at the COLMAP times shifted by a trial offset (-max_offset..max_offset); the offset with
    the smallest trimmed rms wins. Returns scale_mm_per_unit, offset_s, rms_mm, n_used, extent_mm, drift_pct
    (scale from the first half of the path vs the second half; a large value means the VIO path drifts or the
    time matching is off)."""
    t_colmap, t_vio = np.asarray(t_colmap, float), np.asarray(t_vio, float)
    C, P = np.asarray(C_units, float), np.asarray(P_m, float) * 1000.0
    order = np.argsort(t_vio)
    t_vio, P = t_vio[order], P[order]
    extent = float(np.linalg.norm(P.max(0) - P.min(0)))
    if extent < min_extent_mm:
        raise ValueError(f"the VIO path spans only {extent:.0f} mm; the sensors need real travel (at least {min_extent_mm:.0f} mm) to fix a scale")
    best = None
    for off in np.arange(-max_offset, max_offset + step / 2, step):
        tq = t_colmap + off
        ok = (tq >= t_vio[0]) & (tq <= t_vio[-1])
        if ok.sum() < 20:
            continue
        s, R, T, rms, mask = _trimmed_fit(C[ok], _interp(t_vio, P, tq[ok]), keep)
        if best is None or rms < best["rms_mm"]:
            best = {"scale_mm_per_unit": s, "offset_s": float(off), "rms_mm": rms, "n_used": int(mask.sum()), "ok": ok}
    if best is None:
        raise ValueError("fewer than 20 COLMAP frames fall inside the VIO time range; check the two clocks")
    ok = best.pop("ok")
    idx = np.flatnonzero(ok)
    tq = t_colmap[idx] + best["offset_s"]
    Y = _interp(t_vio, P, tq)
    half = len(idx) // 2
    sa = _trimmed_fit(C[idx[:half]], Y[:half], keep)[0]
    sb = _trimmed_fit(C[idx[half:]], Y[half:], keep)[0]
    best["extent_mm"] = extent
    best["drift_pct"] = 100.0 * (sb / sa - 1.0)
    return best


# ------------------------------------------------------------------ self-test (synthetic paths only)
def _path(rng, n, travel_mm=350.0):
    """A hand-held-looking orbit: a sweep with sideways wobble and some height change, in mm."""
    u = np.linspace(0, 1, n)
    return np.c_[travel_mm * u, 60 * np.sin(2 * np.pi * 2 * u) + 20 * rng.normal(size=n).cumsum() / np.sqrt(n),
                 250 + 40 * np.sin(2 * np.pi * 1 * u)]


def selftest():
    rng = np.random.default_rng(3)
    n, k_true = 300, 47.0                                  # frames, mm per SfM unit
    t = np.linspace(0, 20, n)                              # 20 s at 15 frames per second
    path_mm = _path(rng, n)
    th = 0.7                                               # COLMAP's frame is rotated and shifted relative to the VIO's
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    C = (path_mm @ Rz.T + np.array([100.0, -40.0, 5.0])) / k_true
    C = C + rng.normal(0, 0.02, C.shape)                   # COLMAP noise: 0.02 unit, about 1 mm
    t_v = np.linspace(-1, 21, 2200)                        # VIO at 100 Hz, wider than the video
    P_true = np.c_[[np.interp(t_v, t, path_mm[:, j]) for j in range(3)]].T

    # 1. noise only: scale within 1 %, no time offset
    P = (P_true + rng.normal(0, 2.0, P_true.shape)) / 1000.0
    r = scale_from_paths(t, C, t_v, P)
    assert abs(r["scale_mm_per_unit"] / k_true - 1) < 0.01, r
    assert abs(r["offset_s"]) <= 0.011 and abs(r["drift_pct"]) < 2, r

    # 2. the VIO clock is 80 ms ahead: the search finds it and the scale is still right
    P2 = (np.c_[[np.interp(t_v - 0.08, t, path_mm[:, j]) for j in range(3)]].T + rng.normal(0, 2.0, (len(t_v), 3))) / 1000.0
    r = scale_from_paths(t, C, t_v, P2)
    assert abs(r["scale_mm_per_unit"] / k_true - 1) < 0.015, r
    assert abs(r["offset_s"] - 0.08) <= 0.021, r   # offset = seconds added to COLMAP's times to land on the VIO clock

    # 3. 10 % of the VIO samples are glitches of 60 to 150 mm: trimming keeps the scale within 1.5 %
    P3 = P_true.copy()
    bad = rng.random(len(t_v)) < 0.10
    P3[bad] += rng.uniform(60, 150, (bad.sum(), 3)) * rng.choice([-1, 1], (bad.sum(), 3))
    r = scale_from_paths(t, C, t_v, (P3 + rng.normal(0, 2.0, P3.shape)) / 1000.0)
    assert abs(r["scale_mm_per_unit"] / k_true - 1) < 0.015, r

    # 4. a VIO that is itself 3 % too long gives a scale 3 % too large (the method is only as good as the sensors)
    r = scale_from_paths(t, C, t_v, P * 1.03)
    assert abs(r["scale_mm_per_unit"] / (k_true * 1.03) - 1) < 0.01, r

    # 5. a VIO that drifts 5 % over the clip shows up as drift_pct, not silently
    stretch = 1 + 0.05 * np.clip(t_v / 20.0, 0, 1)
    r = scale_from_paths(t, C, t_v, (P_true * stretch[:, None]) / 1000.0)
    assert r["drift_pct"] > 2.0, r

    # 6. too little travel is refused
    try:
        scale_from_paths(t, C[:, :], t_v, P_true[:, :] * 0.0 / 1000.0 + 0.001)
        raise AssertionError("a path with no travel was accepted")
    except ValueError:
        pass
    print("selftest ok: scale within 1 % under 2 mm noise, an 80 ms clock offset found, 10 % glitches trimmed,")
    print("a 3 % wrong VIO passes through as 3 %, 5 % drift reported, a path with no travel refused")
    print("NOT tested: real Spectacular/ARKit paths, their file format, the scale-override in walkaround.py, real COLMAP paths")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    selftest()
