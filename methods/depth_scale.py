"""Marker-free scale test: can a single-image metric-depth network (Apple Depth Pro) give COLMAP's millimetre scale?

COLMAP's poses and points have an arbitrary unit. walkaround.py fixes it with the printed marker (scale.json,
mm per SfM unit). Here the network predicts metric depth for some frames; for each COLMAP point seen in that frame,
(network depth in mm) / (COLMAP depth in units) is one estimate of mm per unit. The median over points, then over
frames, is the marker-free scale. The run's marker scale is the ground truth to compare it with.

  python methods/depth_scale.py runs/<run> [--frames 10] [--net depthpro|moge2|moge2-fov] [--model ...]   (run in WSL; torch + transformers,
                                                                                      or PYTHONPATH=~/moge_pkgs for moge2)
  python methods/depth_scale.py --selftest                                          (no model needed; tests the fit only)

Reads  runs/<run>/methods/walkaround/{sfm.json, scale.json, colmap/sparse/0}  and the frames sfm.json points to
(or runs/<run>/frames). Writes runs/<run>/methods/walkaround/depth_scale.json. Nothing existing is changed.
Heights scale in proportion to the scale, so an X% scale error is an X% height error.
"""
import argparse
import json
import os
import sys

import numpy as np


# ------------------------------------------------------------------ the fit (pure, tested by --selftest)
def project(X, R, t, cam):
    """World points (N,3) -> pixel u, v and camera-frame depth z. cam = SIMPLE_RADIAL or PINHOLE-like COLMAP params."""
    Xc = X @ R.T + t
    z = Xc[:, 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
    p = cam["params"]
    if cam["model"] == "SIMPLE_RADIAL":
        f, cx, cy, k = p
        d = 1 + k * (x * x + y * y)
        return f * x * d + cx, f * y * d + cy, z
    if cam["model"] == "SIMPLE_PINHOLE":
        f, cx, cy = p
        return f * x + cx, f * y + cy, z
    raise SystemExit(f"camera model {cam['model']} not handled (SIMPLE_RADIAL and SIMPLE_PINHOLE are)")


def fit_scale(u, v, z_units, depth_mm):
    """mm per SfM unit from one frame: median of depth_mm[v, u] / z_units over valid points. -> (scale, n, spread_pct)."""
    H, W = depth_mm.shape
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (z_units > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    d = np.where(ok, depth_mm[np.clip(vi, 0, H - 1), np.clip(ui, 0, W - 1)], np.nan)
    r = d / np.where(z_units > 0, z_units, np.nan)
    r = r[np.isfinite(r) & (r > 0)]
    if len(r) < 20:
        return None, len(r), None
    s = float(np.median(r))
    spread = 100.0 * 1.4826 * float(np.median(np.abs(r - s))) / s     # robust sigma of the per-point ratios, % of the scale
    return s, len(r), spread


def combine(per_frame):
    """Frame scales -> (median, robust spread %, n frames)."""
    s = np.array([f["scale"] for f in per_frame if f["scale"]], float)
    if len(s) == 0:
        return None, None, 0
    m = float(np.median(s))
    return m, 100.0 * 1.4826 * float(np.median(np.abs(s - m))) / m, len(s)


# ------------------------------------------------------------------ the network + COLMAP side (not covered by --selftest)
def predict_depth(model_dir, image_paths, device="cuda"):
    """Depth Pro via transformers -> list of (depth_mm HxW float32, focal_px).
    Pre- and post-processing are done here (the model's own processor needs torchvision, which this venv lacks);
    they follow its preprocessor_config.json (1536x1536 bilinear, /255, mean 0.5, std 0.5) and its post_process_depth_estimation."""
    import math
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import DepthProForDepthEstimation
    model = DepthProForDepthEstimation.from_pretrained(model_dir).to(device).eval()
    out = []
    for p in image_paths:
        im = np.asarray(Image.open(p).convert("RGB"))
        H, W = im.shape[:2]
        x = torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255.0).to(device)
        x = F.interpolate(x, size=(1536, 1536), mode="bilinear", antialias=True, align_corners=False)
        x = (x - 0.5) / 0.5
        with torch.no_grad():
            o = model(pixel_values=x)
        inv = o.predicted_depth[0]                                            # canonical inverse depth, model resolution
        fov = float(o.field_of_view[0])
        f = 0.5 * W / math.tan(0.5 * math.radians(fov))
        inv = inv * W / f
        inv = F.interpolate(inv[None, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        d = (1.0 / inv.clamp(1e-4, 1e4)).float().cpu().numpy() * 1000.0
        out.append((d, f))
        del x, o, inv
        torch.cuda.empty_cache()
    return out


def predict_depth_moge2(model_ref, image_paths, device="cuda", fov_x=None):
    """MoGe-2 (metric) -> list of (depth_mm HxW float32, focal_px). model_ref: HF repo id or a local .pt.
    Needs the `moge` and `utils3d_moge` packages (installed with --no-deps into ~/moge_pkgs; put that folder on PYTHONPATH)."""
    import torch
    from moge.model.v2 import MoGeModel
    from PIL import Image
    model = MoGeModel.from_pretrained(model_ref).to(device).eval()
    out = []
    for k, p in enumerate(image_paths):
        im = np.asarray(Image.open(p).convert("RGB")).copy()
        H, W = im.shape[:2]
        x = torch.from_numpy(im).to(device).float().div(255.0).permute(2, 0, 1)
        with torch.no_grad():
            o = model.infer(x, fov_x=None if fov_x is None else fov_x[k])
        d = o["depth"].float().cpu().numpy() * 1000.0                        # metres -> mm; masked pixels are inf/nan, dropped by fit_scale
        f = float(o["intrinsics"][0, 0]) * W                                 # normalised intrinsics -> pixels
        out.append((d, f))
        del x, o
        torch.cuda.empty_cache()
    return out


def run(run_dir, model_dir, n_frames=10, device="cuda", net="depthpro"):
    import pycolmap
    w = os.path.join(run_dir, "methods", "walkaround")
    sfm = json.load(open(os.path.join(w, "sfm.json")))
    marker_scale = json.load(open(os.path.join(w, "scale.json")))["scale_mm_per_unit"]
    img_dir = sfm["image_dir"]
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(run_dir, "frames")
    rec = pycolmap.Reconstruction(os.path.join(w, "colmap", "sparse", "0"))
    by_name = {im.name: im for im in rec.images.values()}
    imgs = sorted(sfm["images"], key=lambda i: i["name"])
    pick = np.unique(np.linspace(0, len(imgs) - 1, min(n_frames, len(imgs))).round().astype(int))
    chosen = [imgs[i] for i in pick]
    paths = [os.path.join(img_dir, c["name"]) for c in chosen]
    if net == "moge2-fov":          # give MoGe-2 the field of view COLMAP estimated (SfM needs no marker), instead of its own guess
        import math
        fovs = [math.degrees(2 * math.atan(sfm["cameras"][c["camera_id"]]["width"] / (2 * sfm["cameras"][c["camera_id"]]["params"][0]))) for c in chosen]
        preds = predict_depth_moge2(model_dir, paths, device, fov_x=fovs)
    elif net == "moge2":
        preds = predict_depth_moge2(model_dir, paths, device)
    else:
        preds = predict_depth(model_dir, paths, device)
    per_frame = []
    for c, (depth_mm, f_pred) in zip(chosen, preds):
        im = by_name[c["name"]]
        ids = [p.point3D_id for p in im.points2D if p.has_point3D()]
        X = np.array([rec.points3D[i].xyz for i in ids], float)
        M = np.array(c["cam_from_world"], float)
        cam = sfm["cameras"][c["camera_id"]]
        u, v, z = project(X, M[:, :3], M[:, 3], cam)
        s, n, spread = fit_scale(u, v, z, depth_mm)
        s_f = s * cam["params"][0] / f_pred if (s and f_pred) else None   # depth scales with focal length: use COLMAP's focal
        per_frame.append({"name": c["name"], "scale": s, "scale_focal_corrected": s_f, "n_points": n,
                          "point_spread_pct": spread, "net_focal_px": f_pred, "colmap_focal_px": cam["params"][0],
                          "error_pct": None if s is None else 100.0 * (s / marker_scale - 1.0),
                          "error_focal_corrected_pct": None if s_f is None else 100.0 * (s_f / marker_scale - 1.0)})
    s_all, sp_all, nf = combine(per_frame)
    s_fc, sp_fc, _ = combine([{"scale": f["scale_focal_corrected"]} for f in per_frame])
    res = {"marker_scale_mm_per_unit": marker_scale, "frames_used": nf,
           "depth_scale_mm_per_unit": s_all, "frame_spread_pct": sp_all,
           "error_pct": None if s_all is None else 100.0 * (s_all / marker_scale - 1.0),
           "depth_scale_focal_corrected": s_fc, "frame_spread_focal_corrected_pct": sp_fc,
           "error_focal_corrected_pct": None if s_fc is None else 100.0 * (s_fc / marker_scale - 1.0),
           "per_frame": per_frame, "model_dir": model_dir, "net": net}
    json.dump(res, open(os.path.join(w, "depth_scale.json" if net == "depthpro" else f"depth_scale_{net}.json"), "w"), indent=1)
    return res


# ------------------------------------------------------------------ self-test (synthetic; no network, no COLMAP)
def selftest():
    rng = np.random.default_rng(1)
    W, H, f = 1600, 900, 1500.0
    cam = {"model": "SIMPLE_RADIAL", "params": [f, W / 2, H / 2, 0.02]}
    R = np.eye(3)
    t = np.zeros(3)
    # a plane about 300 mm in front of the camera, tilted; SfM units are 1 unit = k_true mm
    k_true = 47.0
    n_pts = 400
    xy = rng.uniform(-0.08, 0.08, (n_pts, 2)) * 1000.0                      # mm, stays inside the image
    z_mm = 300.0 + 0.2 * xy[:, 0] + 0.1 * xy[:, 1]
    X_mm = np.c_[xy, z_mm]
    X_units = X_mm / k_true
    u, v, z_units = project(X_units, R, t, cam)
    # build the "network depth" map: true depth of the tilted plane at every pixel, times a 3 % bias plus noise
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    xn, yn = (uu - W / 2) / f, (vv - H / 2) / f
    zp = 300.0 / (1 - 0.2 * xn - 0.1 * yn)                                   # plane z = 300 + 0.2 x + 0.1 y, x = xn z
    for bias in (1.00, 1.03):
        depth = zp * bias * (1 + rng.normal(0, 0.02, zp.shape))
        # 10 % of the points land on a wrong surface (occluder): depth way off
        bad = rng.random(n_pts) < 0.10
        uu_i, vv_i = np.round(u).astype(int), np.round(v).astype(int)
        d2 = depth.copy()
        for a, b in zip(uu_i[bad], vv_i[bad]):
            if 0 <= a < W and 0 <= b < H:
                d2[b, a] *= 2.5
        s, n, spread = fit_scale(u, v, z_units, d2)
        assert s is not None and n > 300, (s, n)
        assert abs(s / (k_true * bias) - 1) < 0.01, (bias, s, k_true * bias)
    # points behind the camera / outside the image are dropped, too few points returns None
    assert fit_scale(u[:5], v[:5], z_units[:5], depth)[0] is None
    s, n, _ = fit_scale(np.r_[u, -50.0], np.r_[v, 10.0], np.r_[z_units, 1.0], depth)
    assert n == n_pts or n == n_pts - 0, n
    m, sp, nf = combine([{"scale": 47.0}, {"scale": 47.5}, {"scale": 46.5}, {"scale": None}])
    assert nf == 3 and abs(m - 47.0) < 1e-9
    print("selftest ok: fit recovers the scale to 1 % with 2 % noise, a 3 % bias and 10 % wrong-surface points")
    print("NOT tested here: Depth Pro itself, the COLMAP reading, real frames (run it on a marked run for that)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--net", choices=["depthpro", "moge2", "moge2-fov"], default="depthpro")
    ap.add_argument("--model", default=None, help="depthpro: model folder (default ~/depthpro); moge2: HF repo id or .pt (default Ruicheng/moge-2-vitl)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest or not a.run:
        selftest()
        sys.exit(0)
    model = a.model or (os.path.expanduser("~/depthpro") if a.net == "depthpro" else "Ruicheng/moge-2-vitl")  # moge2-fov = moge2 told COLMAP's field of view
    r = run(a.run, model, a.frames, a.device, a.net)
    print(f"marker scale {r['marker_scale_mm_per_unit']:.3f} mm/unit | depth scale {r['depth_scale_mm_per_unit']:.3f} "
          f"({r['error_pct']:+.1f} %, frame spread {r['frame_spread_pct']:.1f} %) | focal-corrected "
          f"{r['depth_scale_focal_corrected']:.3f} ({r['error_focal_corrected_pct']:+.1f} %) over {r['frames_used']} frames")
