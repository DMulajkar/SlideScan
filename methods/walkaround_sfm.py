"""walkaround_sfm.py - step 1 of the walk-around method: video frames -> camera poses.

Runs COLMAP's structure-from-motion (through pycolmap) on the frames of one run
and writes the poses and lens model to runs/<run>/methods/walkaround/sfm.json.
Run it where pycolmap installs (WSL, Python 3.12). Step 2 is walkaround.py.

    python methods/walkaround_sfm.py runs/<run> [--images DIR] [--every N]

Nothing in raw/ is modified. Poses are in COLMAP's arbitrary scale; walkaround.py
fixes the scale from the printed tag. pycolmap's wheels are CPU-only, which is
fine here: only sparse SfM runs, no dense stereo.
"""
import argparse
import json
import os
import shutil
import sys
import time

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
OUT_SUB = os.path.join("methods", "walkaround")


def find_images(run_dir, images=None):
    """Return (directory, sorted file names). Looks where slidescan.py frames might put them."""
    cands = [images] if images else [
        os.path.join(run_dir, "frames"),
        os.path.join(run_dir, "raw", "frames"),
        os.path.join(run_dir, "raw"),
    ]
    for d in cands:
        if d and os.path.isdir(d):
            names = sorted(n for n in os.listdir(d) if n.lower().endswith(IMG_EXT))
            if names:
                return d, names
    raise SystemExit(
        "no images found; looked in: " + ", ".join(c for c in cands if c)
        + ". Pass --images DIR (the folder `slidescan.py frames` wrote)."
    )


def rigid_matrix(pose):
    """cam_from_world as a 3x4 list, whether pycolmap exposes it as a property or a method."""
    pose = pose() if callable(pose) else pose
    m = pose.matrix()
    return [[float(v) for v in row] for row in m]


def run(run_dir, images=None, every=1, matcher="auto", max_image_size=3200,
        max_features=8192, camera_model="SIMPLE_RADIAL", overlap=15, focal_35mm=None):
    try:
        import pycolmap
    except ImportError:
        raise SystemExit("pycolmap is not installed here: pip install pycolmap (WSL / Python 3.12)")

    src_dir, names = find_images(run_dir, images)
    names = names[::max(1, every)]
    out_dir = os.path.join(run_dir, OUT_SUB)
    colmap_dir = os.path.join(out_dir, "colmap")
    if os.path.isdir(colmap_dir):
        shutil.rmtree(colmap_dir)  # derived output only; rerunning starts clean
    os.makedirs(colmap_dir)
    db_path = os.path.join(colmap_dir, "database.db")
    sparse_dir = os.path.join(colmap_dir, "sparse")
    os.makedirs(sparse_dir)

    print(f"{len(names)} images from {src_dir}")
    t0 = time.time()

    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = camera_model
    if focal_35mm:
        # starting guess for the focal length from the 35 mm-equivalent value (iPhone main camera: 26).
        # A flat board is a hard case for SfM without it; bundle adjustment still refines the value.
        import cv2
        probe = cv2.imread(os.path.join(src_dir, names[0]), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        h, w = probe.shape[:2]
        f = max(w, h) * focal_35mm / 36.0
        pars = {"SIMPLE_PINHOLE": f"{f},{w / 2},{h / 2}", "SIMPLE_RADIAL": f"{f},{w / 2},{h / 2},0",
                "PINHOLE": f"{f},{f},{w / 2},{h / 2}", "RADIAL": f"{f},{w / 2},{h / 2},0,0",
                "OPENCV": f"{f},{f},{w / 2},{h / 2},0,0,0,0"}
        reader.camera_params = pars[camera_model]
        print(f"focal length guess {f:.0f} px from {focal_35mm} mm (35 mm equivalent)")
    ext = pycolmap.FeatureExtractionOptions()
    ext.max_image_size = max_image_size
    ext.sift.max_num_features = max_features
    # one lens for the whole video: self-calibrates from many views
    pycolmap.extract_features(db_path, src_dir, image_names=names,
                              camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=ext)
    print(f"features: {time.time() - t0:.0f} s")

    if matcher == "auto":
        matcher = "exhaustive" if len(names) <= 200 else "sequential"
    if matcher == "exhaustive":
        pycolmap.match_exhaustive(db_path)
    else:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = overlap
        pycolmap.match_sequential(db_path, pairing_options=pairing)
    print(f"matching ({matcher}): {time.time() - t0:.0f} s")

    recs = pycolmap.incremental_mapping(db_path, src_dir, sparse_dir)
    if not recs:
        raise SystemExit("COLMAP built no model. Typical causes: too little overlap between frames, "
                         "motion blur, or a plain board (add texture) - see WALKAROUND_README.md.")
    best_id = max(recs, key=lambda k: recs[k].num_reg_images())
    rec = recs[best_id]
    print(f"mapping: {time.time() - t0:.0f} s; {len(recs)} model(s), "
          f"using #{best_id} with {rec.num_reg_images()} of {len(names)} images")

    cameras = {}
    for cid, cam in rec.cameras.items():
        cameras[str(cid)] = {
            "model": cam.model_name, "width": int(cam.width), "height": int(cam.height),
            "params": [float(p) for p in cam.params],
        }
    imgs = []
    for iid in rec.reg_image_ids():
        im = rec.images[iid]
        m = rigid_matrix(im.cam_from_world)
        imgs.append({"name": im.name, "camera_id": str(im.camera_id), "cam_from_world": m})
    imgs.sort(key=lambda r: r["name"])

    sfm = {
        "image_dir": os.path.relpath(src_dir, run_dir) if not os.path.isabs(src_dir) else src_dir,
        "n_images_given": len(names),
        "n_registered": len(imgs),
        "n_models": len(recs),
        "n_points3d": int(rec.num_points3D()),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
        "mean_track_length": float(rec.compute_mean_track_length()),
        "cameras": cameras,
        "images": imgs,
        "settings": {"matcher": matcher, "camera_model": camera_model,
                     "max_image_size": max_image_size, "max_features": max_features, "every": every},
    }
    out = os.path.join(out_dir, "sfm.json")
    with open(out, "w") as f:
        json.dump(sfm, f, indent=1)
    print(f"registered {len(imgs)}/{len(names)}; reprojection error "
          f"{sfm['mean_reprojection_error_px']:.2f} px; wrote {out}")
    if len(imgs) < 0.8 * len(names):
        print("WARNING: fewer than 80% of the frames registered. Poses are usable but coverage will be thin.")
    return sfm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run folder, e.g. runs/2026-10-30_walkaround-a")
    ap.add_argument("--images", help="folder of frames (default: frames/, raw/frames/, then raw/)")
    ap.add_argument("--every", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--matcher", choices=["auto", "exhaustive", "sequential"], default="auto")
    ap.add_argument("--max-image-size", type=int, default=3200)
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--camera-model", default="SIMPLE_RADIAL",
                    choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"])
    ap.add_argument("--focal-35mm", type=float,
                    help="35 mm-equivalent focal length as a starting guess (iPhone main camera: 26)")
    a = ap.parse_args(argv)
    run(a.run, images=a.images, every=a.every, matcher=a.matcher,
        max_image_size=a.max_image_size, max_features=a.max_features, camera_model=a.camera_model,
        focal_35mm=a.focal_35mm)


if __name__ == "__main__":
    main()
