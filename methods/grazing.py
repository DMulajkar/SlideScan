"""SlideScan Phase 2, method 6: grazing-light shadows -> methods/grazing/candidates.json and heights.json.

  python methods/grazing.py <run> [--l-min 5]

Per tripod position NN: pNN_pose.jpg gives the board mapping (geometry.json, so run geometry.py first), and
pNN_off.jpg / pNN_on.jpg in the same folder give diff = on - off, rectified to board mm. A shadow is where the
grazing light added under DARK x what it added to the board around it. Marker squares are blanked first:
markers are grading equipment, never detector input.
  blind     every dark blob reaching >= --l-min mm away from the light is a candidate (score = that length, mm);
            the same shadow seen from several positions is merged into one
  targeted  per covered target, the shadow beside it -> length and height, median over positions -> heights.json
Light position comes from meta.json's light fields, measured from targets.json light_ref_mm. Height = L*z/(d+L),
i.e. L*tan(elevation seen from the shadow tip): exact for a point light at height z, d = light to shadow start.
"""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # geometry.py lives one folder up
import geometry as g  # noqa: E402

PPM = 4            # rectified px per mm
# ponytail: starting guess. A 3.2 mm bolt casts about 3.2/tan(elevation) mm (8.8 at 20 deg); calibrate in baseline.py
L_MIN_MM = 5.0
DARK = 0.5         # dark = diff under this fraction of the local lit level
BG_MM = 40         # box window for the local lit level: wider than a shadow, narrower than the light's falloff
MARKER_PAD_MM = 3  # blanked around each marker square; markers sit 0.3-1.7 mm off board.json
STRIP_MM = 1       # half-width of the shadow's center line, where its length is read
MATCH_MM = 10      # merge radius across positions, and how far from a target its shadow may start (= score.py radius)


def light_at(meta, ref):
    """(light x, y in board mm, light height z in mm) from meta's azimuth/distance/elevation, measured from ref."""
    try:
        az, d, el = (float(meta[k]) for k in ("light_azimuth_deg", "light_distance_mm", "light_elevation_deg"))
    except (KeyError, TypeError, ValueError):
        sys.exit("meta.json needs numbers in light_azimuth_deg, light_distance_mm and light_elevation_deg")
    a = math.radians(az)
    return np.array([ref[0] + d * math.cos(a), ref[1] + d * math.sin(a)]), d * math.tan(math.radians(el))


def shadows(diff, valid, xs, ys, light, z):
    """Dark blobs in a rectified diff -> [(board xy of the light-side end, length L mm, height mm)]."""
    vf, box = valid.astype(np.float32), (BG_MM * PPM,) * 2
    lit = cv2.blur(diff * vf, box) / np.maximum(cv2.blur(vf, box), 1e-3)  # local lit level, valid pixels only
    dark = (valid & (diff < DARK * lit)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((PPM, PPM), np.uint8))  # drop specks under 1 mm
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    out = []
    for i in range(1, n):
        x0, y0, w, h = stats[i, :4]
        r, c = np.nonzero(lab[y0:y0 + h, x0:x0 + w] == i)
        pts = np.column_stack([xs[c + x0], ys[r + y0]])
        mid = pts.mean(0)
        u = (mid - light) / np.linalg.norm(mid - light)  # away from the light
        p, q = (pts - mid) @ u, (pts - mid) @ np.array([-u[1], u[0]])
        # length on the center line only: beside a round head the shadow wraps back toward the head's middle
        line = p[np.abs(q) <= STRIP_MM] if (np.abs(q) <= STRIP_MM).any() else p
        L = float(line.max() - line.min()) + 1 / PPM
        d = np.linalg.norm(mid + line.min() * u - light)
        out.append((mid + p.min() * u, L, L * z / (d + L)))  # position: the wrapped end, near the head
    return out


def run(path, l_min=L_MIN_MM):
    path = Path(path)
    if not (path / "geometry.json").exists():
        sys.exit(f"{path}: no geometry.json; run: python geometry.py {path}")
    board = json.loads((path / "targets.json").read_text())
    if "light_ref_mm" not in board:
        sys.exit(f"{path}: targets.json has no light_ref_mm (where the light fields are measured from); "
                 "copy it in from board.json")
    light, z = light_at(json.loads((path / "meta.json").read_text()), board["light_ref_mm"])

    # one rectified grid over the whole board; crop() makes squares, so the short side gets a strip off the board
    bw, bh = board["board_size_mm"]
    size = max(bw, bh)
    n = round(size * PPM)
    xs = bw / 2 - size / 2 + (np.arange(n) + 0.5) / PPM
    ys = bh / 2 + size / 2 - (np.arange(n) + 0.5) / PPM
    X, Y = np.meshgrid(xs, ys)
    keep = (X >= 0) & (X <= bw) & (Y >= 0) & (Y <= bh)
    half = board["marker_mm"] / 2 + MARKER_PAD_MM
    for m in board["markers"]:  # Design Rule 7: blank every marker square before looking
        keep &= (abs(X - m["x_mm"]) > half) | (abs(Y - m["y_mm"]) > half)

    seen, cands, covered, found, used = np.zeros_like(keep), [], set(), {}, 0
    for name, rec in json.loads((path / "geometry.json").read_text())["images"].items():
        if not name.endswith("_pose.jpg") or not rec["H"]:
            continue
        stem = name[:-len("_pose.jpg")]
        off, on = (cv2.imread(str(path / f"{stem}_{k}.jpg"), cv2.IMREAD_GRAYSCALE) for k in ("off", "on"))
        if off is None or on is None or on.shape != off.shape or off.shape[::-1] != tuple(rec["size"]):
            print(f"{name}: skipped; needs {stem}_off.jpg and {stem}_on.jpg at the pose photo's size")
            continue
        H = np.array(rec["H"])
        view = g.crop(np.ones(off.shape, np.float32), H, bw / 2, bh / 2, size, PPM) > 0.999
        diff = g.crop(on.astype(np.float32) - off, H, bw / 2, bh / 2, size, PPM)
        blobs = shadows(diff, view & keep, xs, ys, light, z)
        cands += [{"x_mm": round(float(x), 2), "y_mm": round(float(y), 2), "score": round(L, 2),
                   "est_height_mm": round(float(h), 2), "image": f"{stem}_on.jpg"}
                  for (x, y), L, h in blobs if L >= l_min]
        for t in board["targets"]:
            if t["id"] in rec["covered"]:
                d, L, h = min(((math.dist(xy, (t["x_mm"], t["y_mm"])), L, h) for xy, L, h in blobs),
                              default=(math.inf, 0, 0))
                found.setdefault(t["id"], []).append((L, h) if d <= MATCH_MM else (0, 0))
        seen |= view
        covered |= set(rec["covered"])
        used += 1
    if not used:
        sys.exit(f"{path}: no mapped pNN_pose.jpg with its pNN_off.jpg and pNN_on.jpg")

    kept = []  # score.py counts every background candidate, so one shadow seen N times must be one candidate
    for c in sorted(cands, key=lambda c: -c["score"]):
        if all(math.dist((c["x_mm"], c["y_mm"]), (k["x_mm"], k["y_mm"])) > MATCH_MM for k in kept):
            kept.append(c)
    area = round(float((seen & keep).sum()) / PPM ** 2 / 100, 1)  # board seen, markers blanked
    heights = {t["id"]: {"measured_mm": t["measured_mm"],
                         "shadow_mm": round(statistics.median(L for L, _ in found[t["id"]]), 2),
                         "est_height_mm": round(float(statistics.median(h for _, h in found[t["id"]])), 2),
                         "images": len(found[t["id"]])} for t in board["targets"] if t["id"] in found}

    out = path / "methods" / "grazing"
    out.mkdir(parents=True, exist_ok=True)
    (out / "candidates.json").write_text(json.dumps(
        {"candidates": kept, "covered_targets": sorted(covered), "covered_area_cm2": area}, indent=2))
    (out / "heights.json").write_text(json.dumps(
        {"light_xyz_mm": [round(float(v), 1) for v in (*light, z)], "targets": heights}, indent=2))

    print(f"{used} positions; {len(kept)} candidates with shadows >= {l_min:g} mm ({len(cands)} before merging); "
          f"targets covered {len(covered)} of {len(board['targets'])}; {area:g} cm2 of board seen, markers blanked")
    print("targeted, median over positions (mm):    id  measured  shadow  estimate")
    for i, v in heights.items():
        print(f"{'':39}{i:>4}  {v['measured_mm']:>8g}  {v['shadow_mm']:>6g}  {v['est_height_mm']:>8g}")
    return kept, heights


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--l-min", type=float, default=L_MIN_MM, help="shortest shadow that counts as a candidate, mm")
    args = p.parse_args()
    run(args.run, args.l_min)


if __name__ == "__main__":
    main()
