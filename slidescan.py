"""SlideScan Phase 1: printable markers, run folders, frame extraction, marker check.

  python slidescan.py markers [--start 0] [--mm 40]  -> markers_<first>-<last>.pdf (print at 100% / Actual size)
  python slidescan.py new <label>                    -> runs/<date>_<label>/ with meta.json, targets.json, raw/
  python slidescan.py frames <run> [--every 10]      -> frames/ from raw/ videos (+ JPG copies of HEIC photos), markers -> detections.json
  python slidescan.py check <run>                    -> lists anything missing; exit 1 if the run is incomplete
"""
import argparse
import contextlib
import datetime
import io
import json
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
from PIL import Image

from methods import sled

pillow_heif.register_heif_opener()  # lets PIL open iPhone HEIC photos

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
BOARD = ROOT / "board.json"
DPI = 300
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
DETECTOR = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
VIDEO = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
STILL = {".jpg", ".jpeg", ".png"}
HEIC = {".heic", ".heif"}

# Every field must be filled before a run counts; write "n/a" where a field does not apply.
# Light fields describe the separate grazing light, measured at targets.json light_ref_mm (lean board: its center):
# azimuth = direction from that center toward the light, degrees counterclockwise from board +x;
# distance = horizontal distance from that center to the light; elevation = angle up from the board there.
META = {
    "date": "", "device": "", "capture_app": "", "method": "", "surface": "", "light_level": "",
    "light_elevation_deg": None, "light_azimuth_deg": None, "light_distance_mm": None, "standoff_cm": None,
    "pass_type": "", "texture_help": "none", "repeat": None, "operator": "", "notes": "",
}
OPTIONAL = {"notes"}
# Fixed vocabularies so runs group cleanly later; a typo would split one condition into two.
CHOICES = {
    "method": {"practice", "apartment-pipeline", "grazing-light", "sled", "truedepth", "vslam-colmap", "lidar"},
    "surface": {"matte-flat", "glossy-curved"},
    "light_level": {"bright", "dim", "near-dark"},
    "pass_type": {"static", "slow", "fast", "freehand"},
    "texture_help": {"none", "speckle", "tape"},
}


def marker_page(start=0, mm=40):
    """US Letter at 300 dpi: as many markers as fit, ID labels, and a 100 mm scale bar to check print scale."""
    px = DPI / 25.4
    W, H, m, s, gap = int(8.5 * DPI), int(11 * DPI), 150, round(mm * px), 120
    page = np.full((H, W), 255, np.uint8)

    def put(text, x, y):
        cv2.putText(page, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

    cols = (W - 2 * m + gap) // (s + gap)
    rows = (H - 660) // (s + 110)  # 330 px reserved top (header) and bottom (scale bar)
    ids = range(start, start + cols * rows)
    for k, i in enumerate(ids):
        x, y = m + (k % cols) * (s + gap), 330 + (k // cols) * (s + 110)
        page[y:y + s, x:x + s] = cv2.aruco.generateImageMarker(DICT, i, s)
        put(f"ID {i}", x, y + s + 55)
    put(f"SlideScan markers  DICT_4X4_50  IDs {ids[0]}-{ids[-1]}  side {mm} mm as designed", m, 200)
    put("Print at 100% / Actual size. Caliper one black square -> board.json marker_mm.", m, 260)
    y, L = H - 230, round(100 * px)
    cv2.line(page, (m, y), (m + L, y), 0, 4)
    for t in range(11):
        cv2.line(page, (m + round(t * 10 * px), y), (m + round(t * 10 * px), y - (50 if t % 5 == 0 else 25)), 0, 4)
    put("100 mm", m + L + 30, y)
    return page, ids


def detect(img):
    """{marker id: 4 corner points in pixels} for one image."""
    corners, ids, _ = DETECTOR.detectMarkers(img)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2).round(1).tolist() for i, c in zip(np.ravel(ids), corners)}


def new(label):
    date = datetime.date.today().isoformat()
    run = RUNS / f"{date}_{label}"
    if run.exists():
        sys.exit(f"{run} already exists; pick another label (e.g. add -r2)")  # never overwrite captured data
    (run / "raw").mkdir(parents=True)
    (run / "meta.json").write_text(json.dumps(dict(META, date=date), indent=2))
    shutil.copy(BOARD, run / "targets.json")  # snapshot: the board changes between runs
    return run


def frames(run, every=10):
    """Extract every Nth video frame from raw/, detect markers in those frames and in raw stills."""
    run = Path(run)
    out = run / "frames"
    out.mkdir(exist_ok=True)
    dets = {}
    for f in sorted((run / "raw").iterdir()):
        ext = f.suffix.lower()
        if ext in STILL:
            dets[f"raw/{f.name}"] = detect(cv2.imread(str(f)))
        elif ext in VIDEO:
            cap, i = cv2.VideoCapture(str(f)), 0
            while True:
                ok, img = cap.read()
                if not ok:
                    break
                if i % every == 0:
                    name = f"{f.stem}_{i:06d}.jpg"  # frame index in the name: time = index / fps
                    # ponytail: JPEG q95 to keep disk sane; raw video stays the lossless-as-captured source
                    cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    dets[f"frames/{name}"] = detect(img)
                i += 1
            cap.release()
        elif ext in HEIC:
            # OpenCV can't read HEIC, so frames/ gets a JPG copy that later tools can open; raw/ stays as captured
            img = cv2.cvtColor(np.asarray(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR)
            name = f"{f.stem}.jpg"
            cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            dets[f"frames/{name}"] = detect(img)
    (run / "detections.json").write_text(json.dumps({"every": every, "images": dets}))
    hit, seen = summarize(dets)
    print(f"{len(dets)} images, {hit} with markers; ids seen: {dict(sorted(seen.items()))}")
    return dets


def summarize(dets):
    return sum(1 for d in dets.values() if d), Counter(int(i) for d in dets.values() for i in d)


def check(run):
    """Everything that would make the run unusable later. Empty list = complete."""
    run, problems = Path(run), []
    meta = json.loads((run / "meta.json").read_text())
    for k in META:
        v = meta.get(k)
        if k not in OPTIONAL and v in (None, ""):
            problems.append(f"meta.{k} is empty")
        elif k in CHOICES and v not in CHOICES[k]:
            problems.append(f"meta.{k} = {v!r}; expected one of {sorted(CHOICES[k])}")
    board = json.loads((run / "targets.json").read_text())
    num = (int, float)
    if not isinstance(board.get("marker_mm"), num):
        problems.append("targets.marker_mm not measured (caliper a printed marker)")
    for t in board["targets"]:
        r = t.get("readings_mm") or []
        if len(r) < 3 or not all(isinstance(x, num) for x in r):
            problems.append(f"target {t['id']}: needs 3 caliper readings in readings_mm")
        # 0.005 = half a 0.01 mm caliper step
        elif not isinstance(t.get("measured_mm"), num) or abs(t["measured_mm"] - statistics.median(r)) > 0.005:
            problems.append(f"target {t['id']}: measured_mm should be the median of readings_mm, "
                            f"{statistics.median(r):g}")
        problems += [f"target {t['id']}: {k} missing" for k in ("x_mm", "y_mm") if not isinstance(t.get(k), num)]
    for mk in board["markers"]:
        problems += [f"marker {mk['id']}: {k} missing" for k in ("x_mm", "y_mm") if not isinstance(mk.get(k), num)]
    if meta.get("method") == "sled":  # no images: the typed CSVs, checked by the adapter itself
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sled.run(run)  # ponytail: also rewrites methods/sled/candidates.json, derived and harmless
        except SystemExit as e:
            problems.append(str(e.code))
        return problems
    if not any(f.suffix.lower() in VIDEO | STILL | HEIC for f in (run / "raw").iterdir()):
        problems.append("raw/ has no video or photos")
    det = run / "detections.json"
    if not det.exists():
        problems.append("no detections.json; run: python slidescan.py frames <run>")
    else:
        hit, seen = summarize(json.loads(det.read_text())["images"])
        if not hit:
            problems.append("no markers detected in any image")
        unknown = set(seen) - {mk["id"] for mk in board["markers"]}
        if unknown:
            problems.append(f"markers seen but not in targets.json: {sorted(unknown)}")
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("markers")
    a.add_argument("--start", type=int, default=0)
    a.add_argument("--mm", type=float, default=40)
    sub.add_parser("new").add_argument("label")
    a = sub.add_parser("frames")
    a.add_argument("run")
    a.add_argument("--every", type=int, default=10)
    sub.add_parser("check").add_argument("run")
    args = p.parse_args()

    if args.cmd == "markers":
        page, ids = marker_page(args.start, args.mm)
        out = ROOT / f"markers_{ids[0]}-{ids[-1]}.pdf"
        Image.fromarray(page).save(out, resolution=DPI)
        print(out)
    elif args.cmd == "new":
        print(new(args.label))
    elif args.cmd == "frames":
        frames(args.run, args.every)
    elif args.cmd == "check":
        problems = check(args.run)
        print("\n".join(problems) or "OK: run is complete")
        sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
