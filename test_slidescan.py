"""python test_slidescan.py -- end to end on a synthetic run (no camera needed)."""
import json
import tempfile
from pathlib import Path

import cv2

import slidescan as s

page, ids = s.marker_page()
assert sorted(s.detect(page)) == list(ids) == list(range(16)), "every printed marker must read back"

with tempfile.TemporaryDirectory() as tmp:
    s.RUNS = Path(tmp)
    run = s.new("selftest")
    assert "meta.device is empty" in s.check(run), "blank template must fail"

    meta = json.loads((run / "meta.json").read_text())
    meta.update(device="test", capture_app="test", method="practice", surface="matte-flat", light_level="bright",
                light_elevation_deg="n/a", light_azimuth_deg="n/a", light_distance_mm="n/a", standoff_cm=25, pass_type="static",
                repeat=1, operator="test")
    (run / "meta.json").write_text(json.dumps(meta))
    board = json.loads((run / "targets.json").read_text())
    board["marker_mm"] = 40.0
    board["markers"] = [{"id": i} for i in ids]  # the test page's markers, whatever the real board uses
    for t in board["targets"] + board["markers"]:
        t.update(x_mm=10.0, y_mm=10.0)
    for t in board["targets"]:
        t.update(readings_mm=[3.22, 3.18, 3.21], measured_mm=3.21)
    (run / "targets.json").write_text(json.dumps(board))

    frame = cv2.cvtColor(cv2.resize(page, (960, 1242)), cv2.COLOR_GRAY2BGR)
    vw = cv2.VideoWriter(str(run / "raw" / "clip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (960, 1242))
    assert vw.isOpened()
    for _ in range(20):
        vw.write(frame)
    vw.release()

    s.Image.fromarray(frame[:, :, ::-1]).save(run / "raw" / "photo.heic", quality=90)  # iPhone-style photo

    dets = s.frames(run, every=5)
    assert len(dets) == 5 and all(len(d) == 16 for d in dets.values()), dets.keys()
    assert (run / "frames" / "photo.jpg").exists() and "frames/photo.jpg" in dets, "HEIC must become a readable JPG"
    assert s.check(run) == [], s.check(run)

    t = board["targets"][0]
    t["readings_mm"] = [3.2, 3.2]  # only two readings
    (run / "targets.json").write_text(json.dumps(board))
    assert any("needs 3 caliper readings" in p for p in s.check(run))
    t.update(readings_mm=[3.22, 3.18, 3.21], measured_mm=3.2)  # not the median
    (run / "targets.json").write_text(json.dumps(board))
    assert any("median of readings_mm, 3.21" in p for p in s.check(run))

    meta["surface"] = "matte"  # typo must be caught
    (run / "meta.json").write_text(json.dumps(meta))
    assert any("meta.surface" in p for p in s.check(run))
print("ok")
