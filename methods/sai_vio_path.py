"""Read the camera path (metres, with times) out of a Spectacular Rec recording, for sensor_scale.py.

  python methods/sai_vio_path.py <recording folder> [out.json]     (run with the venv that has spectacularAI; WSL ~/sai_venv)

The folder is what the app exports: data.mov, data.jsonl, calibration.json, metadata.json. Spectacular AI's replay runs
its visual-inertial tracking over the video and the sensors and calls back once per frame; every callback's pose is saved:
  {"t": seconds (same clock as data.jsonl), "p": [x, y, z] metres, "q": [x, y, z, w], "status": "TRACKING"/"INIT"/"LOST_TRACKING"}
Written 2026-11-13 against spectacularAI 1.54.1 and one iPhone 14 recording; not tested on any other phone or version.
"""
import json
import sys


def read_path(folder, out=None):
    import spectacularAI as sai
    rows = []

    def cb(o):
        p = o.pose
        q = p.orientation
        rows.append({"t": p.time, "p": [p.position.x, p.position.y, p.position.z],
                     "q": [q.x, q.y, q.z, q.w], "status": o.status.name})

    rep = sai.Replay(folder)
    rep.setOutputCallback(cb)
    rep.runReplay()
    if out:
        json.dump(rows, open(out, "w"))
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = read_path(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    n_track = sum(r["status"] == "TRACKING" for r in rows)
    print(f"{len(rows)} poses, {n_track} while TRACKING; t {rows[0]['t']:.2f} to {rows[-1]['t']:.2f} s")
