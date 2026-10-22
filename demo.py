"""python demo.py -> demo_report.html: the whole grazing-light pipeline on a synthetic board, no camera needed.

Draws bolt shadows of known length (test_grazing.py), runs grazing.py + score.py on the photos, and puts the photo,
the on-off diff with what the detector flagged, and measured-vs-estimated heights on one page. SYNTHETIC, not a scan.
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

import geometry as g
from methods.grazing import PPM

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "demo_run" / "synthetic-grazing"


def sh(*cmd):
    return subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def png(img):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1]).decode()


shutil.rmtree(ROOT / "demo_run", ignore_errors=True)  # test_grazing.py refuses to overwrite a run
sh("test_grazing.py", "demo_run")  # synthetic run kept in demo_run/, grazing.py already run on it
gate = sh("score.py", RUN, "--method", "grazing", "--gate").split("Gate A")[1].strip().splitlines()
board = json.loads((RUN / "targets.json").read_text())
cands = json.loads((RUN / "methods/grazing/candidates.json").read_text())["candidates"]
heights = json.loads((RUN / "methods/grazing/heights.json").read_text())["targets"]
results = json.loads((RUN / "methods/grazing/results.json").read_text())["targets"]
bw, bh = board["board_size_mm"]

# rectify p01 exactly as grazing.py does, then crop the square to the board
H = np.array(json.loads((RUN / "geometry.json").read_text())["images"]["raw/p01_pose.jpg"]["H"])
size = max(bw, bh)
x0, y1 = bw / 2 - size / 2, bh / 2 + size / 2
sl = (slice(round((y1 - bh) * PPM), round(y1 * PPM)), slice(round(-x0 * PPM), round((bw - x0) * PPM)))
rect = lambda img: g.crop(img.astype(np.float32), H, bw / 2, bh / 2, size, PPM)[sl]
px = lambda x, y: (round((x - 0) * PPM), round((bh - y) * PPM))
gray = lambda name: cv2.imread(str(RUN / "raw" / name), cv2.IMREAD_GRAYSCALE)
off, on = rect(gray("p01_off.jpg")), rect(gray("p01_on.jpg"))
diff = np.clip((on - off) / max(np.percentile(on - off, 99), 1) * 255, 0, 255).astype(np.uint8)
photo = cv2.cvtColor(np.clip(on, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
found = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
for t in board["targets"]:
    cv2.putText(found, f"{t['id']} {t['measured_mm']:g}mm", (px(t["x_mm"], t["y_mm"])[0] - 40, px(t["x_mm"], t["y_mm"])[1] + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
for c in cands:
    cv2.circle(found, px(c["x_mm"], c["y_mm"]), 40, (0, 0, 255), 3)
    cv2.putText(found, f"{c['est_height_mm']:.1f} mm", (px(c["x_mm"], c["y_mm"])[0] + 45, px(c["x_mm"], c["y_mm"])[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

rows = "".join(
    f"<tr><td>{t['id']}</td><td>{t['measured_mm']:g}</td><td>{heights.get(t['id'], {}).get('shadow_mm', '-')}</td>"
    f"<td>{heights.get(t['id'], {}).get('est_height_mm', '-')}</td>"
    f"<td class={results[t['id']]['outcome']}>{results[t['id']]['outcome'] if t['measured_mm'] >= 3.2 else 'below 3.2: not required'}</td></tr>"
    for t in board["targets"])
Path(ROOT / "demo_report.html").write_text(f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>SlideScan Demo</title>
<style>body{{font:16px system-ui;margin:0 auto;max-width:1100px;padding:16px;background:#fff;color:#111}}
@media(prefers-color-scheme:dark){{body{{background:#111;color:#eee}}}}
.row{{display:flex;gap:12px;flex-wrap:wrap}}.row figure{{flex:1 1 320px;margin:0}}img{{width:100%}}
td,th{{padding:4px 12px;text-align:left}}.hit{{color:#2a9d3f}}.miss{{color:#d33}}code{{font-size:.9em}}</style>
<h1>SlideScan: finding bolts that stick out of a playground slide</h1>
<p>Low-angle light makes any bolt taller than the slide's limit (3.2 mm) cast a shadow. Take a photo with the light off,
one with it on, subtract: the shadows are what's left. Shadow length gives the bolt's height.
<b>Synthetic board</b> (shadows drawn at known lengths), not a real scan yet. Same code will run on phone photos.</p>
<div class=row><figure><img src="{png(photo)}"><figcaption>1. Phone photo, grazing light on (board rectified from ArUco markers)</figcaption></figure>
<figure><img src="{png(found)}"><figcaption>2. Light-on minus light-off. Red = what the detector flagged, with its height estimate</figcaption></figure></div>
<h2>Measured vs. estimated height</h2>
<table><tr><th>Bolt<th>Caliper (mm)<th>Shadow (mm)<th>Estimated (mm)<th>Result</tr>{rows}</table>
<h2>Gate A (zero misses at 3.2 mm and taller)</h2><pre>{chr(10).join(gate)}</pre>
<p><small>Rebuild: <code>python demo.py</code> &middot; 3D model of the real board: <code>board_view.html</code></small></p>
""", encoding="utf-8")
print("demo_report.html")
