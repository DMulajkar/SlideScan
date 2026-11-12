"""python yolo/test_labels.py -- box projection, label format and matching on synthetic data (no run folder needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb

# camera 250 mm straight above (100, 100) looking down, board +y up in the image, f = 3500 px, 3840 x 2160
K = np.array([[3500.0, 0, 1920], [0, 3500.0, 1080], [0, 0, 1]])
Rb = np.stack([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
tb = -Rb @ np.array([100.0, 100.0, 250.0])
D0 = np.zeros(4)

# 1. a bolt right under the camera, 10 mm tall: box centred on the image, roughly 9 mm * 3500 / 245 = 129 px wide
b = lb.project_box(K, D0, Rb, tb, (100.0, 100.0), 10.0)
assert b is not None
cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]
assert abs(cx - 1920) < 1 and abs(cy - 1080) < 1, (cx, cy)
assert 125 < w < 160 and abs(w - h) < 1e-6, (w, h)

# 2. direction: a bolt 20 mm to +x lands to the right, 20 mm to +y lands higher in the image (smaller row)
r = lb.project_box(K, D0, Rb, tb, (120.0, 100.0), 3.0)
u = lb.project_box(K, D0, Rb, tb, (100.0, 120.0), 3.0)
assert (r[0] + r[2]) / 2 > 1920 + 250 and (u[1] + u[3]) / 2 < 1080 - 250

# 3. a taller bolt is closer to the camera, so its box is bigger
assert lb.project_box(K, D0, Rb, tb, (100.0, 100.0), 12.0)[2] - lb.project_box(K, D0, Rb, tb, (100.0, 100.0), 12.0)[0] > w

# 4. off the image or behind the camera: None; half in view is kept only above the min_inside fraction
assert lb.project_box(K, D0, Rb, tb, (400.0, 100.0), 3.0) is None
flip = np.stack([Rb[0], Rb[1], -Rb[2]])                                # camera turned to look away from the board
assert lb.project_box(K, D0, flip, -flip @ np.array([100.0, 100.0, 250.0]), (100.0, 100.0), 3.0) is None
xe = 100.0 + 1920 / 3500 * 247                                          # bolt centred on the right image edge
assert lb.project_box(K, D0, Rb, tb, (xe, 100.0), 3.0) is None, "about half inside: below the default 0.7"
edge = lb.project_box(K, D0, Rb, tb, (xe, 100.0), 3.0, min_inside=0.4)
assert edge is not None and edge[2] == 3840.0, edge
assert lb.project_box(K, D0, Rb, tb, (xe - 6.0, 100.0), 3.0) is not None, "mostly inside: kept"

# 5. YOLO format: shrinking by 1/4 and centring on the canvas keeps the box centre at the canvas centre
off = lb.paste_offset((3840, 2160), 0.25)
assert off == ((1920 - 960) // 2, (1088 - 540) // 2)
y = lb.to_yolo(b, 0.25, off)
assert abs(y[0] - 0.5) < 0.002 and abs(y[1] - 0.5) < 0.002, y
assert abs(y[2] - w * 0.25 / 1920) < 1e-9 and abs(y[3] - h * 0.25 / 1088) < 1e-9
back = lb.from_yolo(y)
assert abs((back[0] + back[2]) / 2 - 960) < 0.5 and abs(back[2] - back[0] - w * 0.25) < 1e-6

# 5b. a portrait frame is rotated clockwise to landscape: a mark at (x, y) lands at (h0 - y, x), boxes follow
import cv2
img = np.zeros((300, 200, 3), np.uint8)                                # portrait: 200 wide, 300 high
img[40:60, 10:30] = 255                                                # white patch, box (10, 40, 30, 60)
rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
x0, y0, x1, y1 = lb.rotate_box_cw((10, 40, 30, 60), 300)
assert rot.shape[:2] == (200, 300) and (x0, y0, x1, y1) == (240, 10, 260, 30)
assert rot[y0:y1, x0:x1].min() == 255 and rot.sum() == 255 * 3 * 400, "the rotated box covers exactly the patch"

# 6. matching: one prediction per gt, extras are false, confidence order decides who gets the gt
gts = [(100, 100, 140, 140), (300, 300, 340, 340)]
preds = [(102, 98, 142, 138, 0.9), (110, 110, 150, 150, 0.5), (500, 500, 540, 540, 0.8)]
found, false = lb.match(preds, gts, lb.centre_in)
assert found == {0} and false == 2
assert lb.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0 and lb.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
assert lb.centre_in((0, 0, 10, 10), (3, 3, 13, 13)) and not lb.centre_in((0, 0, 10, 10), (30, 30, 40, 40))
print("ok")
