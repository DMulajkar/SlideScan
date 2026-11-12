"""Pure helpers for the stage-1 bolt detector: boxes from known board positions + camera poses, YOLO label
format, box matching. No pycolmap, no torch: the self-test (test_labels.py) runs anywhere with numpy and OpenCV."""
import cv2
import numpy as np

CANVAS = (1920, 1088)        # every training image is pasted on this canvas (multiple of 32), grey around it


def project_box(K, dist, Rb, tb, xy, z_hi, half=4.5, z_lo=0.0, img_wh=(3840, 2160), min_inside=0.7):
    """Pixel box (x0, y0, x1, y1) around a bolt standing at board position xy (mm), from the corners of a
    (2*half)-mm square column between heights z_lo and z_hi. Pose: x_cam = Rb X + tb. None if the bolt is behind
    the camera or less than min_inside of the box is inside the image."""
    x, y = xy
    P = np.array([[x + dx, y + dy, z] for dx in (-half, half) for dy in (-half, half) for z in (z_lo, z_hi)], float)
    if ((P @ Rb.T + tb)[:, 2] <= 1e-3).any():
        return None
    pix = cv2.projectPoints(P.reshape(-1, 1, 3), cv2.Rodrigues(Rb)[0], np.asarray(tb, float), K, dist)[0].reshape(-1, 2)
    x0, y0 = pix.min(0)
    x1, y1 = pix.max(0)
    w, h = img_wh
    cx0, cy0, cx1, cy1 = max(x0, 0.0), max(y0, 0.0), min(x1, float(w)), min(y1, float(h))
    if cx1 <= cx0 or cy1 <= cy0 or (cx1 - cx0) * (cy1 - cy0) < min_inside * (x1 - x0) * (y1 - y0):
        return None
    return (cx0, cy0, cx1, cy1)


def rotate_box_cw(box, h0):
    """A pixel box in an image of height h0 -> the same box after the image is rotated 90 degrees clockwise
    (cv2.ROTATE_90_CLOCKWISE), which turns a portrait frame into a landscape one."""
    x0, y0, x1, y1 = box
    return h0 - y1, x0, h0 - y0, x1


def paste_offset(img_wh, scale, canvas=CANVAS):
    """Top-left corner that centres an image of img_wh, shrunk by scale, on the canvas."""
    return (canvas[0] - round(img_wh[0] * scale)) // 2, (canvas[1] - round(img_wh[1] * scale)) // 2


def to_yolo(box, scale, off, canvas=CANVAS):
    """Full-resolution pixel box -> normalised (cx, cy, w, h) on the canvas after shrinking by scale and pasting at off."""
    x0, y0, x1, y1 = (v * scale for v in box)
    x0, x1, y0, y1 = x0 + off[0], x1 + off[0], y0 + off[1], y1 + off[1]
    W, H = canvas
    return (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H


def from_yolo(lab, canvas=CANVAS):
    """Normalised (cx, cy, w, h) -> canvas pixel box (x0, y0, x1, y1)."""
    cx, cy, w, h = lab
    W, H = canvas
    return (cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre_in(pred, gt, grow=1.0):
    """True if the centre of pred lies within the gt box grown by `grow` (1.0 = the box itself). A stage-1 proposal
    only has to land on the bolt, so this is the main 'found' rule; IoU is reported next to it."""
    cx, cy = (pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2
    gx, gy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    return abs(cx - gx) <= grow * (gt[2] - gt[0]) / 2 and abs(cy - gy) <= grow * (gt[3] - gt[1]) / 2


def match(preds, gts, rule):
    """Greedy one-to-one matching of predictions (x0, y0, x1, y1, conf), best confidence first, to gt boxes under
    rule(pred_box, gt_box). Returns (gt indices found, number of predictions that matched nothing)."""
    found, false = set(), 0
    for p in sorted(preds, key=lambda p: -p[4]):
        hit = next((j for j, g in enumerate(gts) if j not in found and rule(p[:4], g)), None)
        if hit is None:
            false += 1
        else:
            found.add(hit)
    return found, false
