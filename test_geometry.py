"""python test_geometry.py -- geometry on a synthetic board warped by a known homography (no camera needed)."""
import cv2
import numpy as np

import geometry as g
import slidescan as s

PPM, SIDE, W, H = 5, 38.4, 333, 349  # canvas px per mm; marker side (192 px, a whole number); board mm
MARKERS = {0: (108, 95), 1: (228, 94), 2: (112, 222), 3: (222, 216)}
DOT = (165, 153)
B = np.array([[PPM, 0, -0.5], [0, -PPM, H * PPM - 0.5], [0, 0, 1]])  # board mm -> canvas px (pixel centers)
P = cv2.getPerspectiveTransform(np.float32([[0, 0], [W * PPM, 0], [W * PPM, H * PPM], [0, H * PPM]]),
                                np.float32([[120, 80], [1300, 140], [1250, 1420], [60, 1350]]))  # canvas -> photo
H_TRUE = P @ B


def photo(rotate=None, only=MARKERS):
    """The board as a camera would see it: markers glued label-edge toward -y, a 6 mm black dot at DOT."""
    img = np.full((H * PPM, W * PPM), 255, np.uint8)
    n = round(SIDE * PPM)
    for i in only:
        x, y = MARKERS[i]
        m = cv2.aruco.generateImageMarker(s.DICT, i, n)
        if i == rotate:
            m = cv2.rotate(m, cv2.ROTATE_90_CLOCKWISE)
        u, v = round((x - SIDE / 2) * PPM), round((H - y - SIDE / 2) * PPM)
        img[v:v + n, u:u + n] = m
    u, v = g.to_pixel(B, [DOT])[0]
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    img[(xx - u) ** 2 + (yy - v) ** 2 <= (3 * PPM) ** 2] = 0
    return cv2.warpPerspective(img, P, (1400, 1500), borderValue=255)


img = photo()
Hf, ids, err = g.fit(s.detect(img), MARKERS, SIDE)
assert ids == [0, 1, 2, 3], ids
# slidescan.detect has no subpixel refinement: corners land ~0.7 px inward, so ~0.25 mm residuals here
assert max(err.values()) < 0.5, err
pts = [DOT, *MARKERS.values(), (45, 34), (289, 276)]  # incl. two bolt spots outside the marker hull
off = np.linalg.norm(g.to_pixel(Hf, pts) - g.to_pixel(H_TRUE, pts), axis=1)
assert off.max() < 1.5, off  # px; the photo is about 3.5 px/mm here, the far bolt spot is worst (~0.8)

c = g.crop(img, Hf, *DOT, 30, 10)  # 300 x 300 px, dot dead center
v, u = np.nonzero(c < 128)
assert c.shape == (300, 300) and abs(u.mean() - 149.5) < 1 and abs(v.mean() - 149.5) < 1, (u.mean(), v.mean())
assert np.allclose(g.to_board(Hf, g.to_pixel(Hf, [DOT])), [DOT])

assert g.in_view(Hf, (1400, 1500), *DOT) and not g.in_view(Hf, (1400, 1500), -200, 153)

_, _, err = g.fit(s.detect(photo(rotate=2)), MARKERS, SIDE)
assert max(err, key=err.get) == 2 and err[2] > g.MAX_ERR_MM, err  # a marker glued turned 90 deg is named

Hf, ids, _ = g.fit(s.detect(photo(only=[0])), MARKERS, SIDE)
assert Hf is None and ids == [0], "one marker is not enough for a mapping"
print("ok")
