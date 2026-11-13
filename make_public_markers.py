"""One-page printable marker sheet for the public SlideScan page (new file; slidescan.py is not touched).

  python make_public_markers.py out.pdf          write the sheet (US Letter, 300 dpi)
  python make_public_markers.py --selftest        render, detect the markers back, check the sizes

Markers DICT_4X4_50 ids 0-3, black square 38.3 mm, because the server's board.json uses marker_mm 38.3. The existing
markers_0-15.pdf are designed at 40 mm, so printing them exactly would put the server's scale 4.4 % off.
"""
import sys

import cv2
import numpy as np
from PIL import Image

DPI, MM = 300, 38.3
BAR_Y = 1800                                  # below the second marker row (ends at y = 1544 + its label)
PX = DPI / 25.4
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def sheet():
    W, H = int(8.5 * DPI), int(11 * DPI)
    page = np.full((H, W), 255, np.uint8)
    s, gap, m = round(MM * PX), 220, 150

    def put(text, x, y, scale=1.2):
        cv2.putText(page, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 3)

    put("SlideScan markers 0-3", m, 200, 1.6)
    put("DICT_4X4_50, black square 38.3 mm", m, 270)
    put("Print at 100% / Actual size. Do not use 'fit to page'.", m, 335)
    for k in range(4):
        x, y = m + (k % 2) * (s + gap), 450 + (k // 2) * (s + 190)
        page[y:y + s, x:x + s] = cv2.aruco.generateImageMarker(DICT, k, s)
        put(f"ID {k}", x, y + s + 55)
    y, L = BAR_Y, round(100 * PX)
    cv2.line(page, (m, y), (m + L, y), 0, 4)
    for t in range(11):
        cv2.line(page, (m + round(t * 10 * PX), y), (m + round(t * 10 * PX), y - (50 if t % 5 == 0 else 25)), 0, 4)
    put("This line must measure 100 mm. Each black square must measure 38.3 mm.", m, y + 90)
    put("If not, the print was scaled: reprint.", m, y + 150)
    put("Cut each marker out with a white border of at least 5 mm.", m, y + 270)
    put("Stick it flat on the board with the ID label toward the board's bottom", m, y + 330)
    put("edge, at the position given on the SlideScan page (within 1 mm).", m, y + 390)
    return page


def selftest():
    page = sheet()
    det = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(page)
    assert ids is not None and sorted(ids.ravel().tolist()) == [0, 1, 2, 3], ids
    want = MM * PX
    # the detector's corners sit a few px inside a large marker, so measure the printed black square itself
    sz = round(MM * PX)
    for k in range(4):
        x, y = 150 + (k % 2) * (sz + 220), 450 + (k // 2) * (sz + 190)
        ys, xs = np.where(page[y - 5:y + sz + 5, x - 5:x + sz + 5] == 0)
        assert abs((xs.max() - xs.min() + 1) - want) <= 1 and abs((ys.max() - ys.min() + 1) - want) <= 1, (k, int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    row = page[BAR_Y - 2:BAR_Y + 3].min(0)                                   # the scale bar: dark pixels along its row
    dark = np.flatnonzero(row == 0)
    assert abs((dark.max() - dark.min()) - 100 * PX) < 8, (dark.min(), dark.max())
    print(f"selftest ok: ids 0-3 detected, black squares {want:.0f} px (= {MM} mm at {DPI} dpi), scale bar 100 mm")
    print("NOT tested: a physical print (printer scaling), other readers' PDF rendering")


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) < 2:
        selftest()
    else:
        Image.fromarray(sheet()).save(sys.argv[1], resolution=DPI)
        print(sys.argv[1])
