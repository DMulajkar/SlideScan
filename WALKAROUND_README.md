# Walk-around method (methods 2 and 3, tag-once)

Film a slow walk-around of the board. COLMAP gets the camera poses from the video alone.
The printed tag, visible in a few frames, fixes the metric scale and the board plane once.
A plane sweep then reads each bolt's height against the local board.

Written 2026-10-28 and tested **only on a synthetic board** (see "What was and was not tested").

## New files (nothing existing is changed)
```
SlideScan/
  methods/walkaround_sfm.py   step 1: frames -> camera poses (pycolmap)
  methods/walkaround.py       step 2: poses + tag -> heights, candidates, map
  test_walkaround.py          self-test on a synthetic board
  WALKAROUND_README.md        this file
```
Copy the folder contents over `C:\Users\mmtes\Desktop\SlideScan`. No existing file is overwritten.

## Install (once)
In the WSL venv (`~/vslam/.venv`, Python 3.12):
```
pip install pycolmap opencv-python numpy scipy
```
Run **both** steps in WSL; the run folder is at `/mnt/c/Users/mmtes/Desktop/SlideScan/runs/<run>`.
If it crawls, copy the run into WSL's own disk first. pycolmap's wheels are CPU-only; that is fine, only sparse SfM runs.
Check first: `python test_walkaround.py --quick` (about 2 min) then `python test_walkaround.py` (about 10 min).

## Before you film
1. **Add texture to the board.** The method matches what the camera sees. A plain matte board gives nothing to match.
   Matte tape scraps or a speckle of paint on the board are enough. Keep the bolts, nuts and markers as they are.
2. **The tag.** The existing printed markers are the tag. At least one must be sharp, flat on the board and visible in
   a few frames **from clearly different positions** (move sideways about 10 cm while it is in view). The start and the
   end of the sweep are the easy places. It does not need to be in every frame.
3. **Camera.** iPhone main lens (1x), 4K 30 fps, no zoom change. Lock exposure and focus (long-press on the board).
   Standoff 20 to 30 cm. Soft, steady light with no glare on the bolts, and keep your own shadow off the board.
4. **Motion.** One slow orbit, 15 to 30 s, with real sideways travel (30 cm or more in total) and some change of angle.
   Slow enough that single frames are sharp. Do not rotate the phone between portrait and landscape.
5. Measure the bolt heights with calipers as usual; `targets.json` carries them.

## Run
```
python slidescan.py new ...                       (existing; make the run, then add the video to raw/)
python slidescan.py frames runs/<run> --every 3   (existing; about 10 frames per second)
python methods/walkaround_sfm.py runs/<run> --focal-35mm 26
python methods/walkaround.py runs/<run>
python score.py runs/<run> --method walkaround --gate
```
- `--focal-35mm 26` tells COLMAP the iPhone main camera's 35 mm-equivalent focal length. A flat board is a hard case for
  SfM without a focal-length guess. Leave it off for another lens.
- Frames are read from `frames/`, `raw/frames/`, then `raw/`. Use `--images DIR` if `slidescan.py frames` puts them elsewhere.
- `--max-tag-images 6` makes the second step use the tag from only 6 frames (a tag-once test of your own footage).

## Reading the output (`runs/<run>/methods/walkaround/`)
| File | What it is |
|---|---|
| `scale.json` | How the tag fixed the scale. `rms_mm` is how well the tag corners fit the printed layout (1 mm or so is normal for ruler-placed markers). `drift.difference_pct` compares the scale from the first half of the tagged frames with the second half. |
| `heights.json` | One row per known target: estimate, caliper height, error, score. This is the proof-of-concept number. |
| `candidates.json` | Bumps found anywhere, for `score.py` (same format as the other methods). |
| `heightmap.png` | Height above the local board, targets circled: estimate, then caliper in brackets. |
| `diagnostics.json` | Settings and the frames used. |

My starting thresholds (guesses, set before any real run): mean absolute error under 0.5 mm is a clear pass; over 1 mm means
try Metashape on the same frames, then the known-pose triangulation (option A in the bench-test plan).
A target with a low score or "no local board" had nothing to match; add texture and refilm before blaming the method.

## What to do when it stops
| Message | Meaning |
|---|---|
| `COLMAP built no model` | Too little overlap, blur, or a plain board. Add texture, film slower. |
| `fewer than 80% of the frames registered` | Usable, but coverage is thin. |
| `the tag fixed only N corners` | The tag was not seen sharply from two different positions. |
| `the tag does not fit the board layout` | A marker is glued rotated or off its `board.json` position, or the board y axis is not "toward the marker's top edge". |
| `cameras came out below the board` | The board frame is mirrored against the corner order assumed (see top of `walkaround.py`). |
| `... is WxH but COLMAP's camera is ...` | Frames changed or were rotated since `sfm.json` was written. Rerun step 1. |

## Assumptions I could not check (your files are on your PC, not here)
1. **`board.json` layout.** The loader wants a `markers` list of `{id, x_mm, y_mm}` (it falls back to `planned_xy_mm`)
   and a `marker_mm` size (default 40). It stops with a message naming the keys it found if that does not match.
2. **`targets.json` layout.** A `targets` list (or a bare list) of `{id, x_mm, y_mm, measured_mm}`. IDs starting with `D` are decoys.
3. **Where `slidescan.py frames` writes frames**, and whether `slidescan.py check` accepts `"method": "walkaround"` in
   `meta.json`. If `check` rejects it, add `walkaround` to the method vocabulary in `slidescan.py` (search for `grazing`).
4. **Marker orientation.** Corners are assumed TL, TR, BR, BL of a marker glued square with its top edge toward +y, as in the Phase 1 plan.
5. **Python.** The Windows base env is Python 3.14; pycolmap wheels for 3.14 may not exist, which is why step 1 is WSL.

## What was and was not tested
Tested on a synthetic scene (textured board, 4 markers, 7 flat-topped bolts at 1 to 10 mm, 24 camera positions, noise added):
- the plane sweep with the true poses, the tag-to-board fit, and COLMAP end to end (results in the log of this change).
Not tested: real video, real bolts (heads are shiny and have sides that occlude neighbours; the synthetic ones are matte flat discs),
motion blur, a rolling shutter, a real iPhone lens, and Windows. The apparent-size cue and refining poses with the markers are not built.
The first real run will be harder than the synthetic one; treat its number, not the synthetic one, as the result.

## Known limits
- Heights are read from horizontal patches. Slanted or very glossy surfaces give weaker scores.
- A bolt that occludes its neighbour in oblique views can bias the board reference next to it; the score uses the best 70% of views to blunt this.
- Blind mode only scores cells seen by at least 6 of the used frames, and leaves out the marker squares (plus 3 mm) so markers cannot become detections.
- Everything depends on the SfM poses. If `mean_reprojection_error_px` in `sfm.json` is above about 1 px, look at the frames before trusting the heights.
