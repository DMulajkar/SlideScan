# SlideScan — Phase 1: measurement bench and data logging

Phase 1 of the slide-protrusion build roadmap (Second-Brain: `02 Projects/Slide Protrusion Detection - Build Roadmap.md`).
Everything later (signal tests, detectors) reads the run folders this produces, so nothing has to be recaptured.

**Done when:** one practice run is captured, `check` passes on it, and someone new can read the run folder and know what was captured.

## Quick start: height map from one phone video (walk-around method)
This is the method that has worked so far: film one slow walk-around of the board, COLMAP gets the camera poses, the printed markers fix the scale, and a plane sweep reads the height of every spot on the board. Details and limits: `WALKAROUND_README.md`.

**What it has shown:** four runs on one matte board with seven bolts (0, 1, 3, 3.2, 4, 5, 10 mm). Every bolt of 3.2 mm or taller was found every time, and heights read about 3.4 mm too high by a steady offset. **What it has not shown:** decoys, glossy or curved surfaces, a different board, a real slide. Those four runs are the same four required bolts on one board, so this is not a measured miss rate.

**You need**
- The marker board: print `markers_0-15.pdf` and `markers_16-31.pdf` at 100% scale, glue the markers at the positions in `board.json`, and caliper every bolt (3 readings each, see "Physical build" below). The markers are what give the scale.
- Texture on the board (matte tape scraps or paint speckle). A plain board gives COLMAP nothing to match. `check` refuses `texture_help: none`.
- An iPhone video: main lens (1x), 4K 30 fps, 15-30 s, 20-30 cm above the board, one slow orbit with at least 30 cm of sideways travel, exposure and focus locked, soft light, no glare, no zoom changes.
- A Windows PC with WSL Ubuntu (it was run this way only). Native Windows and Mac are untested. pycolmap has no Linux ARM build, so an ARM Linux server cannot `pip install` it.

**Install once**
```
# Windows
pip install opencv-python pillow pillow-heif
# WSL Ubuntu (Python 3.12; ffmpeg is optional, for the replay video)
python3 -m venv ~/vslam/.venv
~/vslam/.venv/bin/pip install pycolmap opencv-python numpy scipy pillow
```
COLMAP on 4K frames crashed WSL until `C:\Users\<you>\.wslconfig` had `[wsl2]`, `memory=20GB`, `swap=16GB` (run `wsl --shutdown` after editing). Use 16 GB or more of RAM on the PC.

**Run** (the run folder is `runs/<date>_<label>`)
```
# Windows, in this folder
python slidescan.py new <label>                    # makes the run folder
#   copy the video into runs/<run>/raw/ ; fill in meta.json (method "walkaround", texture_help "speckle" or "tape",
#   every field except notes) ; put your caliper readings in the run's targets.json
python slidescan.py frames runs/<run> --every 3    # about 10 frames per second, finds the markers
python slidescan.py check runs/<run>               # lists what is missing; "OK" when complete

# WSL: work on a copy on WSL's own disk (the Windows disk is slow and was unstable)
R=/mnt/c/Users/<you>/Desktop/SlideScan/runs/<run>   # edit the path
W=$HOME/w_run; rm -rf $W; mkdir -p $W
cp -r $R/frames $W/frames; cp $R/meta.json $R/targets.json $R/detections.json $W/
cd /mnt/c/Users/<you>/Desktop/SlideScan
~/vslam/.venv/bin/python methods/walkaround_sfm.py $W --every 3 --max-image-size 2400 --focal-35mm 26 --threads 4   # about 2 min
~/vslam/.venv/bin/python methods/walkaround.py $W                                                                  # about 8 min
mkdir -p $R/methods; cp -r $W/methods/walkaround $R/methods/
~/vslam/.venv/bin/python replay.py $R              # optional: reconstruction_replay.mp4, about 3 min

# Windows
python score.py runs/<run> --method walkaround --gate                  # needs bolts at known positions
python score.py runs/<run> --method walkaround --gate --min-est 5.5    # same, dropping bumps that read under 5.5 mm
```
`--focal-35mm 26` is the iPhone main camera; leave it off for another lens. Frames are 4K, so `--max-image-size 2400` and `--threads 4` keep COLMAP inside memory.

**Read the result** in `runs/<run>/methods/walkaround/`: `heights.json` (each known bolt: estimate against caliper), `heightmap.png`, `candidates.json` (every bump found), `scale.json` (how well the markers fit; about 1-2 mm rms is normal). `replay.py` adds `reconstruction_replay.mp4` and `panel_topdown_map.png` to the run folder.
- Heights are relative to the local board and read high by a steady offset of about 3.4 mm. A flush bolt reads about 3 mm. Judge a bump against that, not against zero.
- The 5.5 mm cut (`--min-est 5.5`) was picked on two runs and held on two others. It leaves the flush and 1 mm bolts unflagged in three of the four runs.
- The first run had 11 false alarms at that cut, the other three 0 or 1. A 3.0 mm bolt reads 5.6-9.0 mm and cannot be told from a 3.2 mm one.
- If the run stops, the message tells you why: see the table in `WALKAROUND_README.md`.

**Self-tests** (no camera needed, each prints `ok`): `python test_slidescan.py`, `python test_score.py`, and in WSL `python test_walkaround.py --quick` (about 2 min), `python test_replay.py`.

## Physical build (not code)
1. **Board.** 1220 x 610 mm (2 x 4 ft) flat matte board. Every target and marker has a `planned_xy_mm` in `board.json` (the layout from the Phase 1 plan). Markers 0 and 1 are planned 1000 mm apart; their measured positions are the scale reference. Curved glossy panel comes later.
2. **Targets.** Mount the targets listed in `board.json` (bolts at 0, 1, 2, 3, 3.2, 4, 5, 10 mm; two-thread bolt; flared head; lifted seam; covered screw; hex head T13, proposed; three decoys). Add duplicates as `T14`, `T15`… if you want more trials per height. Take **three caliper readings** of each height, perpendicular to the surrounding surface, into `readings_mm`, and write their median as `measured_mm`. `check` refuses a run otherwise. Nominal is not enough.
3. **Markers.** The board uses IDs 0-26, so print both sheets at **100% / Actual size**: `python slidescan.py markers` gives `markers_0-15.pdf`, and `python slidescan.py markers --start 16` gives `markers_16-31.pdf` (27-31 are spares). Check the 100 mm bar with calipers, caliper one black square and write it as `marker_mm`. Glue each one flat at its planned spot, **square to the board, with the edge its ID label was printed under toward -y** (the bottom edge). Then measure each center and write `x_mm`, `y_mm` (origin bottom-left, x right, y up).
4. **Jig and light.** Tripod or fixed arm at 20–30 cm. One separate low-angle light (not the phone flash) with elevation and azimuth measured. A 3.2 mm (0.125 in) feeler gauge.

## Each capture run
```
python slidescan.py new practice          # -> runs/<date>_practice/
```
1. Copy the phone video(s) or photos into `raw/` as they come off the phone. iPhone HEIC photos and HEVC `.mov` videos both work, so the camera can stay on **High Efficiency**: `frames` converts each HEIC to a JPG in `frames/` and leaves the original in `raw/`.
2. Fill in `meta.json`. Every field except `notes` is required; write `n/a` where it doesn't apply (for example, the light fields when there is no separate light). The light fields are measured from the board's `light_ref_mm` in `board.json` (lean board: its center, (166.69, 174.63); mark it in pencil):
   - `light_azimuth_deg`: direction from that point toward the light, degrees counterclockwise from board +x (after any board rotation)
   - `light_distance_mm`: horizontal distance from that point to the light
   - `light_elevation_deg`: angle from the board up to the light, seen from that point
3. If the board changed since the run was created, update the run's `targets.json`. It is a snapshot of `board.json`.
```
python slidescan.py frames runs/<run>     # every 10th frame -> frames/, markers -> detections.json
python slidescan.py check runs/<run>      # lists what's missing; "OK" when complete
```

Allowed values (fixed so runs group cleanly later): see `CHOICES` in `slidescan.py`.

## Run folder
| File | What |
|---|---|
| `raw/` | Video or photos exactly as captured. Never edited. |
| `meta.json` | Conditions: device, app, method, surface, light level, light elevation, azimuth and distance, standoff, pass type, texture help, repeat, operator. |
| `targets.json` | Board snapshot: target readings, heights and positions, marker positions, `marker_mm`. |
| `frames/` | Extracted frames, named `<video>_<frame index>.jpg` (time = index / fps), plus `<photo>.jpg` for each HEIC photo. Can be rebuilt from `raw/`. |
| `detections.json` | ArUco corners per image, in pixels. Later used to map pixels to board coordinates. |
| `geometry.json` | Per image: board-mm to pixel homography `H` (null if fewer than 2 markers), markers used, worst corner residual per marker in mm, targets fully in view. Written by `geometry.py`. |
| `geometry_preview.jpg` | The board rectified from the best image, each target's position circled. Bolts inside their circles = mapping and positions agree. |
| `sled.csv`, `sled_lanes.csv` | Sled runs only, typed in by hand: catch or pass per target per pull, and each pull's lane (see Phase 2: sled). |
| `methods/<name>/` | A method's `candidates.json` (board mm, plus coverage) and `score.py`'s `results.json`; grazing also writes `heights.json`. |

## Phase 2: geometry
```
python geometry.py runs/<run>       # -> geometry.json + geometry_preview.jpg
```
Needs a run that passes `check`. Prints how many images got a mapping, each marker's residual (over 2 mm is flagged: misplaced, rotated or mis-measured) and which targets were ever fully in view. One marker alone is not enough (`--min-markers`, default 2): on run B a single-marker mapping was up to 28 mm off. Other tools import `fit`, `to_pixel`, `to_board`, `in_view` and `crop` from it.

## Phase 2: score
```
python score.py runs/* --method grazing --gate   # -> methods/grazing/results.json per run + k/n table + Gate A
```
Grades each run's `methods/<name>/candidates.json` (written by every method, board mm) against `targets.json`: a candidate within 10 mm (`--radius`) of a target is a hit, each target at most once; outcome per target is hit, miss or not covered; near a decoy (`D..`) is a decoy alarm, near nothing a background alarm. Runs whose `meta.json` match apart from date, repeat, operator and notes are repeats of one condition. The table shows hits/covered per measured height (k/n, never percentages) and the smallest height at and above which every covered target was hit in every repeat. `--gate` adds Gate A: zero misses on covered targets of 3.2 mm or taller, with the rule-of-three bound 3/n. Without `--method`, every method found is scored.

## Phase 2: grazing light (method 6)
```
python slidescan.py frames runs/<run>                # markers (HEIC photos become JPGs in frames/)
python geometry.py runs/<run>                        # mapping from each pNN_pose.jpg
python methods/grazing.py runs/<run> [--l-min 5]     # -> methods/grazing/candidates.json + heights.json
python score.py runs/<run> --method grazing --gate
```
Photos per tripod position `NN`, all copied into `raw/`: `pNN_pose.jpg` (room lights, for the markers), `pNN_off.jpg` and `pNN_on.jpg` (test lighting, grazing light off and on; exposure and focus locked, no touching the phone between them). The light fields in `meta.json` must be numbers. The adapter takes `on − off`, rectifies it to board mm, blanks every marker square (plus 3 mm), and marks as shadow whatever the light lit at under half the level of the board around it.
- **Blind (`candidates.json`, scored):** every shadow reaching at least `--l-min` mm away from the light is a candidate at its bolt end, with `score` = that length and `est_height_mm`. One shadow seen from several positions is one candidate.
- **Targeted (`heights.json`, not scored):** per covered target, the shadow length beside it and the height from it, median over positions, next to `measured_mm`.
- Height = L·z/(d + L), i.e. L·tan(elevation seen from the shadow tip), exact for a point light at height z with d from the light to the shadow's start.

## Phase 2: sled (method 7)
```
python methods/sled.py runs/<run>                    # -> methods/sled/candidates.json (no images)
python score.py runs/<run> --method sled --gate
```
Two files typed into the run folder, board mm as in `targets.json`:
- `sled_lanes.csv`: `pull,x0_mm,y0_mm,x1_mm,y1_mm`, the rectangle the bridge swept on each pull. Targets with their center inside are covered; the lanes' area, summed per pull, is `covered_area_cm2`.
- `sled.csv`: `target_id,pull,caught,x_mm,y_mm`, one row per pull for every target in that pull's lane, `caught` 1 or 0. A catch on nothing on the board: `target_id` blank, `caught` 1, and where it stopped in `x_mm,y_mm` (ruler).

Rows and lanes must agree both ways, or the adapter refuses (a forgotten pass row would turn a miss into "not covered"). A target is a candidate only if every pull over it caught it; each catch on a decoy or on nothing is its own candidate (an alarm). Repeats are separate runs (`repeat` in `meta.json`), as for every method. For a sled run (`method` "sled" in `meta.json`), `slidescan.py check` wants `sled.csv` and `sled_lanes.csv` (validated by the adapter) instead of photos in `raw/` and `detections.json`; `score.py` doesn't need it.

## Phase 2: COLMAP free reconstruction (method 3)
```
python slidescan.py new vslam-test                   # copy one slow video sweep of the board into raw/; meta.json method "vslam-colmap"
python slidescan.py frames runs/<run> --every 10     # frames/ + marker detections
python methods/colmap.py runs/<run> [--h-min 2]      # -> methods/colmap/candidates.json + report.json (a few minutes)
python score.py runs/<run> --method colmap --gate
```
Sweep: 30-60 s, phone 20-30 cm above the board, keep moving sideways (a pure rotation gives no baseline), bright light, all four markers in most frames. COLMAP reconstructs in arbitrary scale; the markers' triangulated corners are fitted to their `board.json` positions, so everything comes out in board mm with z = height above the marker plane. The printed report is the answer to "does a plain board reconstruct": frames registered, marker-corner residual (the accuracy floor), plane roughness, and per bolt the points on it and their height next to `measured_mm`. `max z` is outlier-prone; read `median z`. Needs `pip install pycolmap` (Windows wheels exist; no CUDA, sparse only).

Setup: `pip install opencv-python pillow pillow-heif pycolmap`

Self-tests (no camera needed): `python test_slidescan.py`, `python test_geometry.py`, `python test_score.py`, `python test_grazing.py`, `python test_sled.py`, `python test_colmap.py` (about a minute; best-case synthetic render, a ceiling, not a prediction)
