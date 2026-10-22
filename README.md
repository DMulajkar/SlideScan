# SlideScan — Phase 1: measurement bench and data logging

Phase 1 of the slide-protrusion build roadmap (Second-Brain: `02 Projects/Slide Protrusion Detection - Build Roadmap.md`).
Everything later (signal tests, detectors) reads the run folders this produces, so nothing has to be recaptured.

**Done when:** one practice run is captured, `check` passes on it, and someone new can read the run folder and know what was captured.

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
