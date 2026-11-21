# Run-folder format

Everything SlideScan does reads or writes one **run folder**, `runs/<date>_<label>/`. The tools talk to each other only
through these files, so a new method or scorer needs nothing but this page. All lengths are millimetres; board
coordinates have the origin at the board's bottom-left corner, x right, y up.

```
runs/<run>/
  meta.json            how the run was captured               (you write it; `slidescan new` makes the template)
  targets.json         the board: markers and the known bumps (you write it; `slidescan check` validates it)
  raw/                 the videos or photos as captured
  frames/              extracted frames                       (slidescan frames)
  detections.json      marker corners found in each frame     (slidescan frames)
  methods/<name>/      one folder per method run
    candidates.json    what the method flagged                (the contract: score.py reads it)
    results.json       the graded outcome                     (score.py writes it)
    heights.json, scale.json, ...  method-specific extras
```

## meta.json
Flat object. Every field must be filled before `slidescan check` passes (write `"n/a"` where one doesn't apply).
Fields: `date`, `device`, `capture_app`, `method`, `surface`, `light_level`, `light_elevation_deg`,
`light_azimuth_deg`, `light_distance_mm`, `standoff_cm`, `pass_type`, `texture_help`, `repeat`, `operator`, `notes`.
`score.py` treats runs whose meta differs only in `date`, `repeat`, `operator` and `notes` as repeats of one condition.

## targets.json
```json
{"board_size_mm": [333.38, 349.25], "marker_mm": 38.3,
 "markers": [{"id": 0, "x_mm": 108.16, "y_mm": 95.25}],
 "targets": [{"id": "T05", "readings_mm": [3.19, 3.2, 3.21], "measured_mm": 3.2, "x_mm": 44.63, "y_mm": 152.98}]}
```
- `targets` are the known bumps (bolts) and their caliper heights: three `readings_mm`, `measured_mm` is their median.
  An id starting with `D` is a decoy: a bump that is not required, and a flag on it counts as a decoy alarm.
- `markers` are the printed tags (ArUco `DICT_4X4_50`) at their measured centres; `marker_mm` is the printed side.
- `board.json` at the repo root is the board's design layout and is the source for these positions.

## detections.json
`{"every": 3, "images": {"frames/x.jpg": {"<marker id>": [[x, y], [x, y], [x, y], [x, y]]}}}` : the four pixel corners of
each marker seen in each frame.

## `methods/<name>/candidates.json` (the contract)
Every method writes this; `score.py` grades it.
```json
{"candidates": [{"x_mm": 286.2, "y_mm": 276.6, "score": 32.9, "est_height_mm": 10.05, "image": "raw/p01_on.jpg"}],
 "covered_targets": ["T01", "T02"], "covered_area_cm2": 782.9}
```
- `candidates`: every spot the method flagged, in board mm. `est_height_mm` may be `null`; `score` and `image` are
  informational.
- `covered_targets`: target ids the method actually looked at. A target not listed is "not covered", not a miss.
- `covered_area_cm2`: board area looked at, used for false-alarm rates.

## `methods/<name>/results.json` (written by score.py)
Per target `outcome` (`hit`, `miss` or `not covered`), `qualifying`, and for hits `dist_mm` and `est_height_mm`; plus
`decoy_alarms`, `background_alarms` and their per-100 cm2 rates. It records the match radius and `--min-est` used.

## Method extras (not part of the contract)
- `heights.json`: per target `est_height_mm`, `error_mm`, `n_views_seen`, `covered`, ...; a `summary` of the errors.
- `scale.json` (walkaround): `scale_mm_per_unit`, `rms_mm` (marker fit, about 1-2 mm is normal), drift between the
  video's halves, tagged and registered frame counts.
- `sfm.json`, `heightmap.npz`, `diagnostics.json`: the walkaround's reconstruction and settings.

These shapes were read from real run folders (one grazing demo, one walkaround run). Fields not listed above exist and
may change; only `candidates.json` is a contract.
