"""python test_profile.py -- the slide profile and gauge files load, agree with the CPSC figures, and score.py reads them."""
import checks
import equip
import score as sc

p = equip.load("slide")
# the numbers that used to be hard-coded in score.py must come out the same
assert (sc.QUAL_MM, sc.LOW_MM, sc.RADIUS_MM) == (3.2, 2.0, 10.0), (sc.QUAL_MM, sc.LOW_MM, sc.RADIUS_MM)
assert p["flag_min_est_mm"] == 5.5 and p["check"] == "projection"

rings = {g["id"]: g for g in equip.gauge("cpsc_projection_rings")["gauges"]}
assert [rings[k]["hole"] for k in ("small", "medium", "large")] == [0.5, 1.5, 3.0]   # Figure B1
assert [rings[k]["thickness"] for k in ("small", "medium", "large")] == [0.25, 0.75, 1.5]
slide = equip.gauge("cpsc_slide_projection")
assert slide["thickness_max"] == p["limit_in"] == 0.125   # Figure B2 / test B.2.3
# the bench threshold is the 3.2 mm bolt standing in for the 1/8 in limit: within 0.05 mm of it
assert abs(p["limit_in"] * 25.4 - p["qualifying_mm"]) < 0.05
for g in p["gauges"]:
    equip.gauge(g)   # every gauge the profile names exists
try:
    equip.load("no-such-equipment")
    raise AssertionError("missing profile should exit")
except SystemExit:
    pass

# every shipped profile is usable (this is what a contributor's new equipment/<name>.json has to pass)
assert "slide" in equip.names()
for n in equip.names():
    equip.validate(equip.load(n))
assert set(checks.methods_for("projection")) >= {"walkaround", "grazing"}
for bad in ({}, {**p, "check": "gap"}, {**p, "gauges": ["nope"]}):
    try:
        equip.validate(bad)
        raise AssertionError(f"should reject {bad.get('check')}")
    except ValueError:
        pass
try:
    checks.methods_for("gap")
    raise AssertionError("unknown check should exit")
except SystemExit:
    pass

# score.py takes the thresholds as arguments, so another equipment's numbers change the gate, not just the defaults
cells = {0: [1, 1, 0], 4.0: [1, 1, 0], 6.0: [0, 1, 0]}
assert sc.gate(cells) == (1, 2, 0) and sc.gate(cells, 5.0) == (0, 1, 0)
assert sc.low(cells) == (1, 1) and sc.low(cells, 5.0) == (2, 2)
print("ok")
