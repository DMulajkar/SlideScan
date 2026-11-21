"""python test_profile.py -- the slide profile and gauge files load, agree with the CPSC figures, and score.py reads them."""
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
print("ok")
