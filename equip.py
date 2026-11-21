"""Equipment profiles: the rules and tuned numbers for one kind of playground equipment live in equipment/<name>.json,
the gauge shapes they cite in gauges/<name>.json. load("slide") returns the profile dict.
(Named equip, not profile: `profile` is a standard-library module.)"""
import json
from pathlib import Path

import checks

ROOT = Path(__file__).resolve().parent


def load(name="slide"):
    f = ROOT / "equipment" / f"{name}.json"
    if not f.exists():
        raise SystemExit(f"no equipment profile {f}")
    return json.loads(f.read_text())


def gauge(name):
    return json.loads((ROOT / "gauges" / f"{name}.json").read_text())


REQUIRED = ("name", "check", "gauges", "qualifying_mm", "low_mm", "match_radius_mm", "flag_min_est_mm")


def names():
    return sorted(f.stem for f in (ROOT / "equipment").glob("*.json") if not f.stem.startswith("_"))


def validate(p):
    """Raise ValueError if an equipment profile can't be used: a missing key, an unknown check, a missing gauge file."""
    missing = [k for k in REQUIRED if k not in p]
    if missing:
        raise ValueError(f"profile {p.get('name')!r} is missing {missing}")
    if p["check"] not in checks.CHECKS:
        raise ValueError(f"profile {p['name']!r}: unknown check {p['check']!r}; known: {list(checks.CHECKS)}")
    for g in p["gauges"]:
        if not (ROOT / "gauges" / f"{g}.json").exists():
            raise ValueError(f"profile {p['name']!r}: no gauges/{g}.json")
    return p
