"""Equipment profiles: the rules and tuned numbers for one kind of playground equipment live in equipment/<name>.json,
the gauge shapes they cite in gauges/<name>.json. load("slide") returns the profile dict.
(Named equip, not profile: `profile` is a standard-library module.)"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(name="slide"):
    f = ROOT / "equipment" / f"{name}.json"
    if not f.exists():
        raise SystemExit(f"no equipment profile {f}")
    return json.loads(f.read_text())


def gauge(name):
    return json.loads((ROOT / "gauges" / f"{name}.json").read_text())
