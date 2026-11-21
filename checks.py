"""Hazard checks and the methods that implement each one.

A check is a kind of hazard (an equipment profile names one in equipment/<name>.json, key "check"). A method is a
module in methods/ with run(run_dir, ...) that writes methods/<name>/candidates.json in the format score.py documents.
To add a method: write the module, add its name to its check below. To add a check (say "gap"): add a key and its methods.
Method modules are not imported here, so a check's heavy dependencies (pycolmap, scipy) load only when you run it."""

CHECKS = {
    "projection": ["walkaround", "grazing", "colmap", "sled"],  # bumps sticking out of a surface (CPSC Appendix B)
}


def methods_for(check):
    if check not in CHECKS:
        raise SystemExit(f"unknown check {check!r}; known: {', '.join(CHECKS)}")
    return CHECKS[check]


if __name__ == "__main__":
    for c, m in CHECKS.items():
        print(f"{c}: {', '.join(m)}")
