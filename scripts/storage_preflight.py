#!/usr/bin/env python3
"""Fail-fast storage admission check with no model imports or file creation."""
from __future__ import annotations
import math, shutil, sys
from pathlib import Path

def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} TARGET_PATH MIN_FREE_GIB", file=sys.stderr); return 2
    try: minimum = float(argv[2])
    except (TypeError, ValueError):
        print(f"invalid minimum free GiB: {argv[2]}", file=sys.stderr); return 2
    if not math.isfinite(minimum) or minimum < 0:
        print(f"invalid minimum free GiB: {argv[2]}", file=sys.stderr); return 2
    target = Path(argv[1]).expanduser()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try: free = shutil.disk_usage(probe).free
    except OSError as exc:
        print(f"STORAGE_PREFLIGHT_FAILED: cannot inspect {probe}: {exc}", file=sys.stderr); return 28
    free_gib = free / (1024 ** 3)
    if free_gib < minimum:
        print(f"STORAGE_PREFLIGHT_FAILED: free={free_gib:.2f} GiB minimum={minimum:.2f} GiB", file=sys.stderr); return 28
    print(f"STORAGE_PREFLIGHT_OK: free={free_gib:.2f} GiB minimum={minimum:.2f} GiB"); return 0

if __name__ == "__main__": raise SystemExit(main(sys.argv))
