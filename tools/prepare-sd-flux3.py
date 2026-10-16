#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Backport upstream FLUX.3 Image Klein support into the pinned stable-diffusion.cpp."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "tools/patches/stable-diffusion-flux3-klein.patch"
MARKER = "src/model/diffusion/flux3.hpp"


def patch(source: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(["patch", "-p1", "--batch", "--silent", *flags, "-i", str(PATCH)],
                          cwd=source, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".deps/stable-diffusion-vulkan",
                        help="stable-diffusion.cpp source tree (default: the pinned dependency)")
    args = parser.parse_args()
    source = args.source.resolve()
    if not (source / "src/pipeline/diffusion_engine.cpp").is_file():
        raise SystemExit(f"{source}: not a stable-diffusion.cpp source tree")
    if (source / MARKER).is_file():
        if patch(source, "--dry-run", "--reverse").returncode != 0:
            raise SystemExit(f"{source}: {MARKER} exists but differs from {PATCH.name}")
        print(f"{source}: FLUX.3 Image Klein backport present")
        return
    check = patch(source, "--dry-run", "--forward")
    if check.returncode != 0:
        raise SystemExit(f"{source}: {PATCH.name} does not apply\n{check.stdout}{check.stderr}")
    applied = patch(source, "--forward")
    if applied.returncode != 0:
        raise SystemExit(f"{source}: applying {PATCH.name} failed\n{applied.stdout}{applied.stderr}")
    print(f"{source}: FLUX.3 Image Klein backport applied")


if __name__ == "__main__":
    main()
