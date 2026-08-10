#!/usr/bin/env python3
"""Run ``cmake --build build`` if a local editable build directory exists.

Used as a pre-commit hook. Skips (exit 0) when ``build/`` is absent so CI and
fresh checkouts are unaffected. When present, fails with cmake's exit code.
"""
# Copyright (C) TeNPy Developers, Apache license

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / 'build'


def main() -> int:
    if not (BUILD_DIR / 'CMakeCache.txt').is_file():
        print('No local cmake build/ found; skipping cmake --build.')
        return 0
    print(f'Running: cmake --build {BUILD_DIR}')
    result = subprocess.run(['cmake', '--build', str(BUILD_DIR)], cwd=REPO_ROOT)
    return result.returncode


if __name__ == '__main__':
    sys.exit(main())
