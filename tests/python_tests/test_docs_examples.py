"""Run user-guide example scripts so they stay working."""

import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_first_steps_heisenberg_example():
    example = _REPO_ROOT / 'docs' / 'intro' / 'examples' / 'heisenberg_two_site.py'
    runpy.run_path(str(example))
