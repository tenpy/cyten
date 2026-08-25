"""Run user-guide example scripts so they stay working."""

import runpy
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_first_steps_heisenberg_example():
    example = _REPO_ROOT / 'docs' / 'intro' / 'examples' / 'heisenberg_two_site.py'
    runpy.run_path(str(example))


def test_from_npc_trivial_example():
    example = _REPO_ROOT / 'docs' / 'intro' / 'examples' / 'from_npc_trivial.py'
    runpy.run_path(str(example))


def test_from_npc_u1_example():
    example = _REPO_ROOT / 'docs' / 'intro' / 'examples' / 'from_npc_u1.py'
    runpy.run_path(str(example))


def test_from_tenpy_couplings_example():
    example = _REPO_ROOT / 'docs' / 'intro' / 'examples' / 'from_tenpy_couplings.py'
    runpy.run_path(str(example))


def test_from_tenpy_planar_example():
    example = _REPO_ROOT / 'docs' / 'intro' / 'examples' / 'from_tenpy_planar.py'
    runpy.run_path(str(example))
