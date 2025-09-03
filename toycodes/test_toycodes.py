"""Testing if the toycodes can run."""
# Copyright (C) TeNPy Developers, Apache license
import importlib.util
import pytest
import importlib
from pathlib import Path
import os
import sys

import cyten as ct
from toycodes.tenpy_toycodes.a_mps import init_FM_MPS, init_Neel_MPS, init_SU2_sym_MPS
from toycodes.tenpy_toycodes.b_model import (
    HeisenbergModel, TFIModel, heisenberg_finite_gs_energy, tfi_finite_gs_energy
)
from toycodes.tenpy_toycodes.d_dmrg import DMRGEngine


repo_root = Path(__file__).parent.parent
assert repo_root.joinpath('pyproject.toml').exists()  # make sure this is the root!
tenpy_toycodes = repo_root.joinpath('toycodes').joinpath('tenpy_toycodes')
tenpy_toycode_modules = [str(tenpy_toycodes.joinpath(f))
                         for f in os.listdir(tenpy_toycodes)
                         if f.endswith('.py')]


def test_dmrg_heisenberg():
    backend = ct.backends.FusionTreeBackend(ct.backends.NumpyBlockBackend())
    L = 8
    e_exact = heisenberg_finite_gs_energy(L, J=1)
    for conserve in ['none', 'Z2', 'SU2']:
        if conserve == 'SU2':
            psi = init_SU2_sym_MPS(L, backend=backend)
        else:
            psi = init_Neel_MPS(L, backend=backend, conserve=conserve)
        model = HeisenbergModel(L, J=1, backend=backend, conserve=conserve)
        dmrg = DMRGEngine(psi, model)
        e = dmrg.run()
        assert abs(e - e_exact) < 1e-10


def test_dmrg_tfi(np_random):
    backend = ct.backends.AbelianBackend(ct.backends.NumpyBlockBackend())
    L = 16
    J, g = np_random.random(2)
    e_exact = tfi_finite_gs_energy(L, J, g)

    for conserve in ['none', 'Z2']:
        psi = init_FM_MPS(L, backend=backend, conserve=conserve)
        model = TFIModel(L, J, g, backend=backend, conserve=conserve)
        dmrg = DMRGEngine(psi, model)
        e = dmrg.run()
        assert abs(e - e_exact) < 1e-10
    

@pytest.mark.parametrize('module_file', tenpy_toycode_modules)
def test_tenpy_toycodes(module_file):
    spec = importlib.util.spec_from_file_location("module", module_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod"] = mod
    spec.loader.exec_module(mod)

    try:
        mod.main()
    except AttributeError:
        pass
