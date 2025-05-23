import importlib.util
import pytest
import importlib
from pathlib import Path
import os
import sys


repo_root = Path(__file__).parent.parent
assert repo_root.joinpath('pyproject.toml').exists()  # make sure this is the root!
tenpy_toycodes = repo_root.joinpath('toycodes').joinpath('tenpy_toycodes')
tenpy_toycode_modules = [str(tenpy_toycodes.joinpath(f))
                         for f in os.listdir(tenpy_toycodes)
                         if f.endswith('.py')]


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
