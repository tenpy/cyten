"""Ensure generated ``cyten._core`` stubs track the compiled extension API."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO / 'scripts'


def _is_compiled_core() -> bool:
    try:
        spec = importlib.util.find_spec('cyten._core')
    except ImportError, ModuleNotFoundError, ValueError:
        return False
    if spec is None or spec.loader is None:
        return False
    if isinstance(spec.loader, importlib.machinery.ExtensionFileLoader):
        return True
    origin = spec.origin or ''
    return origin.endswith('.so') or origin.endswith('.pyd')


@pytest.mark.skipif(not _is_compiled_core(), reason='compiled cyten._core not available')
def test_core_stub_parity():
    sys.path.insert(0, str(_SCRIPTS))
    try:
        import cyten._core as live
    except ImportError as exc:
        pytest.skip(f'compiled cyten._core not importable: {exc}')

    origin = getattr(live, '__file__', '') or ''
    if origin.endswith('.py'):
        pytest.skip('cyten._core is a pure-Python stub')

    from check_core_stub_parity import compare
    from generate_core_stubs import generate_stub

    stub_src = generate_stub(_REPO, from_doxygen=False, strict_docs=False)
    stub_path = _REPO / 'build' / '_core_stub_parity_pytest.py'
    stub_path.parent.mkdir(parents=True, exist_ok=True)
    stub_path.write_text(stub_src, encoding='utf-8')

    spec = importlib.util.spec_from_file_location('_core_stub_parity_pytest', stub_path)
    assert spec is not None and spec.loader is not None
    stub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stub)

    errors = compare(live, stub)
    assert not errors, 'stub parity errors:\n' + '\n'.join(f'  - {e}' for e in errors[:40])
