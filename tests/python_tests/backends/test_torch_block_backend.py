"""Minimal torch block-backend smoke tests."""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytest.importorskip('torch')

from cyten.backends.backend_factory import get_backend
from cyten.block_backends import TorchBlockBackend
from cyten.block_backends.dtypes import Dtype
from cyten.symmetries import no_symmetry
from cyten.tensors import SymmetricTensor
from cyten.testing import random_tensor


@pytest.mark.torch
def test_torch_block_backend_zeros():
    bb = TorchBlockBackend.from_factory('cpu:0')
    z = bb.zeros([2, 3], Dtype.float64)
    assert bb.get_shape(z) == (2, 3)
    assert bb.get_dtype(z) == Dtype.float64
    assert bb.get_device(z) == 'cpu:0'
    assert float(bb.sum_all(z).as_float64()) == 0.0


@pytest.mark.torch
def test_to_backend_numpy_to_torch():
    np_random = np.random.default_rng(0)
    b1 = get_backend('no_symmetry', 'numpy')
    b2 = get_backend('no_symmetry', 'torch')
    tens = random_tensor(no_symmetry, 1, 1, backend=b1, np_random=np_random, cls=SymmetricTensor)
    res = tens.to_backend(b2)
    res.test_sanity()
    assert isinstance(res.backend.block_backend, TorchBlockBackend)
    recovered = res.to_backend(b1)
    recovered.test_sanity()


_DUAL_IMPORT_SCRIPT = textwrap.dedent(
    r"""
    import os
    import sys

    order = sys.argv[1]
    if order == 'torch_then_cyten':
        import torch
        import cyten  # noqa: F401
    else:
        import cyten  # noqa: F401
        import torch

    from cyten.block_backends import TorchBlockBackend
    from cyten.block_backends.dtypes import Dtype
    import numpy as np

    # Same physical libtorch for cyten._core and torch._C (conda/pip layouts).
    maps = open(f'/proc/{os.getpid()}/maps').read()
    torch_sos = sorted({line.split()[-1] for line in maps.splitlines()
                        if line.endswith('.so') and 'libtorch.so' in line
                        and not line.endswith('(deleted)')})
    # Prefer the realpath of mapped libtorch.so entries (ignore vdso etc.).
    real = {os.path.realpath(p) for p in torch_sos if os.path.exists(p)}
    assert len(real) == 1, f'expected one libtorch.so, got {real!r}'

    bb = TorchBlockBackend.from_factory('cpu:0')
    z = bb.zeros([2, 2], Dtype.float64)
    t = torch.zeros(2, 2, dtype=torch.float64)
    assert bb.get_shape(z) == tuple(t.shape)
    assert float(bb.sum_all(z).as_float64()) == float(t.sum())
    assert np.arange(4).sum() == 6
    print('ok', order, next(iter(real)))
    """
).strip()


@pytest.mark.torch
@pytest.mark.parametrize('order', ['cyten_then_torch', 'torch_then_cyten'])
def test_shared_libtorch_dual_import(order):
    """cyten._core and torch._C must share one libtorch; both import orders must work."""
    env = os.environ.copy()
    # Ensure the just-built/installed package is used.
    proc = subprocess.run(
        [sys.executable, '-c', _DUAL_IMPORT_SCRIPT, order],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
    assert 'ok' in proc.stdout
