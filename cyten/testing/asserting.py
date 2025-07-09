"""Assertion wrappers for testing."""
# Copyright (C) TeNPy Developers, Apache license
from ..tensors import SymmetricTensor


def assert_tensors_almost_equal(a: SymmetricTensor, expect: SymmetricTensor,
                                rtol: float = 1e-12, atol: float = 1e-12):
    assert a.codomain == expect.codomain
    assert a.domain == expect.domain
    assert a.backend.almost_equal(a, expect, rtol=rtol, atol=atol)
