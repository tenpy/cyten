"""Helpers for :class:`~cyten.symmetries.Sector` / :class:`~cyten.symmetries.SectorArray`.

.. todo ::

    these are temporary helpers while converting the codebase to C++ - when everything is migrated,
    the inputs of the helper functions should have well-defined types
    and the python wrappers checking types should no longer be needed.
    Remove this before release.

"""

# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

import numpy as np

from .._core import Sector, SectorArray


def as_sector(obj) -> Sector:
    """Convert a Sector, 1D ndarray, or sequence to :class:`Sector`."""
    if isinstance(obj, Sector):
        return obj
    return Sector(obj)


def as_sector_array(obj, sector_ind_len: int | None = None) -> SectorArray:
    """Convert a SectorArray, 2D ndarray, sequence, or single Sector to :class:`SectorArray`."""
    if isinstance(obj, SectorArray):
        return obj
    if isinstance(obj, Sector):
        return SectorArray.from_sector(obj)
    if obj is None:
        if sector_ind_len is None:
            raise TypeError('as_sector_array(None) requires sector_ind_len')
        return SectorArray.empty(sector_ind_len)
    arr = np.asarray(obj, dtype=int)
    if arr.ndim == 1:
        return SectorArray.from_sector(Sector(arr))
    if arr.ndim != 2:
        raise ValueError(f'Expected 1D or 2D sector data, got shape {arr.shape}')
    if arr.shape[0] == 0 and sector_ind_len is not None:
        return SectorArray.empty(sector_ind_len)
    return SectorArray(arr)


def assert_sectors_equal(a, b, msg: str | None = None):
    """Assert two sectors / sector arrays compare equal (for tests)."""
    if isinstance(a, Sector) or (hasattr(a, 'ndim') and getattr(a, 'ndim', None) == 1):
        sa, sb = as_sector(a), as_sector(b)
        if sa != sb:
            raise AssertionError(msg or f'Sectors differ: {sa!r} != {sb!r}')
        return
    aa, bb = as_sector_array(a), as_sector_array(b)
    if aa != bb:
        raise AssertionError(msg or f'SectorArrays differ: {aa!r} != {bb!r}')


def iter_common_sorted_sector_arrays(a, b, a_strict: bool = True, b_strict: bool = True):
    """Yield ``(i, j)`` for matching rows of lex-sorted SectorArrays."""
    aa = as_sector_array(a)
    bb = as_sector_array(b)
    for i, j in SectorArray.iter_common_sorted(aa, bb, a_strict, b_strict):
        yield int(i), int(j)


__all__ = [
    'Sector',
    'SectorArray',
    'as_sector',
    'as_sector_array',
    'assert_sectors_equal',
    'iter_common_sorted_sector_arrays',
]
