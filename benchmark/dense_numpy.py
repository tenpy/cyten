"""Dense NumPy reference helpers for cyten benchmarks."""

from __future__ import annotations

import numpy as np


def to_numpy(tensor) -> np.ndarray:
    """Export a cyten tensor to a dense ndarray."""
    return tensor.to_numpy(understood_braiding=True)


def as_matrix(tensor) -> np.ndarray:
    """Reshape ``to_numpy(tensor)`` to a matrix (codomain × domain)."""
    arr = to_numpy(tensor)
    n_cod = tensor.num_codomain_legs
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    cod_dim = int(np.prod(arr.shape[:n_cod], dtype=int)) if n_cod else 1
    dom_dim = int(np.prod(arr.shape[n_cod:], dtype=int)) if arr.ndim > n_cod else 1
    return np.reshape(arr, (cod_dim, dom_dim))


def numpy_combine_legs(a: np.ndarray, axes_groups):
    """Combine axis groups via transpose + reshape (TeNPy-style dense combine).

    Parameters
    ----------
    a
        Dense array.
    axes_groups
        Sequence of sequences of axis indices, e.g. ``([0, 1], [2, 3])``.

    Returns
    -------
    combined : ndarray
        Contiguous array with one axis per group.
    pipes : list[list[int]]
        Original axis sizes within each group (for a matching split).
    """
    axes = [list(group) for group in axes_groups]
    pipes = [[int(a.shape[i]) for i in comb] for comb in axes]
    transp: list[int] = []
    newshape: list[int] = []
    for ax in axes:
        transp.extend(ax)
        newshape.append(int(np.prod([a.shape[i] for i in ax])))
    out = np.transpose(a, transp)
    out = np.reshape(out, newshape)
    return np.ascontiguousarray(out).copy(), pipes


def numpy_split_legs(a: np.ndarray, pipes):
    """Inverse of ``numpy_combine_legs``: reshape using stored pipe dims."""
    flat = [d for pipe in pipes for d in pipe]
    return np.ascontiguousarray(a.reshape(flat)).copy()
