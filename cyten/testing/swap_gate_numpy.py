"""Tools for recreating tensor operations on numpy representations.

This uses e.g. explicit SWAP gates.
"""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np

from ..symmetries import Leg, swap_gate
from ..tools.misc import permutation_as_swaps, to_valid_idx


def transpose(arr: np.ndarray, legs: list[Leg], perm: list[int]) -> np.ndarray:
    """Leg permutation that includes swap gates.

    Assumes that all legs are bent to the right (if they are bent).

    Reduces to ``np.transpose(arr, perm)`` if the symmetry has a trivial braid.
    Otherwise, achieves the leg permutation by contraction with :func:`cyten.symmetries.swap_gate` s
    """
    legs = list(legs)  # ensure copy, we will modify it in-place
    N = arr.ndim
    assert len(legs) == N
    if N == 0:
        return arr
    if legs[0].symmetry.has_trivial_braid:
        return np.transpose(arr, perm)

    perm = [to_valid_idx(i, N) for i in perm]
    if set(perm) != set(range(N)):
        raise ValueError('Not a permutation')

    for j in permutation_as_swaps(perm):
        arr = apply_swap_gate(arr, legs, j)
        legs[j], legs[j + 1] = legs[j + 1], legs[j]

    return arr


def apply_swap_gate(arr: np.ndarray, legs: list[Leg], j: int) -> np.ndarray:
    """Applies swap gates on legs ``j, j + 1``"""
    swap = swap_gate(legs[j], legs[j + 1])
    # [0, 1, 2, ..., j, j + 1, ..., N - 1] @ [j + 1, j, j + 1*, j*]
    #   -> [0, 1, ..., j - 1, j + 2, ..., N - 1, j + 1, j]
    res = np.tensordot(arr, swap, ([j, j + 1], [3, 2]))
    return np.transpose(res, [*range(j), -2, -1, *range(j, arr.ndim - 2)])
