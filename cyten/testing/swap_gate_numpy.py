"""Tools for recreating tensor operations on numpy representations.

This uses e.g. explicit SWAP gates.

- combining / splitting legs can be done as usual with np.reshape
- changing leg order
    - with a *cyclic* permutation (i.e. leg bends without braids) is fine with np.transpose
    - with non-cyclic permutation, needs swap gates, see `transpose` and `permute_legs` in this module.

"""

# Copyright (C) TeNPy Developers, Apache license
from collections.abc import Sequence

import numpy as np

from ..symmetries import Leg, SymmetryError, swap_gate, twist_gate
from ..tools.misc import is_iterable, permutation_as_swaps, to_valid_idx


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


def permute_legs(
    arr: np.ndarray,
    num_codomain_legs: int,
    legs: list[Leg],
    codomain: list[int] = None,
    domain: list[int] = None,
    bend_right: bool | Sequence[bool | None] = None,
):
    """Like :func:`cyten.permute_legs`, but on a numpy representation.

    Explicitly includes swap gates and twists as needed.
    """
    N = arr.ndim
    assert 0 <= num_codomain_legs <= N
    assert len(legs) == N
    if N == 0:
        return arr
    symm = legs[0].symmetry

    if codomain is None and domain is None:
        raise ValueError('Need to give at least one of codomain or domain')
    elif codomain is None:
        domain = [to_valid_idx(i, N) for i in domain]
        codomain = [i for i in range(N) if i not in domain]
    elif domain is None:
        codomain = [to_valid_idx(i, N) for i in codomain]
        domain = [i for i in reversed(range(N)) if i not in codomain]
    else:
        codomain = [to_valid_idx(i, N) for i in codomain]
        domain = [to_valid_idx(i, N) for i in domain]
        assert len(codomain) + len(domain) == N
        assert set(codomain).union(domain) == set(range(N))

    bending_legs = [i for i in codomain if i >= num_codomain_legs] + [i for i in domain if i < num_codomain_legs]
    if is_iterable(bend_right):
        assert len(bend_right) == N
    elif bend_right is None:
        bend_right = [None] * N
    elif bend_right in [True, False]:
        bend_right = [bend_right] * N
    else:
        raise ValueError

    # check if those that need to be specified are
    if symm.has_trivial_braid:
        # it doesnt matter which way. choose all right
        bend_right = [True] * N
    else:
        if any(bend_right[l] is None for l in bending_legs):
            raise SymmetryError('Need to specify bend_right!')

    # do twist
    for i in bending_legs:
        if not bend_right[i]:
            arr = apply_twist(arr, legs, i)

    return transpose(arr, legs, [*codomain, *reversed(domain)])


def apply_swap_gate(arr: np.ndarray, legs: list[Leg], j: int) -> np.ndarray:
    """Applies swap gates on legs ``j, j + 1``"""
    swap = swap_gate(legs[j], legs[j + 1])
    # [0, 1, 2, ..., j, j + 1, ..., N - 1] @ [j + 1, j, j + 1*, j*]
    #   -> [0, 1, ..., j - 1, j + 2, ..., N - 1, j + 1, j]
    res = np.tensordot(arr, swap, ([j, j + 1], [3, 2]))
    return np.transpose(res, [*range(j), -2, -1, *range(j, arr.ndim - 2)])


def apply_twist(arr: np.ndarray, legs: list[Leg], j: int) -> np.ndarray:
    """Applies a twist on leg ``j``"""
    # [0, ..., j-1, j, j+1, ..., N-1] @ [j, j*] -> [..., j-1, j+1, ..., N-1, j*]
    res = np.tensordot(arr, twist_gate(legs[j]), (j, 0))
    return np.moveaxis(res, -1, j)
