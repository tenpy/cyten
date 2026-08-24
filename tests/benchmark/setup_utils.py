"""Shared helpers for cyten linalg microbenchmarks."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np

import cyten as ct
from cyten.testing.random_generation import random_symmetry_sectors, randomly_drop_blocks


def symmetry_from_mod_q(mod_q):
    """Map TeNPy-style ``mod_q`` to a cyten ``Symmetry``.

    - ``[]`` → no symmetry
    - ``[1]`` → U(1)
    - ``[n]`` with ``n >= 2`` → Z_n
    - several ints → product, e.g. ``[1, 1]`` → U(1)×U(1)
    """
    if not mod_q:
        return ct.no_symmetry
    factors = []
    for q in mod_q:
        if q == 1:
            factors.append(ct.u1_symmetry)
        elif q >= 2:
            factors.append(ct.ZN(N=int(q)).as_Symmetry())
        else:
            raise ValueError(f'Invalid mod_q entry {q}')
    sym = factors[0]
    for extra in factors[1:]:
        sym = sym * extra
    return sym


def get_backend_from_kwargs(_symmetry, **kwargs):
    """Return a tensor backend from CLI-style kwargs."""
    symmetry_backend = kwargs.get('symmetry_backend', 'abelian')
    block_backend = kwargs.get('block_backend', 'numpy')
    return ct.get_backend(symmetry_backend, block_backend)


def as_cyten_dtype(dtype):
    if dtype is None:
        return ct.float64
    if hasattr(dtype, 'to_numpy_dtype'):
        return dtype
    dt = np.dtype(dtype)
    if dt == np.float64:
        return ct.float64
    if dt == np.float32:
        return ct.float32
    if dt == np.complex128:
        return ct.complex128
    if dt == np.complex64:
        return ct.complex64
    raise TypeError(f'Unsupported dtype {dtype!r}')


def rand_permutation(n):
    """Return a random permutation of length n."""
    perm = list(range(n))
    np.random.shuffle(perm)
    return perm


def rand_distinct_int(a, b, n):
    """Return n distinct integers from a to b inclusive."""
    if n < 0:
        raise ValueError
    if n > b - a + 1:
        raise ValueError
    if n == 0:
        return np.array([], dtype=int)
    return np.sort(np.random.randint(a, b - n + 2, size=n)) + np.arange(n)


def rand_partitions(a, b, n):
    """Return ``[a] + cuts + [b]``, where cuts are ``n-1`` strictly ordered values in between."""
    if b - a <= n:
        return np.array(range(a, b + 1))
    return np.concatenate(([a], rand_distinct_int(a + 1, b - 1, n - 1), [b]))


def _np_rng():
    """Generator seeded from the global ``np.random`` state used by the harness."""
    return np.random.default_rng(np.random.randint(0, 2**31 - 1))


def random_elementary_space(symmetry, size, n_sectors):
    """Random ``ElementarySpace`` with total dim ``size`` and about ``n_sectors`` sectors."""
    size = max(int(size), 1)
    if symmetry == ct.no_symmetry or getattr(symmetry, 'num_sectors', None) == 1:
        return ct.ElementarySpace.from_trivial_sector(dim=size, symmetry=symmetry)

    n_sectors = max(1, min(int(n_sectors), size))
    slices = rand_partitions(0, size, n_sectors)
    mults = np.diff(slices).astype(int)
    n_actual = len(mults)
    rng = _np_rng()
    sectors = random_symmetry_sectors(symmetry, n_actual, np_random=rng)
    n_got = len(sectors)
    if n_got == 0:
        return ct.ElementarySpace.from_trivial_sector(dim=size, symmetry=symmetry)
    if n_got < n_actual:
        extra = int(mults[n_got:].sum())
        mults = np.concatenate([mults[: n_got - 1], [mults[n_got - 1] + extra]])
    else:
        mults = mults[:n_got]
    return ct.ElementarySpace.from_defining_sectors(symmetry, sectors, multiplicities=mults.tolist())


def maybe_drop_blocks(tensor, select_frac):
    """Drop a random subset of blocks if ``select_frac < 1``."""
    if select_frac >= 1.0:
        return tensor
    blocks = getattr(getattr(tensor, 'data', None), 'blocks', None)
    if not blocks:
        return tensor
    num_blocks = len(blocks)
    if num_blocks <= 1:
        return tensor
    max_keep = max(int(num_blocks * select_frac), 1)
    return randomly_drop_blocks(tensor, max_blocks=max_keep, empty_ok=False, np_random=_np_rng())


def random_permute_legs(tensor):
    """Randomly permute legs, keeping the number of (co)domain legs."""
    if not tensor.symmetry.has_symmetric_braid:
        return tensor
    n = tensor.num_legs
    if n <= 1:
        return tensor
    perm = rand_permutation(n)
    n_cod = tensor.num_codomain_legs
    return ct.permute_legs(tensor, perm[:n_cod], perm[n_cod:])


def random_tensor(codomain, domain, labels, backend, dtype=None, select_frac=1.0, permute=True):
    """Random ``SymmetricTensor`` with optional block thinning and a random leg permutation."""
    T = ct.SymmetricTensor.from_random_uniform(
        codomain=codomain,
        domain=domain,
        backend=backend,
        labels=labels,
        dtype=as_cyten_dtype(dtype),
    )
    T = maybe_drop_blocks(T, select_frac)
    if permute:
        T = random_permute_legs(T)
    return T


def to_numpy(tensor):
    return tensor.to_numpy(understood_braiding=True)


def leg_indices(tensor, labels):
    """Positions of ``labels`` in ``tensor.legs`` / ``tensor.labels``."""
    all_labels = list(tensor.labels)
    return [all_labels.index(lab) for lab in labels]


def dense_as_matrix(tensor):
    """Reshape ``to_numpy(tensor)`` to a matrix (codomain × domain)."""
    arr = to_numpy(tensor)
    n_cod = tensor.num_codomain_legs
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    cod_dim = int(np.prod(arr.shape[:n_cod], dtype=int)) if n_cod else 1
    dom_dim = int(np.prod(arr.shape[n_cod:], dtype=int)) if arr.ndim > n_cod else 1
    return np.reshape(arr, (cod_dim, dom_dim))
