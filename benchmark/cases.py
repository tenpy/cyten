"""Named benchmark cases with deterministic legs of controllable dimension."""

from __future__ import annotations

from dataclasses import dataclass

import cyten as ct

from .common import get_tensor_backend


@dataclass(frozen=True)
class CaseSpec:
    """A named symmetry / space family for size sweeps."""

    name: str
    symmetry_name: str
    allowed_symmetry_backends: tuple[str, ...]


CASES: dict[str, CaseSpec] = {
    'nosym': CaseSpec(
        name='nosym',
        symmetry_name='NoSymmetry',
        allowed_symmetry_backends=('no_symmetry',),
    ),
    'u1': CaseSpec(
        name='u1',
        symmetry_name='U1',
        allowed_symmetry_backends=('abelian', 'fusion_tree'),
    ),
    'su2': CaseSpec(
        name='su2',
        symmetry_name='SU2',
        allowed_symmetry_backends=('fusion_tree',),
    ),
}


def get_symmetry(case: str):
    """Return the ``Symmetry`` instance for a named case."""
    if case == 'nosym':
        return ct.NoSymmetry().as_Symmetry()
    if case == 'u1':
        return ct.U1().as_Symmetry()
    if case == 'su2':
        return ct.SU2().as_Symmetry()
    raise ValueError(f'Unknown case {case!r}; choose from {sorted(CASES)}')


def case_compatible(case: str, symmetry_backend: str) -> bool:
    """Whether ``symmetry_backend`` can run the named case."""
    spec = CASES.get(case)
    if spec is None:
        return False
    return symmetry_backend in spec.allowed_symmetry_backends


def _u1_leg(symmetry, dim: int):
    """U(1) leg with charges ``-2..2`` and equal multiplicities ≈ ``dim / 5``."""
    sectors = [[c] for c in range(-2, 3)]
    n = len(sectors)
    mult = max(1, int(dim) // n)
    multiplicities = [mult] * n
    return ct.ElementarySpace.from_defining_sectors(symmetry, sectors, multiplicities=multiplicities)


def _su2_leg(symmetry, dim: int):
    """SU(2) leg with sectors ``0,1,2`` (j=0,1/2,1); total dim ≈ ``6 * mult``."""
    # sector label n has quantum dimension n+1
    sectors = [[0], [1], [2]]
    qdims = [1, 2, 3]
    unit = sum(qdims)
    mult = max(1, int(dim) // unit)
    multiplicities = [mult] * len(sectors)
    return ct.ElementarySpace.from_defining_sectors(symmetry, sectors, multiplicities=multiplicities)


def make_leg(case: str, dim: int):
    """Build a deterministic ``ElementarySpace`` with total dim near ``dim``."""
    if dim < 1:
        raise ValueError('dim must be >= 1')
    symmetry = get_symmetry(case)
    if case == 'nosym':
        return ct.ElementarySpace.from_trivial_sector(int(dim), symmetry=symmetry)
    if case == 'u1':
        return _u1_leg(symmetry, dim)
    if case == 'su2':
        return _su2_leg(symmetry, dim)
    raise ValueError(f'Unknown case {case!r}')


def make_matrix_pair(
    case: str,
    dim: int,
    *,
    symmetry_backend: str,
    block_backend: str,
    device: str,
    dtype,
    seed: int = 0,
):
    """Two square maps on the same leg for contraction benchmarks.

    Returns ``(A, B, actual_dim)`` where ``A @ B`` / ``tdot(A, B, ...)`` is valid.
    """
    if not case_compatible(case, symmetry_backend):
        raise ValueError(f'case {case!r} incompatible with symmetry backend {symmetry_backend!r}')
    leg = make_leg(case, dim)
    actual_dim = int(leg.dim)
    backend = get_tensor_backend(symmetry_backend, block_backend)
    # Seed is recorded by callers; cyten random fills are not currently seeded via
    # a public RNG argument on from_random_uniform, so we only stabilize shape/size.
    _ = seed
    A = ct.SymmetricTensor.from_random_uniform(
        codomain=[leg],
        domain=[leg],
        backend=backend,
        labels=['i', 'j'],
        dtype=dtype,
        device=device,
    )
    B = ct.SymmetricTensor.from_random_uniform(
        codomain=[leg],
        domain=[leg],
        backend=backend,
        labels=['i', 'j'],
        dtype=dtype,
        device=device,
    )
    return A, B, actual_dim


def make_svd_tensor(
    case: str,
    dim: int,
    *,
    symmetry_backend: str,
    block_backend: str,
    device: str,
    dtype,
    seed: int = 0,
):
    """One square map for SVD benchmarks. Returns ``(T, actual_dim)``."""
    if not case_compatible(case, symmetry_backend):
        raise ValueError(f'case {case!r} incompatible with symmetry backend {symmetry_backend!r}')
    leg = make_leg(case, dim)
    actual_dim = int(leg.dim)
    backend = get_tensor_backend(symmetry_backend, block_backend)
    _ = seed
    T = ct.SymmetricTensor.from_random_uniform(
        codomain=[leg],
        domain=[leg],
        backend=backend,
        labels=['i', 'j'],
        dtype=dtype,
        device=device,
    )
    return T, actual_dim
