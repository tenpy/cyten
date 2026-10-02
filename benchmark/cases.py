"""Named benchmark cases with deterministic legs of controllable dimension."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cyten as ct
from cyten.symmetries import su_n_data_file_path, su_n_data_filename

from .common import get_tensor_backend

# Highest weights matching the N=2 files shipped under ``external/SUN_symbols``.
_SUN2_CG_HWEIGHT = 20
_SUN2_F_HWEIGHT = 6
_SUN2_R_HWEIGHT = 6


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
        symmetry_name='SUN',
        allowed_symmetry_backends=('fusion_tree',),
    ),
}


def _repo_sun_symbols_path() -> Path | None:
    """``external/SUN_symbols`` next to the repo root, if present."""
    cand = Path(__file__).resolve().parent.parent / 'external' / 'SUN_symbols'
    if not cand.is_dir():
        return None
    # Require the three N=2 files the benchmark uses.
    needed = [
        su_n_data_filename(2, kind, h)
        for kind, h in (
            ('CG', _SUN2_CG_HWEIGHT),
            ('F', _SUN2_F_HWEIGHT),
            ('R', _SUN2_R_HWEIGHT),
        )
    ]
    if all((cand / name).is_file() for name in needed):
        return cand
    return None


@lru_cache(maxsize=1)
def _sun2_symmetry():
    """SU(2) via ``SUN(N=2, ...)``, not the deprecated test-only ``_SU2``."""
    kwargs = dict(
        f_hweight=_SUN2_F_HWEIGHT,
        r_hweight=_SUN2_R_HWEIGHT,
        descriptive_name='SU(2)',
    )
    try:
        return ct.SUN(2, _SUN2_CG_HWEIGHT, **kwargs).as_Symmetry()
    except FileNotFoundError as first_exc:
        fallback = _repo_sun_symbols_path()
        if fallback is None:
            raise RuntimeError(
                'SU(2) benchmarks need SUN(N=2) Clebsch-Gordan / F / R data. '
                'Install files for N=2 with hweights '
                f'CG={_SUN2_CG_HWEIGHT}, F={_SUN2_F_HWEIGHT}, R={_SUN2_R_HWEIGHT}, e.g. '
                f'{su_n_data_file_path(2, "CG", _SUN2_CG_HWEIGHT)!r}, '
                "or set cyten.set_options(su_n_data_path='...'), or check out "
                'the external/SUN_symbols submodule in this repository.'
            ) from first_exc
        return ct.SUN(2, _SUN2_CG_HWEIGHT, path=str(fallback), **kwargs).as_Symmetry()


def get_symmetry(case: str):
    """Return the ``Symmetry`` instance for a named case."""
    if case == 'nosym':
        return ct.NoSymmetry().as_Symmetry()
    if case == 'u1':
        return ct.U1().as_Symmetry()
    if case == 'su2':
        return _sun2_symmetry()
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
    """SU(2) leg via ``SUN(N=2)`` GT sectors ``[0,0]``, ``[1,0]``, ``[2,0]`` (j=0,½,1).

    Quantum dimensions are ``1, 2, 3``; total dim ≈ ``6 * mult``.
    """
    sectors = [[0, 0], [1, 0], [2, 0]]
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


def make_hermitian_map(
    case: str,
    dim: int,
    *,
    symmetry_backend: str,
    block_backend: str,
    device: str,
    dtype,
    seed: int = 0,
):
    """Hermitian square map ``A + A.hc`` for eigh. Returns ``(H, actual_dim)``."""
    A, actual_dim = make_svd_tensor(
        case,
        dim,
        symmetry_backend=symmetry_backend,
        block_backend=block_backend,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    return A + A.hc, actual_dim


def make_rank4_tensor(
    case: str,
    dim: int,
    *,
    symmetry_backend: str,
    block_backend: str,
    device: str,
    dtype,
    seed: int = 0,
):
    """Rank-4 map with two identical legs per (co)domain. Returns ``(T, actual_dim)``.

    Labels are ``['a', 'b', 'c', 'd']`` with ``codomain=[leg, leg]``, ``domain=[leg, leg]``.
    """
    if not case_compatible(case, symmetry_backend):
        raise ValueError(f'case {case!r} incompatible with symmetry backend {symmetry_backend!r}')
    leg = make_leg(case, dim)
    actual_dim = int(leg.dim)
    backend = get_tensor_backend(symmetry_backend, block_backend)
    _ = seed
    T = ct.SymmetricTensor.from_random_uniform(
        codomain=[leg, leg],
        domain=[leg, leg],
        backend=backend,
        labels=['a', 'b', 'c', 'd'],
        dtype=dtype,
        device=device,
    )
    return T, actual_dim
