"""Contraction (tdot / compose) benchmarks."""

from __future__ import annotations

from .cases import CASES, case_compatible, make_matrix_pair
from .common import (
    BenchmarkRecord,
    device_available,
    device_compatible,
    make_record,
    normalize_block_backend,
    num_blocks_of,
    resolve_dtype,
    time_call,
)


def run_tdot_benchmark(
    *,
    case: str,
    dim: int,
    symmetry_backend: str,
    block_backend: str,
    device: str,
    dtype: str = 'float64',
    warmup: int = 2,
    repeats: int = 5,
    seed: int = 0,
) -> BenchmarkRecord | None:
    """Time ``tdot`` (and record compose-equivalent contraction) for one config.

    Returns ``None`` if the config is incompatible or the device is unavailable.
    """
    if case not in CASES:
        raise ValueError(f'Unknown case {case!r}')
    if not case_compatible(case, symmetry_backend):
        return None
    if not device_compatible(block_backend, device):
        return None
    if not device_available(device):
        return None

    ct_dtype = resolve_dtype(dtype)
    A, B, actual_dim = make_matrix_pair(
        case,
        dim,
        symmetry_backend=symmetry_backend,
        block_backend=block_backend,
        device=device,
        dtype=ct_dtype,
        seed=seed,
    )

    import cyten as ct

    def _once():
        # Explicit tensordot on the shared map legs (same as A @ B for square maps).
        return ct.tdot(A, B, 'j', 'i')

    timing = time_call(_once, warmup=warmup, repeats=repeats, device=device)
    return make_record(
        op='tdot',
        case=case,
        symmetry=CASES[case].symmetry_name,
        symmetry_backend=symmetry_backend,
        block_backend=normalize_block_backend(block_backend),
        device=device,
        dim=dim,
        actual_dim=actual_dim,
        num_blocks=num_blocks_of(A),
        dtype=dtype,
        timing=timing,
    )
