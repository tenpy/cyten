"""SVD benchmarks."""

from __future__ import annotations

from .cases import CASES, case_compatible, make_svd_tensor
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


def run_svd_benchmark(
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
    """Time ``cyten.svd`` for one config.

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
    T, actual_dim = make_svd_tensor(
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
        return ct.svd(T)

    timing = time_call(_once, warmup=warmup, repeats=repeats, device=device)
    return make_record(
        op='svd',
        case=case,
        symmetry=CASES[case].symmetry_name,
        symmetry_backend=symmetry_backend,
        block_backend=normalize_block_backend(block_backend),
        device=device,
        dim=dim,
        actual_dim=actual_dim,
        num_blocks=num_blocks_of(T),
        dtype=dtype,
        timing=timing,
    )
