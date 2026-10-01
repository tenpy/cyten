"""SVD benchmarks."""

from __future__ import annotations

import numpy as np

import cyten as ct

from .cases import CASES, case_compatible, make_svd_tensor
from .common import (
    BenchmarkRecord,
    config_ok,
    make_numpy_record,
    make_record,
    normalize_block_backend,
    num_blocks_of,
    resolve_dtype,
    time_call,
)
from .dense_numpy import as_matrix


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
    numpy_ref: bool = False,
) -> list[BenchmarkRecord]:
    """Time ``cyten.svd``; optionally also time dense ``numpy.linalg.svd``."""
    if case not in CASES:
        raise ValueError(f'Unknown case {case!r}')
    if not config_ok(case, symmetry_backend, block_backend, device, case_compatible_fn=case_compatible):
        return []

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

    def _cyten():
        return ct.svd(T)

    timing = time_call(_cyten, warmup=warmup, repeats=repeats, device=device)
    records = [
        make_record(
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
    ]

    if numpy_ref:
        mat = as_matrix(T)

        def _numpy():
            return np.linalg.svd(mat, full_matrices=False)

        np_timing = time_call(_numpy, warmup=warmup, repeats=repeats, device='cpu')
        records.append(
            make_numpy_record(
                op='svd',
                case=case,
                symmetry=CASES[case].symmetry_name,
                dim=dim,
                actual_dim=actual_dim,
                num_blocks=num_blocks_of(T),
                dtype=dtype,
                timing=np_timing,
            )
        )
    return records
