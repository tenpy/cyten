"""Hermitian eigendecomposition (eigh) benchmarks."""

from __future__ import annotations

import numpy as np

import cyten as ct

from .cases import CASES, case_compatible, make_hermitian_map
from .common import (
    BenchmarkRecord,
    config_ok,
    make_numpy_record,
    make_record,
    normalize_block_backend,
    num_blocks_of,
    resolve_dtype,
    run_numpy_ref,
    time_call,
)
from .dense_numpy import as_matrix


def run_eigh_benchmark(
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
    """Time ``cyten.eigh``; optionally also time dense ``numpy.linalg.eigh``."""
    if case not in CASES:
        raise ValueError(f'Unknown case {case!r}')
    if not config_ok(case, symmetry_backend, block_backend, device, case_compatible_fn=case_compatible):
        return []

    ct_dtype = resolve_dtype(dtype)
    H, actual_dim = make_hermitian_map(
        case,
        dim,
        symmetry_backend=symmetry_backend,
        block_backend=block_backend,
        device=device,
        dtype=ct_dtype,
        seed=seed,
    )

    def _cyten():
        return ct.eigh(H, new_labels=['e'], new_leg_dual=False)

    timing = time_call(_cyten, warmup=warmup, repeats=repeats, device=device)
    records = [
        make_record(
            op='eigh',
            case=case,
            symmetry=CASES[case].symmetry_name,
            symmetry_backend=symmetry_backend,
            block_backend=normalize_block_backend(block_backend),
            device=device,
            dim=dim,
            actual_dim=actual_dim,
            num_blocks=num_blocks_of(H),
            dtype=dtype,
            timing=timing,
        )
    ]

    if numpy_ref:

        def _numpy_record():
            mat = as_matrix(H)

            def _numpy():
                return np.linalg.eigh(mat)

            return make_numpy_record(
                op='eigh',
                case=case,
                symmetry=CASES[case].symmetry_name,
                dim=dim,
                actual_dim=actual_dim,
                num_blocks=num_blocks_of(H),
                dtype=dtype,
                timing=time_call(_numpy, warmup=warmup, repeats=repeats, device='cpu'),
            )

        np_record = run_numpy_ref(_numpy_record)
        if np_record is not None:
            records.append(np_record)
    return records
