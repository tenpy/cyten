"""split_legs benchmarks."""

from __future__ import annotations

import cyten as ct

from .cases import CASES, case_compatible, make_rank4_tensor
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
from .dense_numpy import numpy_combine_legs, numpy_split_legs, to_numpy


def run_split_legs_benchmark(
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
    """Time ``cyten.split_legs``; optionally also time dense reshape-split.

    Combine is performed in setup (not timed) for both cyten and NumPy paths.
    """
    if case not in CASES:
        raise ValueError(f'Unknown case {case!r}')
    if not config_ok(case, symmetry_backend, block_backend, device, case_compatible_fn=case_compatible):
        return []

    ct_dtype = resolve_dtype(dtype)
    T, actual_dim = make_rank4_tensor(
        case,
        dim,
        symmetry_backend=symmetry_backend,
        block_backend=block_backend,
        device=device,
        dtype=ct_dtype,
        seed=seed,
    )
    combined = ct.combine_legs(T, [0, 1], [2, 3])

    def _cyten():
        return ct.split_legs(combined)

    timing = time_call(_cyten, warmup=warmup, repeats=repeats, device=device)
    records = [
        make_record(
            op='split_legs',
            case=case,
            symmetry=CASES[case].symmetry_name,
            symmetry_backend=symmetry_backend,
            block_backend=normalize_block_backend(block_backend),
            device=device,
            dim=dim,
            actual_dim=actual_dim,
            num_blocks=num_blocks_of(combined),
            dtype=dtype,
            timing=timing,
        )
    ]

    if numpy_ref:
        arr = to_numpy(T)
        combined_np, pipes = numpy_combine_legs(arr, ([0, 1], [2, 3]))

        def _numpy():
            return numpy_split_legs(combined_np, pipes)

        np_timing = time_call(_numpy, warmup=warmup, repeats=repeats, device='cpu')
        records.append(
            make_numpy_record(
                op='split_legs',
                case=case,
                symmetry=CASES[case].symmetry_name,
                dim=dim,
                actual_dim=actual_dim,
                num_blocks=num_blocks_of(combined),
                dtype=dtype,
                timing=np_timing,
            )
        )
    return records
