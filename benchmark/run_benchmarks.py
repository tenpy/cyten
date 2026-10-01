#!/usr/bin/env python3
"""CLI entry point for cyten backend benchmarks.

Examples
--------
Run from the repository root (cyten must be importable)::

    python -m benchmark.run_benchmarks \\
        --ops tdot,svd \\
        --cases nosym,u1,su2 \\
        --symmetry-backends no_symmetry,abelian,fusion_tree \\
        --block-backends numpy,torch \\
        --devices cpu \\
        --dims 16,32,64,128 \\
        --output benchmark/results/run.json

Or equivalently::

    python benchmark/run_benchmarks.py ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python benchmark/run_benchmarks.py`` without installing the folder.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ is None:  # pragma: no cover - script entry
    sys.path.insert(0, str(_REPO_ROOT))
    __package__ = 'benchmark'

from benchmark.bench_svd import run_svd_benchmark  # noqa: E402
from benchmark.bench_tdot import run_tdot_benchmark  # noqa: E402
from benchmark.cases import CASES, case_compatible  # noqa: E402
from benchmark.common import (  # noqa: E402
    BLOCK_BACKENDS,
    SYMMETRY_BACKENDS,
    BenchmarkRecord,
    device_available,
    device_compatible,
    normalize_block_backend,
    parse_csv_list,
    parse_int_list,
    save_results,
)

OP_RUNNERS = {
    'tdot': run_tdot_benchmark,
    'svd': run_svd_benchmark,
}

DEFAULT_OPS = 'tdot,svd'
DEFAULT_CASES = 'nosym,u1,su2'
DEFAULT_SYMMETRY_BACKENDS = 'no_symmetry,abelian,fusion_tree'
DEFAULT_BLOCK_BACKENDS = 'numpy,torch'
DEFAULT_DEVICES = 'cpu'
DEFAULT_DIMS = '16,32,64,128,256'


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Benchmark cyten contraction / SVD across backends and devices.',
    )
    p.add_argument(
        '--ops',
        default=DEFAULT_OPS,
        help=f'Comma-separated ops from {sorted(OP_RUNNERS)} (default: {DEFAULT_OPS}).',
    )
    p.add_argument(
        '--cases',
        default=DEFAULT_CASES,
        help=f'Comma-separated cases from {sorted(CASES)} (default: {DEFAULT_CASES}).',
    )
    p.add_argument(
        '--symmetry-backends',
        default=DEFAULT_SYMMETRY_BACKENDS,
        help=f'Comma-separated symmetry backends (default: {DEFAULT_SYMMETRY_BACKENDS}).',
    )
    p.add_argument(
        '--block-backends',
        default=DEFAULT_BLOCK_BACKENDS,
        help=f'Comma-separated block backends / aliases (default: {DEFAULT_BLOCK_BACKENDS}).',
    )
    p.add_argument(
        '--devices',
        default=DEFAULT_DEVICES,
        help=f'Comma-separated devices (default: {DEFAULT_DEVICES}).',
    )
    p.add_argument(
        '--dims',
        default=DEFAULT_DIMS,
        help=f'Comma-separated target leg dimensions (default: {DEFAULT_DIMS}).',
    )
    p.add_argument('--dtype', default='float64', help='Cyten dtype name (default: float64).')
    p.add_argument('--warmup', type=int, default=2, help='Warmup iterations (default: 2).')
    p.add_argument('--repeats', type=int, default=5, help='Timed iterations (default: 5).')
    p.add_argument('--seed', type=int, default=0, help='Reserved seed for reproducibility.')
    p.add_argument(
        '--output',
        default='benchmark/results/run.json',
        help='JSON output path (default: benchmark/results/run.json).',
    )
    p.add_argument(
        '--list-devices',
        action='store_true',
        help='Print which of the requested devices are available and exit.',
    )
    return p


def _iter_configs(args) -> list[dict]:
    ops = parse_csv_list(args.ops)
    cases = parse_csv_list(args.cases)
    symmetry_backends = parse_csv_list(args.symmetry_backends)
    block_backends = parse_csv_list(args.block_backends)
    devices = parse_csv_list(args.devices)
    dims = parse_int_list(args.dims)

    for name in ops:
        if name not in OP_RUNNERS:
            raise SystemExit(f'Unknown op {name!r}; choose from {sorted(OP_RUNNERS)}')
    for name in cases:
        if name not in CASES:
            raise SystemExit(f'Unknown case {name!r}; choose from {sorted(CASES)}')
    for name in symmetry_backends:
        if name not in SYMMETRY_BACKENDS:
            raise SystemExit(f'Unknown symmetry backend {name!r}; choose from {SYMMETRY_BACKENDS}')
    for name in block_backends:
        if name not in BLOCK_BACKENDS:
            raise SystemExit(f'Unknown block backend {name!r}; choose from {BLOCK_BACKENDS}')

    configs = []
    skipped = []
    for op in ops:
        for case in cases:
            for sym_be in symmetry_backends:
                if not case_compatible(case, sym_be):
                    skipped.append(f'{op}/{case}/{sym_be}: incompatible symmetry backend')
                    continue
                for block_be in block_backends:
                    for device in devices:
                        if not device_compatible(block_be, device):
                            skipped.append(
                                f'{op}/{case}/{sym_be}/{block_be}/{device}: device incompatible with block backend'
                            )
                            continue
                        if not device_available(device):
                            skipped.append(f'{op}/{case}/{sym_be}/{block_be}/{device}: device unavailable')
                            continue
                        for dim in dims:
                            configs.append(
                                {
                                    'op': op,
                                    'case': case,
                                    'symmetry_backend': sym_be,
                                    'block_backend': block_be,
                                    'device': device,
                                    'dim': dim,
                                }
                            )
    return configs, skipped


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_devices:
        devices = parse_csv_list(args.devices) or ['cpu', 'cuda:0', 'mps:0']
        for device in devices:
            status = 'available' if device_available(device) else 'unavailable'
            print(f'{device}: {status}')
        return 0

    configs, skipped = _iter_configs(args)
    for msg in skipped:
        print(f'skip: {msg}', file=sys.stderr)

    if not configs:
        print('No compatible benchmark configurations to run.', file=sys.stderr)
        return 1

    print(f'Running {len(configs)} benchmark configurations...')
    records: list[BenchmarkRecord] = []
    for i, cfg in enumerate(configs, start=1):
        runner = OP_RUNNERS[cfg['op']]
        label = (
            f'{cfg["op"]} case={cfg["case"]} sym={cfg["symmetry_backend"]} '
            f'block={normalize_block_backend(cfg["block_backend"])} '
            f'device={cfg["device"]} dim={cfg["dim"]}'
        )
        print(f'[{i}/{len(configs)}] {label} ...', flush=True)
        try:
            record = runner(
                case=cfg['case'],
                dim=cfg['dim'],
                symmetry_backend=cfg['symmetry_backend'],
                block_backend=cfg['block_backend'],
                device=cfg['device'],
                dtype=args.dtype,
                warmup=args.warmup,
                repeats=args.repeats,
                seed=args.seed,
            )
        except Exception as exc:  # noqa: BLE001 - keep suite running
            print(f'  FAILED: {exc}', file=sys.stderr)
            continue
        if record is None:
            print('  skipped (runtime filter)')
            continue
        print(
            f'  mean={record.mean:.4e}s  median={record.median:.4e}s  '
            f'actual_dim={record.actual_dim}  blocks={record.num_blocks}'
        )
        records.append(record)

    out = Path(args.output)
    if not out.is_absolute():
        out = _REPO_ROOT / out
    save_results(out, records)
    print(f'Wrote {len(records)} records to {out}')
    return 0 if records else 1


if __name__ == '__main__':
    raise SystemExit(main())
