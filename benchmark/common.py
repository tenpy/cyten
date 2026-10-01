"""Shared helpers for cyten benchmarks: backends, timing, JSON I/O."""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cyten as ct

# Known string names accepted by ``cyten.get_backend``.
SYMMETRY_BACKENDS = ('no_symmetry', 'abelian', 'fusion_tree')
BLOCK_BACKENDS = ('numpy', 'torch', 'cpu', 'gpu', 'apple_silicon')
DTYPE_MAP = {
    'float32': ct.float32,
    'float64': ct.float64,
    'complex64': ct.complex64,
    'complex128': ct.complex128,
}


@dataclass(frozen=True)
class TimingResult:
    """Statistics from a timed callable."""

    mean: float
    median: float
    min: float
    stdev: float
    repeats: int
    warmup: int


@dataclass
class BenchmarkRecord:
    """One timed benchmark configuration and its timing stats."""

    op: str
    case: str
    symmetry: str
    symmetry_backend: str
    block_backend: str
    device: str
    dim: int
    actual_dim: int
    num_blocks: int | None
    dtype: str
    mean: float
    median: float
    min: float
    stdev: float
    repeats: int
    warmup: int
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d.get('extra') is None:
            d.pop('extra', None)
        return d


def parse_csv_list(value: str | None) -> list[str]:
    """Split a comma-separated CLI string into stripped non-empty parts."""
    if value is None or value.strip() == '':
        return []
    return [part.strip() for part in value.split(',') if part.strip()]


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of integers."""
    return [int(x) for x in parse_csv_list(value)]


def resolve_dtype(name: str):
    """Map a dtype name string to a cyten ``Dtype``."""
    key = name.lower()
    if key not in DTYPE_MAP:
        raise ValueError(f'Unknown dtype {name!r}; choose from {sorted(DTYPE_MAP)}')
    return DTYPE_MAP[key]


def normalize_block_backend(name: str) -> str:
    """Canonicalize block-backend aliases used by ``get_backend``."""
    aliases = {
        'cpu': 'numpy',
        'gpu': 'torch',
        'apple_silicon': 'torch',
    }
    return aliases.get(name, name)


def device_compatible(block_backend: str, device: str) -> bool:
    """Return whether ``device`` is usable with the given block backend."""
    bb = normalize_block_backend(block_backend)
    # Aliases imply a default device; still allow explicit override when sensible.
    if block_backend == 'gpu':
        return device.startswith('cuda')
    if block_backend == 'apple_silicon':
        return device.startswith('mps')
    if bb == 'numpy':
        return device in ('cpu', 'cpu:0')
    if bb == 'torch':
        return device.startswith('cpu') or device.startswith('cuda') or device.startswith('mps')
    return False


def get_tensor_backend(symmetry_backend: str, block_backend: str):
    """Create a tensor backend via ``cyten.get_backend``."""
    return ct.get_backend(symmetry_backend, block_backend)


def sync_device(device: str) -> None:
    """Synchronize asynchronous device work before stopping the timer."""
    if not device.startswith('cuda'):
        return
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def device_available(device: str) -> bool:
    """Check whether a device string is currently usable."""
    if device in ('cpu', 'cpu:0'):
        return True
    if device.startswith('cuda'):
        try:
            import torch
        except ImportError:
            return False
        if not torch.cuda.is_available():
            return False
        if ':' in device:
            index = int(device.split(':', 1)[1])
            return index < torch.cuda.device_count()
        return True
    if device.startswith('mps'):
        try:
            import torch
        except ImportError:
            return False
        return bool(getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())
    return False


def time_call(
    fn: Callable[[], Any],
    *,
    warmup: int = 2,
    repeats: int = 5,
    device: str = 'cpu',
) -> TimingResult:
    """Time ``fn`` with warmups and optional device sync."""
    if repeats < 1:
        raise ValueError('repeats must be >= 1')
    for _ in range(max(0, warmup)):
        fn()
        sync_device(device)

    samples: list[float] = []
    for _ in range(repeats):
        sync_device(device)
        t0 = time.perf_counter()
        fn()
        sync_device(device)
        samples.append(time.perf_counter() - t0)

    stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return TimingResult(
        mean=statistics.mean(samples),
        median=statistics.median(samples),
        min=min(samples),
        stdev=stdev,
        repeats=repeats,
        warmup=warmup,
    )


def num_blocks_of(tensor) -> int | None:
    """Best-effort block count from tensor backend data."""
    data = getattr(tensor, 'data', None)
    if data is None:
        return None
    blocks = getattr(data, 'blocks', None)
    if blocks is not None:
        try:
            return len(blocks)
        except TypeError:
            pass
    block_inds = getattr(data, 'block_inds', None)
    if block_inds is not None:
        try:
            return int(len(block_inds))
        except TypeError:
            pass
    # NoSymmetryBackend stores a single dense block as ``data``.
    if hasattr(data, 'shape') and hasattr(data, 'to_numpy'):
        return 1
    return None


def save_results(path: str | Path, records: Sequence[BenchmarkRecord]) -> None:
    """Write benchmark records as a JSON list."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.to_dict() for r in records]
    out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON results file produced by ``save_results``."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def make_record(
    *,
    op: str,
    case: str,
    symmetry: str,
    symmetry_backend: str,
    block_backend: str,
    device: str,
    dim: int,
    actual_dim: int,
    num_blocks: int | None,
    dtype: str,
    timing: TimingResult,
    extra: dict[str, Any] | None = None,
) -> BenchmarkRecord:
    """Build a ``BenchmarkRecord`` from timing + metadata."""
    return BenchmarkRecord(
        op=op,
        case=case,
        symmetry=symmetry,
        symmetry_backend=symmetry_backend,
        block_backend=normalize_block_backend(block_backend),
        device=device,
        dim=dim,
        actual_dim=actual_dim,
        num_blocks=num_blocks,
        dtype=dtype,
        mean=timing.mean,
        median=timing.median,
        min=timing.min,
        stdev=timing.stdev,
        repeats=timing.repeats,
        warmup=timing.warmup,
        extra=extra,
    )
