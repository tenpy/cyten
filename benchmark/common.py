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
    impl: str = 'cyten'
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


def _read_cmake_cache(cache_path: Path) -> dict[str, str]:
    """Parse selected entries from a CMakeCache.txt file."""
    keys = (
        'CMAKE_BUILD_TYPE',
        'CMAKE_CXX_COMPILER',
        'CMAKE_CXX_FLAGS',
        'CMAKE_CXX_FLAGS_DEBUG',
        'CMAKE_CXX_FLAGS_RELEASE',
        'CMAKE_CXX_FLAGS_RELWITHDEBINFO',
        'CMAKE_CXX_FLAGS_MINSIZEREL',
    )
    wanted = set(keys)
    found: dict[str, str] = {}
    try:
        text = cache_path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return found
    for line in text.splitlines():
        if not line or line.startswith('//') or line.startswith('#'):
            continue
        # KEY:TYPE=VALUE
        if ':' not in line or '=' not in line:
            continue
        key_type, _, value = line.partition('=')
        key = key_type.split(':', 1)[0]
        if key in wanted:
            found[key] = value
    return found


def _guess_cmake_cache_paths() -> list[Path]:
    """Candidate CMakeCache.txt locations for an editable / in-tree build."""
    candidates: list[Path] = []
    # Repo root next to the installed/editable ``cyten`` package.
    pkg_root = Path(ct.__file__).resolve().parent.parent
    candidates.append(pkg_root / 'build' / 'CMakeCache.txt')
    # Current working directory (when launching from the repo).
    candidates.append(Path.cwd() / 'build' / 'CMakeCache.txt')
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def collect_compile_info() -> dict[str, Any]:
    """Best-effort compile / build-type info for reproducibility notes."""
    import os

    info: dict[str, Any] = {
        'python_debug': __debug__,
        'cxxflags_env': os.environ.get('CXXFLAGS') or None,
        'cflags_env': os.environ.get('CFLAGS') or None,
        'cmake_build_type': None,
        'cmake_cxx_compiler': None,
        'cmake_cxx_flags': None,
        'cmake_cxx_flags_for_build_type': None,
        'cmake_cache': None,
    }
    for cache_path in _guess_cmake_cache_paths():
        if not cache_path.is_file():
            continue
        cache = _read_cmake_cache(cache_path)
        if not cache:
            continue
        build_type = cache.get('CMAKE_BUILD_TYPE') or ''
        info['cmake_cache'] = str(cache_path)
        info['cmake_build_type'] = build_type or None
        info['cmake_cxx_compiler'] = cache.get('CMAKE_CXX_COMPILER') or None
        info['cmake_cxx_flags'] = cache.get('CMAKE_CXX_FLAGS') or None
        bt_key = f'CMAKE_CXX_FLAGS_{build_type.upper()}' if build_type else None
        if bt_key and bt_key in cache:
            info['cmake_cxx_flags_for_build_type'] = cache[bt_key] or None
        break
    return info


def collect_run_metadata(*, cli_args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gather host / version / compile metadata at the start of a benchmark run."""
    import platform
    import sys
    from datetime import datetime

    import numpy

    try:
        import torch

        torch_version = getattr(torch, '__version__', None)
    except ImportError:
        torch_version = None

    started = datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M')
    return {
        'cyten_version': getattr(ct, '__version__', None),
        'cyten_commit_id': getattr(ct, '__commit_id__', None),
        'hostname': platform.node(),
        'started_at': started,
        'python_version': sys.version.replace('\n', ' '),
        'numpy_version': numpy.__version__,
        'torch_version': torch_version,
        'platform': platform.platform(),
        'compile': collect_compile_info(),
        'cli': cli_args,
    }


def save_results(
    path: str | Path,
    records: Sequence[BenchmarkRecord],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write benchmark records as JSON with a top-level metadata section."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'metadata': metadata if metadata is not None else {},
        'results': [r.to_dict() for r in records],
    }
    out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')


def load_results(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load a results file.

    Supports the current ``{metadata, results}`` object and legacy bare lists.
    Returns ``(metadata, records)``.
    """
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(raw, list):
        return {}, raw
    if isinstance(raw, dict) and 'results' in raw:
        meta = raw.get('metadata') or {}
        if not isinstance(meta, dict):
            meta = {}
        results = raw['results']
        if not isinstance(results, list):
            raise ValueError(f'Invalid results payload in {path}')
        return meta, results
    raise ValueError(f'Unrecognized benchmark results format in {path}')


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
    impl: str = 'cyten',
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
        impl=impl,
        extra=extra,
    )


def make_numpy_record(
    *,
    op: str,
    case: str,
    symmetry: str,
    dim: int,
    actual_dim: int,
    num_blocks: int | None,
    dtype: str,
    timing: TimingResult,
    extra: dict[str, Any] | None = None,
) -> BenchmarkRecord:
    """Build a dense NumPy reference record (always host CPU)."""
    return make_record(
        op=op,
        case=case,
        symmetry=symmetry,
        symmetry_backend='dense',
        block_backend='numpy',
        device='cpu',
        dim=dim,
        actual_dim=actual_dim,
        num_blocks=num_blocks,
        dtype=dtype,
        timing=timing,
        impl='numpy',
        extra=extra,
    )


def config_ok(
    case: str,
    symmetry_backend: str,
    block_backend: str,
    device: str,
    *,
    case_compatible_fn,
) -> bool:
    """Shared filter for runners: case/backend/device compatibility."""
    if not case_compatible_fn(case, symmetry_backend):
        return False
    if not device_compatible(block_backend, device):
        return False
    if not device_available(device):
        return False
    return True
