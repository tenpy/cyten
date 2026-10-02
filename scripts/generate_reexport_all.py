#!/usr/bin/env python3
"""Generate/refresh ``__all__`` lists for cyten's re-export hub modules.

With ``cyten/py.typed`` present, Pyright/Pylance only treat a name re-exported through
``from x import y`` as public API if it is also listed in that module's ``__all__``
(PEP 484 "implicit re-export" rules). Without it, IDE autocomplete (e.g. ``ct.Sym...``)
cannot discover names like ``SymmetricTensor``.
This script derives ``__all__`` from each hub's own relative imports, so it
does not need to be hand-maintained.

Usage::

    python scripts/generate_reexport_all.py             # write updates in place
    python scripts/generate_reexport_all.py --check     # exit non-zero if stale, no writes

Pre-commit runs the write mode so stale ``__all__`` lists are refreshed automatically
(see ``.pre-commit-config.yaml``). CI uses ``--check`` (see ``.github/workflows/linting.yml``
and ``.github/workflows/pytest_numpy.yml``).
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

# Re-export hub modules: everything that re-exports names via `from .x import y`
# (bare, PEP-484-invisible without __all__) for the public cyten API.
HUB_FILES = [
    'cyten/__init__.py',
    'cyten/tensors/__init__.py',
    'cyten/tensors/_tensors.py',
    'cyten/symmetries/__init__.py',
    'cyten/symmetries/_symmetries.py',
    'cyten/symmetries/spaces.py',
    'cyten/symmetries/trees.py',
    'cyten/backends/__init__.py',
    'cyten/block_backends/__init__.py',
    'cyten/models/__init__.py',
    'cyten/testing/__init__.py',
    'cyten/tools/__init__.py',
]


def _is_public(name: str) -> bool:
    if not name.startswith('_'):
        return True
    return name.startswith('__') and name.endswith('__')  # keep dunders, e.g. __version__


def _collect_reexported_names(tree: ast.Module) -> list[str]:
    """Names bound by top-level relative (or same-package) imports, excluding privates."""
    names: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == '__future__':
            continue
        is_relative = node.level > 0
        is_own_package = node.module is not None and node.module.split('.')[0] == 'cyten'
        if not (is_relative or is_own_package):
            continue
        for alias in node.names:
            if alias.name == '*':
                continue
            name = alias.asname or alias.name
            if _is_public(name):
                names.add(name)
    return sorted(names)


_ALL_COMMENT = '# auto-maintained by scripts/generate_reexport_all.py; do not edit by hand'


def _format_all_block(names: list[str]) -> str:
    lines = [_ALL_COMMENT, '__all__ = [']
    lines += [f"    '{name}'," for name in names]
    lines.append(']')
    return '\n'.join(lines) + '\n'


def _existing_all_node(tree: ast.Module) -> ast.Assign | None:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == '__all__'
        ):
            return node
    return None


def _last_import_end_line(tree: ast.Module) -> int:
    import_nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return max((n.end_lineno for n in import_nodes), default=0)


def _updated_source(path: Path) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    names = _collect_reexported_names(tree)
    block = _format_all_block(names)
    lines = source.splitlines(keepends=True)

    existing = _existing_all_node(tree)
    if existing is not None:
        start, end = existing.lineno - 1, existing.end_lineno
        if start > 0 and lines[start - 1].strip() == _ALL_COMMENT:
            start -= 1
        return ''.join(lines[:start]) + block + ''.join(lines[end:])

    insert_at = _last_import_end_line(tree)
    return ''.join(lines[:insert_at]) + '\n\n' + block + ''.join(lines[insert_at:])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--check', action='store_true', help='only report drift (exit non-zero); do not write'
    )
    args = ap.parse_args()

    stale: list[str] = []
    for rel in HUB_FILES:
        path = _REPO / rel
        updated = _updated_source(path)
        if updated == path.read_text():
            continue
        if args.check:
            stale.append(rel)
        else:
            path.write_text(updated)
            print(f'updated {rel}')

    if args.check and stale:
        print('__all__ lists are stale in:', file=sys.stderr)
        for rel in stale:
            print(f'  {rel}', file=sys.stderr)
        print('Run `python scripts/generate_reexport_all.py` to fix.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
