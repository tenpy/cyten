#!/usr/bin/env python3
"""Fix NumPy section underlines mangled by pybind11_mkdoc reflow.

Stock mkdoc joins ``Parameters`` / ``----------`` into one line. Napoleon only
recognizes the tight NumPy layout, so rewrite common section headers in a
generated ``docstrings.h``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# "Parameters ---------- rest" -> "Parameters\n----------\nrest"
_SECTION_RE = re.compile(
    r'^(Parameters|Returns|Raises|See Also|Notes|Warnings|Attributes|Methods)'
    r'[ \t]+(-{3,}|={3,})[ \t]*',
    re.MULTILINE,
)


def fix_numpy_sections(text: str) -> str:
    return _SECTION_RE.sub(r'\1\n\2\n', text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input', type=Path, help='Generated docstrings.h from mkdoc')
    ap.add_argument('-o', '--output', type=Path, required=True, help='Fixed output path')
    args = ap.parse_args()
    text = args.input.read_text(encoding='utf-8')
    args.output.write_text(fix_numpy_sections(text), encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
