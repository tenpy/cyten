#!/usr/bin/env python3
"""Check that R\"pydoc(...)pydoc\" docstring lines are aligned with the indentation in the start of the line.

Takes a C++ source filename and fixes the base docstring indentation (measured by the second line of the string)
too match the indentation of the first line of the string.
"""
# Copyright (C) TeNPy Developers, Apache license

import sys


def find_pydoc_blocks(content):
    """Find all R\"pydoc( ... )pydoc\" blocks; yield (start_idx, end_idx, start_line_no)."""
    pattern_start = 'R"pydoc('
    pattern_end = ')pydoc"'
    line = 0
    while line < len(content):
        if pattern_start not in content[line]:
            line += 1
            continue
        start = line
        for i, line_i in enumerate(content[line:]):
            if pattern_end in line_i:
                line += i
                if i == 0:
                    continue  # skip: pydoc ends on the same line
                yield start, line
                break
        else:
            raise ValueError(f'No closing )pydoc" found for R"pydoc( on line {line}')


def get_indent_level(line):
    return len(line) - len(line.lstrip(' '))


def fix_docstring_alignment(content):
    updated = False
    for start, end in find_pydoc_blocks(content):
        assert start < end
        indent_level_start = get_indent_level(content[start])
        if not content[start].lstrip().startswith('R"pydoc('):
            indent_level_start += 4
        indent_level_next = get_indent_level(content[start + 1])
        if indent_level_next == 0:
            continue  # missing reference line: cannot fix
        change_indent_level = indent_level_start - indent_level_next
        if change_indent_level == 0:
            continue  # nothing to do
        for line in range(start + 1, end + 1):
            if content[line].strip():  # non-empty line
                new_indent_level = get_indent_level(content[line]) + change_indent_level
                if new_indent_level < 0:
                    raise ValueError(f'Line {line} has less indentation than the start of the R"pydoc( string')
                content[line] = ' ' * new_indent_level + content[line].lstrip()
                updated = True
    return updated


def main():
    if len(sys.argv) != 2:
        raise ValueError('Usage: cpp_docstrings.py <cpp_file>')
    path = sys.argv[1]
    with open(path, 'r') as f:
        content = f.readlines()
    if fix_docstring_alignment(content):
        print(f'Fixed docstring alignment in {path}')
        with open(path, 'w') as f:
            f.writelines(content)
        raise SystemExit(1)
    print(f'No docstring alignment issues found in {path}')
    raise SystemExit(0)


if __name__ == '__main__':
    main()
