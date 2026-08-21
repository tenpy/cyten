#!/usr/bin/env python3
"""Generate pybind ``DOC()`` headers from Doxygen XML + source ``///`` comments.

Doxygen discovers symbols (names, overload order, namespaces). Comment bodies are
read from the C++ header (Doxygen ``@param`` / ``@returns`` + markdown) and
converted to NumPy-style sections for Sphinx Napoleon / Python ``__doc__``.
A ``See Also`` entry with ``:cpp:func:`` / ``:cpp:class:`` is appended so Python
docs link to the matching Breathe page.

Example::

    doxygen <scoped-Doxyfile>
    python scripts/doxygen_xml_to_docstrings.py \\
        --xml-dir build/docstrings_xml/xml \\
        --header-rel tensors/ops_algebra.h \\
        --source-root . \\
        -o pybind/docstrings/tensors/ops_algebra.h
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# @param[in] name1,name2 description…  /  @returns description…
_CMD_RE = re.compile(
    r'^@(?P<cmd>param|paramref|returns?|return|brief)\s*'
    r'(?:\[(?P<dir>[^\]]*)\]\s*)?'
    r'(?P<rest>.*)$'
)

_DOC_PREAMBLE = """\
#ifndef CYTEN_MKDOC_DOC_MACROS
#define CYTEN_MKDOC_DOC_MACROS
#define MKD_EXPAND(x)                                      x
#define MKD_COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define MKD_VA_SIZE(...)                                   MKD_EXPAND(MKD_COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1, 0))
#define MKD_CAT1(a, b)                                     a##b
#define MKD_CAT2(a, b)                                     MKD_CAT1(a, b)
#define MKD_DOC1(n1)                                       mkd_doc_##n1
#define MKD_DOC2(n1, n2)                                   mkd_doc_##n1##_##n2
#define MKD_DOC3(n1, n2, n3)                               mkd_doc_##n1##_##n2##_##n3
#define MKD_DOC4(n1, n2, n3, n4)                           mkd_doc_##n1##_##n2##_##n3##_##n4
#define MKD_DOC5(n1, n2, n3, n4, n5)                       mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5
#define MKD_DOC6(n1, n2, n3, n4, n5, n6)                   mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define MKD_DOC7(n1, n2, n3, n4, n5, n6, n7)               mkd_doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                           MKD_EXPAND(MKD_EXPAND(MKD_CAT2(MKD_DOC, MKD_VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))
#endif /* CYTEN_MKDOC_DOC_MACROS */

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

"""

_BANNER = """\
/*
  Generated from Doxygen XML + include/cyten/{source_rel} /// comments.
  Not committed — produced at build time (requires doxygen).

  cmake --build <build-dir> --target cyten_generate_docstrings
 */

"""

# Kinds we emit as DOC() entries (not typedefs / variables / enums alone).
_EMIT_KINDS = frozenset({'function', 'class', 'struct', 'enum'})


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    namespace_parts: tuple[str, ...]
    line: int
    file: str
    has_docs: bool
    # Comma-separated parameter *types* (no names/defaults), for overload links.
    param_types: str = ''


def _local_tag(tag: str | None) -> str:
    if tag is None:
        return ''
    if '}' in tag:
        return tag.rsplit('}', 1)[-1]
    return tag


def _text_nonempty(elem: ET.Element | None) -> bool:
    if elem is None:
        return False
    return bool(''.join(elem.itertext()).strip())


def _compound_namespace(compound: ET.Element) -> tuple[str, ...]:
    kind = compound.get('kind', '')
    name = (compound.findtext('compoundname') or '').strip()
    if kind == 'namespace' and name:
        return tuple(p for p in name.split('::') if p)
    if kind in ('class', 'struct') and name:
        return tuple(p for p in name.split('::') if p)
    return ()


def _file_matches(location_file: str, header_rel: str) -> bool:
    loc = location_file.replace('\\', '/')
    rel = header_rel.replace('\\', '/')
    return loc == rel or loc.endswith('/' + rel) or loc.endswith(rel)


def _param_types(member: ET.Element) -> str:
    """Return ``Type1, Type2, …`` from a ``memberdef`` (no names/defaults)."""
    types: list[str] = []
    for child in member:
        if _local_tag(child.tag) != 'param':
            continue
        type_el = None
        for sub in child:
            if _local_tag(sub.tag) == 'type':
                type_el = sub
                break
        if type_el is None:
            continue
        t = re.sub(r'\s+', ' ', ''.join(type_el.itertext()).strip())
        if t:
            types.append(t)
    return ', '.join(types)


def collect_symbols(xml_dir: Path, header_rel: str) -> list[Symbol]:
    symbols: list[Symbol] = []
    for path in sorted(xml_dir.glob('*.xml')):
        if path.name in {'index.xml', 'Doxyfile.xml'} or path.name.endswith('.xsd'):
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for compound in root.iter():
            if _local_tag(compound.tag) != 'compounddef':
                continue
            ns_parts = _compound_namespace(compound)
            compound_kind = compound.get('kind', '')
            # Class/struct documented on the compound itself.
            if compound_kind in ('class', 'struct'):
                loc = None
                for child in compound:
                    if _local_tag(child.tag) == 'location':
                        loc = child
                        break
                if loc is not None and _file_matches(loc.get('file', ''), header_rel):
                    brief = None
                    detailed = None
                    for child in compound:
                        t = _local_tag(child.tag)
                        if t == 'briefdescription':
                            brief = child
                        elif t == 'detaileddescription':
                            detailed = child
                    qname = ns_parts
                    if qname:
                        symbols.append(
                            Symbol(
                                kind=compound_kind,
                                name=qname[-1],
                                namespace_parts=qname[:-1],
                                line=int(loc.get('line', '0') or 0),
                                file=loc.get('file', ''),
                                has_docs=_text_nonempty(brief) or _text_nonempty(detailed),
                            )
                        )
            for md in compound.iter():
                if _local_tag(md.tag) != 'memberdef':
                    continue
                kind = md.get('kind', '')
                if kind not in _EMIT_KINDS:
                    continue
                loc = None
                name = None
                brief = None
                detailed = None
                for child in md:
                    t = _local_tag(child.tag)
                    if t == 'location':
                        loc = child
                    elif t == 'name':
                        name = (child.text or '').strip()
                    elif t == 'briefdescription':
                        brief = child
                    elif t == 'detaileddescription':
                        detailed = child
                if loc is None or not name:
                    continue
                if not _file_matches(loc.get('file', ''), header_rel):
                    continue
                member_ns = ns_parts
                # Nested class members: compound is the class; keep full path.
                if compound_kind in ('class', 'struct') and ns_parts:
                    # methods: DOC(ns, Class, method) → namespace_parts = ns + Class
                    pass
                symbols.append(
                    Symbol(
                        kind=kind,
                        name=name,
                        namespace_parts=member_ns,
                        line=int(loc.get('line', '0') or 0),
                        file=loc.get('file', ''),
                        has_docs=_text_nonempty(brief) or _text_nonempty(detailed),
                        param_types=_param_types(md) if kind == 'function' else '',
                    )
                )
    # Dedupe identical (kind, path, name, line)
    uniq: dict[tuple, Symbol] = {}
    for s in symbols:
        key = (s.kind, s.namespace_parts, s.name, s.line)
        uniq[key] = s
    return sorted(uniq.values(), key=lambda s: (s.line, s.name))


def extract_slash_comment(source_lines: list[str], decl_line: int) -> str | None:
    """Return the ``///`` block immediately above 1-based ``decl_line``, or None."""
    if decl_line < 1 or decl_line > len(source_lines):
        return None
    # Skip attribute / empty lines directly above the declaration.
    i = decl_line - 2  # 0-based index of line above decl
    while i >= 0:
        stripped = source_lines[i].strip()
        if stripped == '' or stripped.startswith('[['):
            i -= 1
            continue
        break
    if i < 0 or not source_lines[i].lstrip().startswith('///'):
        return None
    block: list[str] = []
    while i >= 0 and source_lines[i].lstrip().startswith('///'):
        line = source_lines[i].lstrip()
        if line.startswith('/// '):
            block.append(line[4:])
        elif line.startswith('///'):
            block.append(line[3:])
        else:
            break
        i -= 1
    block.reverse()
    # Trim trailing blank comment lines
    while block and block[-1].strip() == '':
        block.pop()
    text = '\n'.join(block).rstrip() + '\n'
    return text if text.strip() else None


def _split_param_rest(rest: str) -> tuple[str, str]:
    """Split ``name[,name…] description`` after ``@param``."""
    rest = rest.strip()
    if not rest:
        return '', ''
    # First token is the name list (may contain commas, no spaces inside names).
    parts = rest.split(None, 1)
    names = parts[0]
    desc = parts[1] if len(parts) > 1 else ''
    return names, desc


def doxygen_comment_to_numpy(doc: str) -> str:
    """Convert Doxygen ``@param`` / ``@returns`` (+ markdown) to NumPy sections.

    Prose before the first command is kept as the summary/body. Markdown
    backticks are upgraded to reST double-backticks for Napoleon/Sphinx.
    """
    prose: list[str] = []
    params: list[tuple[str, list[str]]] = []
    returns: list[str] = []
    mode: str | None = None  # None | 'param' | 'returns'
    cur_names = ''
    cur_lines: list[str] = []

    def flush() -> None:
        nonlocal cur_names, cur_lines, mode
        if mode == 'param' and cur_names:
            # ``a,b`` → ``a, b`` for NumPy readability
            display = ', '.join(p.strip() for p in cur_names.split(',') if p.strip())
            params.append((display, list(cur_lines)))
        elif mode == 'returns' and cur_lines:
            returns.extend(cur_lines)
        cur_names = ''
        cur_lines = []
        mode = None

    for raw in doc.splitlines():
        # Normalize backslash commands (\param → @param, \returns → @returns).
        line = raw
        for old, new in (
            ('\\paramref', '@paramref'),
            ('\\param', '@param'),
            ('\\returns', '@returns'),
            ('\\return', '@return'),
            ('\\brief', '@brief'),
        ):
            if line.lstrip().startswith(old):
                line = line.replace(old, new, 1)
                break
        stripped = line.strip()
        m = _CMD_RE.match(stripped) if stripped.startswith('@') else None
        if m:
            cmd = m.group('cmd')
            rest = m.group('rest') or ''
            if cmd in ('param', 'paramref'):
                flush()
                names, desc = _split_param_rest(rest)
                mode = 'param'
                cur_names = names
                cur_lines = [desc] if desc else []
            elif cmd in ('return', 'returns'):
                flush()
                mode = 'returns'
                cur_lines = [rest] if rest else []
            elif cmd == 'brief':
                flush()
                if rest:
                    prose.append(rest)
                mode = None
            continue

        if mode == 'param':
            if stripped == '':
                if cur_lines:
                    cur_lines.append('')
            else:
                cur_lines.append(stripped)
            continue
        if mode == 'returns':
            if stripped == '':
                if cur_lines:
                    cur_lines.append('')
            else:
                cur_lines.append(stripped)
            continue

        prose.append(raw)

    flush()

    # Drop trailing blanks in prose
    while prose and prose[-1].strip() == '':
        prose.pop()

    out: list[str] = []
    if prose:
        out.extend(prose)
    if params:
        if out:
            out.append('')
        out.append('Parameters')
        out.append('----------')
        for names, desc_lines in params:
            # Trim trailing empty continuation lines
            while desc_lines and desc_lines[-1].strip() == '':
                desc_lines.pop()
            if not desc_lines:
                out.append(names)
                continue
            first, *rest = desc_lines
            out.append(f'{names}')
            if first:
                out.append(f'    {first}')
            for cont in rest:
                if cont.strip() == '':
                    out.append('')
                else:
                    out.append(f'    {cont}')
    if returns:
        while returns and returns[-1].strip() == '':
            returns.pop()
        if out:
            out.append('')
        out.append('Returns')
        out.append('-------')
        for line in returns:
            out.append(line if line.strip() == '' else line)

    text = '\n'.join(out).rstrip() + '\n'
    # Markdown `code` → reST ``code`` (avoid touching already-doubled ticks).
    text = re.sub(r'(?<!`)`([^`]+)`(?!`)', r'``\1``', text)
    return text


def _cpp_see_also(sym: Symbol, *, has_overloads: bool) -> str:
    """NumPy ``See Also`` linking to the Breathe / Sphinx C++ domain target."""
    qname = '::'.join([*sym.namespace_parts, sym.name])
    if sym.kind in ('class', 'struct'):
        role = 'class'
        target = qname
    elif sym.kind == 'enum':
        role = 'enum'
        target = qname
    elif sym.kind == 'function':
        role = 'func'
        if has_overloads:
            target = f'{qname}({sym.param_types})'
        else:
            target = f'{qname}()'
    else:
        return ''
    return f'\nSee Also\n--------\n:cpp:{role}:`{target}`\n'


def _macro_name(parts: list[str], overload_suffix: int | None) -> str:
    base = '_'.join(parts)
    if overload_suffix is None:
        return f'mkd_doc_{base}'
    return f'mkd_doc_{base}_{overload_suffix}'


def _escape_raw_string(doc: str) -> str:
    # Avoid terminating R"doc( ... )doc"
    if ')doc"' in doc:
        raise SystemExit('docstring contains )doc" which breaks R"doc(...)doc"')
    return doc


def format_entry(macro: str, doc: str) -> str:
    doc = _escape_raw_string(doc)
    if '\n' not in doc.rstrip('\n'):
        # Single-line: keep compact like mkdoc for short briefs
        one = doc.rstrip('\n')
        return f'static const char* {macro} =\n  R"doc({one})doc";\n'
    # Multi-line: content starts on same line as R"doc(
    body = doc if doc.endswith('\n') else doc + '\n'
    return f'static const char* {macro} =\n  R"doc({body})doc";\n'


def generate(xml_dir: Path, header_rel: str, source_root: Path) -> str:
    source_path = source_root / 'include' / 'cyten' / header_rel
    if not source_path.is_file():
        raise SystemExit(f'source header not found: {source_path}')
    source_lines = source_path.read_text(encoding='utf-8').splitlines()

    symbols = collect_symbols(xml_dir, header_rel)
    # Prefer symbols that have doxygen docs OR a /// block we can extract.
    entries: list[tuple[list[str], int | None, str]] = []
    # Group by qualified path for overload suffixes
    by_key: dict[tuple[str, ...], list[Symbol]] = defaultdict(list)
    for s in symbols:
        key = (*s.namespace_parts, s.name)
        by_key[key].append(s)

    # Emit in first-seen (line) order across all keys
    ordered: list[tuple[tuple[str, ...], Symbol, int | None, bool]] = []
    for key, group in by_key.items():
        group = sorted(group, key=lambda s: s.line)
        has_overloads = len(group) > 1
        for idx, s in enumerate(group):
            suffix = None if idx == 0 else idx + 1  # 2, 3, ...
            ordered.append((key, s, suffix, has_overloads))
    ordered.sort(key=lambda t: t[1].line)

    for key, sym, suffix, has_overloads in ordered:
        doc = extract_slash_comment(source_lines, sym.line)
        if doc is None:
            continue
        doc = doxygen_comment_to_numpy(doc)
        see = _cpp_see_also(sym, has_overloads=has_overloads)
        if see:
            doc = doc.rstrip() + '\n' + see
            if not doc.endswith('\n'):
                doc += '\n'
        parts = list(key)
        entries.append((parts, suffix, doc))

    if not entries:
        raise SystemExit(f'no documented symbols found for {header_rel} in {xml_dir}')

    out: list[str] = [_BANNER.format(source_rel=header_rel), _DOC_PREAMBLE]
    for parts, suffix, doc in entries:
        out.append(format_entry(_macro_name(parts, suffix), doc))
        out.append('\n')
    out.append('#if defined(__GNUG__)\n#pragma GCC diagnostic pop\n#endif\n')
    return ''.join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml-dir', type=Path, required=True, help='Doxygen XML output directory')
    ap.add_argument(
        '--header-rel',
        required=True,
        help='Path relative to include/cyten/ (e.g. tensors/ops_algebra.h)',
    )
    ap.add_argument(
        '--source-root',
        type=Path,
        default=Path('.'),
        help='Repository root containing include/cyten/',
    )
    ap.add_argument('-o', '--output', type=Path, required=True, help='Output docstring header')
    args = ap.parse_args()
    if not args.xml_dir.is_dir():
        raise SystemExit(f'XML dir not found: {args.xml_dir}')
    text = generate(args.xml_dir, args.header_rel, args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding='utf-8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
