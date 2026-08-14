#!/usr/bin/env python3
"""Audit and recover Python docstrings/comments into C++ / pybind for layers 2–3.

Reads ``scripts/py_cpp_comment_map.yaml`` and original sources under ``tmp/orig_cyten/``.

Usage:
  python scripts/recover_py_comments.py report
  python scripts/recover_py_comments.py apply-docstrings [--dry-run]
  python scripts/recover_py_comments.py apply-hints [--dry-run]
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import tokenize
from collections import defaultdict
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit('PyYAML is required') from e

# Python method name → C++ method name when renamed during conversion.
METHOD_NAME_ALIASES: dict[str, list[str]] = {
    '_valid_block_inds': ['valid_block_inds'],
    '_calc_basis_perm': ['calc_basis_perm'],
    '_calc_sectors': ['calc_sectors', 'prepare'],
    '_get_fusion_outcomes_perm': ['get_fusion_outcomes_perm', 'fusion_outcomes_perm'],
    '__repr__': ['repr', 'operator<<'],
    '__str__': ['str'],
    '__eq__': ['operator=='],
    '__init__': [],  # handled specially as constructors
}

REPO = Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO / 'scripts' / 'py_cpp_comment_map.yaml'
DEFAULT_ORIG = REPO / 'tmp' / 'orig_cyten'
DEFAULT_REPORT = REPO / 'docs' / 'cpp_conversion' / 'comment_recovery_report.md'

SEPARATOR_RE = re.compile(r'^#?\s*[-=*]{3,}\s*$')
HASH_BANG_ONLY = re.compile(r'^#+\s*$')
PYDOC_START = 'R"pydoc('
PYDOC_END = ')pydoc"'


def normalize_text(s: str) -> str:
    """Collapse whitespace for fuzzy presence checks."""
    s = s.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s)
    return s.strip().lower()


def distinctive_snippet(comment: str, max_len: int = 60) -> str:
    """Pick a searchable substring from a comment (drop leading #/OPTIMIZE tags)."""
    t = comment.lstrip('#').strip()
    t = re.sub(r'^(OPTIMIZE|FIXME|TODO|NOTE|HACK)\s*[:?]?\s*', '', t, flags=re.I)
    t = re.sub(r'\s+', ' ', t).strip()
    if len(t) > max_len:
        t = t[:max_len]
    return t


def is_noise_comment(text: str) -> bool:
    body = text.lstrip('#').strip()
    if not body:
        return True
    if SEPARATOR_RE.match(text) or HASH_BANG_ONLY.match(text):
        return True
    if body.startswith('type:'):
        return True
    # Tiny label fragments left over from diagrams / math scribbles
    if len(body) <= 8 and not re.search(r'OPTIMIZE|FIXME|TODO|NOTE\b|HACK', body, re.I):
        return True
    # Commented-out code: starts like an assignment/call with no spaces of prose
    if re.match(r'^[a-zA-Z_][\w.]*\s*[=(]', body) and ' ' not in body.split('=')[0]:
        # still keep if it has OPTIMIZE/TODO embedded
        if not re.search(r'OPTIMIZE|FIXME|TODO|NOTE\b', body, re.I):
            return True
    return False


@dataclass
class CommentItem:
    lineno: int
    text: str  # including leading '#'
    relative: float = 0.0  # 0..1 within function body


@dataclass
class SymbolInfo:
    qualname: str  # Class or Class.method or function
    kind: str  # class|method|function|property
    lineno: int
    end_lineno: int
    docstring: str | None = None
    comments: list[CommentItem] = field(default_factory=list)
    bind_name: str | None = None  # Python name used in pybind .def("...")
    class_name: str | None = None


@dataclass
class ModuleMap:
    py: str
    cpp: list[str]
    pybind: list[str]


def load_map(path: Path) -> list[ModuleMap]:
    data = yaml.safe_load(path.read_text())
    out = []
    for m in data['modules']:
        out.append(
            ModuleMap(
                py=m['py'],
                cpp=list(m.get('cpp') or []),
                pybind=list(m.get('pybind') or []),
            )
        )
    return out


def collect_comments_by_line(source: str) -> dict[int, list[str]]:
    """Map lineno -> list of full comment strings on that line (tokenize)."""
    by_line: dict[int, list[str]] = defaultdict(list)
    try:
        tokens = tokenize.generate_tokens(StringIO(source).readline)
        for tok in tokens:
            if tok.type == tokenize.COMMENT:
                by_line[tok.start[0]].append(tok.string)
    except tokenize.TokenError:
        pass
    return by_line


class SymbolCollector(ast.NodeVisitor):
    def __init__(self, comments_by_line: dict[int, list[str]]):
        self.comments_by_line = comments_by_line
        self.symbols: list[SymbolInfo] = []
        self._class_stack: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        doc = ast.get_docstring(node)
        info = SymbolInfo(
            qualname=node.name,
            kind='class',
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            docstring=doc,
            comments=[],
            bind_name=node.name,
            class_name=node.name,
        )
        self.symbols.append(info)
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add_function(node)
        # Do not recurse into nested functions.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add_function(node)

    def _add_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        class_name = self._class_stack[-1] if self._class_stack else None
        if class_name:
            qualname = f'{class_name}.{node.name}'
            kind = 'method'
        else:
            qualname = node.name
            kind = 'function'
        for dec in node.decorator_list:
            dec_name = ''
            if isinstance(dec, ast.Name):
                dec_name = dec.id
            elif isinstance(dec, ast.Attribute):
                dec_name = dec.attr
            if dec_name in ('property', 'cached_property'):
                kind = 'property'
        doc = ast.get_docstring(node)
        end = node.end_lineno or node.lineno
        comments = self._comments_in_span(node.lineno, end)
        span = max(end - node.lineno, 1)
        for c in comments:
            c.relative = (c.lineno - node.lineno) / span
        self.symbols.append(
            SymbolInfo(
                qualname=qualname,
                kind=kind,
                lineno=node.lineno,
                end_lineno=end,
                docstring=doc,
                comments=comments,
                bind_name=node.name,
                class_name=class_name,
            )
        )

    def _comments_in_span(self, start: int, end: int) -> list[CommentItem]:
        items = []
        for lineno in range(start, end + 1):
            for text in self.comments_by_line.get(lineno, []):
                if is_noise_comment(text):
                    continue
                items.append(CommentItem(lineno=lineno, text=text))
        return items


def parse_python_module(path: Path) -> list[SymbolInfo]:
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(path))
    comments_by_line = collect_comments_by_line(source)
    collector = SymbolCollector(comments_by_line)
    collector.visit(tree)
    # Deduplicate comments claimed by both class and methods: methods keep their span;
    # strip method-span comments from class-level list.
    method_lines = set()
    for s in collector.symbols:
        if s.kind in ('method', 'property', 'function'):
            for c in s.comments:
                method_lines.add(c.lineno)
    for s in collector.symbols:
        if s.kind == 'class':
            s.comments = [c for c in s.comments if c.lineno not in method_lines]
    return collector.symbols


def read_files(paths: list[str]) -> dict[str, str]:
    out = {}
    for p in paths:
        fp = REPO / p
        if fp.is_file():
            out[p] = fp.read_text(encoding='utf-8')
        else:
            out[p] = ''
    return out


def combined_text(files: dict[str, str]) -> str:
    return '\n'.join(files.values())


def extract_pydoc_blocks(text: str) -> list[str]:
    blocks = []
    start = 0
    while True:
        i = text.find(PYDOC_START, start)
        if i < 0:
            break
        j = text.find(PYDOC_END, i + len(PYDOC_START))
        if j < 0:
            break
        blocks.append(text[i + len(PYDOC_START) : j])
        start = j + len(PYDOC_END)
    return blocks


def docstring_present(doc: str, pybind_text: str) -> bool:
    if not doc or not doc.strip():
        return True
    # Check against all pydoc blocks and raw text
    norm = normalize_text(doc)
    # Use first meaningful line(s) — full match is often too strict due to indent
    first_lines = [ln.strip() for ln in doc.strip().splitlines() if ln.strip()]
    if not first_lines:
        return True
    # Prefer checking first non-empty line + maybe second
    probes = [first_lines[0]]
    if len(first_lines) > 1 and len(first_lines[0]) < 40:
        probes.append(first_lines[0] + ' ' + first_lines[1])
    pybind_norm = normalize_text(pybind_text)
    for probe in probes:
        if normalize_text(probe) in pybind_norm:
            return True
    # Also try full normalized containment for short docs
    if len(norm) < 200 and norm in pybind_norm:
        return True
    return False


def comment_present(comment: str, cpp_text: str) -> bool:
    snippet = distinctive_snippet(comment)
    raw = comment.lstrip('#').strip()
    if len(snippet) < 12:
        # Short comments: require exact-ish presence of the raw phrase
        if len(raw) < 8:
            # very short ("blocks", "OPTIMIZE?") — too ambiguous; treat as present
            # only if the full `// raw` or `# raw` form exists
            return f'// {raw}' in cpp_text or f'// {raw.lower()}' in cpp_text.lower()
        snippet = raw
    return normalize_text(snippet) in normalize_text(cpp_text)


def find_def_sites(pybind_text: str, name: str) -> list[tuple[int, int, bool]]:
    """Return list of (start_idx, end_idx_of_stmt, has_pydoc) for .def/.def_static/etc.

    end_idx points at the closing paren of the .def( call (approx).
    """
    patterns = [
        rf'\.def\(\s*"{re.escape(name)}"\s*,',
        rf'\.def_static\(\s*"{re.escape(name)}"\s*,',
        rf'\.def_property_readonly\(\s*"{re.escape(name)}"\s*,',
        rf'\.def_property\(\s*"{re.escape(name)}"\s*,',
        rf'\.def_prop_ro\(\s*"{re.escape(name)}"\s*,',
        rf'\.def_prop_rw\(\s*"{re.escape(name)}"\s*,',
        rf'\.def_readonly\(\s*"{re.escape(name)}"\s*,',
        rf'\.def_readwrite\(\s*"{re.escape(name)}"\s*,',
    ]
    sites = []
    for pat in patterns:
        for m in re.finditer(pat, pybind_text):
            start = m.start()
            # scan forward for matching paren of .def(
            # find the '(' right after .def
            open_paren = pybind_text.find('(', start)
            depth = 0
            i = open_paren
            in_str = None
            has_pydoc = False
            while i < len(pybind_text):
                ch = pybind_text[i]
                # detect R"pydoc
                if pybind_text.startswith(PYDOC_START, i):
                    has_pydoc = True
                    end_pd = pybind_text.find(PYDOC_END, i)
                    if end_pd < 0:
                        break
                    i = end_pd + len(PYDOC_END)
                    continue
                if in_str:
                    if ch == '\\' and i + 1 < len(pybind_text):
                        i += 2
                        continue
                    if ch == in_str:
                        in_str = None
                    i += 1
                    continue
                if ch in ('"', "'"):
                    # raw/ordinary string — but R"pydoc handled above
                    in_str = ch
                    i += 1
                    continue
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        sites.append((start, i, has_pydoc))
                        break
                i += 1
    return sites


def find_class_binding(pybind_text: str, class_name: str) -> list[tuple[int, int, str, bool]]:
    """Find py::class_ bindings for class_name.

    Returns list of (start_idx, end_of_ctor_call_idx, var_name, has_doc).
    Supports both::

        py::class_<Foo>(m, "Foo")
        py::class_<Foo> var(m, "Foo");
    """
    results = []
    # Allow optional C++ variable name between > and (
    pat = (
        rf'py::class_<[^>]*\b{re.escape(class_name)}\b[^>]*>\s*'
        rf'(?:(?P<var1>\w+)\s*)?\(\s*m\s*,\s*"{re.escape(class_name)}"'
    )
    for m in re.finditer(pat, pybind_text):
        var = m.group('var1') or 'cls'
        # find end of the (m, "Name", ...) call that starts at m's '('
        open_paren = pybind_text.find('(', m.start())
        depth = 0
        i = open_paren
        while i < len(pybind_text):
            if pybind_text[i] == '(':
                depth += 1
            elif pybind_text[i] == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        else:
            continue
        # Look for .doc() = after ctor, before next py::class_
        next_class = pybind_text.find('py::class_<', i + 1)
        window_end = next_class if next_class >= 0 else min(len(pybind_text), i + 2500)
        window = pybind_text[m.start() : window_end]
        has_doc = f'{var}.doc()' in window and PYDOC_START in window
        # Also docstring as third ctor arg
        ctor = pybind_text[open_paren : i + 1]
        if PYDOC_START in ctor:
            has_doc = True
        results.append((m.start(), i, var, has_doc))
    return results


def find_method_def_for_class(pybind_text: str, class_name: str, method_name: str) -> list[tuple[int, int, bool]]:
    """Find .def("method"...) sites that clearly belong to class_name via &Class::method."""
    sites = []
    # Require &ClassName::method somewhere in the .def(...) call
    for start, end, has_pydoc in find_def_sites(pybind_text, method_name):
        chunk = pybind_text[start : end + 1]
        if re.search(rf'&{re.escape(class_name)}\s*::\s*{re.escape(method_name)}\b', chunk):
            sites.append((start, end, has_pydoc))
            continue
        # Lambdas / free wrappers: still OK if within class binding region that uses this class
        # Heuristic: previous py::class_ mentioning ClassName and no intervening other class_
        prev = pybind_text.rfind('py::class_<', 0, start)
        if prev < 0:
            continue
        region = pybind_text[prev:start]
        if f'"{class_name}"' not in region and not re.search(rf'py::class_<\s*{re.escape(class_name)}\s*[,>]', region):
            continue
        # ensure no other class_ between
        if region.count('py::class_<') != 1:
            continue
        # accept readwrite/property without &Class:: as well
        sites.append((start, end, has_pydoc))
    return sites


def format_pydoc(doc: str, indent: str) -> str:
    """Format docstring as R\"pydoc block with given indent for the R\" line."""
    lines = doc.strip('\n').splitlines()
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty:
        common = min(len(ln) - len(ln.lstrip(' ')) for ln in non_empty)
        lines = [ln[common:] if len(ln) >= common else ln for ln in lines]
    body = '\n'.join(indent + (ln if ln.strip() else '') for ln in lines)
    return f'{indent}R"pydoc(\n{body}\n{indent})pydoc"'


def apply_docstring_to_pybind(
    path: str,
    text: str,
    bind_name: str,
    doc: str,
    is_class: bool,
    class_name: str | None = None,
) -> tuple[str, bool]:
    """Insert docstring into pybind file text. Returns (new_text, changed)."""
    if not doc or not doc.strip():
        return text, False

    if is_class:
        bindings = find_class_binding(text, bind_name)
        if not bindings:
            return text, False
        idx, end_ctor, var_name, has_doc = bindings[0]
        if has_doc:
            return text, False
        # If this is a chained temporary `py::class_<T>(m, "T").def...`
        # without a variable, rewrite to a named binding first.
        line_start = text.rfind('\n', 0, idx) + 1
        header = text[line_start : end_ctor + 1]
        needs_named = f'> {var_name}(' not in text[max(0, idx - 80) : end_ctor + 1] and (
            re.search(rf'py::class_<[^>]*>\s*\(\s*m\s*,\s*"{re.escape(bind_name)}"', header) is not None
        )
        if needs_named or var_name == 'cls':
            # Choose a stable variable name
            var_name = f'{bind_name[0].lower()}{bind_name[1:]}_cls' if bind_name else 'cls'
            # Avoid keywords / collisions
            if var_name in ('class_cls',):
                var_name = 'cls'
            old_ctor = text[idx : end_ctor + 1]
            # Build named declaration
            # Extract template args from py::class_<...>
            tm = re.match(r'(py::class_<[^>]*>)', old_ctor)
            if not tm:
                return text, False
            named = f'{tm.group(1)} {var_name}(m, "{bind_name}");'
            # Replace the ctor call; if it was followed by chaining `.def`, the
            # caller must use the named var — only safe when next nonws is `;` or newline then `.`
            text = text[:idx] + named + text[end_ctor + 1 :]
            # Recompute insert_at after named decl
            insert_at = idx + len(named)
            j = insert_at
            while j < len(text) and text[j] in ' \t':
                j += 1
            # If chaining `.def` follows immediately, insert `var_name\n` before it later via doc insert
            if j < len(text) and text[j] == '.':
                # Will insert doc then need `var_name` before `.def` — handle below
                pass
        else:
            insert_at = end_ctor + 1
            j = insert_at
            while j < len(text) and text[j] in ' \t':
                j += 1
            if j < len(text) and text[j] == ';':
                insert_at = j + 1

        next_nl = text.find('\n', insert_at)
        indent = '    '
        if next_nl >= 0:
            rest = text[next_nl + 1 :]
            m_ind = re.match(r'([ \t]*)\S', rest)
            if m_ind:
                indent = m_ind.group(1) or '    '
        block = format_pydoc(doc, indent)
        insertion = f'\n{indent}{var_name}.doc() = {block[len(indent) :]};\n'
        # If the next token is a chained `.def`, prefix with `var_name`
        k = insert_at + len(insertion)
        while k < len(text) and text[k] in ' \t\n\r':
            k += 1
        new_text = text[:insert_at] + insertion + text[insert_at:]
        # After insertion, if we see `.def` without receiver, add var_name
        k2 = insert_at + len(insertion)
        while k2 < len(new_text) and new_text[k2] in ' \t\n\r':
            k2 += 1
        if new_text.startswith('.def', k2) or new_text.startswith('.def_', k2):
            new_text = new_text[:k2] + f'{var_name}\n{indent}  ' + new_text[k2:]
        return new_text, True

    # Methods / properties / free functions
    owner = class_name
    if owner:
        sites = find_method_def_for_class(text, owner, bind_name)
    else:
        sites = find_def_sites(text, bind_name)
    if not sites:
        return text, False
    for start, end, has_pydoc in sites:
        if has_pydoc:
            continue
        chunk = text[start : end + 1]
        if docstring_present(doc, chunk):
            continue
        line_start = text.rfind('\n', 0, end) + 1
        indent = re.match(r'[ \t]*', text[line_start:end]).group(0) or '      '
        block = format_pydoc(doc, indent)
        insertion = ',\n' + block
        return text[:end] + insertion + text[end:], True
    return text, False


def find_cpp_function_body_start(cpp_text: str, qualname: str) -> int | None:
    """Return index of '{' opening the function body for Class::method or free function."""
    candidates: list[str] = []
    if '.' in qualname:
        class_name, method = qualname.split('.', 1)
        if method == '__init__':
            # constructors: Class::Class(
            candidates.append(rf'\b{re.escape(class_name)}\s*::\s*{re.escape(class_name)}\s*\(')
        else:
            methods = [method] + METHOD_NAME_ALIASES.get(method, [])
            for meth in methods:
                if not meth:
                    continue
                candidates.append(rf'\b{re.escape(class_name)}\s*::\s*{re.escape(meth)}\s*\(')
    else:
        methods = [qualname] + METHOD_NAME_ALIASES.get(qualname, [])
        for meth in methods:
            if meth:
                # Definition at beginning of a line only (not calls like ->name( ).
                candidates.append(rf'(?m)^[ \t]*{re.escape(meth)}\s*\(')

    for pat in candidates:
        for m in re.finditer(pat, cpp_text):
            if m.start() > 0 and cpp_text[m.start() - 1] in '.>':
                continue
            if '.' not in qualname:
                before = cpp_text[: m.start()].rstrip()
                prev_nl = before.rfind('\n')
                prev = before[prev_nl + 1 :] if prev_nl >= 0 else before
                prev_s = prev.strip()
                if prev_s and not (
                    prev_s.startswith('//')
                    or prev_s.startswith('/*')
                    or prev_s.startswith('*')
                    or prev_s.startswith('[[')
                    or prev_s.endswith('>')
                    or re.match(r'^[\w:<>\[\],\s\*&]+$', prev_s)
                ):
                    continue
            rest = cpp_text[m.end() - 1 : m.end() + 8000]
            depth = 0
            i = 0
            while i < len(rest):
                if rest[i] == '(':
                    depth += 1
                elif rest[i] == ')':
                    depth -= 1
                    if depth == 0:
                        after = rest[i + 1 : i + 200]
                        bm = re.search(
                            r'(?:(?:const|override|final|noexcept|volatile|&\s*&?)\s*)*\{',
                            after,
                        )
                        if bm:
                            return m.end() - 1 + i + 1 + bm.end() - 1
                        break
                i += 1
    return None


def make_hint_block(qualname: str, comments: list[CommentItem], indent: str) -> str:
    lines = [f'{indent}// --- hints from Python {qualname} ---']
    seen = set()
    for c in comments:
        t = c.text.lstrip('#').strip()
        # convert to // comment
        key = normalize_text(t)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f'{indent}// {t}')
    lines.append(f'{indent}// ---')
    return '\n'.join(lines) + '\n'


def apply_hints_to_cpp(path: str, text: str, qualname: str, missing_comments: list[CommentItem]) -> tuple[str, bool]:
    if not missing_comments:
        return text, False
    # Skip if hint block already present
    marker = f'hints from Python {qualname}'
    if marker in text:
        return text, False

    brace_idx = find_cpp_function_body_start(text, qualname)
    if brace_idx is None:
        return text, False
    nl = text.find('\n', brace_idx)
    if nl < 0:
        return text, False
    rest = text[nl + 1 :]
    m_ind = re.match(r'([ \t]*)\S', rest)
    indent = m_ind.group(1) if m_ind else '    '
    block = make_hint_block(qualname, missing_comments, indent)
    return text[: nl + 1] + block + text[nl + 1 :], True


@dataclass
class AuditResult:
    module_py: str
    symbol: SymbolInfo
    doc_status: str  # PRESENT|MISSING|N/A|NO_BINDING
    missing_comments: list[CommentItem]
    present_comments: list[CommentItem]
    pybind_files: list[str]
    cpp_files: list[str]


def audit_module(mod: ModuleMap, orig_root: Path) -> list[AuditResult]:
    py_path = orig_root / mod.py
    if not py_path.is_file():
        print(f'WARNING: missing {py_path}', file=sys.stderr)
        return []
    symbols = parse_python_module(py_path)
    pybind_files = read_files(mod.pybind)
    cpp_files = read_files(mod.cpp)
    pybind_text = combined_text(pybind_files)
    cpp_text = combined_text(cpp_files)

    results = []
    for sym in symbols:
        # Docstring status
        if not sym.docstring or not sym.docstring.strip():
            doc_status = 'N/A'
        else:
            bound = False
            if sym.kind == 'class':
                bound = bool(find_class_binding(pybind_text, sym.bind_name or sym.qualname))
            else:
                owner = sym.class_name or ''
                if owner:
                    bound = bool(find_method_def_for_class(pybind_text, owner, sym.bind_name or ''))
                if not bound:
                    # inherited binding on a base class (.def name only)
                    bound = bool(find_def_sites(pybind_text, sym.bind_name or ''))
            if not bound and (sym.bind_name or '').startswith('_'):
                if sym.qualname.split('.')[-1] in cpp_text:
                    doc_status = 'N/A'
                else:
                    doc_status = 'NO_BINDING'
            elif not bound:
                doc_status = 'NO_BINDING'
            elif docstring_present(sym.docstring, pybind_text):
                doc_status = 'PRESENT'
            else:
                # Bound on this class specifically without docstring?
                owner = sym.class_name or (sym.bind_name if sym.kind == 'class' else '')
                if sym.kind == 'class':
                    doc_status = 'MISSING'
                elif owner and find_method_def_for_class(pybind_text, owner, sym.bind_name or ''):
                    # Check if those sites lack pydoc
                    sites = find_method_def_for_class(pybind_text, owner, sym.bind_name or '')
                    if any(not hp for _, _, hp in sites):
                        doc_status = 'MISSING'
                    else:
                        doc_status = 'PRESENT'
                else:
                    # Only inherited base binding — treat as present if base has any doc,
                    # else MISSING_INHERITED (report as MISSING only if no doc at all)
                    if docstring_present(sym.docstring.split('\n')[0], pybind_text):
                        doc_status = 'PRESENT'
                    else:
                        doc_status = 'INHERITED'  # docstring lives on base binding / absent

        missing_c = []
        present_c = []
        for c in sym.comments:
            if comment_present(c.text, cpp_text):
                present_c.append(c)
            else:
                missing_c.append(c)

        results.append(
            AuditResult(
                module_py=mod.py,
                symbol=sym,
                doc_status=doc_status,
                missing_comments=missing_c,
                present_comments=present_c,
                pybind_files=mod.pybind,
                cpp_files=mod.cpp,
            )
        )
    return results


def write_report(results: list[AuditResult], path: Path) -> None:
    by_mod: dict[str, list[AuditResult]] = defaultdict(list)
    for r in results:
        by_mod[r.module_py].append(r)

    lines = [
        '# Comment / docstring recovery report',
        '',
        'Generated by `scripts/recover_py_comments.py`.',
        '',
    ]
    total_doc_missing = sum(1 for r in results if r.doc_status == 'MISSING')
    total_doc_present = sum(1 for r in results if r.doc_status == 'PRESENT')
    total_doc_inherited = sum(1 for r in results if r.doc_status == 'INHERITED')
    total_c_missing = sum(len(r.missing_comments) for r in results)
    total_c_present = sum(len(r.present_comments) for r in results)
    lines += [
        '## Summary',
        '',
        f'- Docstrings PRESENT: **{total_doc_present}**',
        f'- Docstrings MISSING: **{total_doc_missing}**',
        f'- Docstrings INHERITED (bound only on base): **{total_doc_inherited}**',
        f'- Comments PRESENT: **{total_c_present}**',
        f'- Comments MISSING: **{total_c_missing}**',
        '',
    ]

    for mod, items in by_mod.items():
        lines += [f'## `{mod}`', '']
        doc_miss = [r for r in items if r.doc_status == 'MISSING']
        if doc_miss:
            lines += ['### Missing docstrings', '']
            for r in doc_miss:
                first = (r.symbol.docstring or '').strip().splitlines()[0][:80]
                lines.append(f'- `{r.symbol.qualname}` ({r.symbol.kind}, L{r.symbol.lineno}): {first!r}')
                lines.append(f'  - pybind: {", ".join(r.pybind_files)}')
            lines.append('')

        c_miss = [r for r in items if r.missing_comments]
        if c_miss:
            lines += ['### Missing comments', '']
            for r in c_miss:
                lines.append(
                    f'- `{r.symbol.qualname}` ({len(r.missing_comments)} missing / '
                    f'{len(r.present_comments)} present) → cpp: {", ".join(r.cpp_files)}'
                )
                for c in r.missing_comments[:40]:
                    lines.append(f'  - L{c.lineno}: `{c.text[:100]}`')
                if len(r.missing_comments) > 40:
                    lines.append(f'  - … +{len(r.missing_comments) - 40} more')
            lines.append('')

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Wrote {path}')
    print(
        f'Summary: docs missing={total_doc_missing} present={total_doc_present} '
        f'inherited={total_doc_inherited}; '
        f'comments missing={total_c_missing} present={total_c_present}'
    )


def cmd_report(args: argparse.Namespace) -> int:
    mods = load_map(Path(args.map))
    results: list[AuditResult] = []
    for mod in mods:
        results.extend(audit_module(mod, Path(args.orig)))
    write_report(results, Path(args.report))
    return 0


def cmd_apply_docstrings(args: argparse.Namespace) -> int:
    mods = load_map(Path(args.map))
    changed_files: dict[str, str] = {}
    n_applied = 0
    n_skip = 0
    for mod in mods:
        results = audit_module(mod, Path(args.orig))
        file_texts = read_files(mod.pybind)
        for r in results:
            sym = r.symbol
            if not sym.docstring or not sym.docstring.strip():
                continue
            # Apply for MISSING, and also retry PRESENT/INHERITED in case class-owned
            # binding still lacks an attached R"pydoc (idempotent).
            if r.doc_status in ('N/A', 'NO_BINDING'):
                continue
            owners: list[str | None]
            if sym.kind == 'class':
                owners = [sym.bind_name]
            elif sym.class_name:
                owners = [sym.class_name]
                owners.extend(DOC_ALIAS_CLASSES.get(sym.class_name, []))
            else:
                owners = [None]
            applied_any = False
            for owner in owners:
                for p, text in list(file_texts.items()):
                    text = changed_files.get(p, text)
                    new_text, changed = apply_docstring_to_pybind(
                        p,
                        text,
                        sym.bind_name or sym.qualname.split('.')[-1],
                        sym.docstring or '',
                        is_class=(sym.kind == 'class' and owner == sym.bind_name),
                        class_name=owner if sym.kind != 'class' else owner,
                    )
                    if changed:
                        changed_files[p] = new_text
                        file_texts[p] = new_text
                        applied_any = True
                        n_applied += 1
                        print(f'  + docstring {sym.qualname} → {p} (as {owner})')
            if not applied_any and r.doc_status == 'MISSING':
                n_skip += 1
                print(f'  ! could not place docstring for {sym.qualname}')
    if args.dry_run:
        print(f'DRY-RUN: would update {len(changed_files)} files, applied={n_applied}, skip={n_skip}')
        return 0
    for p, text in changed_files.items():
        (REPO / p).write_text(text, encoding='utf-8')
        print(f'Updated {p}')
    print(f'Applied {n_applied} docstrings; skipped {n_skip}')
    return 0


def build_doc_index(orig_root: Path, mods: list[ModuleMap]) -> dict[tuple[str | None, str], str]:
    """Map (class_name|None, bind_name) -> docstring text."""
    index: dict[tuple[str | None, str], str] = {}
    for mod in mods:
        py_path = orig_root / mod.py
        if not py_path.is_file():
            continue
        for sym in parse_python_module(py_path):
            if not sym.docstring or not sym.docstring.strip():
                continue
            if sym.kind == 'class':
                index[(sym.bind_name, '')] = sym.docstring  # class doc under ("Class", "")
            else:
                index[(sym.class_name, sym.bind_name or '')] = sym.docstring
    return index


# Subclass → Python base class names to fall back to for method docs
DOC_BASE_CLASSES: dict[str, list[str]] = {
    'AbelianBackend': ['TensorBackend'],
    'NoSymmetryBackend': ['TensorBackend'],
    'FusionTreeBackend': ['TensorBackend'],
    'AbelianLegPipe': ['LegPipe', 'Space', 'Leg'],
    'LegPipe': ['Space', 'Leg'],
    'ElementarySpace': ['Leg', 'Space'],
    'TensorProduct': ['Space'],
    'Space': ['Leg'],
    'Symmetry': ['BaseSymmetry', 'SymmetryFactor'],
    'SymmetryFactor': ['BaseSymmetry'],
    'Group': ['SymmetryFactor', 'BaseSymmetry'],
    'AbelianGroup': ['Group', 'SymmetryFactor', 'BaseSymmetry'],
    'SUN': ['Group', 'SymmetryFactor', 'BaseSymmetry'],
    'SU2': ['Group', 'SymmetryFactor', 'BaseSymmetry'],
    'U1': ['AbelianGroup', 'Group', 'SymmetryFactor', 'BaseSymmetry'],
    'ZN': ['AbelianGroup', 'Group', 'SymmetryFactor', 'BaseSymmetry'],
    'NoSymmetry': ['AbelianGroup', 'Group', 'SymmetryFactor', 'BaseSymmetry'],
    'TreePairMapping': ['TensorMapping'],
    'FactorizedTreeMapping': ['TensorMapping'],
    # Layer 4 tensors
    'Tensor': ['LabelledLegs'],
    'SymmetricTensor': ['Tensor', 'LabelledLegs'],
    'DiagonalTensor': ['SymmetricTensor', 'Tensor', 'LabelledLegs'],
    'Identity': ['DiagonalTensor', 'SymmetricTensor', 'Tensor', 'LabelledLegs'],
    'Mask': ['Tensor', 'LabelledLegs'],
    'ChargedTensor': ['Tensor', 'LabelledLegs'],
}


def lookup_doc(index: dict[tuple[str | None, str], str], class_name: str | None, method: str) -> str | None:
    if class_name is None:
        return index.get((None, method))
    if method == '':
        return index.get((class_name, ''))
    doc = index.get((class_name, method))
    if doc:
        return doc
    for base in DOC_BASE_CLASSES.get(class_name, []):
        doc = index.get((base, method))
        if doc:
            return doc
    return index.get((None, method))


def iter_class_regions(text: str) -> list[tuple[str, int, int]]:
    """Return (class_name, start, end) for each py::class_ region until next class_ or EOF."""
    regions = []
    pat = re.compile(r'py::class_<[^>]*>\s*(?:\w+\s*)?\(\s*m\s*,\s*"(?P<name>[^"]+)"')
    matches = list(pat.finditer(text))
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        regions.append((m.group('name'), start, end))
    return regions


def trailing_cstring_doc_span(text: str, open_paren: int, close_paren: int) -> tuple[int, int] | None:
    """If the last argument of a .def( call is a plain \"...\" docstring, return its [start, end)."""
    chunk = text[open_paren : close_paren + 1]
    # Find last non-whitespace before ')'
    i = close_paren - 1
    while i > open_paren and text[i] in ' \t\n\r':
        i -= 1
    if i <= open_paren or text[i] != '"':
        return None
    # Walk back over a C string literal (no raw/R" support here — those are pydoc)
    end = i + 1
    i -= 1
    while i > open_paren:
        if text[i] == '"' and text[i - 1] != '\\':
            # check not part of R" or u8"
            start = i
            # ensure preceded by comma (docstring arg), not identifier
            j = start - 1
            while j > open_paren and text[j] in ' \t\n\r':
                j -= 1
            if text[j] != ',':
                return None
            return start, end
        if text[i] == '\\':
            i -= 2
            continue
        i -= 1
    return None


def insert_or_replace_pydoc(text: str, close_paren: int, doc: str) -> tuple[str, int]:
    """Insert R\"pydoc before close_paren, or replace a trailing short \"...\" docstring.

    Returns (new_text, new_close_paren_index_after_edit) — actually just new_text;
    caller works reverse so indices of earlier sites stay valid if we only edit at end.
    """
    open_paren = text.rfind('(', 0, close_paren)
    # Prefer: find matching open for this def — close_paren is known; scan back for depth
    depth = 0
    open_paren = None
    for i in range(close_paren, -1, -1):
        if text.startswith(PYDOC_END, i - len(PYDOC_END) + 1) if i >= len(PYDOC_END) - 1 else False:
            pass
        ch = text[i]
        if ch == ')':
            depth += 1
        elif ch == '(':
            depth -= 1
            if depth == 0:
                open_paren = i
                break
    if open_paren is None:
        open_paren = text.rfind('(', 0, close_paren)

    line_start = text.rfind('\n', 0, close_paren) + 1
    indent = re.match(r'[ \t]*', text[line_start:close_paren]).group(0) or '      '
    block = format_pydoc(doc, indent)

    span = trailing_cstring_doc_span(text, open_paren, close_paren)
    if span is not None:
        start, end = span
        # Replace `"short"` with R"pydoc block (keep surrounding commas/whitespace structure)
        # expand to include optional whitespace before the string so indent is clean
        j = start
        while j > open_paren and text[j - 1] in ' \t':
            j -= 1
        # keep the comma before; replace from after comma
        k = j
        while k > open_paren and text[k - 1] in ' \t\n\r':
            k -= 1
        # k should be at comma
        if text[k - 1] == ',':
            replacement = '\n' + block
            return text[:k] + replacement + text[end:]
        return text[:start] + block.lstrip() + text[end:]

    insertion = ',\n' + block
    return text[:close_paren] + insertion + text[close_paren:]


def fill_docs_in_region(text: str, region_start: int, region_end: int, class_name: str, index) -> tuple[str, int]:
    """Fill missing pydocs for .def bindings inside [region_start, region_end)."""
    region = text[region_start:region_end]
    n = 0
    class_doc = lookup_doc(index, class_name, '')
    if class_doc and not (f'.doc() = {PYDOC_START}' in region[:1500] or region[:800].count(PYDOC_START) > 0):
        new_text, changed = apply_docstring_to_pybind('', text, class_name, class_doc, True, class_name)
        if changed:
            text = new_text
            n += 1
            regions = {n: (s, e) for n, s, e in iter_class_regions(text)}
            if class_name in regions:
                region_start, region_end = regions[class_name]
            region = text[region_start:region_end]

    sites_abs: list[tuple[int, int, str, bool]] = []
    for m in re.finditer(
        r'\.def(?:_static|_prop_ro|_property_readonly|_readwrite|_readonly)?\(\s*"(?P<name>[^"]+)"',
        region,
    ):
        name = m.group('name')
        if name.startswith('__') and name.endswith('__'):
            continue
        abs_start = region_start + m.start()
        open_paren = text.find('(', abs_start)
        depth = 0
        i = open_paren
        has_pydoc = False
        while i < len(text):
            if text.startswith(PYDOC_START, i):
                has_pydoc = True
                j = text.find(PYDOC_END, i)
                i = j + len(PYDOC_END) if j >= 0 else i + 1
                continue
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    sites_abs.append((abs_start, i, name, has_pydoc))
                    break
            i += 1

    for start, end, name, has_pydoc in reversed(sites_abs):
        if has_pydoc:
            continue
        doc = lookup_doc(index, class_name, name)
        if not doc:
            continue
        text = insert_or_replace_pydoc(text, end, doc)
        n += 1
        print(f'  + fill {class_name}.{name}')
    return text, n


def cmd_fill_bound_docs(args: argparse.Namespace) -> int:
    """Fill R\"pydoc on every bound .def that lacks one, using Python (+ base class) docs."""
    mods = load_map(Path(args.map))
    index = build_doc_index(Path(args.orig), mods)
    # unique pybind files
    pybind_paths: list[str] = []
    seen = set()
    for mod in mods:
        for p in mod.pybind:
            if p not in seen:
                seen.add(p)
                pybind_paths.append(p)

    changed_files: dict[str, str] = {}
    total = 0
    for p in pybind_paths:
        text = (REPO / p).read_text(encoding='utf-8') if (REPO / p).is_file() else ''
        if not text:
            continue
        n_file = 0
        # Process class regions from end to start
        regions = iter_class_regions(text)
        for class_name, start, end in reversed(regions):
            text, n = fill_docs_in_region(text, start, end, class_name, index)
            n_file += n
        # Module-level m.def("...")
        for m in reversed(list(re.finditer(r'\bm\.def\(\s*"(?P<name>[^"]+)"', text))):
            name = m.group('name')
            open_paren = text.find('(', m.start())
            depth = 0
            i = open_paren
            has_pydoc = False
            while i < len(text):
                if text.startswith(PYDOC_START, i):
                    has_pydoc = True
                    j = text.find(PYDOC_END, i)
                    i = j + len(PYDOC_END) if j >= 0 else i + 1
                    continue
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            else:
                continue
            if has_pydoc:
                continue
            doc = lookup_doc(index, None, name)
            if not doc:
                continue
            text = insert_or_replace_pydoc(text, i, doc)
            n_file += 1
            print(f'  + fill module.{name} → {p}')
        if n_file:
            changed_files[p] = text
            total += n_file
            print(f'  ({n_file} fills in {p})')

    if args.dry_run:
        print(f'DRY-RUN: would update {len(changed_files)} files, fills={total}')
        return 0
    for p, text in changed_files.items():
        (REPO / p).write_text(text, encoding='utf-8')
        print(f'Updated {p}')
    print(f'Filled {total} docstrings across {len(changed_files)} files')
    return 0


def cmd_apply_hints(args: argparse.Namespace) -> int:
    mods = load_map(Path(args.map))
    changed_files: dict[str, str] = {}
    n_applied = 0
    n_skip = 0
    # leftover comments per primary cpp file
    leftovers: dict[str, list[tuple[str, list]]] = defaultdict(list)

    for mod in mods:
        results = audit_module(mod, Path(args.orig))
        file_texts = read_files(mod.cpp)
        primary_cpp = next((p for p in mod.cpp if p.endswith('.cpp')), None)
        for r in results:
            comments = r.symbol.comments
            if not comments:
                continue
            # Skip if already have a dedicated hint block anywhere in mapped cpp
            marker = f'hints from Python {r.symbol.qualname}'
            combined_now = '\n'.join(changed_files.get(p, file_texts.get(p, '')) for p in mod.cpp)
            if marker in combined_now:
                continue
            cpp_paths = [p for p in mod.cpp if p.endswith('.cpp')] + [p for p in mod.cpp if not p.endswith('.cpp')]
            placed = False
            for p in cpp_paths:
                text = changed_files.get(p, file_texts.get(p, ''))
                if not text:
                    continue
                new_text, changed = apply_hints_to_cpp(p, text, r.symbol.qualname, comments)
                if changed:
                    changed_files[p] = new_text
                    file_texts[p] = new_text
                    n_applied += 1
                    placed = True
                    print(f'  + hints {r.symbol.qualname} → {p} ({len(comments)} comments)')
                    break
            if not placed:
                # Only orphan comments that are still truly missing from cpp
                still = [c for c in comments if not comment_present(c.text, combined_now)]
                if not still:
                    continue
                n_skip += 1
                if args.verbose:
                    print(f'  ! could not place hints for {r.symbol.qualname}')
                if primary_cpp:
                    leftovers[primary_cpp].append((r.symbol.qualname, still))

    # Append leftover hint catalogs at end of primary cpp files
    for p, items in leftovers.items():
        text = changed_files.get(p, (REPO / p).read_text(encoding='utf-8'))
        if 'ORPHANED PYTHON COMMENT HINTS' in text:
            continue
        block_lines = [
            '',
            '// =============================================================================',
            '// ORPHANED PYTHON COMMENT HINTS (no matching C++ function body found)',
            '// =============================================================================',
        ]
        for qualname, comments in items:
            block_lines.append(f'// --- {qualname} ---')
            seen = set()
            for c in comments:
                t = c.text.lstrip('#').strip()
                key = normalize_text(t)
                if key in seen:
                    continue
                seen.add(key)
                block_lines.append(f'// {t}')
        block_lines.append('// =============================================================================')
        text = text.rstrip() + '\n' + '\n'.join(block_lines) + '\n'
        changed_files[p] = text
        print(f'  + orphaned hints catalog → {p} ({len(items)} symbols)')

    if args.dry_run:
        print(f'DRY-RUN: would update {len(changed_files)} files, applied={n_applied}, skip={n_skip}')
        return 0
    for p, text in changed_files.items():
        (REPO / p).write_text(text, encoding='utf-8')
        print(f'Updated {p}')
    print(f'Applied hint blocks to {n_applied} symbols; skipped {n_skip}')
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', default=str(DEFAULT_MAP))
    ap.add_argument('--orig', default=str(DEFAULT_ORIG))
    ap.add_argument('--report', default=str(DEFAULT_REPORT))
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_report = sub.add_parser('report', help='Write markdown audit report')
    p_report.set_defaults(func=cmd_report)

    p_doc = sub.add_parser('apply-docstrings', help='Insert missing R"pydoc into pybind files')
    p_doc.add_argument('--dry-run', action='store_true')
    p_doc.set_defaults(func=cmd_apply_docstrings)

    p_fill = sub.add_parser(
        'fill-bound-docs',
        help='Fill R"pydoc on all bound defs lacking docs (uses base-class docs)',
    )
    p_fill.add_argument('--dry-run', action='store_true')
    p_fill.set_defaults(func=cmd_fill_bound_docs)

    p_hints = sub.add_parser('apply-hints', help='Insert function-top hint blocks for missing comments')
    p_hints.add_argument('--dry-run', action='store_true')
    p_hints.add_argument('-v', '--verbose', action='store_true')
    p_hints.set_defaults(func=cmd_apply_hints)

    args = ap.parse_args()
    return args.func(args)


if __name__ == '__main__':
    # Fix duplicate visit_FunctionDef from editing — clean class at runtime if needed
    sys.exit(main())
