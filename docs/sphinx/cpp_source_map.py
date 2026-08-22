"""Map C++ qualified names to repo-relative source locations from Doxygen XML."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

_SKIP_XML = frozenset({'index.xml', 'Doxyfile.xml'})
_COMPOUND_KINDS = frozenset({'class', 'struct'})
_FUNCTION_KIND = 'function'

# Sphinx C++ domain v1 id fragments for fundamental / alias types.
_FUNDAMENTAL_IDS = {
    'bool': 'b',
    'char': 'c',
    'signed char': 'a',
    'unsigned char': 'h',
    'wchar_t': 'w',
    'short': 's',
    'unsigned short': 't',
    'int': 'i',
    'unsigned': 'j',
    'unsigned int': 'j',
    'long': 'l',
    'unsigned long': 'm',
    'long long': 'x',
    'unsigned long long': 'y',
    'float': 'f',
    'double': 'd',
    'long double': 'e',
    'void': 'v',
    'std::string': 'ss',
}


@dataclass(frozen=True)
class CppLocation:
    """Repo-relative path and 1-based line range for a C++ symbol."""

    relpath: str
    start: int
    end: int | None
    has_body: bool = False
    encoded_params: str = ''


_maps: dict[tuple[str, str], dict[str, CppLocation]] = {}


def _local_tag(tag: str | None) -> str:
    if tag is None:
        return ''
    if '}' in tag:
        return tag.rsplit('}', 1)[-1]
    return tag


def _child_text(elem: ET.Element, tag: str) -> str:
    for child in elem:
        if _local_tag(child.tag) == tag:
            return (child.text or '').strip()
    return ''


def _first_child(elem: ET.Element, tag: str) -> ET.Element | None:
    for child in elem:
        if _local_tag(child.tag) == tag:
            return child
    return None


def _int_attr(elem: ET.Element, name: str) -> int:
    raw = elem.get(name) or ''
    try:
        return int(raw)
    except ValueError:
        return 0


def _split_top_level(text: str, sep: str = ',') -> list[str]:
    """Split ``text`` on ``sep`` not nested in ``<>``, ``()``, or ``[]``."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch in '<([{':
            depth += 1
            cur.append(ch)
        elif ch in '>)]}':
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == sep and depth == 0:
            piece = ''.join(cur).strip()
            if piece:
                parts.append(piece)
            cur = []
        else:
            cur.append(ch)
    piece = ''.join(cur).strip()
    if piece:
        parts.append(piece)
    return parts


def encode_param_type(dtype: str) -> str:
    """Encode a Doxygen parameter type like Sphinx C++ domain v1 ids.

    Examples::

        DiagonalTensorCPtr              → DiagonalTensorCPtr
        BlockBackend::Scalar const &    → BlockBackend::ScalarCR
        std::string const &             → ssCR
        std::vector<TensorCPtr> const & → std::vector:TensorCPtr:CR
        bool                            → b
    """
    s = re.sub(r'\s+', ' ', dtype.strip())
    if not s:
        return ''

    suffix = ''
    while True:
        s = s.rstrip()
        if s.endswith('&&'):
            suffix = 'RR' + suffix
            s = s[:-2]
        elif s.endswith('&'):
            suffix = 'R' + suffix
            s = s[:-1]
        elif s.endswith('*'):
            suffix = 'P' + suffix
            s = s[:-1]
        else:
            break

    tokens = s.split()
    core_tokens: list[str] = []
    for tok in tokens:
        if tok == 'const':
            suffix = 'C' + suffix
        elif tok == 'volatile':
            suffix = 'V' + suffix
        else:
            core_tokens.append(tok)
    s = ' '.join(core_tokens).strip()

    lt = s.find('<')
    if lt != -1 and s.endswith('>'):
        name = s[:lt].strip().replace(' ', '')
        inner = s[lt + 1 : -1]
        inner_enc = '.'.join(encode_param_type(p) for p in _split_top_level(inner))
        core = f'{name}:{inner_enc}:'
    else:
        key = s
        core = _FUNDAMENTAL_IDS.get(key, key.replace(' ', ''))
    return core + suffix


def encode_param_list(types: list[str]) -> str:
    """Join encoded parameter types with ``.``, matching Sphinx v1 function ids."""
    return '.'.join(encode_param_type(t) for t in types if t)


def _param_types(member: ET.Element) -> list[str]:
    """Return parameter type strings from a Doxygen ``memberdef``."""
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
    return types


def to_repo_relpath(doxy_path: str, repo_root: Path) -> str | None:
    """Return a POSIX path relative to ``repo_root``, or None if outside the repo.

    Doxygen emits absolute paths or paths relative to the Doxyfile directory
    (``docs/``).
    """
    if not doxy_path:
        return None
    raw = Path(doxy_path.replace('\\', '/'))
    repo_root = repo_root.resolve()
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (repo_root / 'docs' / raw).resolve()
    try:
        rel = resolved.relative_to(repo_root)
    except ValueError:
        return None
    return rel.as_posix()


def _location_from_elem(loc: ET.Element, repo_root: Path) -> CppLocation | None:
    """Prefer the definition body when it is inside the repo; else the declaration."""
    bodyfile = loc.get('bodyfile') or ''
    bodystart = _int_attr(loc, 'bodystart')
    bodyend = _int_attr(loc, 'bodyend')
    if bodyfile and bodystart > 0:
        rel = to_repo_relpath(bodyfile, repo_root)
        if rel is not None:
            end = bodyend if bodyend >= bodystart else None
            return CppLocation(rel, bodystart, end, has_body=True)

    decl_file = loc.get('file') or ''
    line = _int_attr(loc, 'line')
    if decl_file and line > 0:
        rel = to_repo_relpath(decl_file, repo_root)
        if rel is not None:
            return CppLocation(rel, line, None, has_body=False)
    return None


def _member_qname(member: ET.Element, ns_parts: tuple[str, ...]) -> str:
    qname = _child_text(member, 'qualifiedname')
    if qname:
        return qname
    name = _child_text(member, 'name')
    if not name:
        return ''
    if ns_parts:
        return '::'.join((*ns_parts, name))
    return name


def _pick_location(candidates: list[CppLocation]) -> CppLocation:
    """For a bare name, prefer a definition with a body, else the first declaration."""
    for loc in candidates:
        if loc.has_body:
            return loc
    return candidates[0]


def load_cpp_source_map(xml_dir: str | Path, repo_root: str | Path) -> dict[str, CppLocation]:
    """Parse Doxygen XML into ``qualified-name -> CppLocation``.

    Indexes class/struct compounds and function memberdefs (methods and free
    functions). Overloads are also stored under Sphinx v1 ids
    (``cyten::sqrt__DiagonalTensorCPtr``). Missing or unreadable XML yields an
    empty map.
    """
    xml_path = Path(xml_dir)
    root_path = Path(repo_root)
    if not xml_path.is_dir():
        return {}

    candidates: dict[str, list[CppLocation]] = {}

    def _add(qname: str, loc: CppLocation) -> None:
        if qname:
            candidates.setdefault(qname, []).append(loc)

    for path in sorted(xml_path.glob('*.xml')):
        if path.name in _SKIP_XML or path.name.endswith('.xsd'):
            continue
        try:
            xml_root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for compound in xml_root.iter():
            if _local_tag(compound.tag) != 'compounddef':
                continue
            kind = compound.get('kind', '')
            compoundname = _child_text(compound, 'compoundname')
            ns_parts = (
                tuple(p for p in compoundname.split('::') if p)
                if kind in ('namespace', 'class', 'struct') and compoundname
                else ()
            )

            if kind in _COMPOUND_KINDS and compoundname:
                loc_el = _first_child(compound, 'location')
                if loc_el is not None:
                    loc = _location_from_elem(loc_el, root_path)
                    if loc is not None:
                        _add(compoundname, loc)

            for member in compound.iter():
                if _local_tag(member.tag) != 'memberdef':
                    continue
                if member.get('kind') != _FUNCTION_KIND:
                    continue
                loc_el = _first_child(member, 'location')
                if loc_el is None:
                    continue
                loc = _location_from_elem(loc_el, root_path)
                if loc is None:
                    continue
                encoded = encode_param_list(_param_types(member))
                loc = CppLocation(loc.relpath, loc.start, loc.end, loc.has_body, encoded)
                _add(_member_qname(member, ns_parts), loc)

    result: dict[str, CppLocation] = {}
    for qname, locs in candidates.items():
        result[qname] = _pick_location(locs)
        for loc in locs:
            if loc.encoded_params:
                result[f'{qname}__{loc.encoded_params}'] = loc
    return result


def lookup_cpp_location(
    mapping: dict[str, CppLocation],
    qname: str | None,
    ids: list[str] | str | None = None,
) -> CppLocation | None:
    """Resolve a location, preferring an overload-specific Sphinx v1 id."""
    if isinstance(ids, str):
        ids = [ids]
    for id_ in reversed(list(ids or [])):
        if not isinstance(id_, str) or id_.startswith('_CPP') or '::' not in id_:
            continue
        loc = mapping.get(id_)
        if loc is not None:
            return loc
    if not qname:
        return None
    loc = mapping.get(qname)
    if loc is not None:
        return loc
    if '__' in qname:
        loc = mapping.get(qname.split('__', 1)[0])
        if loc is not None:
            return loc
    if qname.endswith('C'):
        return mapping.get(qname[:-1])
    return None


def get_cpp_source_map(xml_dir: str | Path, repo_root: str | Path) -> dict[str, CppLocation]:
    """Return a cached ``load_cpp_source_map`` result."""
    key = (str(xml_dir), str(repo_root))
    cached = _maps.get(key)
    if cached is None:
        cached = load_cpp_source_map(xml_dir, repo_root)
        _maps[key] = cached
    return cached
