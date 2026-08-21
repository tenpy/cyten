"""Inject compact ``[C++]`` signature badges from ``.. cyten-cpp-ref::`` markers.

Markers are produced by ``scripts/doxygen_xml_to_docstrings.py`` and
``doc_cpp_ref()`` in ``pybind/doc_plus.h``. This extension:

1. On ``autodoc-process-docstring`` (**before** Napoleon), records the C++
   target and strips the marker so Napoleon cannot treat it as a parameter.
2. On ``doctree-read``, appends a ``[C++]`` link next to the signature (same
   presentation as ``sphinx.ext.linkcode``'s ``[source]``).
"""

from __future__ import annotations

import re
from typing import Any

from docutils import nodes
from docutils.nodes import Node
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.locale import _

_MARKER_RE = re.compile(r'^\.\.\s+cyten-cpp-ref::\s*(.+?)\s*$')
_ROLE_RE = re.compile(r'^:role:\s*(\S+)\s*$')


def _normalize_cpp_target(target: str) -> str:
    """Map marker text to a Sphinx ``cpp:function`` inventory key."""
    target = target.strip()
    if '(' in target:
        target = target[: target.index('(')].rstrip()
    return target


# env.cyten_cpp_refs: fullname -> (role, target)
_ENV_KEY = 'cyten_cpp_refs'

# Run before Napoleon (default priority 500) so the marker is not eaten as :param:.
_BEFORE_NAPOLEON = 400


def _ensure_map(env) -> dict[str, tuple[str, str]]:
    if not hasattr(env, _ENV_KEY):
        setattr(env, _ENV_KEY, {})
    return getattr(env, _ENV_KEY)


def _extract_marker(lines: list[str]) -> tuple[str, str] | None:
    """Return ``(role, target)`` if a marker block is present; remove it from ``lines``."""
    marker_idx = None
    for i, line in enumerate(lines):
        if _MARKER_RE.match(line.strip()):
            marker_idx = i
            break
    if marker_idx is None:
        return None

    m = _MARKER_RE.match(lines[marker_idx].strip())
    assert m is not None
    target = _normalize_cpp_target(m.group(1))
    role = 'func'
    end = marker_idx

    # Optional ":role: class" on the following non-empty line.
    j = marker_idx + 1
    while j < len(lines) and lines[j].strip() == '':
        j += 1
    if j < len(lines):
        rm = _ROLE_RE.match(lines[j].strip())
        if rm:
            role = rm.group(1)
            end = j

    start = marker_idx
    if start > 0 and lines[start - 1].strip() == '':
        start -= 1
    del lines[start : end + 1]
    while lines and lines[-1].strip() == '':
        lines.pop()
    return role, target


def autodoc_process_docstring(
    app: Sphinx,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    lines: list[str],
) -> None:
    """Record and strip ``cyten-cpp-ref`` markers from autodoc docstrings."""
    del what, obj, options  # unused
    extracted = _extract_marker(lines)
    if extracted is None:
        return
    role, target = extracted
    _ensure_map(app.env)[name] = (role, target)


def doctree_read(app: Sphinx, doctree: Node) -> None:
    """Append ``[C++]`` signature badges from recorded C++ xref targets."""
    refs = getattr(app.env, _ENV_KEY, None)
    if not refs:
        return

    node_only_expr = getattr(app.builder, 'supported_linkcode', 'html')

    for objnode in list(doctree.findall(addnodes.desc)):
        if objnode.get('domain') != 'py':
            continue
        for signode in objnode:
            if not isinstance(signode, addnodes.desc_signature):
                continue
            mod = signode.get('module') or ''
            fullname = signode.get('fullname') or ''
            if not fullname:
                continue
            key = f'{mod}.{fullname}' if mod else fullname
            entry = refs.get(key) or refs.get(fullname)
            if entry is None:
                continue
            role, target = entry

            xref = addnodes.pending_xref(
                '',
                refdomain='cpp',
                reftype=role,
                reftarget=target,
                refexplicit=True,
                refwarn=True,
            )
            xref['refdoc'] = app.env.docname
            xref += nodes.inline('', _('[C++]'), classes=['viewcode-link', 'cyten-cpp-link'])
            onlynode = addnodes.only(expr=node_only_expr)
            onlynode += xref
            signode += onlynode


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the extension with Sphinx."""
    app.connect('autodoc-process-docstring', autodoc_process_docstring, priority=_BEFORE_NAPOLEON)
    app.connect('doctree-read', doctree_read)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
