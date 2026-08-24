"""Embed Doxygen's built-in C++ inheritance graphs into the Sphinx C++ reference.

Sphinx ``inheritance-diagram`` (Graphviz) spreads many siblings across a wide
rank and collides labels. Doxygen's built-in CLASS_GRAPH generator stacks those
labels on a vertical bus — the same diagrams as ``build_docs/doxy_html``.

This extension copies those PNGs into the Sphinx HTML output and rewires the
imagemap to ``:cpp:class:`` pages. ``.. doxygen-inheritance-diagram::`` with no
arguments uses every ``doxygenclass`` / ``doxygenstruct`` on the page and keeps
the smallest set of graphs that cover them.
"""

from __future__ import annotations

import html
import os
import re
import shutil
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.osutil import ensuredir

_DOXY_CLASS_RE = re.compile(
    r'^\.\.\s+doxygen(?:class|struct)::\s+(\S+)\s*$',
    re.M,
)
_INHERIT_MAP_RE = re.compile(
    r'Inheritance diagram for.*?<map[^>]*>(.*?)</map>',
    re.S | re.I,
)
_AREA_TAG_RE = re.compile(r'<area\b[^>]*>', re.I)
_ATTR_RE = re.compile(r'\b(alt|coords|href)="([^"]*)"', re.I)

_ENV_INDEX = 'doxygen_inheritance_index'


class doxygen_inheritance_diagram(nodes.General, nodes.Element):
    """Placeholder: PNG path, imagemap areas, pending C++ xrefs as children."""


def _html_dir(app: Sphinx) -> Path:
    configured = getattr(app.config, 'doxygen_html_dir', '') or ''
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else Path(app.srcdir) / path
    projects = getattr(app.config, 'breathe_projects', None) or {}
    xml = next(iter(projects.values()), None)
    if xml:
        xml_path = Path(xml)
        if not xml_path.is_absolute():
            xml_path = Path(app.srcdir) / xml_path
        return xml_path.parent / 'doxy_html'
    return Path(app.srcdir) / 'build_docs' / 'doxy_html'


def _xml_dir(app: Sphinx) -> Path:
    projects = getattr(app.config, 'breathe_projects', None) or {}
    xml = next(iter(projects.values()), None)
    if xml:
        xml_path = Path(xml)
        return xml_path if xml_path.is_absolute() else Path(app.srcdir) / xml_path
    return Path(app.srcdir) / 'build_docs' / 'doxy_xml'


def _compound_index(app: Sphinx) -> dict[str, str]:
    """Map ``cyten::Tensor`` → Doxygen refid ``classcyten_1_1Tensor``."""
    env = app.env
    cached = getattr(env, _ENV_INDEX, None)
    if cached is not None:
        return cached
    index: dict[str, str] = {}
    index_xml = _xml_dir(app) / 'index.xml'
    if index_xml.is_file():
        root = ET.parse(index_xml).getroot()
        for compound in root.findall('compound'):
            kind = compound.get('kind')
            if kind not in {'class', 'struct', 'interface'}:
                continue
            refid = compound.get('refid')
            name_el = compound.find('name')
            if refid and name_el is not None and name_el.text:
                index[name_el.text.strip()] = refid
    setattr(env, _ENV_INDEX, index)
    return index


def _parse_areas(html_text: str) -> list[dict[str, str]]:
    match = _INHERIT_MAP_RE.search(html_text)
    if not match:
        return []
    areas: list[dict[str, str]] = []
    for tag in _AREA_TAG_RE.findall(match.group(1)):
        attrs = {key.lower(): html.unescape(val) for key, val in _ATTR_RE.findall(tag)}
        alt = attrs.get('alt', '').strip()
        coords = attrs.get('coords', '').strip()
        if not alt or not coords:
            continue
        areas.append({'alt': alt, 'coords': coords, 'href': attrs.get('href', '')})
    return areas


def _graph_for_class(html_dir: Path, refid: str, qname: str) -> tuple[Path, list[dict[str, str]], set[str]] | None:
    png = html_dir / f'{refid}.png'
    page = html_dir / f'{refid}.html'
    if not png.is_file() or not page.is_file():
        return None
    areas = _parse_areas(page.read_text(encoding='utf-8', errors='replace'))
    if not areas:
        return None
    labels = {area['alt'] for area in areas}
    labels.add(qname)
    return png, areas, labels


def _select_graphs(
    names: list[str], html_dir: Path, index: dict[str, str]
) -> list[tuple[str, Path, list[dict[str, str]]]]:
    """Greedy cover: fewest Doxygen graphs that still show every class on the page."""
    candidates: list[tuple[str, Path, list[dict[str, str]], set[str]]] = []
    for name in names:
        refid = index.get(name)
        if not refid:
            continue
        graph = _graph_for_class(html_dir, refid, name)
        if graph is None:
            continue
        png, areas, labels = graph
        candidates.append((name, png, areas, labels))

    remaining = set(names)
    chosen: list[tuple[str, Path, list[dict[str, str]]]] = []
    while remaining and candidates:
        best = max(candidates, key=lambda item: len(item[3] & remaining))
        if not (best[3] & remaining):
            break
        chosen.append((best[0], best[1], best[2]))
        remaining -= best[3]
        candidates = [item for item in candidates if item[0] != best[0]]
    return chosen


def _collect_from_rst(path: str) -> list[str]:
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError:
        return []
    seen: set[str] = set()
    names: list[str] = []
    for name in _DOXY_CLASS_RE.findall(text):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _xref(qname: str) -> addnodes.pending_xref:
    inner = nodes.literal(qname, qname)
    ref = addnodes.pending_xref(
        '',
        inner,
        refdomain='cpp',
        reftype='class',
        reftarget=qname,
        refexplicit=True,
    )
    return ref


class DoxygenInheritanceDiagram(SphinxDirective):
    """Include Doxygen CLASS_GRAPH PNGs for classes documented on this page."""

    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    has_content = False

    def run(self) -> list[nodes.Node]:
        if self.arguments and str(self.arguments[0]).strip():
            names = self.arguments[0].split()
        else:
            source, _lineno = self.state_machine.get_source_and_line(self.lineno)
            names = _collect_from_rst(source) if source else []
        if not names:
            return []

        html_dir = _html_dir(self.env.app)
        if not html_dir.is_dir():
            return [
                self.state.document.reporter.warning(
                    f'doxygen-inheritance-diagram: Doxygen HTML output not found at {html_dir}',
                    line=self.lineno,
                )
            ]

        graphs = _select_graphs(names, html_dir, _compound_index(self.env.app))
        if not graphs:
            return []

        rubric = nodes.rubric('', 'Inheritance Diagram')
        container = nodes.container(classes=['doxygen-inheritance-diagram'])
        for qname, png, areas in graphs:
            node = doxygen_inheritance_diagram()
            node['png'] = str(png)
            node['areas'] = areas
            node['title'] = qname
            seen: set[str] = set()
            for area in areas:
                alt = area['alt']
                if alt in seen or alt.startswith('std::'):
                    continue
                seen.add(alt)
                node += _xref(alt)
            container += node
        return [rubric, container]


def _copy_png(builder, png: Path) -> str:
    """Copy *png* into ``_images/`` and return a URI relative to the current doc."""
    imagedir = Path(builder.outdir) / '_images'
    ensuredir(str(imagedir))
    dest = imagedir / png.name
    if not dest.is_file() or dest.stat().st_mtime < png.stat().st_mtime:
        shutil.copyfile(png, dest)
    docdir = Path(builder.get_target_uri(builder.current_docname)).parent
    # URI from this HTML file to _images/<png>
    prefix = Path(os.path.relpath('_images', start=str(docdir) if str(docdir) != '.' else '.'))
    return str(prefix / png.name).replace('\\', '/')


def html_visit_doxygen_inheritance(self, node: doxygen_inheritance_diagram) -> None:
    """Emit the Doxygen PNG and an imagemap whose areas link to ``:cpp:class:`` pages."""
    urls: dict[str, str] = {}
    for child in node.findall(nodes.Element):
        qname = child.get('reftarget') or child.get('reftitle') or ''
        if not qname and isinstance(child, nodes.Text):
            continue
        if not qname:
            text = child.astext().strip()
            if text.startswith('cyten::') or '::' in text:
                qname = text
        uri = None
        if child.get('refuri'):
            uri = child['refuri']
        elif child.get('refid'):
            uri = '#' + child['refid']
        if qname and uri:
            urls[qname] = uri

    png = Path(node['png'])
    uri = _copy_png(self.builder, png)
    map_name = 'doxyinh_' + png.stem
    title = node.get('title', '')
    alt = f'Inheritance diagram of {title}' if title else 'Inheritance diagram'
    esc = html.escape
    self.body.append('<div class="graphviz">\n')
    self.body.append(f'<img src="{esc(uri)}" usemap="#{map_name}" alt="{esc(alt)}"/>\n')
    self.body.append(f'<map id="{map_name}" name="{map_name}">\n')
    for area in node['areas']:
        qname = area['alt']
        href = urls.get(qname, '')
        href_attr = f' href="{esc(href)}"' if href else ''
        self.body.append(
            f'<area shape="rect" coords="{esc(area["coords"])}" alt="{esc(qname)}" title="{esc(qname)}"{href_attr}/>\n'
        )
    self.body.append('</map>\n</div>\n')
    raise nodes.SkipNode


def latex_visit_doxygen_inheritance(self, node: doxygen_inheritance_diagram) -> None:
    """Omit the diagram from LaTeX output."""
    raise nodes.SkipNode


def skip_node(self, node: doxygen_inheritance_diagram) -> None:
    """Omit the diagram from text, man, and texinfo builders."""
    raise nodes.SkipNode


def _auto_insert(app: Sphinx, docname: str, source: list[str]) -> None:
    """Add a diagram to C++ pages that document classes, matching the Python reference."""
    if not docname.startswith('cpp/'):
        return
    text = source[0]
    if 'doxygen-inheritance-diagram::' in text:
        return
    if not _DOXY_CLASS_RE.search(text):
        return
    match = _DOXY_CLASS_RE.search(text)
    assert match is not None
    insert = '.. doxygen-inheritance-diagram::\n\n'
    source[0] = text[: match.start()] + insert + text[match.start() :]


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the Doxygen inheritance-diagram directive and HTML visitor."""
    app.add_config_value('doxygen_html_dir', '', 'html', [str])
    app.add_node(
        doxygen_inheritance_diagram,
        html=(html_visit_doxygen_inheritance, None),
        latex=(latex_visit_doxygen_inheritance, None),
        text=(skip_node, None),
        man=(skip_node, None),
        texinfo=(skip_node, None),
    )
    app.add_directive('doxygen-inheritance-diagram', DoxygenInheritanceDiagram)
    app.connect('source-read', _auto_insert)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
