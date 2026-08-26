"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import importlib.machinery
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path

from sphinx import addnodes
from sphinx.ext.linkcode import add_linkcode_domain

_DOCS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DOCS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Local Sphinx extensions under docs/sphinx/
sys.path.insert(0, str(_DOCS_DIR / 'sphinx'))


def _extension_module_available(name: str = 'cyten._core') -> bool:
    """True if a compiled extension (not a pure-Python stub) can be loaded."""
    try:
        spec = importlib.util.find_spec(name)
    except ImportError, ModuleNotFoundError, ValueError:
        return False
    if spec is None or spec.origin is None:
        return False
    # Extension modules use ExtensionFileLoader; .py stubs use SourceFileLoader.
    return isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)


def _ensure_version_module() -> None:
    version_py = _REPO_ROOT / 'cyten' / '_version.py'
    if version_py.is_file():
        return
    try:
        from setuptools_scm import get_version

        get_version(root=str(_REPO_ROOT), write_to=str(version_py))
    except Exception:
        version_py.write_text(
            "version = __version__ = '0.0.0+unknown'\n"
            "version_tuple = __version_tuple__ = (0, 0, 0, 'unknown')\n"
            'commit_id = __commit_id__ = None\n',
            encoding='utf-8',
        )


def _ensure_core_stub() -> None:
    """Generate ``cyten/_core.py`` when the compiled extension is unavailable."""
    if _extension_module_available():
        return
    stub = _REPO_ROOT / 'cyten' / '_core.py'
    if stub.is_file():
        return
    script = _REPO_ROOT / 'scripts' / 'generate_core_stubs.py'
    cmd = [sys.executable, str(script), '-o', str(stub)]
    # Prefer existing pybind/docstrings; fall back to scoped Doxygen.
    docstrings = _REPO_ROOT / 'pybind' / 'docstrings'
    if not any(docstrings.rglob('*.h')):
        cmd.append('--from-doxygen')
    subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))


_ensure_version_module()
_ensure_core_stub()

import cyten  # noqa: E402

project = 'Cyten'
copyright = '2024, Cyten developer team'
author = 'Cyten developer team'
release = cyten.__version__

GITHUBBASE = 'https://github.com/tenpy/cyten'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.extlinks',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.mathjax',
    'sphinx.ext.linkcode',
    'sphinx.ext.githubpages',
    'cyten_cpp_link',  # after autodoc; connects with priority before napoleon
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'cyten_inheritance',  # pybind re-exports; after inheritance_diagram
    'doxygen_inheritance',  # Doxygen CLASS_GRAPH PNGs on C++ pages
    'sphinx_rtd_theme',
    'sphinx_copybutton',
]

templates_path = ['sphinx/templates']
exclude_patterns = ['build_docs', 'Thumbs.db', '.DS_Store']

source_suffix = {'.rst': 'restructuredtext'}  # can add markdown if needed
master_doc = 'index'  # The master toctree document.
language = 'en'  # no translations
pygments_style = 'sphinx'  # syntax highlighting style


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

#  html_logo = 'images/cyten_logo.png'
#  html_favicon = "images/logo.ico"
html_static_path = ['sphinx/static']
html_last_updated_fmt = '%b %d, %Y'

html_css_files = [
    'custom.css',  # to highlight targets
]

html_context = {
    'display_github': True,  # Integrate GitHub
    'github_user': 'tenpy',  # Username
    'github_repo': 'cyten',  # Repo name
    'github_version': 'main',  # Version
    'conf_py_path': '/docs/',  # Path in the checkout to the docs root
}

# -- breathe (including doxygen docs) -------------------------------------

breathe_projects = {'cyten': 'build_docs/doxy_xml'}

breathe_default_project = 'cyten'

breathe_default_members = ('members', 'undoc-members')

# Map header extensions to the C++ domain so :cpp:func: links resolve.
breathe_domain_by_extension = {
    'h': 'cpp',
    'hpp': 'cpp',
    'cpp': 'cpp',
}

# -- sphinx.ext.autodoc ---------------------------------------------------

autodoc_default_options = {
    'member-order': 'bysource',
}
autodoc_member_order = 'bysource'
# some options are included in the templates under
# sphinx_templates/autosummary/class.rst
# for example :inherited-members: and :show-inheritance:
autosummary_generate = True
autodoc_docstring_signature = True

# -- sphinx.ext.todo ------------------------------------------------------

todo_include_todos = True  # show todo-boxes in output

# -- sphinx.ext.doctest ---------------------------------------------------

doctest_global_setup = """
import numpy as np
import scipy
import scipy.linalg
import cyten
np.set_printoptions(suppress=True)
"""

trim_doctest_flag = True

# -- sphinx.ext.napoleon --------------------------------------------------
# numpy-like doc strings

napoleon_use_admonition_for_examples = True
napoleon_use_ivar = False  # otherwise :attr:`...` doesn't work anymore
napoleon_custom_sections = ['Options']

# -- sphinx.ext.inheritance_diagram ---------------------------------------

inheritance_graph_attrs = {
    'rankdir': 'TB',  # top-to-bottom
    'fontsize': 14,
    'ratio': 'compress',
}

# -- sphinx.ext.intersphinx -----------------------------------------------
# cross links to other sphinx documentations
# this makes  e.g. :class:`numpy.ndarray` work
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'tenpy': ('https://tenpy.readthedocs.org/en/stable', None),
    'tenpy_v1': ('https://tenpy.readthedocs.org/en/v1.1.1', None),
    'matplotlib': ('https://matplotlib.org', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
}

# -- sphinx.ext.extlinks --------------------------------------------------
# allows to use, e.g., :arxiv:`1805.00055`
extlinks = {
    'arxiv': ('https://arxiv.org/abs/%s', 'arXiv:%s'),
    'doi': ('https://dx.doi.org/%s', 'doi:%s'),
    'issue': (GITHUBBASE + '/issues/%s', 'issue #%s'),
    'pull': (GITHUBBASE + '/pulls/%s', 'PR #%s'),
}

# -- sphinx.ext.linkcode --------------------------------------------------
# linkcode to put links to the github repository from the documentation

# C++ domain does not populate signode['names']; we fill it on doctree-read.
# ids are a fallback (readable ``cyten::…`` entries, not ``_CPPv4…``).
add_linkcode_domain('cpp', ['names', '_toc_parts', 'ids'], override=True)

_DOXY_XML = str(_DOCS_DIR / 'build_docs' / 'doxy_xml')
_REPO_ROOT_STR = str(_REPO_ROOT)


def _github_ref():
    """GitHub ref for source links: a release tag, else the build commit."""
    if 'dev' not in cyten.__version__:
        return f'v{cyten.__version__}'
    # setuptools_scm writes ``g`` + git node (e.g. ``g2ce59cf1a``).
    commit_id = cyten._version.commit_id
    if not commit_id:
        return 'main'
    return commit_id[1:] if commit_id.startswith('g') else commit_id


def _blob_url(repo_relpath, linespec):
    """GitHub blob URL for a path relative to the repository root."""
    return f'{GITHUBBASE}/blob/{_github_ref()}/{repo_relpath}{linespec}'


def _strip_func_parens(name):
    """Drop the ``()`` Sphinx appends to C++ function TOC entries."""
    if name.endswith('()'):
        return name[:-2]
    return name


def _toc_as_qname(toc):
    """Normalize ``_toc_parts`` (tuple or string) to a C++ qualified name."""
    if not toc:
        return ''
    if isinstance(toc, (list, tuple)):
        parts = [str(p) for p in toc if p]
        if parts:
            parts[-1] = _strip_func_parens(parts[-1])
        return '::'.join(parts)
    return _strip_func_parens(str(toc))


def _qname_from_ids(ids):
    """Readable Sphinx C++ id, including overload suffix when present.

    ``cyten::sqrt__DiagonalTensorCPtr`` is kept intact so overloads can be
    distinguished. ``_CPPv4…`` mangled ids are ignored.
    """
    if isinstance(ids, str):
        ids = [ids]
    for id_ in reversed(list(ids or [])):
        if not isinstance(id_, str) or id_.startswith('_CPP') or '::' not in id_:
            continue
        return id_
    return None


def _cpp_qname(info):
    """Qualified C++ name from linkcode info (``cyten::sqrt__DiagonalTensorCPtr``)."""
    from_id = _qname_from_ids(info.get('ids'))
    if from_id and '__' in from_id:
        return from_id
    names = info.get('names') or []
    if isinstance(names, str):
        names = [names]
    for name in names:
        if isinstance(name, str) and '::' in name:
            return name
    if from_id:
        return from_id
    toc = _toc_as_qname(info.get('_toc_parts'))
    if '::' in toc:
        return toc
    for name in names:
        if isinstance(name, str) and name and not name.startswith('_CPP'):
            return name
    return toc or None


def _cpp_prepare_linkcode(app, doctree):
    """Set ``signode['names']`` to a fully qualified C++ name before linkcode runs.

    Sphinx's C++ domain leaves ``names`` empty. ``_toc_parts`` is a short name
    such as ``dagger()`` for free functions and methods; class signatures use
    ``cyten::Tensor``. Nested members are prefixed with the enclosing class.
    """
    del app  # unused; signature required by Sphinx
    for objnode in doctree.findall(addnodes.desc):
        if objnode.get('domain') != 'cpp':
            continue
        parent_qname = ''
        parent = objnode.parent
        while parent is not None:
            if (
                isinstance(parent, addnodes.desc)
                and parent.get('domain') == 'cpp'
                and parent.get('objtype') in ('class', 'struct')
            ):
                for sig in parent:
                    if isinstance(sig, addnodes.desc_signature):
                        parent_qname = _toc_as_qname(sig.get('_toc_parts')) or _qname_from_ids(sig.get('ids'))
                        break
                if parent_qname:
                    break
            parent = parent.parent

        objtype = objnode.get('objtype')
        for signode in objnode:
            if not isinstance(signode, addnodes.desc_signature):
                continue
            local = _toc_as_qname(signode.get('_toc_parts'))
            from_id = _qname_from_ids(signode.get('ids'))
            if from_id and '__' in from_id:
                qname = from_id
            elif '::' in local:
                qname = local
            elif parent_qname and local and objtype in ('function', 'type', 'enum', 'member', 'var'):
                qname = f'{parent_qname}::{local}'
            else:
                qname = from_id or local
            if qname:
                signode['names'] = [qname]


def _cpp_url_for_qname(qname, ids=None):
    """GitHub URL for a C++ qualified name, or None if unknown."""
    if not qname and not ids:
        return None
    import cpp_source_map

    mapping = cpp_source_map.get_cpp_source_map(_DOXY_XML, _REPO_ROOT_STR)
    loc = cpp_source_map.lookup_cpp_location(mapping, qname, ids)
    if loc is None:
        return None
    if loc.start and loc.end:
        linespec = f'#L{loc.start}-L{loc.end}'
    elif loc.start:
        linespec = f'#L{loc.start}'
    else:
        linespec = ''
    return _blob_url(loc.relpath, linespec)


def _python_source_url(obj):
    """URL for a pure-Python object whose source file lives in this repo.

    The generated autodoc stub ``cyten/_core.py`` is skipped so linkcode can
    fall back to the C++ GitHub location (same target as the ``[C++]`` badge).
    """
    try:
        obj = inspect.unwrap(obj)
    except Exception:
        pass
    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn or not str(fn).endswith('.py'):
        return None
    fn = os.path.abspath(fn)
    try:
        rel = os.path.relpath(fn, start=_REPO_ROOT_STR)
    except ValueError:
        return None
    rel = rel.replace('\\', '/')
    if rel.startswith('..'):
        return None
    # Generated stub for RTD — not a useful [source] target.
    if rel == 'cyten/_core.py':
        return None
    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None
        source = []
    if lineno:
        linespec = f'#L{lineno}-L{lineno + len(source) - 1}'
    else:
        linespec = ''
    return _blob_url(rel, linespec)


def _linkcode_resolve_py(info):
    """URL for a Python object, falling back to C++ source for pybind types."""
    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    obj = None
    if submod is not None:
        obj = submod
        for part in fullname.split('.'):
            try:
                obj = getattr(obj, part)
            except Exception:
                obj = None
                break

    if obj is not None:
        py_obj = obj
        if isinstance(obj, property):
            py_obj = obj.fget or obj
        elif isinstance(obj, (staticmethod, classmethod)):
            py_obj = getattr(obj, '__func__', obj)
        url = _python_source_url(py_obj)
        if url:
            return url

    import cyten_cpp_link

    qname = None
    if obj is not None:
        qname = cyten_cpp_link.cpp_ref_from_object(obj)
    if not qname:
        qname = cyten_cpp_link.lookup_cpp_ref(modname, fullname)
    return _cpp_url_for_qname(qname)


def _linkcode_resolve_cpp(info):
    """URL for a C++ class, function, or method from Doxygen XML locations."""
    return _cpp_url_for_qname(_cpp_qname(info), info.get('ids'))


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python or C++ object."""
    if domain == 'py':
        return _linkcode_resolve_py(info)
    if domain == 'cpp':
        return _linkcode_resolve_cpp(info)
    return None


def _cpp_inline_source_links(app, doctree):
    """Move C++ ``[source]`` onto the signature line, matching Python layout.

    The C++ domain always uses multiline signatures. The HTML writer emits the
    permalink then ``<br />`` at the end of each ``desc_signature_line``, while
    ``sphinx.ext.linkcode`` appends ``[source]`` to the parent ``desc_signature``.
    That puts ``[source]`` on the following line. Attach it to the declarator
    line instead so it appears before the permalink, as on Python pages.
    """
    del app  # unused; signature required by Sphinx
    for objnode in doctree.findall(addnodes.desc):
        if objnode.get('domain') != 'cpp':
            continue
        for signode in objnode:
            if not isinstance(signode, addnodes.desc_signature):
                continue
            sources = [child for child in list(signode) if isinstance(child, addnodes.only)]
            if not sources:
                continue
            lines = [child for child in signode if isinstance(child, addnodes.desc_signature_line)]
            target = next((line for line in reversed(lines) if line.get('add_permalink')), None)
            if target is None and lines:
                target = lines[-1]
            if target is None:
                continue
            for src in sources:
                signode.remove(src)
                target.append(src)


def setup(app):
    """Fill C++ signature names before ``sphinx.ext.linkcode`` (priority 500)."""
    app.connect('doctree-read', _cpp_prepare_linkcode, priority=400)
    # After linkcode (500): keep [source] on the signature line, not after <br />.
    app.connect('doctree-read', _cpp_inline_source_links, priority=501)
