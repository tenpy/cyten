"""Make Sphinx inheritance diagrams work for pybind11 types re-exported from ``cyten._core``.

TeNPy's autosummary templates use ``.. inheritance-diagram:: <module>``. That only
picks up classes whose ``__module__`` matches the documented module. Cyten classes
are bound in ``cyten._core`` and imported into public modules, so the stock
directive would draw empty graphs.

This extension:

* Includes ``cyten._core`` classes that a module actually imports (or re-exports
  from a private ``_foo`` submodule).
* Maps ``cyten._core.Name`` to the public name used in autodoc so nodes link.
* Skips pybind/enum/typing noise in the graph.
* Adds ``cyten-inheritance-diagram``, which defaults to the ``autoclass`` /
  ``autoexception`` / ``automodule`` targets in the current page (``:parts: 1``,
  like TeNPy).
"""

from __future__ import annotations

import inspect
import re
import sys
from typing import Any

from sphinx.application import Sphinx
from sphinx.ext.inheritance_diagram import (
    PY_BUILTINS,
    InheritanceDiagram,
    InheritanceException,
    InheritanceGraph,
    try_import,
)

_CORE = 'cyten._core'

_SKIP_ROOT_MODULES = frozenset(
    {
        'abc',
        'builtins',
        '__builtin__',
        'enum',
        'pybind11',
        'pybind11_builtins',
        'typing',
        'typing_extensions',
    }
)
_SKIP_NAMES = frozenset({'Protocol', 'Generic', 'instance', 'pybind11_object'})

_AUTO_CLASS_RE = re.compile(
    r'^\.\.\s+auto(?:class|exception)::\s+([A-Za-z_][\w.]*)\s*$',
    re.M,
)
_AUTO_MODULE_RE = re.compile(
    r'^\.\.\s+automodule::\s+([A-Za-z_][\w.]*)\s*$',
    re.M,
)
_CORE_IMPORT_RE = re.compile(
    r'from\s+[\w.]*_core\s+import\s+(\([^)]*\)|[A-Za-z_][\w,\s\\]*)',
    re.S,
)


def _skip_class(cls: type) -> bool:
    module = getattr(cls, '__module__', '') or ''
    root = module.split('.', 1)[0]
    if root in _SKIP_ROOT_MODULES:
        return True
    return cls.__name__ in _SKIP_NAMES


def _identifier_names(chunk: str) -> list[str]:
    names: list[str] = []
    stripped = '\n'.join(line.split('#', 1)[0] for line in chunk.splitlines())
    for raw in stripped.replace('(', ' ').replace(')', ' ').replace('\\', ' ').split(','):
        part = raw.strip()
        if not part:
            continue
        if ' as ' in part:
            part = part.split(' as ')[-1].strip()
        if part.isidentifier():
            names.append(part)
    return names


def _names_imported_from_core(mod: Any) -> set[str]:
    try:
        src = inspect.getsource(mod)
    except (OSError, TypeError):
        return set()
    names: set[str] = set()
    for match in _CORE_IMPORT_RE.finditer(src):
        names.update(_identifier_names(match.group(1)))
    return names


def _is_package(mod: Any) -> bool:
    return inspect.ismodule(mod) and hasattr(mod, '__path__')


def _public_submodule_core_ids(mod: Any) -> set[int]:
    ids: set[int] = set()
    if not _is_package(mod):
        return ids
    prefix = mod.__name__ + '.'
    for name, sub in list(vars(mod).items()):
        if name.startswith('_') or not inspect.ismodule(sub):
            continue
        if not (sub.__name__ or '').startswith(prefix):
            continue
        for obj in vars(sub).values():
            if inspect.isclass(obj) and getattr(obj, '__module__', '') == _CORE:
                ids.add(id(obj))
    return ids


def _private_submodule_exports(mod: Any) -> list[type]:
    """Classes re-exported from ``_foo`` submodules (e.g. ``_symmetries``)."""
    if not _is_package(mod):
        return []
    prefix = mod.__name__ + '.'
    private_ids: set[int] = set()
    for name, sub in list(vars(mod).items()):
        if not name.startswith('_') or not inspect.ismodule(sub):
            continue
        if not (sub.__name__ or '').startswith(prefix):
            continue
        for obj in vars(sub).values():
            if inspect.isclass(obj) and getattr(obj, '__module__', '') == _CORE:
                private_ids.add(id(obj))
    return [
        cls
        for name, cls in vars(mod).items()
        if inspect.isclass(cls) and not name.startswith('_') and id(cls) in private_ids
    ]


def classes_in_module(mod: Any, *, imported_members: bool = False) -> list[type]:
    """Classes that belong on an inheritance diagram for *mod*."""
    found: list[type] = []
    seen: set[int] = set()

    def add(cls: type) -> None:
        if id(cls) in seen or _skip_class(cls):
            return
        seen.add(id(cls))
        found.append(cls)

    core_imported = _names_imported_from_core(mod)
    skip_from_public_subs = set() if imported_members else _public_submodule_core_ids(mod)

    for name, obj in list(vars(mod).items()):
        if not inspect.isclass(obj) or name.startswith('_'):
            continue
        module = getattr(obj, '__module__', '')
        if module == mod.__name__:
            add(obj)
        elif module == _CORE and (imported_members or name in core_imported):
            if id(obj) not in skip_from_public_subs:
                add(obj)
        elif imported_members and isinstance(module, str) and module.startswith('cyten.'):
            add(obj)

    if not imported_members:
        for cls in _private_submodule_exports(mod):
            add(cls)

    return found


def import_classes(name: str, currmodule: str) -> list[type[Any]]:
    """Like Sphinx's importer, but includes pybind classes re-exported from ``_core``."""
    target = try_import(f'{currmodule}.{name}') if currmodule else None
    if target is None:
        target = try_import(name)
    if target is None:
        raise InheritanceException(f'Could not import class or module {name!r} specified for inheritance diagram')
    if inspect.isclass(target):
        return [target]
    if inspect.ismodule(target):
        return classes_in_module(target)
    raise InheritanceException(f'{name!r} specified for inheritance diagram is not a class or module')


def _patched_class_info(self, classes, show_builtins, private_bases, parts, aliases, top_classes):
    """Like ``InheritanceGraph._class_info``, but skips pybind/enum ancestors."""
    all_classes: dict[type, tuple] = {}

    def recurse(cls: type) -> None:
        if _skip_class(cls):
            return
        if not show_builtins and cls in PY_BUILTINS:
            return
        if not private_bases and cls.__name__.startswith('_'):
            return

        nodename = self.class_name(cls, parts, aliases)
        fullname = self.class_name(cls, 0, aliases)
        tooltip = None
        try:
            if cls.__doc__:
                doc = cls.__doc__.strip().split('\n')[0]
                if doc:
                    tooltip = f'"{doc.replace('"', '\\"')}"'
        except Exception:
            pass

        baselist: list[str] = []
        all_classes[cls] = (nodename, fullname, baselist, tooltip)
        if fullname in top_classes:
            return
        for base in cls.__bases__:
            if _skip_class(base):
                continue
            if not show_builtins and base in PY_BUILTINS:
                continue
            if not private_bases and base.__name__.startswith('_'):
                continue
            baselist.append(self.class_name(base, parts, aliases))
            if base not in all_classes:
                recurse(base)

    for cls in classes:
        recurse(cls)

    return [
        (cls_name, fullname, tuple(bases), tooltip) for (cls_name, fullname, bases, tooltip) in all_classes.values()
    ]


def _cyten_modules() -> list[Any]:
    modules = []
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name == 'cyten' or name.startswith('cyten.'):
            modules.append(module)
    return modules


def build_inheritance_alias() -> dict[str, str]:
    """Map ``cyten._core.Class`` to the public name autodoc uses."""
    import cyten

    try:
        import cyten._core as core
    except ImportError:
        return {}

    documented: dict[str, str] = {}
    from pathlib import Path

    docs_python = Path(__file__).resolve().parent.parent / 'python'
    if docs_python.is_dir():
        for rst in docs_python.rglob('*.rst'):
            text = rst.read_text(encoding='utf-8')
            for full in _AUTO_CLASS_RE.findall(text):
                documented[full.rsplit('.', 1)[-1]] = full
            for modname in _AUTO_MODULE_RE.findall(text):
                try:
                    mod = __import__(modname, fromlist=['*'])
                except Exception:
                    continue
                for attr, obj in vars(mod).items():
                    if inspect.isclass(obj) and getattr(obj, '__module__', '') == _CORE:
                        documented.setdefault(attr, f'{modname}.{attr}')

    alias: dict[str, str] = {}
    for name, obj in vars(core).items():
        if not inspect.isclass(obj):
            continue
        qual = f'{_CORE}.{obj.__qualname__}'
        if name in documented:
            alias[qual] = documented[name]
            continue
        public = None
        for module in _cyten_modules():
            modname = getattr(module, '__name__', '') or ''
            if not modname.startswith('cyten') or modname == _CORE:
                continue
            if any(part.startswith('_') for part in modname.split('.')[1:]):
                continue
            if getattr(module, name, None) is obj:
                if public is None or modname.count('.') > public.count('.'):
                    public = modname
        if public:
            alias[qual] = f'{public}.{name}'
        elif getattr(cyten, name, None) is obj:
            alias[qual] = f'cyten.{name}'
    return alias


def _collect_from_rst(path: str) -> list[str]:
    try:
        text = open(path, encoding='utf-8').read()
    except OSError:
        return []
    names = list(_AUTO_CLASS_RE.findall(text))
    for modname in _AUTO_MODULE_RE.findall(text):
        try:
            mod = __import__(modname, fromlist=['*'])
        except Exception:
            continue
        for cls in classes_in_module(mod, imported_members=True):
            attr = next((key for key, val in vars(mod).items() if val is cls), None)
            names.append(f'{modname}.{attr}' if attr else f'{cls.__module__}.{cls.__qualname__}')
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


class CytenInheritanceDiagram(InheritanceDiagram):
    """``inheritance-diagram`` that defaults to classes documented on this page."""

    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True

    def run(self):
        if 'parts' not in self.options:
            self.options['parts'] = 1
        if not self.arguments or not str(self.arguments[0]).strip():
            source, _lineno = self.state_machine.get_source_and_line(self.lineno)
            names = _collect_from_rst(source) if source else []
            if not names:
                return [
                    self.state.document.reporter.warning(
                        'cyten-inheritance-diagram: no autoclass/automodule targets found',
                        line=self.lineno,
                    )
                ]
            self.arguments = [' '.join(names)]
        return super().run()


def setup(app: Sphinx) -> dict[str, Any]:
    """Patch stock inheritance diagrams and register ``cyten-inheritance-diagram``."""
    import sphinx.ext.inheritance_diagram as ext

    ext.import_classes = import_classes
    InheritanceGraph._class_info = _patched_class_info  # type: ignore[method-assign]

    app.add_directive('cyten-inheritance-diagram', CytenInheritanceDiagram)

    def _set_alias(app_):
        alias = build_inheritance_alias()
        if not app_.config.inheritance_alias:
            app_.config.inheritance_alias = alias
        else:
            merged = dict(alias)
            merged.update(app_.config.inheritance_alias)
            app_.config.inheritance_alias = merged

    app.connect('builder-inited', _set_alias)
    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
