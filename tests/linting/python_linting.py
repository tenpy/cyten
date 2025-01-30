"""Perform linting checks.

These are checks for coding guidelines, best practices etc.
The code may still run fine even if these checks fail.
We therefore consider them part of a linting routine and do *not* call them from pytest.
"""
# Copyright (C) TeNPy Developers, Apache license

import cyten
import types
import os


def main():
    """Called when this script is called, e.g. via `python linting.py`"""
    print('Checking __all__ attributes')
    check_all_attribute()
    print('Checking copyright notices')
    check_copyright_notice()
    print('Done')


# Exceptions: these do not need to be listed in __all__
_dunder_all_exceptions = {
    cyten.tensors: [cyten.tensors.T],
}


def check_all_attribute(check_module=cyten):
    """Recursively check the `__all__` attribute of a module.

    In each *.py file under cyten/, there should be an __all__, which fulfills::
        - Each entry should be valid (i.e. importable from that module).
        - Each object declared in that module should be listed (unless it starts with ``_``)
    """
    _name_ = check_module.__name__
    if not hasattr(check_module, '__all__'):
        raise AssertionError("module {0} has no line __all__ = [...]".format(_name_))
    _all_ = check_module.__all__
    exceptions = _dunder_all_exceptions.get(check_module, [])

    # print("test __all__ of", _name_)
    # find entries in __all__ but not in the module
    nonexistent = [n for n in _all_ if not hasattr(check_module, n)]
    if len(nonexistent) > 0:
        raise AssertionError("found entries {0!s} in __all__ but not in module {1}".format(
            nonexistent, _name_))

    # find objects in the module, which are not listed in __all__ (although they should be)
    missing_objects = []
    missing_modules = []
    for n in dir(check_module):
        if n[0] == '_' or n in _all_:  # private or listed in __all__
            continue
        obj = getattr(check_module, n)
        if obj in exceptions:
            continue
        if getattr(obj, "__module__", None) == _name_:
            missing_objects.append(obj)
            continue
        if hasattr(obj, "__package__") and obj.__name__.startswith(_name_):
            missing_modules.append(obj)
            continue
    if missing_objects or missing_modules:
        print()
        print(f'File "{check_module.__file__}:"')
        print(f'Missing objects (if any):')
        for obj in missing_objects:
            print(f'  {obj.__name__}')
        print(f'Missing modules (if any):')
        for mod in missing_modules:
            print(f'  {mod.__name__}')
        print()
        msg = f'In module {_name_}, the items listed before the stacktrace are missing from __all__'
        raise AssertionError(msg)

    # recurse into submodules
    submodules = [getattr(check_module, n, None) for n in _all_]
    for m in submodules:
        if isinstance(m, types.ModuleType) and m.__name__.startswith('cyten'):
            check_all_attribute(m)


def get_python_files(top):
    """return list of all python files in a directory. Recursive."""
    python_files = []
    for dirpath, dirnames, filenames in os.walk(top):
        if '__pycache__' in dirnames:
            del dirnames[dirnames.index('__pycache__')]
        for fn in filenames:
            if fn.endswith('.py') and fn != '_npc_helper.py':
                # exclude _npc_helper.py generated in the egg by ``python setup.py install``
                python_files.append(os.path.join(dirpath, fn))
    return python_files


def check_copyright_notice():
    expected_notice = '# Copyright (C) TeNPy Developers, Apache license'
    cyten_files = get_python_files(os.path.dirname(cyten.__file__))
    for fn in cyten_files:
        with open(fn, 'r') as f:
            for line in f:
                if line.startswith(expected_notice):
                    break
            else:  # no break
                raise AssertionError(f'No/wrong copyright notice in {fn}.')


if __name__ == '__main__':
    main()
