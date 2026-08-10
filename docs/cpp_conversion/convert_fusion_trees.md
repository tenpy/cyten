# Conversion of fusion_trees

## metadata

- original python name: `fusion_trees`
- original python file: `cyten/symmetries/trees.py`
- original python module: `cyten.symmetries.trees`
- declaration in C++ header file: `include/cyten/symmetries/trees.h` (same as FusionTree)
- definition in C++ file: `src/symmetries/trees.cpp`
- pybind11 binding: `pybind/symmetries/py_trees.cpp`
- first line of docstring: Iterable over all FusionTrees with given uncoupled and coupled sectors.

## Notes

- Depends on completed `FusionTree` C++ type.
- C++ does not inherit a Python ABC; expose `__iter__` / `__len__` / `index` via pybind11.
- `__iter__` materializes `all_trees()` into a Python list then iterates (tests are fine with this).
- Efficient `size()` / `index` / `compute_index` preserved.
- Monkey-patch keeps original Python class body; C++ imported at end of `trees.py`.

## TODO list for conversion

- [x] Wait until FusionTree conversion is green
- [x] generate / improve declaration (append to `trees.h`)
- [x] generate / improve definitions
- [x] pybind11 bindings (no trampoline)
- [x] monkey-patch `from .._core import fusion_trees` (Python body kept)
- [x] pytest `test_trees.py` (157 passed, 1 xfailed)
- [ ] remove original python class body (deferred)
- [ ] wrap up / suggest merge to `main_cpp`
