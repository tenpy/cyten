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
- Python subclasses `Iterable[FusionTree]`; C++ need not inherit a Python ABC — expose `__iter__` / `__len__` / `index` via pybind11.
- Efficient `__len__` and `index` / `_compute_index` must be preserved (tests rely on them).
- Convert **after** `FusionTree` is monkey-patched and `test_trees.py` passes for FusionTree methods.

## TODO list for conversion

- [ ] Wait until FusionTree conversion is green
- [ ] generate / improve declaration (append to `trees.h`)
- [ ] generate / improve definitions
- [ ] pybind11 bindings (no trampoline)
- [ ] monkey-patch `from .._core import fusion_trees`
- [ ] pytest `test_trees.py` + full suite as needed
- [ ] remove original python class body
- [ ] wrap up / suggest merge to `main_cpp`
