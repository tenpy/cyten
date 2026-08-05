# Conversion of FusionTree

## metadata

- original python name: `FusionTree`
- original python file: `cyten/symmetries/trees.py`
- original python module: `cyten.symmetries.trees`
- declaration in C++ header file: `include/cyten/symmetries/trees.h`
- definition in C++ file: `src/symmetries/trees.cpp`
- pybind11 binding: `pybind/symmetries/py_trees.cpp`
- first line of docstring: A fusion tree, which represents the map from uncoupled to coupled sectors.

## Module context (`trees.py`)

Convert objects in dependency order:

1. **`_concat_sector_arrays`** — private helper; **do not bind**. Use existing C++ `concat_sector_arrays` from `sector_ops.h` (loop / small variadic helper in `trees.cpp` if needed).
2. **`FusionTree`** — this file (large; ~1k lines of methods).
3. **`fusion_trees`** — see [convert_fusion_trees.md](convert_fusion_trees.md); depends on `FusionTree`.

## Design notes

- Hold `Symmetry::Ptr symmetry` (same pattern as other Layer 2 types).
- Store `SectorArray uncoupled`, `Sector coupled`, `std::vector<bool> are_dual` (or `std::vector<char>` if mutability/indexing of `vector<bool>` is painful), `SectorArray inner_sectors`, `std::vector<int64> multiplicities`.
- Cache `num_uncoupled`, `num_vertices`, `num_inner_edges`, and style flags (`fusion_style`, `is_abelian`, `braiding_style`) like Python.
- Linear-combination returns (`braid`, `bend_leg`, `insert_at`, `outer`, `twist`) → `std::map` / unordered_map with `FusionTree` keys and `complex128` (or `std::complex<double>`) coeffs. `bend_leg` uses pair keys `(Y_i, X_i)`.
- `to_dense_block`: `TensorBackend` is still Python (Layer 3). Accept optional `BlockBackend*` / `py::object` for backend; default to `NumpyBlockBackend::from_factory("cpu")`. Return `BlockBackend::BlockPtr`.
- No trampoline: no Python subclasses of `FusionTree` in the library.
- ASCII helpers (`_ascii_diagram`, `ascii_diagram`, `__str__`) can stay; use `std::vector<std::string>` or a 2D char buffer instead of NumPy string arrays where practical.

## Dependencies (already in C++)

- `Sector` / `SectorArray`, `sector_ops` (`concat_sector_arrays`, `rows_equal`, `sector_array_from_sector`, …)
- `Symmetry` / `BaseSymmetry` (F/R/C/B symbols, fusion tensors, qdims, …)
- `BlockBackend` / `NumpyBlockBackend`, `Dtype`
- `to_valid_idx` in `tools.h`
- `SymmetryError`, styles

## TODO list for conversion

- [x] initial setup (clean tree, `list_python_names`, pytest `test_trees.py` green: 157 passed, 1 xfailed; branch `convert_trees`)
- [x] planning (this file)
- [ ] generate the declaration draft (`gen_cpp_declaration`)
- [ ] improve and fix the declaration draft (namespace, types, C++23 / pre-commit)
- [ ] generate the C++ definitions (`gen_cpp_definition`; add to `src/CMakeLists.txt`)
- [ ] improve and fix the definition drafts (CHECKME/FIXME; compile + ctest)
- [ ] generate pybind11 bindings (`gen_pyb11_binding`; register in `py_symmetries.cpp` / `pybind/CMakeLists.txt`)
- [ ] generate pybind11 trampoline — **skip** (no subclasses)
- [ ] monkey-patch `from .._core import FusionTree` in `trees.py`
- [ ] run python tests (`test_trees.py`, then broader)
- [ ] remove original python `FusionTree` class body (keep import)
- [ ] wrap up (then convert `fusion_trees`)
