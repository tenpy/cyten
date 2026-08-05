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
- Store `SectorArray uncoupled`, `Sector coupled`, `std::vector<uint8_t> are_dual`, `SectorArray inner_sectors`, `std::vector<int64> multiplicities`.
- Cache `num_uncoupled`, `num_vertices`, `num_inner_edges`, and style flags (`fusion_style`, `is_abelian`, `braiding_style`) like Python.
- Linear-combination returns → `std::map` with `FusionTree` keys and `complex128` coeffs. `bend_leg` uses pair keys `(Y_i, X_i)`.
- `to_dense_block`: optional `BlockBackend*`; Python `TensorBackend` resolved in bindings via `.block_backend`.
- No trampoline: no Python subclasses of `FusionTree` in the library.
- ASCII grid stores one Unicode codepoint (`std::string`) per cell (UTF-8 box-drawing).
- Bindings accept SymmetryFactor via `as_Symmetry()`, and SectorArray from lists / empty sequences / ndarrays.
- Monkey-patch keeps original Python class body in `trees.py`; C++ is imported below it.

## Dependencies (already in C++)

- `Sector` / `SectorArray`, `sector_ops`, `Symmetry` / `BaseSymmetry`, `BlockBackend` / `NumpyBlockBackend`, `Dtype`, `to_valid_idx`, `SymmetryError`, styles

## TODO list for conversion

- [x] initial setup (clean tree, `list_python_names`, pytest `test_trees.py` green; branch `convert_trees`)
- [x] planning (this file)
- [x] generate the declaration draft (`gen_cpp_declaration`)
- [x] improve and fix the declaration draft (namespace, types, C++23 / pre-commit)
- [x] generate the C++ definitions (`gen_cpp_definition`; add to `src/CMakeLists.txt`)
- [x] improve and fix the definition drafts (CHECKME/FIXME; compile)
- [x] generate pybind11 bindings (`gen_pyb11_binding`; register in `py_symmetries.cpp` / `pybind/CMakeLists.txt`)
- [x] generate pybind11 trampoline — **skipped** (no subclasses)
- [x] monkey-patch `from .._core import FusionTree` in `trees.py` (**original Python class body kept**)
- [x] run python tests (`test_trees.py`: 157 passed, 1 xfailed)
- [ ] remove original python `FusionTree` class body (deferred per user request)
- [ ] wrap up (then convert `fusion_trees`)
