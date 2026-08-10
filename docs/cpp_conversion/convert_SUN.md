# Conversion of SUN

## Status

**Done for monkey-patch.** C++ `SUN` + bindings; imported from `_core`. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped (`test_suN_symmetry` skips without local HDF5 data files).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `SUN` |
| original python file | `cyten/symmetries/_symmetries.py` |
| declaration | `include/cyten/symmetries/factors/sun.h` |
| definition | `src/symmetries/factors/sun.cpp` |
| pybind11 binding | `pybind/symmetries/factors/py_sun.cpp` |
| trampoline | none |
| first line of docstring | SU(N) group symmetry |

## Design notes

- Subclass C++ `Group` with `FusionStyle::general`.
- `CGfile` / `Ffile` / `Rfile` held as `py::object` (`h5py.File`); symbol lookups use HDF5 via the Python API from C++.
- Pure C++ for GT validation, `sector_dim`, `dual_sector`, `S_index_irrep_weight`, `highest_irrep_in_decomp`.
- Extra helpers bound: `clebschgordan`, `_f_symbol_from_CG`, `_r_symbol_from_CG`, `dims_of_irreps`, etc.

## TODO checklist

- [x] setup / plan / declaration / definitions / bindings / monkey-patch / pytest
- [ ] wrap up / merge or continue (`FermionNumber`, `FermionParity`, anyons, …)
