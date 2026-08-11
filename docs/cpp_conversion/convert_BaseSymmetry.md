# Conversion of BaseSymmetry

## Status

**Done / monkey-patched.** C++ `BaseSymmetry` + trampoline; imported from `_core` in `_symmetries.py` together with `SymmetryFactor` / `Symmetry` and concretes. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped.

See also [convert_SymmetryFactor.md](convert_SymmetryFactor.md), [convert_Symmetry.md](convert_Symmetry.md).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `BaseSymmetry` |
| original python file | `cyten/symmetries/_symmetries.py` |
| original python module | `cyten.symmetries` |
| declaration | `include/cyten/symmetries/base_symmetry.h` |
| definition | `src/symmetries/base_symmetry.cpp` |
| pybind11 binding | `pybind/symmetries/py_base_symmetry.cpp` |
| trampoline | yes (`PyBaseSymmetry`) |
| first line of docstring | Common method implementations for both SymmetryFactor and Symmetry. |

## Design notes

- Non-templated; uses owning `Sector` / `SectorArray` (see [convert_Sector.md](convert_Sector.md)).
- `num_sectors` is `float64` so `inf` works like Python.
- Declare pure virtual `is_valid_sector` / `fusion_outcomes` on the C++ base (Python only defines them on subclasses, but BaseSymmetry methods call them).
- `as_Symmetry()` → `std::shared_ptr<Symmetry>`.
- Topological arrays stay `py::array` for now (F/R/… symbols).
- `enable_shared_from_this` + `py::smart_holder` like BlockBackend.
- Factor helpers may use `std::span<const int16_t, N>` internally later; public API stays `Sector`.
- Fix Python bug in `sector_dim`: call `qdim(a)` not `qdim()`.
- Monkey-patch **BaseSymmetry** together with subclasses (`SymmetryFactor` / `Symmetry` / Group hierarchy); replacing only the base while Python subclasses remained caused inheritance crashes.

## TODO checklist

- [x] initial setup / planning
- [x] improve and fix the declaration draft
- [x] generate C++ definitions
- [x] improve definition drafts; compile + sector CTest
- [x] pybind11 bindings + trampoline (exported from `_core`)
- [x] monkey-patch with SymmetryFactor/Symmetry; pytest
- [x] remove original Python `BaseSymmetry` (imported from `_core`)
