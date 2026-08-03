# Conversion of BaseSymmetry

## Status

**C++ declaration + definitions + bindings/trampoline done.** Not monkey-patched into Python yet: replacing Python `BaseSymmetry` while `Symmetry` / `SymmetryFactor` remain Python subclasses segfaults / fails inheritance. Keep Python `BaseSymmetry` until those subclasses are converted (or trampoline inheritance is fixed). C++ type is still exported as `cyten._core.BaseSymmetry` for further Layer 2 work.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `BaseSymmetry` |
| original python file | `cyten/symmetries/_symmetries.py` |
| original python module | `cyten.symmetries` |
| declaration | `include/cyten/symmetries/base_symmetry.h` |
| definition | `src/symmetries/base_symmetry.cpp` |
| pybind11 binding | `pybind/symmetries/py_base_symmetry.cpp` |
| trampoline | yes (`SymmetryFactor` / `Symmetry` / concretes remain Python until converted) |
| first line of docstring | Common method implementations for both SymmetryFactor and Symmetry. |

## Design notes

- Non-templated; uses owning `Sector` / `SectorArray` (see [convert_Sector.md](convert_Sector.md)).
- `num_sectors` is `float64` so `inf` works like Python.
- Declare pure virtual `is_valid_sector` / `fusion_outcomes` on the C++ base (Python only defines them on subclasses, but BaseSymmetry methods call them).
- `as_Symmetry()` → `std::shared_ptr<Symmetry>` with forward declaration of `Symmetry`.
- Topological arrays stay `py::array` for now (F/R/… symbols).
- `enable_shared_from_this` + `py::smart_holder` like BlockBackend.
- Factor helpers may use `std::span<const int16_t, N>` internally later; public API stays `Sector`.
- Fix Python bug in `sector_dim`: call `qdim(a)` not `qdim()`.

## TODO checklist

- [x] initial setup / planning
- [x] improve and fix the declaration draft
- [x] generate C++ definitions
- [x] improve definition drafts; compile + sector CTest
- [x] pybind11 bindings + trampoline (exported from `_core`)
- [ ] monkey-patch; pytest — **blocked** until SymmetryFactor/Symmetry converted (Python subclass of C++ BaseSymmetry crashes)
- [ ] remove original Python `BaseSymmetry` once green
- [ ] wrap up / suggest merge

Interim: `as_Symmetry()` returns `py::object` until C++ `Symmetry` exists.