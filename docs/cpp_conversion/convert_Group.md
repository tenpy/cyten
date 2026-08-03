# Conversion of Group

## Status

**In progress.** Thin layer on C++ `SymmetryFactor`: forces bosonic braiding; overrides `swap_gate`, `qdim`, `batch_qdim`, `topological_twist`; pure-virtual `_fusion_tensor`. Python `AbelianGroup` / `SU2` / `SUN` will subclass via `PyGroup` trampoline.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `Group` |
| original python file | `cyten/symmetries/_symmetries.py` |
| original python module | `cyten.symmetries` |
| declaration | `include/cyten/symmetries/group.h` |
| definition | `src/symmetries/group.cpp` |
| pybind11 binding | `pybind/symmetries/py_group.cpp` |
| trampoline | `PyGroup` in `pybind/symmetries/py_trampolines.hpp` (required: AbelianGroup, SU2, SUN) |
| first line of docstring | Base-class for symmetries that are described by a group. |

## Design notes

- Inherits `SymmetryFactor`; ctor always passes `BraidingStyle::bosonic`.
- `num_sectors`: `float64` like other symmetry classes (`+inf` allowed).
- `descriptive_name`: `std::optional<std::string>`.
- Pure virtual `_fusion_tensor` (Python `@abstractmethod`); other abstracts remain on `SymmetryFactor` / `BaseSymmetry`.
- `swap_gate`: Kronecker of identity blocks `[b,a,b*,a*]`.
- `qdim` / `batch_qdim`: delegate to `sector_dim` / `batch_sector_dim`.
- `topological_twist`: always `1`.
- Next: `AbelianGroup`, then concretes (`NoSymmetry`, `U1`, `ZN`, …).

## Dependencies

- Done: `Sector`, styles, `BaseSymmetry`, `SymmetryFactor`, product `Symmetry`.
- Still Python: `AbelianGroup` and concretes (will hold via `PyGroup`).

## TODO checklist

- [x] initial setup (clean tree, list_python_names, pytest green, branch `convert_Group`)
- [x] planning (this file)
- [ ] generate declaration draft
- [ ] improve declaration
- [ ] generate definitions
- [ ] improve definitions; compile + ctest
- [ ] pybind11 bindings + trampoline
- [ ] monkey-patch `from .._core import Group`
- [ ] pytest `test_symmetries.py`
- [ ] remove Python `Group` class body
- [ ] wrap up / suggest merge
