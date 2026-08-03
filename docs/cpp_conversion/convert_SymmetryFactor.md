# Conversion of SymmetryFactor

## Status

**In progress.** Builds on C++ `BaseSymmetry` ([convert_BaseSymmetry.md](convert_BaseSymmetry.md)). Goal: C++ `SymmetryFactor` + trampoline so Python `Group` / anyons can subclass; then monkey-patch `BaseSymmetry` and `SymmetryFactor` together.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `SymmetryFactor` |
| original python file | `cyten/symmetries/_symmetries.py` |
| declaration | `include/cyten/symmetries/symmetry_factor.h` |
| definition | `src/symmetries/symmetry_factor.cpp` |
| pybind11 binding | `pybind/symmetries/py_symmetry_factor.cpp` |
| trampoline | `PySymmetryFactor` in `pybind/symmetries/py_trampolines.hpp` |
| first line of docstring | Base class for symmetries that impose a block-structure on tensors |

## Design notes

- Inherits `BaseSymmetry`; adds `group_name`, `descriptive_name`, `fusion_tensor_dtype` (`std::optional<Dtype>`).
- Pure virtual: `__repr__` → `std::string repr() const`, `_is_equivalent_factor`, plus inherited abstracts (`dual_sector`, …).
- `as_Symmetry()` builds Python `Symmetry([self])` via pybind until product `Symmetry` is converted.
- `__mul__` / `__eq__` / `__str__` / HDF5 bound on the Python side of the class binding.
- Monkey-patch **both** `BaseSymmetry` and `SymmetryFactor` in one step; if Python `Group(SymmetryFactor)` still crashes, convert `Group`/`AbelianGroup` next before concretes.

## TODO checklist

- [x] initial setup / planning
- [ ] declaration draft + improve
- [ ] definitions + compile
- [ ] bindings + trampoline
- [ ] monkey-patch BaseSymmetry + SymmetryFactor; pytest
- [ ] remove Python SymmetryFactor (and BaseSymmetry) when green
