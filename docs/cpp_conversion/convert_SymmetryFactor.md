# Conversion of SymmetryFactor

## Status

**Done for monkey-patch.** C++ `SymmetryFactor` + trampoline; `BaseSymmetry` and `SymmetryFactor` imported from `_core` in `_symmetries.py`. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped.

Python `Group` / concretes / product `Symmetry` still subclass the C++ bases. Next Layer 2 step: convert `Group` → `AbelianGroup` → concretes → product `Symmetry`.

### Pitfalls fixed during monkey-patch

- `py::reinterpret_steal` on owning `py::object` temporaries → use `.cast<py::array>()`.
- Do not trampoline methods bound as `def_property_readonly` (`is_abelian`, …) — `PYBIND11_OVERRIDE` then raises `TypeError: bool is not an instance of function`.
- `super()._c_symbol` from a Python override re-enters the trampoline; use module helper `_default_c_symbol` instead.
- Finite `num_sectors` must cast to Python `int` (stored as `float64` in C++).
- `fusion_outcomes_broadcast` must throw `AssertionError`, not C `assert` (Debug abort).

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
- [x] declaration draft + improve
- [x] definitions + compile
- [x] bindings + trampoline
- [x] monkey-patch BaseSymmetry + SymmetryFactor; pytest
- [ ] remove Python SymmetryFactor (and BaseSymmetry) leftovers when hierarchy is fully C++
