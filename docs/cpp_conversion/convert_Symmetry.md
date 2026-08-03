# Conversion of Symmetry

## Status

**Planning / declaration.** Product symmetry on top of C++ `BaseSymmetry` + `SymmetryFactor` (already monkey-patched). Python `Group` / concretes remain Python subclasses of C++ `SymmetryFactor` and can be held as `SymmetryFactor::Ptr` via trampolines.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `Symmetry` |
| original python file | `cyten/symmetries/_symmetries.py` |
| original python module | `cyten.symmetries` |
| declaration | `include/cyten/symmetries/symmetry.h` |
| definition | `src/symmetries/symmetry.cpp` |
| pybind11 binding | `pybind/symmetries/py_symmetry.cpp` |
| trampoline | `PySymmetry` in `pybind/symmetries/py_trampolines.hpp` (only if Python subclasses Symmetry; currently none do — optional) |
| first line of docstring | Describes a symmetry of a space or tensor. |

## Design notes

- Inherits `BaseSymmetry` (not `SymmetryFactor`). Holds `std::vector<SymmetryFactor::Ptr> factors`.
- `sector_slices`: length `num_factors + 1` cumulative offsets into product sectors (`std::vector<std::uint8_t>` or `int`); Python exposes 1D ndarray.
- `fusion_tensor_dtype`: `std::optional<Dtype>` like factors (`Dtype::common` when all defined).
- Constructor flattens nested `Symmetry` factors; warns on multiple fermionic factors.
- Implements / overrides product logic: `is_valid_sector`, `fusion_outcomes`, Kronecker-style `_f_symbol` / `_r_symbol` / fusion tensors, `is_equivalent_to`, `__mul__`, etc.
- After conversion, update `SymmetryFactor::as_Symmetry()` to build C++ `Symmetry` instead of importing Python.
- Empty-factor `Symmetry([])`: Python `max()` on empty factors would fail; keep defensive handling if needed for HDF5 / edge cases.
- No Python subclasses of `Symmetry` today → trampoline optional; still useful if tests monkey-patch. Prefer trampoline for consistency with `BaseSymmetry`.

## Dependencies

- Done: `Sector`, `SectorArray`, styles, exceptions, `BaseSymmetry`, `SymmetryFactor`, `Dtype`.
- Still Python: `Group`, `AbelianGroup`, concretes (held via `PySymmetryFactor`).

## TODO checklist

- [x] initial setup (clean tree, list_python_names, pytest green)
- [ ] planning (this file)
- [ ] generate declaration draft
- [ ] improve declaration
- [ ] generate definitions
- [ ] improve definitions; compile + ctest
- [ ] pybind11 bindings (+ trampoline if needed)
- [ ] monkey-patch `from .._core import Symmetry`; update `as_Symmetry`
- [ ] pytest `test_symmetries.py` then broader suite
- [ ] remove Python `Symmetry` class body
- [ ] wrap up / suggest merge to `main_cpp`
