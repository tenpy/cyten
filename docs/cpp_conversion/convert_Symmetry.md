# Conversion of Symmetry

## Status

**Done for monkey-patch.** C++ product `Symmetry` + pybind bindings; imported from `_core` in `_symmetries.py`. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped. No trampoline (no Python subclasses of `Symmetry`).

Codegen `gen_cpp_declaration --py-name Symmetry` fails with `KeyError: 'BaseSymmetry'` (base already imported from `_core`); declaration/definitions/bindings were hand-written.

### Pitfalls fixed during monkey-patch

- `SymmetryFactor::as_Symmetry` / `__mul__` must not use `shared_from_this()` with smart_holder trampolines (`bad_weak_ptr`). Bindings take `py::object self` and `cast<SymmetryFactor::Ptr>()`.
- `libcyten` must not `py::cast` `Sector` / `SectorArray` without type casters — use `sector_numpy.h` helpers (`sector_to_numpy` / `sector_array_from_numpy`).
- `is_valid_sector` / `are_valid_sectors` bindings: invalid Python types → `False` (match Python), not a cast `TypeError`.
- Product `fusion_tensor_dtype`: read each factor’s dtype via Python `attr` (class attributes on subclasses); C++ `optional` member may be empty.
- Trampoline for `_multiple_fusion_broadcast`: Python overrides take `*sectors`; unpack `std::vector<SectorArray>` as `*args` (do not pass one list).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `Symmetry` |
| original python file | `cyten/symmetries/_symmetries.py` |
| original python module | `cyten.symmetries` |
| declaration | `include/cyten/symmetries/symmetry.h` |
| definition | `src/symmetries/symmetry.cpp` |
| pybind11 binding | `pybind/symmetries/py_symmetry.cpp` |
| trampoline | skipped (no Python subclasses) |
| first line of docstring | Describes a symmetry of a space or tensor. |

## Design notes

- Inherits `BaseSymmetry` (not `SymmetryFactor`). Holds `std::vector<SymmetryFactor::Ptr> factors`.
- `sector_slices`: `std::vector<std::uint8_t>`; Python exposes int64 ndarray.
- `fusion_tensor_dtype`: `std::optional<Dtype>` (`Dtype::common` when all factors define one).
- Constructor / pybind init flatten nested `Symmetry` factors; warn on multiple fermionic factors (group_name heuristic).
- `from_hdf5` remains a thin Python `classmethod` attached after import; C++ has `save_hdf5`.
- `SymmetryFactor::as_Symmetry` / `mul` build C++ `Symmetry`.

## Dependencies

- Done: `Sector`, `SectorArray`, styles, exceptions, `BaseSymmetry`, `SymmetryFactor`, `Dtype`, product `Symmetry`.
- Still Python: `Group` → `AbelianGroup` → concretes / anyons (held via `PySymmetryFactor`).

## TODO checklist

- [x] initial setup (clean tree, list_python_names, pytest green)
- [x] planning (this file)
- [x] declaration draft (hand-written; codegen KeyError)
- [x] improve declaration
- [x] definitions + compile
- [x] improve definitions
- [x] pybind11 bindings (no trampoline)
- [x] monkey-patch `from .._core import Symmetry`; update `as_Symmetry`
- [x] pytest `test_symmetries.py`
- [x] remove Python `Symmetry` class body
- [ ] broader suite / wrap up / suggest merge to `main_cpp`
