# Conversion of NoSymmetry (with U1, ZN)

## Status

**Done for monkey-patch.** C++ `NoSymmetry`, `U1`, `ZN` + bindings; imported from `_core`. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped.

Hand-written after codegen declaration drafts (types were placeholders).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `NoSymmetry` (+ `U1`, `ZN`) |
| original python file | `cyten/symmetries/_symmetries.py` |
| declaration | `include/cyten/symmetries/{no_symmetry,u1,zn}.h` |
| definition | `src/symmetries/{no_symmetry,u1,zn}.cpp` |
| pybind11 binding | `pybind/symmetries/py_{no_symmetry,u1,zn}.cpp` |
| trampoline | none (no further Python subclasses in tree) |
| first line of docstring | Trivial symmetry group that doesn't do anything. |

## Design notes

- Subclass C++ `AbelianGroup`; implement `repr` / `_is_equivalent_factor` plus sector/fusion overrides.
- Leaf bindings: `py::class_<…, AbelianGroup, py::smart_holder>` (no trampoline).
- `ZN::are_valid_sectors`: Python has a bug (`np.all(0 < self.N)`); C++ uses `0 <= sectors < N`.
- Modular arithmetic for `ZN` matches Python (non-negative remainder).
- Keep Python `__repr__` strings: `NoSymmetry()`, `U1Symmetry(...)`, `ZNSymmetry(N...)`.
- `is_valid_sector` / `are_valid_sectors` bindings (BaseSymmetry / SymmetryFactor / Symmetry): require a NumPy ndarray and return `False` on cast failure (lists/scalars are not sectors).

## TODO checklist

- [x] setup / plan / declaration / definitions / bindings / monkey-patch / pytest
- [ ] wrap up / merge or continue with next concretes (`SU2`, …)
