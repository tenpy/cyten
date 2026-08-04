# Conversion of SU2

## Status

In progress on branch `convert_SU2`.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `SU2` |
| original python file | `cyten/symmetries/_symmetries.py` |
| declaration | `include/cyten/symmetries/su2.h` |
| definition | `src/symmetries/su2.cpp` |
| pybind11 binding | `pybind/symmetries/py_su2.cpp` |
| trampoline | none (no Python subclasses of `SU2` in tree) |
| first line of docstring | SU(2) symmetry. |

## Design notes

- Subclass C++ `Group` with `FusionStyle::multiple_unique`, infinite sectors, bosonic braid (via Group ctor).
- Class attrs `spin_zero` / `spin_half` / `spin_one` exposed on the pybind class; `fusion_tensor_dtype = Float64` set in ctor.
- `_f_symbol`, `_fusion_tensor`, `Z_iso` call Python `cyten.symmetries._su2data` (same pattern as NumPy helpers in AbelianGroup). Full `_su2data` C++ conversion is deferred.
- `_r_symbol` / `frobenius_schur` / `qdim` / fusion combinatorics implemented in C++.
- Leaf binding: `py::class_<SU2, Group, py::smart_holder>`.

## TODO checklist

- [x] setup (branch, pytest green)
- [ ] plan / declaration / definitions / bindings / monkey-patch / pytest
- [ ] wrap up
