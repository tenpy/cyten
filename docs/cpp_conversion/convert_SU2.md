# Conversion of SU2

## Status

**Done for monkey-patch.** C++ `SU2` + bindings; imported from `_core`. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped.

Hand-written after codegen drafts. No trampoline (no Python subclasses of `SU2`).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `SU2` |
| original python file | `cyten/symmetries/_symmetries.py` |
| declaration | `include/cyten/symmetries/su2.h` |
| definition | `src/symmetries/su2.cpp` |
| pybind11 binding | `pybind/symmetries/py_su2.cpp` |
| trampoline | none |
| first line of docstring | SU(2) symmetry. |

## Design notes

- Subclass C++ `Group` with `FusionStyle::multiple_unique`, infinite sectors.
- Class attrs `spin_zero` / `spin_half` / `spin_one` set on the pybind class; `fusion_tensor_dtype = Float64` in ctor.
- `_f_symbol`, `_fusion_tensor`, `Z_iso` call Python `cyten.symmetries._su2data` (deferred full C++ conversion of that module).
- Combinatorics (`fusion_outcomes`, `can_fuse_to`, dims, `_r_symbol`, `frobenius_schur`) in C++.

## TODO checklist

- [x] setup / plan / declaration / definitions / bindings / monkey-patch / pytest
- [ ] wrap up / merge to `main_cpp` or continue (`SUN`, fermion parity, anyons, …)
