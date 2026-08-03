# Conversion of AbelianGroup

## Status

**Done for monkey-patch.** C++ `AbelianGroup` + `PyAbelianGroup`; imported from `_core`. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped. Python `NoSymmetry` / `U1` / `ZN` subclass C++ `AbelianGroup`.

Hand-written after codegen declaration draft (bindings for bases from `_core` still broken in codegen).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `AbelianGroup` |
| original python file | `cyten/symmetries/_symmetries.py` |
| declaration | `include/cyten/symmetries/abelian_group.h` |
| definition | `src/symmetries/abelian_group.cpp` |
| pybind11 binding | `pybind/symmetries/py_abelian_group.cpp` |
| trampoline | `PyAbelianGroup` in `pybind/symmetries/py_trampolines.hpp` |
| first line of docstring | Base-class for abelian symmetry groups. |

## Design notes

- Ctor → `Group(FusionStyle::single, …, has_complex=false)`; sets `fusion_tensor_dtype = Dtype::Float64`.
- Constant topo data via NumPy ones helpers (mirror `one_1D` … `one_4D_float`).
- Next: concretes (`NoSymmetry`, `U1`, `ZN`, …).

## TODO checklist

- [x] setup / plan / declaration / definitions / bindings+trampoline / monkey-patch / pytest
- [ ] wrap up / merge or continue with `NoSymmetry`
