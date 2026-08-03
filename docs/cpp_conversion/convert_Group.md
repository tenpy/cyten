# Conversion of Group

## Status

**Done for monkey-patch.** C++ `Group` + `PyGroup` trampoline; imported from `_core` in `_symmetries.py`. `pytest tests/python_tests/test_symmetries.py`: 48 passed, 1 skipped. Python `AbelianGroup` / `SU2` / `SUN` subclass C++ `Group`.

### Codegen notes

- `gen_cpp_declaration` / `gen_cpp_definition` worked.
- `gen_pyb11_binding --py-name Group ...` failed: `AttributeError: 'NoneType' object has no attribute 'name'` when resolving base `SymmetryFactor` (already from `_core`). Bindings/trampoline were hand-written.

### Pitfalls

- `PyGroup` must trampoline all pure virtuals inherited from `SymmetryFactor`, not only Group’s own overrides.
- `_multiple_fusion_broadcast` trampoline must unpack `*args` (same as `PySymmetryFactor`).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `Group` |
| original python file | `cyten/symmetries/_symmetries.py` |
| original python module | `cyten.symmetries` |
| declaration | `include/cyten/symmetries/group.h` |
| definition | `src/symmetries/group.cpp` |
| pybind11 binding | `pybind/symmetries/py_group.cpp` |
| trampoline | `PyGroup` in `pybind/symmetries/py_trampolines.hpp` |
| first line of docstring | Base-class for symmetries that are described by a group. |

## Design notes

- Inherits `SymmetryFactor`; ctor always passes `BraidingStyle::bosonic`.
- Pure virtual `_fusion_tensor`; overrides `swap_gate`, `qdim`, `batch_qdim`, `topological_twist`.
- Next: `AbelianGroup`, then concretes.

## TODO checklist

- [x] initial setup
- [x] planning
- [x] declaration draft + improve
- [x] definitions + compile + ctest
- [x] pybind11 bindings + trampoline
- [x] monkey-patch; pytest
- [x] remove Python `Group` class body
- [ ] wrap up / suggest merge to `main_cpp` (or continue with `AbelianGroup`)
