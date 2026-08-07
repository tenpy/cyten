# Conversion of AbelianBackend

## Status

**Not started** (data class done separately). `AbelianBackendData` is in C++ — see [convert_AbelianBackendData.md](convert_AbelianBackendData.md). The ~2k-line `AbelianBackend` class remains Python.

## Metadata (planned)

| Field | Value |
| --- | --- |
| original python name | `AbelianBackend` (+ `_valid_block_inds`) |
| original python file | `cyten/backends/abelian.py` |
| declaration | `include/cyten/backends/abelian.h` (extend existing) |
| definition | `src/backends/abelian.cpp` |
| pybind11 binding | `pybind/backends/py_abelian.cpp` |
| trampoline | only if Python subclasses remain |

## Dependencies

- Done: `TensorBackend`, `AbelianBackendData`, spaces (`AbelianLegPipe`, …), `BlockBackend`
- Still Python: tensor classes (Layer 4) — use `py::object` interim

## TODO

- [ ] gen_cpp_declaration / improve for `AbelianBackend`
- [ ] implement overrides (~same surface as TensorBackend)
- [ ] bindings; monkey-patch after FusionTreeBackend or with it
- [ ] pytest
