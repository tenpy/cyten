# Conversion of tensor elementwise ops (batch 11)

## metadata

- original python names: `angle`, `cutoff_inverse`, `complex_conj`, `imag`, `real`, `real_if_close`, `sqrt`, `stable_log`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration: `include/cyten/tensors/ops_elementwise.h`
- definition: `src/tensors/ops_elementwise.cpp`
- pybind11 binding: `pybind/tensors/py_ops_elementwise.cpp`
- trampoline: n/a (free functions)
- note: skip `_elementwise_function` (decorator — skill / convert_tensors.md: do not port; emit concrete functions)

## Module context

Batch 10 constructors done. This is **free-function batch 11** (Elementwise ops).
Next: algebra → legs → decompositions.

Stay on branch **`convert_tensors`**.

## Design notes

- Mirror the decorator: DiagonalTensor → `block_backend.<block_func>` via `_elementwise_unary`; scalar via original numeric body; else `TypeError`.
- Args are `py::object` so Python DiagonalTensor / Identity / numbers work before class monkey-patch.
- Defaults that the decorator baked into `func_kwargs`: `real_if_close(tol=100)`, `stable_log(cutoff=1e-30)`.
- `is_scalar` still Python (call via import) until a later misc/algebra cleanup.
- Monkey-patch after pytest; keep Python decorated defs for reference.
- Full docstrings in bindings.

## Dependencies

- Done: DiagonalTensor `_elementwise_unary`, BlockBackend `angle`/`conj`/`imag`/`real`/`real_if_close`/`sqrt`/`stable_log`/`cutoff_inverse`
- Still Python: `is_scalar`

## TODO list for conversion

- [x] planning (this file)
- [x] declaration + definitions
- [x] pybind11 bindings
- [x] monkey-patch
- [x] pytest (`-m "not slow"`) — 4341 passed, 596 xfailed
- [x] wrap up → algebra batch
