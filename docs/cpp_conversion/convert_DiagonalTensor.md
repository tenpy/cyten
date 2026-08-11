# Conversion of DiagonalTensor

## metadata

- original python name: `DiagonalTensor`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration in C++ header file: `include/cyten/tensors/diagonal_tensor.h`
- definition in C++ file: `src/tensors/diagonal_tensor.cpp`
- pybind11 binding: `pybind/tensors/py_diagonal_tensor.cpp`
- trampoline: `PyDiagonalTensor` in `pybind/tensors/py_trampolines.hpp` (required — `Identity` still Python)
- first line of docstring: Special case of a SymmetricTensor that is diagonal in the computational basis.

## Module context

1. Label helpers + `LabelledLegs` — done
2. `Tensor` ABC — done (monkey-patch deferred)
3. `SymmetricTensor` — done (monkey-patch deferred)
4. **`DiagonalTensor`** — this conversion
5. Next: `Identity` → `Mask` → `ChargedTensor`

Keep original Python `DiagonalTensor` until `Identity` is converted. Export `cyten._core.DiagonalTensor`; **defer monkey-patch**.

## Design notes

- Inherit `SymmetricTensor`; `using Ptr = std::shared_ptr<DiagonalTensor>`.
- Ctor: `(DataPtr data, Space::Ptr|py::object leg, backend=None, labels=None)` — reject `LegPipe`.
- Override `verify_dtype` (no-op — real dtypes always allowed).
- Override `_forbidden_dtypes` behavior: empty list (bool allowed). Prefer virtual `forbidden_dtypes()` on `Tensor` so `test_sanity` respects subclass.
- `ascii_diagram_type_name()` → `"Diag"`; `class_name()` → `"DiagonalTensor"`.
- Factories: `from_zero`, `from_eye`, `from_block_func`, `from_sector_block_func`, `from_diag_block`, `from_dense_block`, `from_random_*`, `from_tensor`, hdf5.
- Overrides of Tensor abstracts: `as_dtype`, `as_SymmetricTensor` (builds full data via backend), `copy`, `to_backend`, `to_dense_block`, `move_to_device`, `_get_item`.
- Elementwise API: `_elementwise_unary` / `_elementwise_binary` / `_binary_operand`; bind dunders similar to Tensor (may call into these).
- `diagonal()` returns `self`; `as_DiagonalTensor` returns self/copy.
- `leg` property → `Space::Ptr` from `codomain->factors[0]`.
- Update toml: `DiagonalTensor = "DiagonalTensor::Ptr"`.
- Optionally update `SymmetricTensor::diagonal` to construct C++ `DiagonalTensor` (still OK while Python class remains).
- Full docstrings in bindings; preserve OPTIMIZE/TODO comments.

## Dependencies

- Done: `SymmetricTensor`, `Tensor`, backends (`diagonal_*` APIs), spaces, `Dtype`
- Still Python: `Identity`, free ops (`linear_combination`, `is_scalar`, `item`), Mask

## TODO list for conversion

- [x] initial setup (branch `convert_tensors`, listed, pytest smoke 504 Diagonal-related passed)
- [ ] planning (this file)
- [ ] generate / improve declaration
- [ ] generate / improve definitions
- [ ] pybind11 bindings + trampoline
- [ ] monkey-patch — deferred
- [ ] pytest (Python DiagonalTensor still used)
- [ ] remove Python body — later (after Identity)
- [ ] wrap up → Identity
