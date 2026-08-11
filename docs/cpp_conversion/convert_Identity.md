# Conversion of Identity

## metadata

- original python name: `Identity`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration in C++ header file: `include/cyten/tensors/diagonal_tensor.h` (same file as DiagonalTensor)
- definition in C++ file: `src/tensors/diagonal_tensor.cpp` (same file)
- pybind11 binding: `pybind/tensors/py_diagonal_tensor.cpp` (extend) or `py_identity.cpp`
- trampoline: not needed (no library subclasses of Identity)
- first line of docstring: (none in Python — special-case DiagonalTensor that is exactly the identity)

## Module context

1–4. Labels / Tensor / SymmetricTensor / DiagonalTensor — done (monkey-patch deferred)
5. **`Identity`** — this conversion
6. Next: `Mask` → `ChargedTensor`

After Identity, DiagonalTensor has no remaining Python subclasses → can monkey-patch `SymmetricTensor` + `DiagonalTensor` + `Identity` together (optional wrap-up; Tensor/Mask/Charged still Python).

## Design notes

- Inherit `DiagonalTensor`; `using Ptr = std::shared_ptr<Identity>`.
- Ctor: `(leg, backend=None, dtype=None, device=None, labels=None)` — builds dummy `eye_data` like Python.
- `test_sanity`: only `Tensor::test_sanity` + `verify_dtype` (skip diagonal data checks).
- Unsupported factories (`from_block_func`, `from_dense_block`, …) → `TypeError` / `std::invalid_argument`.
- `from_eye` → `Identity(leg, backend, labels)` (ignores dtype/device like Python).
- Many DiagonalTensor methods virtualized for Identity overrides (`all`/`any`/`abs`/`as_*`/`copy`/`diagonal*`/`elementwise*`/`max`/`min`/`move_to_device`/`to_backend`/`to_dense_block`/`_get_item`/`_binary_operand`).
- `ascii_diagram_type_name` / `class_name` → `"Id"` / `"Identity"` (or keep Diag? Python uses type name Identity for class_name; ascii may fall through — check Python: no override, so `"Diag"` from DiagonalTensor). Keep `"Diag"` for ascii, `"Identity"` for class_name.
- Bindings: `py::class_<Identity, DiagonalTensor, py::smart_holder>` (no trampoline).
- Defer monkey-patch until wrap-up (with DiagonalTensor).
- Preserve OPTIMIZE/TODO; full docstrings where present.

## Dependencies

- Done: `DiagonalTensor`, backends `eye_data`, `Dtype` one/zero scalars
- Still Python: Mask, ChargedTensor, free ops (`is_scalar`)

## TODO list for conversion

- [x] initial setup
- [ ] planning (this file)
- [ ] generate / improve declaration
- [ ] generate / improve definitions
- [ ] pybind11 bindings (no trampoline)
- [ ] monkey-patch — deferred (with DiagonalTensor)
- [ ] pytest
- [ ] remove Python body — later with DiagonalTensor
- [ ] wrap up → Mask
