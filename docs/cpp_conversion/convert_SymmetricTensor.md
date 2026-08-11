# Conversion of SymmetricTensor

## metadata

- original python name: `SymmetricTensor`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration in C++ header file: `include/cyten/tensors/symmetric_tensor.h`
- definition in C++ file: `src/tensors/symmetric_tensor.cpp`
- pybind11 binding: `pybind/tensors/py_symmetric_tensor.cpp`
- trampoline: `PySymmetricTensor` in `pybind/tensors/py_trampolines.hpp` (required — `DiagonalTensor` / `Identity` still Python)
- first line of docstring: A tensor that is symmetric, i.e. invariant under the symmetry.

## Module context

1. Label helpers + `LabelledLegs` — done
2. `Tensor` ABC — done (monkey-patch deferred)
3. **`SymmetricTensor`** — this conversion
4. Next: `DiagonalTensor` → `Identity` → `Mask` → `ChargedTensor`

Keep original Python `SymmetricTensor` until `DiagonalTensor` / `Identity` are converted. Export `cyten._core.SymmetricTensor`; **defer monkey-patch**.

## Design notes

- Inherit `Tensor`; `using Ptr = std::shared_ptr<SymmetricTensor>`.
- Member: `TensorBackend::DataPtr data`.
- Ctor: `(DataPtr data, py::object|TensorProduct codomain, domain=None, backend=None, labels=None)` — dtype/device from `backend->get_*_from_data(data)`.
- Override all 7 Tensor abstracts; `as_SymmetricTensor` returns `py::object` (cast of `Ptr`) to match `Tensor` until we tighten the base signature.
- `ascii_diagram_type_name()` → `"Symm"`; `class_name()` → `"SymmetricTensor"`.
- Factories: implement `from_zero`, `from_eye`, `from_block_func`, `from_sector_block_func`, `from_sector_projection`, `from_dense_block`, `from_random_uniform`, `from_random_normal` (mean path via Python `+` if needed), `from_tree_pairs`, `_parse_default_dtype`.
- `from_dense_block_trivial_sector`: keep Python `NotImplementedError`.
- `diagonal`: return via Python `DiagonalTensor.from_tensor` until DiagonalTensor is C++.
- `to_backend`: implement common paths; pipes / FT↔abelian via Python helpers when needed (`// OPTIMIZE` preserved).
- `save_hdf5` / `from_hdf5`: implement with `py::object` hdf5 saver/loader like other types.
- Full docstrings in bindings; preserve OPTIMIZE/TODO comments.

## Dependencies

- Done: `Tensor`, `LabelledLegs`, `TensorBackend` (+ `conventional_leg_order`), spaces, `Dtype`, backends
- Still Python: `DiagonalTensor`, free ops (`split_legs`, `combine_legs`, `_convert_*`), hdf5 stack

## TODO list for conversion

- [x] initial setup
- [ ] planning (this file)
- [ ] generate / improve declaration
- [ ] generate / improve definitions
- [ ] pybind11 bindings + trampoline
- [ ] monkey-patch — deferred
- [ ] pytest (Python SymmetricTensor still used)
- [ ] remove Python body — later
- [ ] wrap up → DiagonalTensor
