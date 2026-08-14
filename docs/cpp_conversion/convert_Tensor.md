# Conversion of Tensor

## metadata

- original python name: `Tensor`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration in C++ header file: `include/cyten/tensors/tensor.h`
- definition in C++ file: `src/tensors/tensor.cpp`
- pybind11 binding: `pybind/tensors/py_tensor.cpp`
- trampoline: `PyTensor` in `pybind/tensors/py_trampolines.hpp` (required)
- first line of docstring: Common base class for tensors.

## Module context

1. Label helpers + `LabelledLegs` — **done** (C++ + helpers monkey-patched; Python `LabelledLegs` kept). See [convert_LabelledLegs.md](convert_LabelledLegs.md).
2. **`Tensor`** — this conversion (abstract base).
3. Next: `SymmetricTensor` → `DiagonalTensor` → `Identity` → `Mask` → `ChargedTensor` → free functions.

Keep original Python `Tensor` until all subclasses are converted. Export `cyten._core.Tensor`; **defer monkey-patch**.

## Design notes

- Inherit `LabelledLegs`; use `std::enable_shared_from_this<Tensor>` + `Tensor::Ptr` / `py::smart_holder`.
- Members: `TensorProduct::Ptr codomain/domain`, `TensorBackend::Ptr backend`, `Symmetry::Ptr symmetry`, `Dtype dtype`, `std::string device`, `std::vector<float64> shape` (Python dims can be non-integer for some symmetries — Python uses `sp.dim` which is float64 in C++ Space).
- Class attr `_forbidden_dtypes`: `static` vector, default `{Dtype::bool_}` (match Python `Dtype.bool`).
- Pure virtual (trampoline): `as_dtype`, `as_SymmetricTensor`, `copy`, `to_backend`, `to_dense_block`, `move_to_device`, `_get_item`.
- `as_SymmetricTensor` return type: `SymmetricTensor::Ptr` (see [convert_tensor_typed_api.md](convert_tensor_typed_api.md)).
- Other Tensor-returning abstracts: `Tensor::Ptr`.
- `to_dense_block` → `BlockBackend::BlockPtr`.
- Free-function-backed API (`dagger`, `T`, `__add__`, `__matmul__`, …): implement in C++ by calling free functions once they exist; until then `throw NotImplemented(...)` in C++ methods, and/or bind operators in pybind to Python free functions. Prefer NotImplemented stubs with `// FIXME` so subclasses compile.
- `ascii_diagram` / `__str__` / `__repr__`: port fully; subclass name map uses `dynamic_cast` / typeid once subclasses exist — for now map unknown → `"???"`.
- `num_parameters`: use `SectorArray::iter_common_sorted` on (co)domain sector decompositions.
- `_init_parse_args`: `get_backend` returns `py::object` → cast to `TensorBackend::Ptr`.
- Labels input for ctor / `set_labels`: accept flexible formats in bindings via `py::object`; C++ API takes `LegLabels` or a dedicated parse helper that also accepts nested lists via overloads / `py::object` helper.
- Override `test_sanity`, `set_labels` (virtual from `LabelledLegs`).
- Carry full Python docstrings into pybind bindings (`R"pydoc(...)"`).
- Preserve `# OPTIMIZE` / TODO comments from Python in C++ (`// OPTIMIZE`, `// TODO`).

## Dependencies

- Done: `LabelledLegs`, label helpers, `TensorBackend`, `get_backend`, spaces (`TensorProduct`, `Leg`, `LegPipe`, `Space`), `Symmetry`, `Dtype`, `Block`, `Scalar`, `get_config`, `SectorArray::iter_common_sorted`
- Still Python: tensor subclasses, free ops (`dagger`, `compose`, …), `iter_common_sorted_arrays` in tools (use SectorArray C++ instead), `vert_join` (string tools — may call Python or reimplement briefly for `__str__`)

## TODO list for conversion

- [x] initial setup (branch `convert_tensors`, Tensor listed, pytest smoke)
- [x] planning (this file)
- [x] generate the declaration draft
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions
- [x] improve and fix the definition drafts
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline (`PyTensor`)
- [x] monkey-patch — **deferred** until subclasses converted
- [x] run python tests (still using Python `Tensor`; 4341 passed)
- [ ] remove original python `Tensor` — later
- [ ] wrap up (then `SymmetricTensor`)
