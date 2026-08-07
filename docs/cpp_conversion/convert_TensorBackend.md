# Conversion of TensorBackend

## Status

**C++ declaration + definitions + bindings/trampoline done** on branch `convert_backends`. Exported as `cyten._core.TensorBackend` (plus `conventional_leg_order`, `get_same_backend`). **Not monkey-patched** into `cyten.backends` yet — wait for concrete backends.

Layer overview: [convert_backends.md](convert_backends.md).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `TensorBackend` (+ `conventional_leg_order`, `get_same_backend`) |
| original python file | `cyten/backends/_backend.py` |
| original python module | `cyten.backends` |
| declaration | `include/cyten/backends/tensor_backend.h` |
| definition | `src/backends/tensor_backend.cpp` |
| pybind11 binding | `pybind/backends/py_tensor_backend.cpp` |
| trampoline | yes — `pybind/backends/py_trampolines.hpp` (`PyTensorBackend`) |
| first line of docstring | Abstract base class for tensor-backends. |

Skip binding `HasBackend` (typing `Protocol` only).

## Design notes

- `enable_shared_from_this` + `Ptr` / `CPtr`; hold `BlockBackend*` or `std::shared_ptr<BlockBackend>` as `block_backend` (match how BlockBackend is owned elsewhere).
- Nested abstract `Data` with `DataPtr` / `DataCPtr`; Python `Data` / `DiagonalData` / `MaskData` TypeVars → `DataPtr`.
- Tensor args (`SymmetricTensor`, `DiagonalTensor`, `Mask`) → interim `py::object`.
- Callables → `py::function`.
- ~73 pure virtual methods; ~11 concrete (`__init__`, repr/str, `item`, sanity checks, `make_pipe`, `_truncate_singular_values_selection`, `is_real`, HDF5).
- Free functions in same header/source: `conventional_leg_order`, `get_same_backend`.

## Dependencies

- Done: `BlockBackend`, `Block`, `Scalar`, `Dtype`, spaces (`Leg`, `Space`, `TensorProduct`, …), `Symmetry`, `FusionTree`.
- Still Python: tensor classes (Layer 4).

## TODO checklist

- [x] initial setup (clean tree on `convert_backends`; no new branch; list_python_names; pytest smoke)
- [x] planning (this file + convert_backends.md)
- [x] generate the declaration draft (`gen_cpp_declaration`)
- [x] improve and fix the declaration draft (namespaces, types, `Data` nested type, C++23 / pre-commit)
- [x] generate the C++ definitions (`gen_cpp_definition` + CMake)
- [x] improve and fix the definition drafts (concrete methods; pure virtuals stay `= 0`; compile + ctest)
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline
- [ ] monkey-patch — **deferred** until concrete backends converted
- [ ] run python tests with pytest
- [ ] remove original python code — deferred with monkey-patch
- [ ] wrap up / continue with NoSymmetryBackend

## Type mappings to add (`pybind11_codegen.toml`)

- `TensorBackend` → `TensorBackend::Ptr`
- `Data` / `DiagonalData` / `MaskData` → `TensorBackend::DataPtr`
- `SymmetricTensor` / `DiagonalTensor` / `Mask` → `py::object`
- `BlockBackend` → `std::shared_ptr<BlockBackend>` (if missing)
