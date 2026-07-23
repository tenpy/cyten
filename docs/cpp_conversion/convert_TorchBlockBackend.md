# Conversion of TorchBlockBackend

## metadata

- original python name: TorchBlockBackend
- original python file: cyten/block_backends/torch.py
- original python module: cyten.block_backends.torch
- declaration in C++ header file: include/cyten/block_backend/torch.h
- definition in C++ file: src/block_backend/torch.cpp
- pybind11 binding: pybind/block_backend/py_torch.cpp (included from py_block_backend.cpp)
- first line of docstring: A block-backend using PyTorch

## Context

`BlockBackend` / `NumpyBlockBackend` / `Block` / `Scalar` are already in C++
(see [convert_BlockBackend.md](convert_BlockBackend.md)). Torch is still a Python subclass of the
**Python** ABC in `_block_backend.py` and uses `torch.Tensor` as the block type.

This conversion mirrors the numpy pattern:

- Nested `TorchBlockBackend::Block` wrapping a `torch::Tensor` (prefer libtorch C++ API).
- Nearly-singleton `from_factory(device)` / `from_factory_shared(device)` (devices like `cpu:0`, `cuda:0`).
- Scalar-valued ops return `BlockBackend::Scalar` (0-d torch tensors), not bare `float`/`complex`.
- No trampoline: like numpy, not intended to be subclassed from Python.

LibTorch is already linked (`find_package(Torch)` in top-level CMake; smoke-tested via
`cyten._core.check_torch_array`).

## Design notes

1. **Block storage:** `torch::Tensor tensor_` plus cached `device_` string matching Python
   `as_device` / `get_device` (`cpu:0` style).
2. **Dtype maps:** C++ helpers `dtype_to_torch` / `dtype_from_torch` (no Python dtype dicts).
3. **Methods missing from Python torch.py but pure-virtual in C++:** `apply_mask`,
   `cutoff_inverse`, `scale_axis`, `multiply_blocks` — implement via torch ops (same semantics
   as Python ABC defaults / numpy).
4. **`to_same_dtype`:** Keep as a private helper (Python had it as a method); promote types with
   `torch::promoteTypes` / `to(...)`.
5. **`getitem` / `setitem`:** Prefer libtorch indexing where practical; may fall back to
   pybind/`torch` Python API for arbitrary keys if needed (numpy uses `py::array` `__getitem__`).
6. **HDF5:** Round-trip via `to_numpy()` / `block_from_numpy`, same idea as numpy block save/load.
7. **Factory wiring:** Update `backend_factory.py` to use `TorchBlockBackend.from_factory` like
   numpy; export from `cyten/block_backends/__init__.py` via `_core`.
8. **Tests:** Enable torch path in `test_to_backend` (remove xfail); run pytest with
   `--block-backends=torch` once monkey-patched.

## TODO list for conversion

- [x] initial setup
- [ ] planning
- [ ] generate the declaration draft
- [ ] improve and fix the declaration draft
- [ ] generate the C++ definitions
- [ ] improve and fix the definition drafts
- [ ] generate pybind11 bindings
- [ ] generate pybind11 trampoline (skip — not subclassed)
- [ ] monkey-patch the python binding into the Python library
- [ ] run python tests with pytest
- [ ] remove original python code for the object converted
- [ ] wrap up
