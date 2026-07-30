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
(see [convert_BlockBackend.md](convert_BlockBackend.md)). This conversion mirrors the numpy pattern:

- Nested `TorchBlockBackend::Block` wrapping a `torch::Tensor` (libtorch C++ API).
- Nearly-singleton `from_factory(device)` / `from_factory_shared(device)` (devices like `cpu:0`, `cuda:0`).
- Scalar-valued ops return `BlockBackend::Scalar` (0-d torch tensors), not bare `float`/`complex`.
- No trampoline: like numpy, not intended to be subclassed from Python.

## Shared libtorch with Python `torch`

Users will `import cyten` and `import torch` in the same process. Cyten uses the **hybrid**
approach: C++ `TorchBlockBackend` calls the libtorch C++ API, but `_core` must load the **same**
shared `libtorch` / `libc10` as `torch._C`:

1. CMake finds Torch via `torch.utils.cmake_prefix_path` from the build Python (not a separate
   standalone LibTorch tree).
2. `_core` links `TORCH_LIBRARIES` from that package and sets `INSTALL_RPATH` so runtime
   resolution prefers env/`site-packages/torch` libs (`$ORIGIN/../../..` and
   `$ORIGIN/../torch/lib`).
3. Do **not** ship a second copy of libtorch next to `_core`.

Smoke-tested via `cyten._core.check_torch_array` and dual-import regression tests.

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
   `--block-backends=torch` once monkey-patched. Allow `import torch` alongside cyten.

## TODO list for conversion

- [x] initial setup
- [x] planning
- [x] generate the declaration draft
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions
- [x] improve and fix the definition drafts
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline (skip — not subclassed)
- [x] monkey-patch the python binding into the Python library
- [x] run python tests with pytest
- [x] remove original python code for the object converted (`torch.py` is a `_core` re-export)
- [x] wrap up (merge/PR left to the user)
