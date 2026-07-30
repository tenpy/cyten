# Conversion of ArrayApiBlockBackend

## metadata

- original python name: ArrayApiBlockBackend
- original python file: cyten/block_backends/array_api.py
- original python module: cyten.block_backends.array_api
- declaration in C++ header file: include/cyten/block_backend/array_api.h
- definition in C++ file: src/block_backend/array_api.cpp
- pybind11 binding: pybind/block_backend/py_array_api.cpp (included from py_block_backend.cpp)
- trampoline: pybind/block_backend/py_trampolines.hpp (`PyArrayApiBlockBackend`)
- first line of docstring: A block-backend based on a generic Array API compliant library

## Context

`BlockBackend` / `NumpyBlockBackend` / `TorchBlockBackend` / `Block` / `Scalar` are already in
C++ (see [convert_BlockBackend.md](convert_BlockBackend.md),
[convert_TorchBlockBackend.md](convert_TorchBlockBackend.md)).

`ArrayApiBlockBackend` is a thin adapter over any [Array API](https://data-apis.org/array-api/)
namespace (`numpy`, future `jax.numpy`, etc.). It is currently unused by `backend_factory`, but
must remain a **Python-subclassable** base so libraries can specialize missing ops (e.g. `kron`).

## Design notes

1. **Subclass C++ `BlockBackend`.** Same virtual surface as numpy/torch.
2. **Nested `ArrayApiBlockBackend::Block`** wraps a `py::object` (the Array-API array). Arithmetic /
   indexing go through Python operators / `__getitem__` on that object (same idea as
   `NumpyBlockBackend::Block` with `py::array`).
3. **Hold `py::object api_`** (the Array API namespace passed to the ctor). Dtype maps are built from
   `api_.attr("float32")` etc., stored as `std::map` / parallel lookups.
4. **No nearly-singleton factory.** Construct with `(api_namespace, default_device='cpu')`. Public
   ctor (not protected) so Python subclasses can call `super().__init__(api, device)`.
5. **Scalar-valued ops return `BlockBackend::Scalar`** (0-d wrapped blocks), not bare
   `float`/`complex`.
6. **Trampoline required.** Bind with `PyArrayApiBlockBackend` using `PYBIND11_OVERRIDE` (not
   `_PURE`) so C++ defaults run unless a Python subclass overrides. Existing `PyBlockBackend` stays
   for the abstract base.
7. **Methods missing from Python ArrayApi but pure-virtual in C++** (`abs`, `apply_mask`,
   `cutoff_inverse`, `scale_axis`, `multiply_blocks`, `tile`, …): implement via Array API where
   obvious (`api.abs`, elementwise `*`, …); otherwise `throw NotImplemented(...)` matching the
   Python stubs (`angle`, `kron`, `matrix_exp`, …).
8. **`to_numpy`:** convert via `numpy.asarray(obj)` (Array API arrays that support the buffer /
   `__array__` protocol).
9. **HDF5:** round-trip via `to_numpy()` / `block_from_numpy`, like numpy/torch.
10. **Tests:** no dedicated ArrayApi pytest file today. Add a small smoke test constructing
    `ArrayApiBlockBackend(numpy)` (numpy is Array-API compatible enough for basic ops) and a
    Python subclass that overrides one method to verify the trampoline.

## TODO list for conversion

- [x] initial setup
- [x] planning
- [x] generate the declaration draft
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions
- [x] improve and fix the definition drafts
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline (required — Python subclassable)
- [x] monkey-patch the python binding into the Python library
- [x] run python tests with pytest
- [x] remove original python code for the object converted (`array_api.py` is a `_core` re-export)
- [ ] wrap up (merge/PR left to the user)
