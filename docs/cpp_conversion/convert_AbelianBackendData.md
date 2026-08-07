# Conversion of AbelianBackendData

## Status

**In progress** on branch `convert_backends`. Export as `cyten._core.AbelianBackendData`. **Not monkey-patched** into `cyten.backends`. `AbelianBackend` deferred.

Layer overview: [convert_backends.md](convert_backends.md). Abstract base: [convert_TensorBackend.md](convert_TensorBackend.md).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `AbelianBackendData` |
| original python file | `cyten/backends/abelian.py` |
| original python module | `cyten.backends.abelian` |
| declaration | `include/cyten/backends/abelian.h` |
| definition | `src/backends/abelian.cpp` |
| pybind11 binding | `pybind/backends/py_abelian.cpp` |
| trampoline | none (data class; no Python subclasses) |
| first line of docstring | Data stored in a Tensor for :class:`AbelianBackend`. |

## Design notes

### Class layout

- Top-level `AbelianBackendData : TensorBackend::Data` (not nested under `AbelianBackend`).
- Forward-declare `AbelianBackend` in the same header for later; do **not** convert the backend class yet.
- Members:
  - `Dtype dtype`
  - `std::string device`
  - `std::vector<BlockBackend::BlockPtr> blocks`
  - `py::array_t<int64>` `block_inds` (2D; keep numpy-compatible for `np.lexsort`)

### Methods

- Ctor `(dtype, device, blocks, block_inds, is_sorted=false)` — if not sorted, permute via `np.lexsort(block_inds.T)`.
- `get_block_num(block_inds) -> optional<int64>`
- `get_block(block_inds) -> BlockPtr` (nullable / optional)
- `save_hdf5` / `from_hdf5` via `py::object` hdf5 loader/saver (same pattern as symmetries / Dtype).

### HDF5

- Save `block_inds`, `blocks`, `dtype.to_numpy_dtype()`, `device` (match Python).
- Load and reconstruct with `Dtype.from_numpy_dtype`; call `memorize_load`.

## Dependencies

- Done: `TensorBackend::Data`, `BlockBackend::BlockPtr`, `Dtype`.
- Deferred: `AbelianBackend` itself.

## TODO checklist

- [x] initial setup (on `convert_backends`; list_python_names; read Python source)
- [x] planning (this file)
- [ ] generate the declaration draft (`gen_cpp_declaration`)
- [ ] improve and fix the declaration draft
- [ ] generate the C++ definitions (`gen_cpp_definition` + CMake)
- [ ] improve and fix the definition drafts (implement; compile)
- [ ] generate pybind11 bindings (+ register in CMake / `_core` / header)
- [ ] fix bindings
- [ ] trampoline — skip
- [ ] monkey-patch — **deferred** (per convert_backends / user)
- [ ] run python tests — deferred until monkey-patch / AbelianBackend
- [ ] remove original python code — deferred
