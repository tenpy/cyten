# Conversion of AbelianBackendData

## Status

**Done / monkey-patched** on branch `convert_backends`. Exported as `cyten._core.AbelianBackendData`. Imported in `cyten/backends/abelian.py` from `_core` (with `AbelianBackend`).

Layer overview: [convert_backends.md](convert_backends.md). Abstract base: [convert_TensorBackend.md](convert_TensorBackend.md). Backend: [convert_AbelianBackend.md](convert_AbelianBackend.md).

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
- Members:
  - `Dtype dtype`
  - `std::string device`
  - `std::vector<BlockBackend::BlockPtr> blocks`
  - `py::array_t<int64>` `block_inds` (2D; keep numpy-compatible for `np.lexsort`)

### Methods

- Ctor `(dtype, device, blocks, block_inds, is_sorted=false)` — if not sorted, permute via `np.lexsort(block_inds.T)`.
- `get_block_num(block_inds) -> optional<int64>` (Python returns `None`)
- `get_block(block_inds) -> BlockPtr` (nullable)
- `save_hdf5` / `from_hdf5` via `py::object` hdf5 loader/saver (same pattern as symmetries / Dtype).

### HDF5

- Save `block_inds`, `blocks`, `dtype.to_numpy_dtype()`, `device` (match Python).
- Load and reconstruct with `Dtype.from_numpy_dtype`; call `memorize_load`.
- `from_hdf5` constructs with `is_sorted=true` (data was sorted when saved).

## Dependencies

- Done: `TensorBackend::Data`, `BlockBackend::BlockPtr`, `Dtype`, `AbelianBackend`.

## TODO checklist

- [x] initial setup (on `convert_backends`; list_python_names; read Python source)
- [x] planning (this file)
- [x] generate the declaration draft (`gen_cpp_declaration`)
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions (`gen_cpp_definition` + CMake)
- [x] improve and fix the definition drafts (implement; compile)
- [x] generate pybind11 bindings (+ register in CMake / `_core` / header)
- [x] fix bindings (`shared_ptr` ctor, inherit `TensorBackend::Data`, members, static `from_hdf5`)
- [x] trampoline — skip
- [x] monkey-patch via `abelian.py`
- [x] run python tests
- [x] remove original python class body — module re-exports from `_core`
