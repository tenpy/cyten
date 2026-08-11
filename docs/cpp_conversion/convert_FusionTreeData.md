# Conversion of FusionTreeData

## Status

**Done / monkey-patched** on branch `convert_backends`. Exported as `cyten._core.FusionTreeData`. Imported in `cyten/backends/fusion_tree_backend.py` from `_core` (with `FusionTreeBackend` and mapping helpers).

Layer overview: [convert_backends.md](convert_backends.md). Abstract base: [convert_TensorBackend.md](convert_TensorBackend.md). Backend: [convert_FusionTreeBackend.md](convert_FusionTreeBackend.md).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `FusionTreeData` |
| original python file | `cyten/backends/fusion_tree_backend.py` |
| original python module | `cyten.backends.fusion_tree_backend` |
| declaration | `include/cyten/backends/fusion_tree_backend.h` |
| definition | `src/backends/fusion_tree_backend.cpp` |
| pybind11 binding | `pybind/backends/py_fusion_tree_backend.cpp` |
| trampoline | none (data class; no Python subclasses expected) |
| first line of docstring | Data stored in a Tensor for :class:`FusionTreeBackend`. |

## Design notes

### Inheritance

- `FusionTreeData : public TensorBackend::Data` (standalone class, not nested in backend).
- `Ptr` / `CPtr` via `shared_ptr`.

### Members

| Python | C++ |
| --- | --- |
| `block_inds` (2D ndarray) | `py::array block_inds` |
| `blocks` (list of Block) | `std::vector<BlockBackend::BlockPtr> blocks` |
| `dtype` | `Dtype dtype` |
| `device` | `std::string device` |

Ctor takes `is_sorted`; if false, lexsort `block_inds` (like `np.lexsort(block_inds.T)`) and permute `blocks` (via numpy). `block_inds` must have shape `(N, 2)`.

### Methods

- `block_ind_from_coupled(Sector, TensorProduct::Ptr) -> optional<int64>` — uses `domain->sector_decomposition_where`.
- `block_ind_from_domain_sector_ind(int64) -> optional<int64>` — `np.searchsorted` on column 1.
- `discard_zero_blocks(shared_ptr<BlockBackend>, float64 eps)` — binding converts factory backends via `as_shared_block_backend`.
- `save_hdf5` / `from_hdf5` via Python `hdf5_saver` / `hdf5_loader` (`py::object`); saves `dtype` as C++ `Dtype` (has `save_hdf5`).

## Dependencies

- Done: `TensorBackend::Data`, `BlockBackend`, `Dtype`, `Sector`, `TensorProduct`, `FusionTreeBackend`.
- Still Python: tensor classes (Layer 4).

## TODO checklist

- [x] initial setup (on `convert_backends`; list_python_names)
- [x] planning (this file)
- [x] generate the declaration draft (`gen_cpp_declaration`)
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions (`gen_cpp_definition` + CMake)
- [x] improve and fix the definition drafts (compile)
- [x] generate pybind11 bindings (+ register in CMake / `_core` / header)
- [x] fix bindings (BlockBackend shared_ptr helper for `discard_zero_blocks`)
- [x] trampoline — skip
- [x] monkey-patch via `fusion_tree_backend.py`
- [x] run python tests
- [x] remove original python class body — module re-exports from `_core`
