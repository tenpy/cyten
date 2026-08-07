# Conversion of FusionTreeData

## Status

**In progress** on branch `convert_backends`. C++ `FusionTreeData` only (not `FusionTreeBackend`). Exported as `cyten._core.FusionTreeData`. **Not monkey-patched** into `cyten.backends`.

Layer overview: [convert_backends.md](convert_backends.md). Abstract base: [convert_TensorBackend.md](convert_TensorBackend.md).

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
- Backend class (`FusionTreeBackend`) deferred to a later conversion.

### Members

| Python | C++ |
| --- | --- |
| `block_inds` (2D ndarray) | `py::array block_inds` |
| `blocks` (list of Block) | `std::vector<BlockBackend::BlockPtr> blocks` |
| `dtype` | `Dtype dtype` |
| `device` | `std::string device` |

Ctor takes `is_sorted`; if false, lexsort `block_inds` (like `np.lexsort(block_inds.T)`) and permute `blocks`.

### Methods

- `block_ind_from_coupled(Sector, TensorProduct::Ptr) -> optional<int64>` — uses `domain->sector_decomposition_where`.
- `block_ind_from_domain_sector_ind(int64) -> optional<int64>` — `searchsorted` on column 1.
- `discard_zero_blocks(shared_ptr<BlockBackend>, float64 eps)`.
- `save_hdf5` / `from_hdf5` via Python `hdf5_saver` / `hdf5_loader` (`py::object`).

### Out of scope

- Do **not** convert `FusionTreeBackend` or Instruction/Mapping helpers.
- Do **not** monkey-patch Python.

## Dependencies

- Done: `TensorBackend::Data`, `BlockBackend`, `Dtype`, `Sector`, `TensorProduct`.
- Still Python: tensor classes, `FusionTreeBackend`.

## TODO checklist

- [x] initial setup (on `convert_backends`; list_python_names)
- [ ] planning (this file)
- [ ] generate the declaration draft (`gen_cpp_declaration`)
- [ ] improve and fix the declaration draft
- [ ] generate the C++ definitions (`gen_cpp_definition` + CMake)
- [ ] improve and fix the definition drafts (compile)
- [ ] generate pybind11 bindings (+ register in CMake / `_core` / header)
- [ ] fix bindings
- [ ] trampoline — skip
- [ ] monkey-patch — **deferred**
- [ ] run python tests — deferred (not monkey-patched)
- [ ] remove original python code — deferred
- [ ] wrap up / continue with FusionTreeBackend
