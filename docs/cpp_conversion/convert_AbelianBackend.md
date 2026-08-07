# Conversion of AbelianBackend

## Status

**In progress** on branch `convert_backends`. `AbelianBackendData` already in C++ — [convert_AbelianBackendData.md](convert_AbelianBackendData.md). Do **not** monkey-patch until FusionTreeBackend is ready or trampoline inheritance is proven for mixed use.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `AbelianBackend` (+ `_valid_block_inds`) |
| original python file | `cyten/backends/abelian.py` |
| original python module | `cyten.backends` |
| declaration | `include/cyten/backends/abelian.h` (extend) |
| definition | `src/backends/abelian.cpp` (extend) |
| pybind11 binding | `pybind/backends/py_abelian.cpp` (extend) |
| trampoline | no (no Python subclasses of AbelianBackend) |
| first line of docstring | Backend for Abelian group symmetries. |

## Design notes

- Inherit `TensorBackend`; `DataCls` = pybind type of `AbelianBackendData`.
- Helpers: `wrap` / `unwrap` / `data_from_tensor` for `AbelianBackendData::Ptr`.
- Free function `_valid_block_inds(codomain, domain) -> py::array_t<int64>` (or nested static).
- Override `make_pipe` → `AbelianLegPipe`; override `save_hdf5` / `from_hdf5` (Python only saves `DataCls` — match that, or also save `block_backend` via base).
- Tensor args stay interim `py::object`; callables → `py::function`.
- Complex kernels (`_compose_worker`, `combine_legs`, `split_legs`, QR/SVD/LQ): translate carefully; may call `cyten.tools.misc` via pybind for `iter_common_*` / `make_grid` / `list_to_dict_list` initially.
- Stubs matching Python: `state_tensor_product`, `to_dense_block_trivial_sector` → `NotImplemented`.
- Bindings return `AbelianBackendData` (not unwrapped Block) — Python already stores Data objects.

## Suggested implementation order

1. Header: class + all TensorBackend overrides + `make_pipe` / HDF5 / `_valid_block_inds`
2. Thin methods (zeros, copy, mul, accessors, supports_symmetry, …)
3. `_valid_block_inds` + constructors (`from_*`, `eye_data`, …)
4. Diagonal / mask ops
5. `combine_legs` / `split_legs` / `_compose_worker`
6. Decompositions
7. Bindings + smoke test; no monkey-patch yet

## TODO checklist

- [x] planning (this file)
- [ ] declaration
- [ ] definitions + compile + ctest
- [ ] bindings
- [ ] monkey-patch — deferred
- [ ] pytest — deferred
