# Conversion of AbelianBackend

## Status

**In progress** on branch `convert_backends`. `AbelianBackendData` already in C++ — [convert_AbelianBackendData.md](convert_AbelianBackendData.md). Do **not** monkey-patch until FusionTreeBackend is ready or trampoline inheritance is proven for mixed use.

C++ `AbelianBackend` + `valid_block_inds` are defined in `src/backends/abelian.cpp` and the library builds / ctest passes. Many complex methods still **delegate to the Python `AbelianBackend`** via `py_call_data` / `py_abelian` (WIP); thin methods and helpers are native.

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

- Inherit `TensorBackend`; `DataCls` = pybind type of `AbelianBackendData` (set in bindings; ctor leaves `py::none()` for now).
- Helpers: `wrap` / `unwrap` / `data_from_tensor` for `AbelianBackendData::Ptr` (also accepts Python duck-typed data).
- Free function `valid_block_inds(codomain, domain)` (native; uses `make_grid` + fusion broadcast).
- Override `make_pipe` → `AbelianLegPipe`; override `save_hdf5` / `from_hdf5` (Python only saves `DataCls` — match that).
- Tensor args stay interim `py::object`; callables → `py::function`.
- Stubs matching Python: `state_tensor_product`, `to_dense_block_trivial_sector` → `NotImplemented`.
- `partial_trace` scalar path returns `Data` with one scalar block + `nullptr` domains.
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
- [x] declaration
- [x] definitions + compile + ctest (native thin methods; heavy kernels still Python-delegated — see FIXMEs in `abelian.cpp`)
- [ ] replace Python-delegated methods with native C++ ports
- [ ] bindings
- [ ] monkey-patch — deferred
- [ ] pytest — deferred

## Methods still Python-delegated (FIXME)

`combine_legs`, `_compose_worker` (via Python `compose`), `diagonal_elementwise_binary`, `diagonal_to_mask`, `eigh`, `eye_data`, `from_dense_block`, `from_grid`, `from_tree_pairs`, `get_element*`, `inner`, `inv_part_*`, `linear_combination`, `lq`, `mask_binary_operand`, `_mask_contract`, `mask_from_block`, `mask_to_block`, `mask_to_diagonal`, `mask_transpose`, `mask_unary_operand`, `outer`, `partial_compose`, `partial_trace`, `qr`, `reduce_DiagonalTensor`, `scale_axis`, `split_legs`, `svd`, `to_dense_block`, `trace_full`, `truncate_singular_values`.
