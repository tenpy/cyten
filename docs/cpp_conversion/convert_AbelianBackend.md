# Conversion of AbelianBackend

## Status

**C++ declaration + definitions + bindings done** on branch `convert_backends`. Exported as `cyten._core.AbelianBackend` (+ `valid_block_inds` / `_valid_block_inds`). `get_backend('abelian', …)` constructs the C++ backend. **Not monkey-patched** into `cyten.backends` yet.

`AbelianBackendData`: [convert_AbelianBackendData.md](convert_AbelianBackendData.md).

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
- Free helper `abelian_compose_worker` (anonymous namespace) used by `_compose_worker` and `partial_compose`.
- Override `make_pipe` → `AbelianLegPipe`; override `save_hdf5` / `from_hdf5` (Python only saves `DataCls` — match that).
- Tensor args stay interim `py::object`; callables → `py::function`.
- Call `cyten.tools.misc` via `py::module_::import` for helpers (`make_grid`, `iter_common_*`, `list_to_dict_list`, `find_row_differences`, `rank_data`, `make_stride`, `inverse_permutation`).
- Stubs matching Python: `state_tensor_product`, `to_dense_block_trivial_sector` → `NotImplemented`.
- `partial_trace` scalar path returns `Data` with one scalar block + `nullptr` domains.
- Bindings return `AbelianBackendData` (not unwrapped Block) — Python already stores Data objects.
- `mask_binary_operand`: C++ advances `mask2` block_inds correctly (Python had a typo using `mask1_block_inds`).
- `mask_unary_operand`: uses `block_inds` (Python typo was `blocks_inds`).

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
- [x] definitions + compile (native ports; no remaining `py_call_data` / `py_abelian` delegations)
- [x] replace Python-delegated methods with native C++ ports
- [x] bindings (`cyten._core.AbelianBackend`, `valid_block_inds`; factory wired)
- [ ] monkey-patch — deferred
- [ ] pytest — deferred

## Methods still Python-delegated (FIXME)

None. All previously delegated methods are native.

## Intentionally not implemented (match Python)

- `state_tensor_product` → `NotImplemented`
- `to_dense_block_trivial_sector` → `NotImplemented`
