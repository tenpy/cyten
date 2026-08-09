# Conversion of FusionTreeBackend

## Status

**In progress** on branch `convert_backends`. `FusionTreeData` already in C++ — [convert_FusionTreeData.md](convert_FusionTreeData.md). Do **not** monkey-patch until the full permute/mapping stack is solid and pytest is green.

### Python delegation remaining (intentional)

| Method / helper | Delegates to |
| --- | --- |
| `_partial_trace_helper` | Python module function (called from native `partial_trace`) |

`apply_instructions` and `permute_legs` are **native C++** (TreePairMapping / FactorizedTreeMapping / PermuteLegsInstructionEngine). See [convert_TreePairMapping.md](convert_TreePairMapping.md).

All other former `ft_py().FusionTreeBackend(...)` stubs (outer, partial_compose, partial_trace body, from_tree_pairs, get_element, from_grid, scale_axis, mask_*, diagonal_to_mask) are **native C++** in `src/backends/_ft_native_methods.inc` (included from `fusion_tree_backend.cpp`).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `FusionTreeBackend` (+ Instruction / Engine / TensorMapping helpers) |
| original python file | `cyten/backends/fusion_tree_backend.py` |
| original python module | `cyten.backends` |
| declaration | `include/cyten/backends/fusion_tree_backend.h` (+ optional `fusion_tree_permute.h`, `fusion_tree_mapping.h`) |
| definition | `src/backends/fusion_tree_backend.cpp` (may split) |
| pybind11 binding | `pybind/backends/py_fusion_tree_backend.cpp` |
| trampoline | no (no Python subclasses) |
| first line of docstring | A backend based on fusion trees. |

## Design notes

- Inherit `TensorBackend`; ctor `(block_backend, eps=5e-14)`; `can_decompose_tensors = true`.
- Helpers: `wrap` / `unwrap` / `data_from_tensor` for `FusionTreeData::Ptr`.
- No `make_pipe` / HDF5 / `is_real` overrides (inherit base).
- Tensor args interim `py::object`; callables → `py::function`.
- Stubs matching Python: `from_dense_block_trivial_sector`, `inv_part_*`, `state_tensor_product`, `to_dense_block_trivial_sector` → `NotImplemented`.
- **Permute stack** (required by `permute_legs` / `partial_trace`):
  1. Instruction POD (`Braid` / `Bend` / `Twist`)
  2. `PermuteLegsInstructionEngine`
  3. `TensorMapping` / `TreePairMapping` / `FactorizedTreeMapping` — **done** ([convert_TreePairMapping.md](convert_TreePairMapping.md))
- Forest helpers `_add_forest_block_entries` / `_get_forest_block_contribution` for dense I/O.
- Prefer calling `cyten.tools.misc` / mappings via pybind for iterators when clearer than a full port.
- **SparseMapping** is available in C++ (`include/cyten/tools/mappings.h`); next step for removing `ft_py()` Mapping delegation.

## Suggested implementation order

1. Header: `FusionTreeBackend` + all TensorBackend overrides + `eps`
2. Thin native methods (compose, dagger, eye, zeros, QR/SVD/LQ, …)
3. Instruction + Engine (+ bind for existing unit tests)
4. TensorMapping hierarchy (or Python-delegate `apply_instructions` / `permute_legs` initially)
5. Forest dense↔sparse helpers
6. Remaining complex methods; replace any Python delegation — **done** (except permute/mapping stack + `_partial_trace_helper`)
7. Bindings + wire `get_backend('fusion_tree', …)`; smoke test; no monkey-patch yet

## TODO checklist

- [x] planning (this file)
- [x] declaration
- [x] definitions + compile
- [x] bindings + factory
- [ ] monkey-patch — deferred
- [ ] pytest — deferred
