# Conversion of FusionTreeBackend

## Status

**Not started** (data class done separately). `FusionTreeData` is in C++ — see [convert_FusionTreeData.md](convert_FusionTreeData.md). The ~3.7k-line `FusionTreeBackend` plus helpers (`Instruction`, `TensorMapping`, …) remain Python.

## Metadata (planned)

| Field | Value |
| --- | --- |
| original python name | `FusionTreeBackend` (+ helpers as needed) |
| original python file | `cyten/backends/fusion_tree_backend.py` |
| declaration | `include/cyten/backends/fusion_tree_backend.h` (extend existing) |
| definition | `src/backends/fusion_tree_backend.cpp` (may split) |
| pybind11 binding | `pybind/backends/py_fusion_tree_backend.cpp` |
| trampoline | only if Python subclasses remain |

## Suggested internal order

1. `FusionTreeBackend` core overrides
2. `PermuteLegsInstructionEngine` / `Instruction` hierarchy if still required
3. `TensorMapping` / `TreePairMapping` / `FactorizedTreeMapping`
4. Helpers `_partial_trace_helper`, `_tree_block_iter`

## TODO

- [ ] Plan split of headers/sources if unit size is too large
- [ ] declaration / definitions / bindings
- [ ] monkey-patch with other backends
- [ ] pytest (`tests/python_tests/backends/test_fusion_tree_backend.py`)
