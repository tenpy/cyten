# Conversion of TreePairMapping (+ FactorizedTreeMapping, Instructions, Engine)

## Status

**Done / monkey-patched** on branch `convert_backends`. Exported on `cyten._core` and re-exported from `cyten.backends.fusion_tree_backend` (`TreePairMapping`, `FactorizedTreeMapping`, Instructions, `PermuteLegsInstructionEngine`).

| Component | C++ location |
| --- | --- |
| `BraidInstruction`, `BendInstruction`, `TwistInstruction` | `include/cyten/backends/fusion_tree_mapping.h` |
| `TensorMapping` (ABC), `TreePairMapping`, `FactorizedTreeMapping` | same + `src/backends/fusion_tree_mapping.cpp` |
| `PermuteLegsInstructionEngine` | `include/cyten/backends/fusion_tree_permute.h` + `src/backends/fusion_tree_permute.cpp` |
| pybind11 | `pybind/backends/py_fusion_tree_mapping.cpp` |

`FusionTreeBackend::apply_instructions` and `permute_legs` now call the native mapping stack (no Python `TreePairMapping` / `PermuteLegsInstructionEngine` delegation).

## Design notes

- `Instruction = std::variant<BraidInstruction, BendInstruction, TwistInstruction>`.
- `TreePairMapping` holds `SparseMappingFusionTreePair`.
- `FactorizedTreeMapping` uses `std::variant<IdentityMappingFusionTree, SparseMappingFusionTree>` for splitting/fusion maps; `IdentityMapping::pre_compose(Sparse)` promotes to sparse after braid/twist.
- `FusionTreeLinearCombination` / `FusionTreePairLinearCombination` (`std::map`) converted to sparse inner rows via `to_inner` / `to_inner_pair`.
- `transform_tensor` uses `misc.iter_common_sorted_arrays`, `inverse_permutation` (local helper), and `BlockBackend::permute_combined_*`.
- `PermuteLegsInstructionEngine` ports Python logic including native `permutation_as_swaps`.

## TODO checklist

- [x] headers + sources + CMake
- [x] wire `apply_instructions` / `permute_legs`
- [x] bindings + smoke
- [x] monkey-patch via `fusion_tree_backend.py`
- [x] pytest fusion_tree permute suite
