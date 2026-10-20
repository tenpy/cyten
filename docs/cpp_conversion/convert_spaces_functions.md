# Conversion of spaces.py free functions

## metadata

- original python names: `swap_gate`, `twist_gate`, `_twist_gate_diag`, `_flat_leg_permutation`, `_unique_sorted_sectors`, `_sort_sectors`, `_parse_inputs_drop_symmetry`
- original python file: `lasyten/symmetries/spaces.py`
- declaration: `include/lasyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`

## Notes

- `_sort_sectors` / `_parse_inputs_drop_symmetry` already existed as anonymous helpers; exposed as `sort_sectors_public` / `parse_inputs_drop_symmetry_public`.
- `swap_gate` / `twist_gate` take `Leg::Ptr`. Plain `LegPipe` (not also `ElementarySpace`) uses the recursive pipe path; `AbelianLegPipe` follows the ES path (Python `isinstance`).
- Module-level `_flat_leg_permutation` is imported by `fusion_tree_backend.py`.
## Monkey-patch / cleanup

Python class and free-function bodies were removed from `lasyten/symmetries/spaces.py`.
The module is now a thin re-export of the C++ types and functions from `lasyten._core`.
