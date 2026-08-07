# Conversion of spaces.py free functions

## metadata

- original python names: `swap_gate`, `twist_gate`, `_twist_gate_diag`, `_flat_leg_permutation`, `_unique_sorted_sectors`, `_sort_sectors`, `_parse_inputs_drop_symmetry`
- original python file: `cyten/symmetries/spaces.py`
- declaration: `include/cyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`

## Notes

- `_sort_sectors` / `_parse_inputs_drop_symmetry` already exist as anonymous helpers used by class methods; expose thin public wrappers for bindings.
- `_unique_sorted_sectors` → `SectorArray::unique_sorted`.
- `swap_gate` / `twist_gate` take `Leg::Ptr`; plain `LegPipe` (not also `ElementarySpace`) uses the recursive pipe path — `AbelianLegPipe` follows the ES path (Python `isinstance`).
- Module-level `_flat_leg_permutation` is imported by `fusion_tree_backend.py` — must monkey-patch.
- Public exports: `swap_gate`, `twist_gate`; underscore names bound to match Python.

## TODO

- [ ] declaration
- [ ] definitions
- [ ] bindings + monkey-patch
- [ ] pytest
