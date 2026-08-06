# Conversion of TensorProduct

## metadata

- original python name: `TensorProduct`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration: `include/cyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyTensorProduct` (optional — no Python subclass of TensorProduct in the library; still useful for consistency)
- first line of docstring: Represents a tensor product of Spaces, e.g. the (co-)domain of a tensor.

## Design notes

### Inheritance

Python/C++: `TensorProduct(Space)` only — no Leg MI. Override `dual_space`, `change_symmetry`, `drop_symmetry`, `operator==`.

### Heterogeneous `factors`

Python type is `list[Space | LegPipe]`. In practice: `ElementarySpace`, nested `TensorProduct`, `LegPipe` / `AbelianLegPipe`. Store as `std::vector<py::object>` so Python and C++ objects mix until monkey-patch; dispatch `flat_spaces` / `dual` / `change_symmetry` / `drop_symmetry` via attributes or typed casts when possible.

### Sector computation (`_calc_sectors`)

Hard part: abelian path uses `make_grid` (Python `cyten.tools.misc`) + `multiple_fusion_broadcast`; non-abelian recurses with `fusion_outcomes` / `_n_symbol`. Flatten pipes via `flat_spaces` then `as_Space`. Reuse existing C++ `_sort_sectors` / `_unique_sorted_sectors` helpers already in `spaces.cpp`.

### Iterators

`iter_uncoupled` / `iter_forest_blocks` / `iter_tree_blocks` are generators. Prefer C++ methods that fill `std::vector` of result structs (or accept a callback); expose Python generators in bindings. Depend on C++ `fusion_trees` / `FusionTree` in `trees.h`.

### Slices

Python `slice` → `std::array<int64,2>` `{start, stop}` or a small struct; bindings convert to `py::slice`.

### Monkey-patch

Deferred until `AbelianLegPipe` (and remaining free fns) if following the spaces module plan. Export `cyten._core.TensorProduct`.

## TODO

- [ ] setup / plan
- [ ] declaration
- [ ] definitions
- [ ] bindings (+ trampoline if useful)
- [ ] build / pytest
- [ ] monkey-patch deferred
