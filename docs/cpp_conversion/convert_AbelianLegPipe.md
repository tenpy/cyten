# Conversion of AbelianLegPipe

## metadata

- original python name: `AbelianLegPipe`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration: `include/cyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyAbelianLegPipe` (optional; no further Python subclasses expected)
- first line of docstring: Special case of a LegPipe for abelian group symmetries.

## Design notes

### Diamond MI

Python: `AbelianLegPipe(LegPipe, ElementarySpace)`. C++:

```cpp
class AbelianLegPipe : public LegPipe, public ElementarySpace
```

`Leg` is already `virtual` on both parents → one `Leg` subobject. `LegOrSpace` likewise. Construction: `prepare(...)` computes sectors / maps / `basis_perm`, then delegated ctor runs `LegPipe(...)` then `ElementarySpace(...)` (second `init_leg` wins with the fusion `basis_perm`, matching Python).

### Extra members

- `sector_strides` — 1D int
- `fusion_outcomes_sort` — 1D int (permutation)
- `block_ind_map_slices` — 1D int, length `num_sectors + 1`
- `block_ind_map` — 2D int shape `(M, 3 + num_legs)` — store as `std::vector<std::vector<int64>>` or flat + ncol; expose as numpy in bindings

### Hard helpers

`_calc_sectors`, `_calc_basis_perm`, `_get_fusion_outcomes_perm` — port using C++ `make_grid`/`make_stride` (implement locally or call Python `cyten.tools.misc`) and `Symmetry::multiple_fusion_broadcast`.

### Overrides

- `is_abelian_leg_pipe()` → true
- `flat_spaces()` → `{ shared_leg() }` (do not flatten nested AbelianLegPipes)
- `as_Space` → return self as py::object
- `dual_leg` / `dual_space` / `dual_es` → new `AbelianLegPipe` with reversed dual legs
- Unsupported ES factories → `TypeError`
- `set_basis_perm` → `TypeError`
- `take_slice` → warn + delegate to ElementarySpace
- `operator==` — LegPipe equality plus `combine_cstyle` (already in LegPipe? check)

### Monkey-patch

This is the last MI class in `spaces.py`. Export `cyten._core.AbelianLegPipe`. Prefer monkey-patching **only** `AbelianLegPipe` first; full hierarchy monkey-patch still optional if interop is fragile.

## TODO

- [ ] setup / plan
- [ ] declaration
- [ ] definitions
- [ ] bindings (+ trampoline)
- [ ] build / pytest / parity
- [ ] monkey-patch (AbelianLegPipe)
