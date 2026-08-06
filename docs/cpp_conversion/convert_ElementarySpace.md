# Conversion of ElementarySpace

## metadata

- original python name: `ElementarySpace`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration: `include/cyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyElementarySpace` (required — `AbelianLegPipe` subclasses)
- first line of docstring: A Space that is defined as (the dual of) a direct sum of sectors.

## Design notes

### Multiple inheritance

Python: `ElementarySpace(Space, Leg)`. C++: `class ElementarySpace : public Space, public virtual Leg`.

`LegPipe` updated to `public virtual Leg` so future `AbelianLegPipe(LegPipe, ElementarySpace)` has a single `Leg` subobject.

### Conflicting `dual()` return types

`Space::dual` → `Space::Ptr` and `Leg::dual` → `Leg::Ptr` cannot both be overridden under one name. Resolve with intermediate hooks:

- `Space`: keep `dual()` as non-pure wrapper calling pure `dual_space()`
- `Leg`: keep `dual()` as non-pure wrapper calling pure `dual_leg()`
- `ElementarySpace`: implement both hooks to return the same `ElementarySpace::Ptr`

### Shared state

`Space` and `Leg` both store `symmetry` / `dim`; ctor initializes `Space` first, then `Leg` with `Space::dim`. `is_dual` lives on `Leg`. Extra member: `defining_sectors`.

### Factories

`from_basis`, `from_defining_sectors`, `from_sector_decomposition`, `from_null_space`, `from_trivial_sector`, `from_largest_common_subspace`, `from_independent_symmetries` — port helpers `_sort_sectors` into `spaces.cpp`. Prefer C++ `SectorArray` ops; call Python for `rank_data` / `format_like_list` where useful.

### Return types

`as_ElementarySpace` / `change_symmetry` / `drop_symmetry` on `Space`/`Leg` become able to return real `ElementarySpace` (`py::object` or `Ptr`). Update `Space::as_ElementarySpace` to construct C++ `ElementarySpace`.

### Monkey-patch

Deferred until `AbelianLegPipe` is converted (MI). Export `cyten._core.ElementarySpace`.

## TODO

- [x] setup / plan
- [ ] declaration (hooks for dual; virtual Leg)
- [ ] definitions
- [ ] bindings + trampoline
- [ ] build / pytest
- [ ] monkey-patch deferred
