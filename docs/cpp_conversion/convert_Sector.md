# Conversion of Sector / SectorArray (C++ foundation)

## Status

**Done (types locked).** `Sector` / `SectorArray` are C++ types exposed as Python classes via pybind; NumPy remains available through `to_numpy()` and load casters.

## Metadata

| Object | Original Python | C++ header | C++ source | Bindings |
| --- | --- | --- | --- | --- |
| `Sector` | type alias → class | `include/cyten/symmetries/sector.h` | + `sector_numpy.cpp` (HDF5 / NumPy) | `pybind/symmetries/py_sector.cpp` + casters |
| `SectorArray` | type alias → class | same | `src/symmetries/sector_array.cpp` | same |

## Locked design decisions

- **`max_sector_ind_len = 7`**, components are **`int16_t`**.
- Owning value type:

```cpp
class Sector {
  std::array<int16_t, 7> q{};
  // private len_; set only by constructors / from_span / zeros (n <= max_sector_ind_len)
};
static_assert(sizeof(Sector) == 16);  // 128-bit-sized
```

- **`SectorArray`**: `class SectorArray : public std::vector<Sector>` with a separate `sector_ind_len_` so empty arrays still know their width (`SectorArray::empty(N)`). Uniform sector length is checked at construction / typed mutators (`push_back`, `resize`).
- **Element access**: `operator[]` returns `Sector&` / `const Sector&` (no copy via `from_span`).
- **Ops as methods**: former `sector_ops` free functions are methods/statics (`lexsort_indices`, `sorted`, `concat`, `from_sector`, `repeat`, `take`, `slice`, `unique_sorted`, `iter_common_sorted`, …). No free-function wrappers.
- **Compile-time N**: factor helpers take `std::span<const int16_t, N>` via `Sector::as_span<N>()` / `subspan<N>()`.
- **Owning vs view**: virtual / public C++ APIs take and return owning `Sector` / `SectorArray`. Spans are non-owning internal views only.
- **Do not template `Symmetry` / Spaces / Backends / Tensors on `N`.**

## Pybind contract

| Boundary | Conversion |
| --- | --- |
| Arg `Sector` / `SectorArray` | Bound instance, or ndarray/sequence via load caster |
| Return | Bound `Sector` / `SectorArray` (never bare ndarray) |
| Explicit NumPy | `.to_numpy()` copies to `int64` arrays |

Casters live in `pybind/symmetries/casters.hpp`.

## TODO checklist

- [x] Lock `max_sector_ind_len = 7`, `int16_t` components, 128-bit-sized `Sector`
- [x] Implement `Sector` / `SectorArray` (`vector<Sector>` + width)
- [x] Ops as `SectorArray` methods; delete free `sector_ops` API
- [x] Python class bindings + load casters + `to_numpy` / HDF5
- [x] CTest for Sector / SectorArray
- [x] Migrate C++ and Python call sites to the method API
