# Conversion of Sector / SectorArray (C++ foundation)

## Status

**In progress (types locked).** Python still uses ndarray type aliases; C++ owns the value types and pybind casters. Full Symmetry conversion (Layer 2) builds on this.

## Metadata

| Object | Original Python | C++ header | C++ source | Bindings |
| --- | --- | --- | --- | --- |
| `Sector` | type alias in `cyten/symmetries/_symmetries.py` | `include/cyten/symmetries/sector.h` | header-only | casters in `pybind/symmetries/casters.hpp` (not a Python class) |
| `SectorArray` | type alias in `cyten/symmetries/_symmetries.py` | same | header-only | same |

Python public API remains `Sector` / `SectorArray` as **NumPy ndarrays**. C++ does **not** expose `Sector` as a `py::class_`.

## Locked design decisions

- **`max_sector_ind_len = 7`**, components are **`int16_t`**.
- Owning value type:

```cpp
struct Sector {
  std::array<int16_t, 7> q{};
  std::uint8_t len = 0;
};
static_assert(sizeof(Sector) == 16);  // 128-bit-sized
```

- **`SectorArray`**: contiguous row-major `int16_t` buffer, shape `(num_sectors, sector_ind_len)` — not `std::vector<Sector>`.
- **Compile-time N**: factor helpers take `std::span<const int16_t, N>` (and mutable out-spans) via `Sector::as_span<N>()`, `Sector::subspan<N>(offset)`, or `SectorArray::row_as_span<N>(i)`. No separate `SectorFixed` / copy conversion layer.
- **Owning vs view**: virtual / public C++ APIs take and return owning `Sector` / `SectorArray`. Spans are non-owning internal views only.
- **No packed `int64_t` / `_compress_sector`**.
- **Do not template `Symmetry` / Spaces / Backends / Tensors on `N`.** Single non-templated `Sector` at the virtual boundary.

## Pybind caster contract

| Boundary | Conversion |
| --- | --- |
| Arg `Sector` | 1D integer sequence / ndarray, length ≤ 7 → copy into `Sector` (narrow to `int16_t` with range check) |
| Return `Sector` | `numpy.ndarray` shape `(len,)`, dtype compatible with Python `np.int_` (typically `int64`) |
| Arg/return `SectorArray` | 2D ndarray shape `(M, len)` ↔ contiguous `int16_t` buffer |
| `trivial_sector` (later) | property returning ndarray |

Casters live in `pybind/symmetries/casters.hpp`. Include that header from any binding translation unit that passes `Sector` / `SectorArray` across the Python boundary.

## Symmetry API constraint (for later conversion)

When converting Layer 2:

- `BaseSymmetry` / `SymmetryFactor` / `Symmetry` virtual methods use **`Sector` and `SectorArray` only** (non-templated).
- Concrete factor arithmetic may use `std::span<const int16_t, N>` into existing storage (product slices via `sector_slices`).
- Product `Symmetry` keeps runtime `sector_slices` and concatenates factor spans — same as Python.
- Trampolines / `py::smart_holder` follow the BlockBackend pattern; fixed-extent spans stay C++-only.

## Layer 2 conversion order

Convert in this order (Sector foundation first):

1. **`Sector` / `SectorArray`** (this doc) — C++ types + casters + CTest.
2. Exceptions / enums (`SymmetryError`, `FusionStyle`, `BraidingStyle`).
3. `BaseSymmetry` (virtual API on `Sector` / `SectorArray`).
4. `SymmetryFactor` and concrete factors (`NoSymmetry`, `U1`, `ZN`, `SU2`, …).
5. Product `Symmetry`.
6. Free functions in `_symmetries.py`.
7. Then `_su2data.py` → `trees.py` → `spaces.py`.

Use the usual pybind11-codegen workflow per object (`docs/cpp_conversion/convert_<pyname>.md`).

## TODO checklist

- [x] Lock `max_sector_ind_len = 7`, `int16_t` components, 128-bit-sized `Sector`
- [x] Implement `include/cyten/symmetries/sector.h`
- [x] Implement pybind type casters (no Python `Sector` class)
- [x] CTest for Sector / SectorArray and caster round-trips
- [x] Document Layer 2 order and non-templated Symmetry API constraint
- [ ] Monkey-patch / remove Python type aliases only when Symmetry is converted and tests pass (aliases stay as documentation until then)
