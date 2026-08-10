# Conversion of Space

## metadata

- original python name: `Space`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration in C++ header file: `include/cyten/symmetries/spaces.h` (same file as `Leg`)
- definition in C++ file: `src/symmetries/spaces.cpp`
- pybind11 binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PySpace` in `pybind/symmetries/py_trampolines.hpp` (required — Python subclasses remain)
- first line of docstring: Base class for symmetry spaces, see ElementarySpace for the standard case.

## Module context (`spaces.py`)

1. `Leg` — **done** (C++ + bindings; monkey-patch deferred). See [convert_Leg.md](convert_Leg.md).
2. **`Space`** — this conversion (abstract base; parallel to `Leg`).
3. `LegPipe` → `ElementarySpace` → `TensorProduct` → `AbelianLegPipe` → free functions.

Keep original Python `Space` until subclasses (`ElementarySpace`, `TensorProduct`) are converted. Export `cyten._core.Space`; **defer monkey-patch**.

## Design notes

- Parallel hierarchy to `Leg` (not a subclass). `ElementarySpace` will later inherit both.
- `enable_shared_from_this` + `Space::Ptr` / `py::smart_holder`.
- Members:
  - `Symmetry::Ptr symmetry`
  - `SectorArray sector_decomposition`
  - `std::vector<int64> multiplicities`
  - `std::optional<std::string> sector_order` (`"sorted"` / `"dual_sorted"` / nullopt) — match Python literals
  - `int64 num_sectors`
  - `std::optional<std::vector<int64>> sector_dims` (only if `can_be_dropped`)
  - `std::vector<float64> sector_qdims` (always; equals sector_dims when droppable)
  - `std::optional<std::vector<std::array<int64, 2>>> slices` or flat `vector` of pairs / `py::array` — prefer `std::optional<std::vector<std::pair<int64,int64>>>` or 2×N structure
  - `float64 dim`
- Pure virtual: `dual()`, `change_symmetry(...)`, `drop_symmetry(...)`.
- `operator==`: default throws `TypeError`-like (`std::invalid_argument` / custom); subclasses may override. Python base raises `TypeError`.
- `as_ElementarySpace` / `change_symmetry` / `drop_symmetry` return `py::object` until `ElementarySpace` exists.
- `as_Space()` → `Space::Ptr` via `shared_from_this()`.
- `sector_map` for `change_symmetry`: `std::function<SectorArray(SectorArray const&)>` (bindings accept Python callable).
- `drop_symmetry(which)`: Python `'all' | int | list[int]` → C++ overload or `std::variant` / optional vector of factor indices (`nullopt` = all).
- Trampoline required (`ElementarySpace`, `TensorProduct` still Python).
- Reuse C++ `SectorArray::{lexsort_indices,row_where,unique_sorted,operator==}` and `Symmetry::{batch_sector_dim,batch_qdim,dual_sectors,are_valid_sectors}`.

## Dependencies

- Done: `Leg` (same header), `Symmetry`, `Sector` / `SectorArray`, `SymmetryError`
- Still Python: `ElementarySpace`, `TensorProduct`, `as_sector` / `as_sector_array` helpers (C++ has SectorArray ctors / casters)

## TODO list for conversion

- [x] initial setup (clean tree, `Space` listed, pytest green; branch `convert_Space` from `convert_Leg`)
- [x] planning (this file)
- [x] generate the declaration draft
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions
- [x] improve and fix the definition drafts
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline (`PySpace`)
- [x] monkey-patch — done with full `spaces.py` cleanup (see convert_spaces_functions.md)
- [x] run python tests (`test_spaces.py`; still using Python `Space`)
- [x] remove original python `Space`
- [ ] wrap up (then `LegPipe` / `ElementarySpace`)
