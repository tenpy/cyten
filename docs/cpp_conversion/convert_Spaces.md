# Conversion of Space (and related symmetries)

## Purpose of this file

Reminder and notes for when the Python `Space` type (and related symmetry types from `cyten.symmetries.spaces`) are converted to C++. This is not yet scheduled; the file exists to document a dependency left in place during the BlockBackend conversion.

## BlockBackend dependency: `apply_basis_perm`

During the conversion of `BlockBackend` to C++, the method **`apply_basis_perm(block, legs, inv)`** depends on `Space`: it takes `legs: list[Space]` and uses each leg’s `basis_perm` / `inverse_basis_perm` (index arrays) to permute the block’s axes.

**Current state:**

- **C++**: `BlockBackend::apply_basis_perm(block, legs, inv)` is **implemented in C++**. It accepts `std::vector<py::object> legs` (Python Space/leg objects). For each leg it reads `leg.attr("inverse_basis_perm")` or `leg.attr("basis_perm")`, builds `std::vector<py::array_t<cyten_int>> perms`, and calls `apply_leg_permutations(block, perms)`. No Python wrapper is needed; Python callers pass the same `list[Space]` and the binding converts it to `std::vector<py::object>`.
- **Python**: Callers use `backend.block_backend.apply_basis_perm(block, legs, inv=False)` as before; the C++ implementation is used directly.

**When converting Space to C++:**

1. Implement the real C++ `Space` type (and any needed subtypes, e.g. `ElementarySpace`) with at least `basis_perm` and `inverse_basis_perm` (e.g. as `std::vector<cyten_int>` or array views).
2. Optionally change `BlockBackend::apply_basis_perm` to accept `std::vector<Space const*>` (or C++ Space references) instead of `std::vector<py::object>`, and build perms from the C++ Space objects to avoid Python attribute access.
3. Bind the C++ Space type and have Python pass C++ Space objects (or keep passing Python Space and convert at the boundary via `py::object` until Space is fully migrated).

## Where Space is used (Python)

- `cyten.symmetries.spaces` (and related modules) define `Space` and subtypes.
- `BlockBackend.apply_basis_perm(self, block, legs: list[Space], inv=False)` is the only BlockBackend method that takes `Space` directly; `apply_leg_permutations(block, perms)` takes only index arrays and is backend/block-level.

## Metadata (to fill when conversion is planned)

- original python name: (e.g. Space, ElementarySpace)
- original python file: cyten/symmetries/spaces.py (or as identified)
- original python module: cyten.symmetries.spaces
- declaration in C++ header file: (TBD, e.g. include/cyten/symmetries/spaces.h)
- definition in C++ file: (TBD)
- pybind11 binding: (TBD)

## TODO (when conversion starts)

- [ ] Identify all Space-related types and their usage in block_backends and symmetries.
- [ ] Design C++ Space API (basis_perm / inverse_basis_perm, construction, etc.).
- [ ] Implement C++ `apply_basis_perm` in BlockBackend using C++ Space.
- [ ] Remove Python wrapper for `apply_basis_perm` and wire Python to C++.
