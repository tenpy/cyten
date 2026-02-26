# Conversion of Space (and related symmetries)

## Purpose of this file

Reminder and notes for when the Python `Space` type (and related symmetry types from `cyten.symmetries.spaces`) are converted to C++. This is not yet scheduled; the file exists to document a dependency left in place during the BlockBackend conversion.

## BlockBackend dependency: `apply_basis_perm`

During the conversion of `BlockBackend` to C++, the method **`apply_basis_perm(block, legs, inv)`** depends on `Space`: it takes `legs: list[Space]` and uses each leg’s `basis_perm` / `inverse_basis_perm` (index arrays) to permute the block’s axes.

**Current (planned) state until Space is converted:**

- **C++**: `BlockBackend` declares `apply_basis_perm` with a **forward-declared** `Space` type (e.g. `const std::vector<Space>&` or `const std::vector<const Space*>&` once `Space` exists). The C++ implementation **throws `NotImplemented`** so the signature is in place but no real logic runs in C++.
- **Python**: The original `apply_basis_perm` logic remains in Python behind a **small wrapper**: the Python `BlockBackend` (or a thin wrapper around the C++ backend) still implements `apply_basis_perm` using Python `Space` objects (building the list of perms from `leg.basis_perm` / `leg.inverse_basis_perm` and calling `apply_leg_permutations`). So Python callers keep the current behavior.

**When converting Space to C++:**

1. Implement the real C++ `Space` type (and any needed subtypes, e.g. `ElementarySpace`) with at least `basis_perm` and `inverse_basis_perm` (e.g. as `std::vector<cyten_int>` or array views).
2. In `BlockBackend`, replace the `apply_basis_perm` stub with the actual implementation: build the perms from the C++ `Space` legs and call `apply_leg_permutations(block, perms)`.
3. Remove the Python-side wrapper for `apply_basis_perm` and have Python call the C++ implementation (with Space objects bound from C++ or converted at the boundary).

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
