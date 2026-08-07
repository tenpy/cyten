# Conversion of Space (and related symmetries) — BlockBackend follow-up

## Purpose

Notes left during the BlockBackend conversion for when `Space` / legs existed only in Python.
The spaces hierarchy is now C++ (`Leg`, `Space`, `ElementarySpace`, …); this follow-up wires
`BlockBackend::apply_basis_perm` to those types.

## `apply_basis_perm`

- Takes `std::vector<BlockBackend::LegCPtr>` (`shared_ptr<const Leg>`).
- Uses `Leg::basis_perm()` / `Leg::inverse_basis_perm()` (C++), then `apply_leg_permutations`.
- Python binding casts each sequence element to `Leg` (covers `ElementarySpace`, `LegPipe`,
  `AbelianLegPipe`).
- No separate Python wrapper: callers already use the C++ method via pybind.

`basis_perm` lives on :class:`Leg`, not bare :class:`Space` — hence `Leg` rather than `Space*`.

## Metadata

- declaration: `include/cyten/block_backend/block_backend.h`
- definition: `src/block_backend/block_backend.cpp`
- binding: `pybind/block_backend/py_block_backend.cpp`
- related spaces API: `include/cyten/symmetries/spaces.h`

## TODO

- [x] Identify Space-related usage in block backends (`apply_basis_perm` only)
- [x] C++ Space/Leg API with `basis_perm` / `inverse_basis_perm` (done in spaces conversion)
- [x] Implement `apply_basis_perm` using C++ `Leg`
- [x] No Python wrapper to remove (already C++-only)
