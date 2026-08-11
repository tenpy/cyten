# Conversion of SparseMapping (+ IdentityMapping)

## Status

**Done** on branch `convert_backends`. C++ templates + bindings on `cyten._core` as `SparseMappingFusionTree`, `SparseMappingFusionTreePair`, `IdentityMappingFusionTree`, `IdentityMappingFusionTreePair`. Used by the FusionTree mapping stack; no Python `cyten.tools.mappings` module (removed — nothing called it after Layer 3 monkey-patch).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `SparseMapping`, `IdentityMapping` |
| original python file | `cyten/tools/mappings.py` (removed) |
| original python module | `cyten.tools` |
| declaration | `include/cyten/tools/mappings.h` |
| definition | header-only templates + `src/tools/mappings.cpp` (explicit instantiations) |
| pybind11 binding | `pybind/tools/py_mappings.cpp` |
| trampoline | no |
| first line of docstring | A sparse matrix, where the labels of basis states are a structured type, not just int. |

## Design notes

- Template: `SparseMapping<KT, Scalar = complex128>` wrapping nested `std::unordered_map<KT, std::unordered_map<KT, Scalar>>`.
- Storage is **unordered** (Python was `dict`); do not use `std::map`.
- `prune(tol)` is **in-place and returns `void`** (Python returns `self`; C++ drops fluent chaining).
- `IdentityMapping<KT>` holds `std::unordered_set<KT> keys`; same method surface; `prune` is a no-op.
- Concrete aliases (and only these are bound):
  - `SparseMappingFusionTree` / `IdentityMappingFusionTree` — `KT = FusionTree`
  - `SparseMappingFusionTreePair` / `IdentityMappingFusionTreePair` — `KT = std::pair<FusionTree, FusionTree>`
- `std::hash` for `std::pair<FusionTree, FusionTree>` in `trees.h`.
- No generic `SparseMapping<py::object>` binding — no monkey-patch of a generic Python API.

## TODO checklist

- [x] planning (this file)
- [x] declaration (templated header + pair hash)
- [x] definitions + explicit instantiations + compile
- [x] bindings + smoke
- [x] remove unused Python `cyten/tools/mappings.py` (no monkey-patch)
- [x] pytest — covered indirectly by FT backend / permute tests
