# Conversion of SparseMapping (+ IdentityMapping)

## Status

**In progress** on branch `convert_backends`. Templated C++ `SparseMapping` / `IdentityMapping` with `std::unordered_map` storage. Exported as concrete `_core` aliases only — **not** monkey-patched into `cyten.tools`. Used by the FusionTreeBackend mapping stack (still Python until that stack is ported).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `SparseMapping`, `IdentityMapping` |
| original python file | `cyten/tools/mappings.py` |
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
- Need `std::hash` for `std::pair<FusionTree, FusionTree>` (combine of tree hashes).
- No generic `SparseMapping<py::object>` binding.

## TODO checklist

- [x] planning (this file)
- [ ] declaration (templated header + pair hash)
- [ ] definitions + explicit instantiations + compile
- [ ] bindings + smoke
- [ ] monkey-patch — deferred
- [ ] pytest — deferred (covered indirectly by FT backend tests later)
