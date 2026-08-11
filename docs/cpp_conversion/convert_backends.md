# Layer 3 — Tensor backends conversion

## Status

**Done / monkey-patched** on branch `convert_backends`. Layer 3 backends and `get_backend` are imported from `cyten._core` into `cyten.backends`.

Monkey-patched:

- `TensorBackend` + `conventional_leg_order` + `get_same_backend` — [convert_TensorBackend.md](convert_TensorBackend.md)
- `NoSymmetryBackend` — [convert_NoSymmetryBackend.md](convert_NoSymmetryBackend.md)
- `AbelianBackendData` + `AbelianBackend` + `valid_block_inds` — [convert_AbelianBackendData.md](convert_AbelianBackendData.md), [convert_AbelianBackend.md](convert_AbelianBackend.md)
- `FusionTreeData` + `FusionTreeBackend` (+ Instruction / Mapping helpers; keeps `_tree_block_iter` in Python) — [convert_FusionTreeData.md](convert_FusionTreeData.md), [convert_FusionTreeBackend.md](convert_FusionTreeBackend.md), [convert_TreePairMapping.md](convert_TreePairMapping.md)
- `get_backend` — [convert_get_backend.md](convert_get_backend.md)

Still Python (not Layer 3 backends):

- Layer 4 tensor classes — backends reach them via `py::object`

Related (not a backend; C++ only, Python module removed): [convert_SparseMapping.md](convert_SparseMapping.md).

## Conversion order

```mermaid
flowchart TD
  TB[TensorBackend plus helpers]
  NS[NoSymmetryBackend]
  ABD[AbelianBackendData]
  AB[AbelianBackend]
  FTD[FusionTreeData]
  FT[FusionTreeBackend plus helpers]
  BF[get_backend]
  TB --> NS
  TB --> ABD --> AB
  TB --> FTD --> FT
  NS --> BF
  AB --> BF
  FT --> BF
```

| Order | Object(s) | Python source | Notes |
| --- | --- | --- | --- |
| 1 | `TensorBackend`, `conventional_leg_order`, `get_same_backend` | `cyten/backends/_backend.py` | Skip `HasBackend` Protocol |
| 2 | `NoSymmetryBackend` | `cyten/backends/no_symmetry.py` | Smallest concrete backend |
| 3 | `AbelianBackendData` then `AbelianBackend` (+ `_valid_block_inds`) | `cyten/backends/abelian.py` | |
| 4 | `FusionTreeData` then `FusionTreeBackend` (+ Instruction/Mapping helpers as needed) | `cyten/backends/fusion_tree_backend.py` | Largest; may split headers |
| 5 | `get_backend` | `cyten/backends/backend_factory.py` | |

## File layout

| Kind | Path |
| --- | --- |
| Header | `include/cyten/backends/tensor_backend.h` (then `no_symmetry.h`, `abelian.h`, `fusion_tree_backend.h`, …) |
| Source | `src/backends/tensor_backend.cpp` (+ register in `src/CMakeLists.txt`) |
| Bindings | `pybind/backends/py_tensor_backend.cpp` |
| Trampoline | `pybind/backends/py_trampolines.hpp` |
| Optional umbrella | `include/cyten/backends.h` (later) |

## Design decisions

### `Data` / `DiagonalData` / `MaskData`

- Nested abstract `TensorBackend::Data` (`enable_shared_from_this`, `DataPtr` / `DataCPtr`), analogous to `BlockBackend::Block`.
- Concrete backends own concrete data types inheriting that base (`AbelianBackendData`, `FusionTreeData`; NoSymmetry uses a thin `Data` holding `BlockBackend::BlockPtr`).
- On the abstract API, `Data` / `DiagonalData` / `MaskData` all map to `DataPtr`.

### Tensor circular dependency (Layer 4 not ready)

- Map `SymmetricTensor` / `DiagonalTensor` / `Mask` → `py::object` interim in declarations and `pybind11_codegen.toml`.
- Keep C++ types for: `BlockBackend`, `Block`, `Scalar`, `Dtype`, spaces, `Symmetry`, `FusionTree`, `Sector`.
- Callables → `py::function` until a clearer C++ callback typedef is justified.
- When Layer 4 lands, replace `py::object` tensor args with real C++ tensor types.

### Trampoline + monkey-patch timing

- Generate trampoline so Python subclasses can remain until converted.
- Monkey-patch of `TensorBackend` and concrete backends into `cyten.backends` is **done** (concrete backends are C++).
- Original Python backend bodies were removed; modules re-export from `_core` (FT keeps `_tree_block_iter`).

## Codegen

Use micromamba Python (system Python lacks `ast_comments`):

```bash
/opt/micromamba/envs/cyten_py314/bin/python .cursor/skills/pybind11-codegen/pybind11_codegen.py <cmd> ...
```

## Dependencies

- Layer 1: `BlockBackend`, `NumpyBlockBackend`, `Block`, `Scalar`, `Dtype` — done.
- Layer 2: symmetries, trees, spaces — largely done / monkey-patched.
- Layer 4 tensors — still Python; backends reach them via `py::object` for now.
