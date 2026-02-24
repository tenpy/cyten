# C++ conversion tracking

This folder stores per-object conversion plans and notes for the Python-to-C++ migration of the cyten core library.

Each conversion follows the workflow in [.cursor/skills/pybind11-codegen/SKILL.md](../../.cursor/skills/pybind11-codegen/SKILL.md) and is documented in a file `convert_<pyname>.md` (e.g. `convert_format_like_list.md`).

Conversion order follows the plan: Layer 0 (tools) → Layer 1 (block_backends) → Layer 2 (symmetries) → Layer 3 (backends) → Layer 4 (tensors) → Layer 5 (models).


## Conversion workflow (per object)

The project uses a **one-class-or-function-at-a-time** workflow with the pybind11 codegen in [.cursor/skills/pybind11-codegen/](.cursor/skills/pybind11-codegen/) (see [SKILL.md](.cursor/skills/pybind11-codegen/SKILL.md)):

1. **Setup** — Clean git, branch `convert_<pyname>`, run pytest for the module’s tests.
2. **Plan** — Create `docs/cpp_conversion/convert_<pyname>.md` from the template (e.g. in `.cursor/skills/pybind11-codegen/assets/convert_pyobj.md`), list dependencies and TODOs.
3. **Declaration** — `gen_cpp_declaration --py-name <pyname> --header-file <header>`, then fix types/namespaces (C++23, use `pybind11_codegen.toml` type mappings).
4. **Definitions** — `gen_cpp_definition` into a `.cpp`, fix CHECKME/FIXME, get clang-tidy clean and CTest passing.
5. **Bindings** — `gen_pyb11_binding` (and trampoline for virtual classes), adjust if C++ API diverged.
6. **Monkey-patch** — In the original Python file, right below the (removed) definition, add `from .._core import <pyname>`.
7. **Tests** — Run pytest for the converted object, then full suite; remove original Python only after all tests pass.

Script entry point: run the codegen from the repo root; the script lives under `.cursor/skills/pybind11-codegen/` (e.g. `python .cursor/skills/pybind11-codegen/pybind11_codegen.py list_python_names`). Config: [pybind11_codegen.toml](pybind11_codegen.toml).

## Dependency order and module breakdown

Convert in **dependency order** so that C++ types exist before use. Suggested layers:

```mermaid
flowchart LR
  subgraph layer0 [Layer 0 - Tools]
    mappings
    misc
    string
    dtypes
    cost_polynomials
    math
  end
  subgraph layer1 [Layer 1 - Block backends]
    block_backend
    numpy
    array_api
  end
  subgraph layer2 [Layer 2 - Symmetries]
    symmetries
    su2data
    trees
    spaces
  end
  subgraph layer3 [Layer 3 - Backends]
    backend
    no_symmetry
    abelian
    fusion_tree_backend
    backend_factory
  end
  subgraph layer4 [Layer 4 - Tensors]
    tensors
    planar
    sparse
    krylov_based
  end
  subgraph layer5 [Layer 5 - Models]
    dof
    sites
    couplings
    tenpy_models
  end
  layer0 --> layer1
  layer1 --> layer2
  layer2 --> layer3
  layer3 --> layer4
  layer4 --> layer5
```



### Layer 0 — Minimal tools

- **cyten/tools/mappings.py** — `IdentityMapping`, `SparseMapping`, etc. (small).
- **cyten/tools/misc.py** — `to_iterable`, `rank_data`, `argsort`, `combine_constraints`, `as_immutable_array`, `duplicate_entries`, `inverse_permutation`, `iter_common_sorted_arrays`, `to_valid_idx`, `is_iterable` (used everywhere).
- **cyten/tools/string.py** — `format_like_list` (tiny).
- **cyten/block_backends/dtypes.py** — `Dtype`, numpy/cyten dtype mapping (needed by block_backends).
- **cyten/tools/cost_polynomials.py** — `BigOPolynomial` (used by planar/tensors).
- **cyten/tools/math.py** — `speigs`, `speigsh` (used by sparse/tensors).

Convert **functions and classes** in each file in dependency order within the file (e.g. helpers before public API). Use `list_python_names` to get the exact list of convertible names per module.

### Layer 1 — Block backends (no torch)

- **cyten/block_backends/_block_backend.py** — `BlockBackend` (abstract), `Block` type. Many methods; convert base class first, then concrete backends.
- **cyten/block_backends/numpy.py** — `NumpyBlockBackend`.
- **cyten/block_backends/array_api.py** — `ArrayApiBlockBackend`.

Skip `block_backends/torch.py` per scope.

### Layer 2 — Symmetries

- **cyten/symmetries/_symmetries.py** — Large (~2.6k lines). Contains `SymmetryError`, `Sector`/`SectorArray`, `FusionStyle`, `Symmetry` (abstract), `TensorProduct`, concrete symmetries (`U1`, `no_symmetry`, etc.), and many functions. Convert in file order: exceptions → types → enums → base `Symmetry` → concrete symmetries → free functions.
- **cyten/symmetries/_su2data.py** — SU2 data (small).
- **cyten/symmetries/trees.py** — `FusionTree`, `fusion_trees` (~1k lines).
- **cyten/symmetries/spaces.py** — `ElementarySpace`, `Leg`, `LegPipe`, `Space`, `TensorProduct`, `AbelianLegPipe` (~2.2k lines). Depends on symmetries and trees.

### Layer 3 — Backends

- **cyten/backends/_backend.py** — `TensorBackend` (abstract), `Data`/`DiagonalData`/`MaskData`, `conventional_leg_order`, `get_same_backend`, etc. (~787 lines).
- **cyten/backends/no_symmetry.py** — `NoSymmetryBackend` (~476 lines).
- **cyten/backends/abelian.py** — `AbelianBackend`, `AbelianBackendData` (~1.9k lines).
- **cyten/backends/fusion_tree_backend.py** — `FusionTreeBackend`, `FusionTreeData` (~3.3k lines).
- **cyten/backends/backend_factory.py** — `get_backend` (~65 lines).

### Layer 4 — Tensors

- **cyten/tensors/_tensors.py** — **Very large** (~6.8k lines). Many classes: `LabelledLegs`, `Tensor`, `ChargedTensor`, `SymmetricTensor`, `DiagonalTensor`, `Mask`, and many functions (`add`, `permute_legs`, `combine_legs`, `compose`, etc.). **Strategy:** Convert in dependency order within the file (e.g. `LabelledLegs` → `Tensor` → `ChargedTensor` → `SymmetricTensor` → …). Consider multiple headers/sources (e.g. `tensors/core.h/.cpp`, `tensors/ops.h/.cpp`) and multiple bindings files to keep units manageable. Use `list_python_names` to get the full list and split into batches.
- **cyten/tensors/planar.py** — Planar diagram utilities (~1.2k lines), depends on `_tensors` and tools.
- **cyten/tensors/sparse.py** — `LinearOperator`, `ProjectedLinearOperator`, etc. (~627 lines).
- **cyten/tensors/krylov_based.py** — `Arnoldi`, `LanczosEvolution`, etc. (~521 lines).

### Layer 5 — Models

- **cyten/models/degrees_of_freedom.py** — DOF classes, `Site` (~714 lines).
- **cyten/models/sites.py** — `SpinSite`, `GoldenSite`, etc. (~655 lines).
- **cyten/models/couplings.py** — `Coupling`, `gold_coupling`, etc. (~560 lines).
- **cyten/models/tenpy_models.py** — Tenpy model helpers (~130 lines).

## C++ layout

- **Headers:** [include/cyten/](include/cyten/) — add e.g. `tools.hpp`, `block_backend.hpp`, `symmetries.hpp`, `spaces.hpp`, `trees.hpp`, `backend.hpp`, `tensors.hpp`, `models.hpp` (or split further as needed).
- **Sources:** Add under a `src/` (or equivalent) and register in CMake.
- **Bindings:** [pybind/](pybind/) — extend `_core.cpp` or add separate `.cpp` files and include them in the same module.

Keep namespaces and include structure consistent with [include/cyten/cyten.h](include/cyten/cyten.h) and existing [include/cyten/config.h](include/cyten/config.h) / [include/cyten/version.h](include/cyten/version.h).



## Practical recommendations

1. **Create `docs/cpp_conversion/`** — Store one `convert_<pyname>.md` per conversion (and optionally high-level `convert_tensors.md`, `convert_symmetries.md`) to track progress and decisions.
2. **Extend type mappings** — Update [pybind11_codegen.toml](pybind11_codegen.toml) `[types_py_to_cpp]` as you find recurring types (e.g. `npt.NDArray` → appropriate C++ container or pybind11 buffer).
3. **Trampolines** — Use `gen_pyb11_trampoline` only for classes that are subclassed in Python (e.g. `TensorBackend`, `BlockBackend`, `Symmetry`); for internal-only bases, bindings may not need trampolines.
4. **Tests** — For each converted object, run the relevant pytest under `tests/`; keep the full suite passing before removing the corresponding Python implementation.
5. **Incremental commits** — Per the skill: WIP commits with `--no-verify` (e.g. "WIP: convert : ") so pre-commit doesn’t block mid-conversion.
