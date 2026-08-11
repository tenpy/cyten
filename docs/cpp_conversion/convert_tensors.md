# Layer 4 — Tensors conversion (`_tensors.py`)

## Status

Branch: **`convert_tensors`** (reuse for the whole module; no per-object branches).

Python source: [`cyten/tensors/_tensors.py`](../../cyten/tensors/_tensors.py) (~7500 lines).

Layer 3 backends still take tensor args as interim `py::object` in [`tensor_backend.h`](../../include/cyten/backends/tensor_backend.h). Replace those after tensor classes exist in C++.

## Conversion order

```mermaid
flowchart TD
  labels[Label helpers plus LabelledLegs]
  tensor[Tensor ABC plus trampoline]
  sym[SymmetricTensor]
  diag[DiagonalTensor]
  ident[Identity]
  mask[Mask]
  charged[ChargedTensor]
  helpers[Private helpers]
  ctors[Constructors]
  elem[Elementwise ops]
  algebra[Algebra ops]
  legs[Leg permutation ops]
  decomp[Decompositions]
  backendFix[Replace py object in TensorBackend]
  labels --> tensor
  tensor --> sym
  sym --> diag
  diag --> ident
  tensor --> mask
  sym --> charged
  diag --> mask
  labels --> helpers
  sym --> helpers
  helpers --> ctors
  helpers --> elem
  helpers --> algebra
  helpers --> legs
  helpers --> decomp
  charged --> algebra
  mask --> algebra
  decomp --> backendFix
```

| # | Object(s) | Status |
| --- | --- | --- |
| 1–2 | Label helpers + `LabelledLegs` | **C++ + bindings**; helpers monkey-patched; Python `LabelledLegs` kept — [convert_LabelledLegs.md](convert_LabelledLegs.md) |
| 3 | `Tensor` ABC + trampoline | **C++ + bindings + trampoline**; monkey-patch deferred — [convert_Tensor.md](convert_Tensor.md) |
| 4 | `SymmetricTensor` | **C++ + bindings + trampoline**; monkey-patch deferred — [convert_SymmetricTensor.md](convert_SymmetricTensor.md) |
| 5 | `DiagonalTensor` | **C++ + bindings + trampoline**; monkey-patch deferred — [convert_DiagonalTensor.md](convert_DiagonalTensor.md) |
| 6 | `Identity` | **C++ + bindings** (no trampoline); monkey-patch deferred — [convert_Identity.md](convert_Identity.md) |
| 7 | `Mask` | **C++ + bindings** (no trampoline); monkey-patch deferred — [convert_Mask.md](convert_Mask.md) |
| 8 | `ChargedTensor` | **C++ + bindings** (no trampoline); monkey-patch deferred — [convert_ChargedTensor.md](convert_ChargedTensor.md) |
| 9 | Private helpers (`_check_compatible_legs`, `_compose_*`, `_convert_*`, `_decomposition_*`, `_svd_new_labels`) | **C++ + bindings + monkey-patched** — [convert_tensor_helpers.md](convert_tensor_helpers.md) |
| 10 | Constructors (`eye`, `tensor`, `add_trivial_leg`, `zero_like`, `tensor_from_grid`) | **C++ + bindings + monkey-patched** — [convert_tensor_constructors.md](convert_tensor_constructors.md) |
| 11 | Elementwise ops (`angle`, `cutoff_inverse`, `complex_conj`, `imag`, `real`, `real_if_close`, `sqrt`, `stable_log`) | **C++ + bindings + monkey-patched** — [convert_tensor_elementwise.md](convert_tensor_elementwise.md) |
| 12 | Algebra ops (`almost_equal`, `compose`, `dagger`, `inner`, `item`, `linear_combination`, `norm`, `outer`, `partial_compose`, `partial_trace`, `pinv`, `scalar_multiply`, `scale_axis`, `tdot`, `trace`, `transpose`, `is_scalar`, `get_same_device`, `on_device`) | **C++ + bindings + monkey-patched** — [convert_tensor_algebra.md](convert_tensor_algebra.md) |
| 13 | Leg permutation ops | pending |
| 14 | Decompositions | pending |
| 15 | Backend `py::object` cleanup | pending |

Keep Python class bodies until **all** `Tensor` subclasses are converted (skill rule). `LabelledLegs` / label helpers may be monkey-patched earlier once bindings work; keep Python `LabelledLegs` until `Tensor` subclasses no longer need it, or monkey-patch carefully so `Tensor` still subclasses the C++ type.

## File layout

```
include/cyten/tensors.h
include/cyten/tensors/
  labels.h              # constants + label helpers + LabelledLegs
  tensor.h
  symmetric_tensor.h
  diagonal_tensor.h     # DiagonalTensor + Identity
  mask.h
  charged_tensor.h
  helpers.h
  constructors.h
  ops_elementwise.h
  ops_algebra.h
  ops_legs.h
  decompositions.h
src/tensors/            # matching .cpp; register in src/CMakeLists.txt
pybind/tensors/         # matching py_*.cpp; register in pybind/CMakeLists.txt
```

Namespace: `cyten::`.

## Design notes

- Label type: `std::optional<std::string>` (`None` ↔ empty optional).
- Label lists: `std::vector<std::optional<std::string>>`.
- `LabelledLegs` holder: `py::smart_holder` (subclasses will use `shared_ptr` / inheritance).
- Do **not** port `_elementwise_function` decorator; emit overloads later.
- `_dual_label_list` is not listed by codegen; convert manually with the other label helpers.
- `duplicate_entries` remains Python for now; implement a small C++ helper in `labels.cpp` (or inline) for `LabelledLegs`.

## Codegen

```bash
/opt/micromamba/envs/cyten_py314/bin/python .cursor/skills/pybind11-codegen/pybind11_codegen.py <cmd> ...
```
