# Conversion of tensor private helpers (batch 9)

## metadata

- original python names: `_check_compatible_legs`, `_compose_with_Mask`, `_compose_SymmetricTensors`, `_convert_abelian_to_FT`, `_convert_FT_to_abelian`, `_decomposition_prepare`, `_decomposition_labels`, `_svd_new_labels`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration: `include/cyten/tensors/helpers.h`
- definition: `src/tensors/helpers.cpp`
- pybind11 binding: `pybind/tensors/py_helpers.cpp`
- trampoline: n/a (free functions)
- note: skip `_elementwise_function` (decorator — skill: do not port)

## Module context

Classes 1–8 done. This is **free-function batch 9** (Private helpers) per [convert_tensors.md](convert_tensors.md).
Next batches: constructors → elementwise → algebra → legs → decompositions.

Stay on branch **`convert_tensors`**.

## Design notes

- Free functions in namespace `cyten::`.
- Tensor / Mask args for compose helpers: `py::object` so Python and C++ tensors both work until the Tensor hierarchy is monkey-patched. Results are built via the **Python** `SymmetricTensor` / `ChargedTensor` constructors so `isinstance` keeps working.
- NoSymmetry: C++ `DataPtr` is `BlockData`; unwrap to `Block` before constructing Python tensors (`data_as_python`).
- `_compose_with_Mask`: rebind `leg_idx` from `_parse_leg_idx` (same as Python) before recursive ChargedTensor call / backend contract.
- `_check_compatible_legs`: compare via Python `__eq__` (not `py::object::equal` / pointer identity); legs may be `Space` / `TensorProduct`.
- `_compose_SymmetricTensors`: scalar case calls Python `inner`; duplicate labels via local set helper (same role as `duplicate_entries`).
- `_decomposition_prepare`: calls Python `combine_legs` until legs batch; returns one-factor `TensorProduct` as `new_co_domain`.
- `_convert_*`: typed `FusionTreeData` / `AbelianBackendData` returns; `SymmetricTensor::to_backend` calls C++ directly.
- Monkey-patched into `_tensors.py` after each def (Python bodies kept for reference).
- Full docstrings in bindings.

## Dependencies

- Done: tensor classes, backends (`compose`, `mask_contract_*`, `zero_data`, …), `FusionTreeData`, `AbelianBackendData`
- Still Python: `inner`, `combine_legs` (called via import)

## TODO list for conversion

- [x] initial setup / planning (this file)
- [x] declaration + definitions for all 8 helpers
- [x] pybind11 bindings
- [x] monkey-patch into `_tensors.py`
- [x] pytest (`-m "not slow"`) — 4341 passed, 596 xfailed
- [x] wrap up → constructors batch next
