# Conversion of tensor decompositions (batch 14)

## metadata

- original python names: `eigh`, `entropy`, `qr`, `lq`, `svd`, `svd_apply_mask`, `truncate_singular_values`, `truncated_svd`, `apply_mask_DiagonalTensor`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration: `include/cyten/tensors/decompositions.h`
- definition: `src/tensors/decompositions.cpp`
- pybind11 binding: `pybind/tensors/py_decompositions.cpp`
- trampoline: n/a (free functions)

## Module context

Batch 13 leg ops done. This is **free-function batch 14** (Decompositions) per [convert_tensors.md](convert_tensors.md).
Next: backend `py::object` cleanup.

Stay on branch **`convert_tensors`**.

## Design notes

- Args/returns are `py::object` (and tuples thereof) so Python Tensor subclasses work before class monkey-patch.
- Reuse C++ `_decomposition_prepare` / `_decomposition_labels` / `_svd_new_labels` / `_compose_with_Mask` / `combine_legs` / `split_legs` / `move_leg` / `dagger` / `trace` / `stable_log` / `norm`.
- Backend calls: `eigh` / `qr` / `lq` / `svd` / `truncate_singular_values` / `apply_mask_to_DiagonalTensor`.
- `apply_mask_DiagonalTensor` is small and required by `svd_apply_mask` — include here.
- `entropy`: DiagonalTensor path via C++ ops; Sequence path via NumPy.
- `truncated_svd`: call C++ `svd` + `truncate_singular_values` + `svd_apply_mask`.
- Fix `pinv` to call `truncated_svd(..., svd_min=cutoff)` (Python/`options=` was never a real API).
- Monkey-patch after pytest; keep Python defs for reference.

## Dependencies

- Done: helpers (`_decomposition_*`, `_svd_new_labels`, `_compose_with_Mask`), algebra, legs, elementwise, constructors
- Still Python: class monkey-patches deferred

## TODO list for conversion

- [x] planning (this file)
- [x] declaration + definitions
- [x] pybind11 bindings
- [x] monkey-patch + fix `pinv`
- [x] pytest (`-m "not slow"`) — 6352 passed, 642 xfailed
- [x] wrap up
