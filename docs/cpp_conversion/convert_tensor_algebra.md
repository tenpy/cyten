# Conversion of tensor algebra ops (batch 12)

## metadata

- original python names: `almost_equal`, `compose`, `dagger`, `inner`, `item`, `linear_combination`, `norm`, `outer`, `partial_compose`, `partial_trace`, `pinv`, `scalar_multiply`, `scale_axis`, `tdot`, `trace`, `transpose`, `is_scalar`, `get_same_device`, `on_device`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration: `include/cyten/tensors/ops_algebra.h`
- definition: `src/tensors/ops_algebra.cpp`
- pybind11 binding: `pybind/tensors/py_ops_algebra.cpp`
- trampoline: n/a (free functions)

## Module context

Batch 11 elementwise done. This is **free-function batch 12** (Algebra / scalar) per [convert_tensors.md](convert_tensors.md).
Next: leg ops → decompositions.

Stay on branch **`convert_tensors`**.

## Design notes

- Args/returns are `py::object` so Python Tensor subclasses work before class monkey-patch.
- Build result tensors via Python ctors + NoSymmetry `data_as_python` unwrap (same as helpers/constructors).
- Call C++ for same-batch / already-converted helpers (`_check_compatible_legs`, `_compose_*`, `complex_conj`, `cutoff_inverse`, …).
- Call Python for leg-ops batch still pending: `permute_legs`, `bend_legs`, `move_leg`, `check_same_legs`.
- Call Python for decompositions still pending: `truncated_svd` (used by `pinv`).
- `get_same_backend` via C++ `get_same_backend({…})`.
- Relabel maps: `std::optional<std::map<std::string, std::string>>` with pybind None ↔ nullopt.
- `partial_trace(*pairs)` / `get_same_device(*tensors)`: bind with `py::args`.
- Python `SymmetryError` from `permute_legs` arrives as `py::error_already_set`; rewrite to `_USE_PERMUTE_LEGS_ERR_MSG`.
- NoSymmetry `item`: call `tensor.backend.item(tensor)` via Python (C++ `TensorBackend::item` cannot cast Block data).
- Monkey-patch after pytest; keep Python defs for reference.
- Full docstrings in bindings (short form).

## Dependencies

- Done: helpers, constructors, elementwise, Tensor subclass C++ types (not monkey-patched)
- Still Python: leg ops, decompositions, `duplicate_entries`

## TODO list for conversion

- [x] planning (this file)
- [x] declaration + definitions
- [x] pybind11 bindings
- [x] monkey-patch
- [x] pytest (`-m "not slow"`) — 4341 passed, 596 xfailed
- [x] wrap up → leg ops batch
