# Conversion of tensor leg ops (batch 13)

## metadata

- original python names: `bend_legs`, `check_same_legs`, `combine_legs`, `combine_to_matrix`, `move_leg`, `permute_legs`, `split_legs`, `squeeze_legs`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration: `include/cyten/tensors/ops_legs.h`
- definition: `src/tensors/ops_legs.cpp`
- pybind11 binding: `pybind/tensors/py_ops_legs.cpp`
- trampoline: n/a (free functions)

## Module context

Batch 12 algebra done. This is **free-function batch 13** (Leg permutation ops) per [convert_tensors.md](convert_tensors.md).
Next: decompositions.

Stay on branch **`convert_tensors`**.

## Design notes

- Args/returns are `py::object` so Python Tensor subclasses work before class monkey-patch.
- Build result tensors via Python ctors + NoSymmetry `data_as_python` unwrap (same as algebra).
- Call C++ backend methods (`permute_legs` / `combine_legs` / `split_legs` / `squeeze_legs` / `make_pipe`) via `TensorBackend::Ptr`.
- Call already-converted C++ for `transpose` (DiagonalTensor/Mask permute special case).
- `combine_to_matrix` is a thin wrapper — include with this batch.
- Variadic `combine_legs(*which_legs)`: bind with `py::args`.
- Flexible `levels` / `bend_right` / leg-index args: keep as `py::object` where Python accepts list|dict|bool|None.
- Label helpers: C++ `_combine_leg_labels` / `_split_leg_label`.
- `duplicate_entries` / `inverse_permutation`: call Python `cyten.tools.misc` for now.
- `check_same_legs` warning: Python `logging` via import.
- After monkey-patch, algebra's `tensors_mod().attr("permute_legs")` / `move_leg` automatically hit C++.

## Dependencies

- Done: helpers, constructors, elementwise, algebra, Tensor subclass C++ types (not monkey-patched), label helpers, backends’ leg methods
- Still Python: decompositions, `duplicate_entries`

## TODO list for conversion

- [x] planning (this file)
- [x] declaration + definitions
- [x] pybind11 bindings
- [x] monkey-patch
- [x] pytest (`-m "not slow"`) — 6352 passed, 642 xfailed
- [x] wrap up → decompositions batch
