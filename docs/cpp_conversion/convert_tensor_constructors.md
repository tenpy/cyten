# Conversion of tensor constructors (batch 10)

## metadata

- original python names: `eye`, `tensor`, `add_trivial_leg`, `zero_like`, `tensor_from_grid`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration: `include/cyten/tensors/constructors.h`
- definition: `src/tensors/constructors.cpp`
- pybind11 binding: `pybind/tensors/py_constructors.cpp`
- trampoline: n/a (free functions)

## Module context

Batch 9 helpers done. This is **free-function batch 10** (Constructors) per [convert_tensors.md](convert_tensors.md).
Next: elementwise → algebra → legs → decompositions.

Stay on branch **`convert_tensors`**.

## Design notes

- Free functions in `cyten::`; return `py::object` and build results via **Python** tensor classmethods / ctors (hierarchy not monkey-patched yet).
- NoSymmetry: unwrap `BlockData` → `Block` when constructing Python `SymmetricTensor` from C++ `DataPtr` (same as helpers).
- `eye` / `zero_like` / `tensor`: thin wrappers over Python `from_eye` / `from_zero` / `from_dense_block` / `as_SymmetricTensor` (keeps isinstance correct).
- `add_trivial_leg`: port position parsing + recursive ChargedTensor path in C++; call backend `add_trivial_leg` with `py::object` tensor.
- `tensor_from_grid`: port in C++; call Python `get_same_device` / `get_same_backend`; backend `from_grid`.
- Monkey-patch after pytest.
- Full docstrings in bindings.

## Dependencies

- Done: DiagonalTensor/SymmetricTensor/Mask/ChargedTensor factories, backends `add_trivial_leg` / `from_grid`, helpers patterns
- Still Python: `get_same_device` (until algebra/misc batch)

## TODO list for conversion

- [x] planning (this file)
- [x] declaration + definitions
- [x] pybind11 bindings
- [x] monkey-patch
- [x] pytest (`-m "not slow"`) — 4341 passed, 596 xfailed
- [x] wrap up → elementwise batch
