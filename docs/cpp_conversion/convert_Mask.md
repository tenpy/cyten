# Conversion of Mask

## metadata

- original python name: `Mask`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration in C++ header file: `include/cyten/tensors/mask.h`
- definition in C++ file: `src/tensors/mask.cpp`
- pybind11 binding: `pybind/tensors/py_mask.cpp`
- trampoline: not needed (no library subclasses of Mask)
- first line of docstring: A boolean mask that can be used to project or enlarge a leg.

## Module context

1–6. Labels / Tensor / SymmetricTensor / DiagonalTensor / Identity — done (monkey-patch deferred)
7. **`Mask`** — this conversion
8. Next: `ChargedTensor` → free functions

Stay on branch **`convert_tensors`**. Defer monkey-patch until ChargedTensor (or all Tensor subclasses) are converted.

## Design notes

- Inherit `Tensor` (not DiagonalTensor); `using Ptr = std::shared_ptr<Mask>`.
- Member: `bool is_projection`; `TensorBackend::DataPtr data`.
- Ctor: `(data, space_in, space_out, is_projection=nullopt, backend=nullptr, labels=none)` — reject `LegPipe`; infer projection from dims when omitted.
- `_forbidden_dtypes`: float/complex (bool only); override `forbidden_dtypes()`.
- `ascii_diagram_type_name` / `class_name` → `"Mask"` / `"Mask"` (check Python: no override → Tensor default; prefer `"Mask"`).
- Factories: `from_eye`, `from_block_mask`, `from_DiagonalTensor`, `from_indices`, `from_random`, `from_zero`, hdf5.
- Properties: `large_leg` / `small_leg` from domain/codomain depending on `is_projection`.
- Boolean ops: `_binary_operand` / `_unary_operand`; bind `__and__`/`__or__`/`__xor__`/`__invert__`/etc.
- Overrides: `test_sanity`, `as_dtype`, `as_DiagonalTensor`, `as_SymmetricTensor`, `copy`, `move_to_device`, `to_backend`, `to_dense_block`, `to_numpy`, `_get_item`.
- Use `as_py_object()` when calling backend methods that still take `py::object` Mask.
- Free functions still Python: `dagger`, `get_same_backend` — call via Python or C++ equivalents already available.
- Update toml: `Mask = "Mask::Ptr"`.
- Preserve OPTIMIZE/TODO; full docstrings in bindings.
- Python `from_block_mask` raises `SymmetricTensor(msg)` (bug) → C++ use `SymmetryError`.

## Dependencies

- Done: `Tensor`, `DiagonalTensor`, backends (`mask_*`, `zero_mask_data`, `diagonal_to_mask`, …), spaces, `Dtype`
- Still Python: `ChargedTensor`, free ops (`dagger`, `apply_mask`, …)

## TODO list for conversion

- [x] initial setup (branch `convert_tensors`, listed, pytest `-k Mask`)
- [x] planning (this file)
- [x] generate / improve declaration
- [x] generate / improve definitions
- [x] pybind11 bindings (no trampoline)
- [x] monkey-patch — **deferred** (Python Mask still used by library)
- [x] pytest (4341 passed, 596 xfailed with `-m "not slow"`; Python Mask still used)
- [x] C++ smoke: `_core.Mask.from_eye` / `from_indices` / `orthogonal_complement` / `dagger`
- [ ] remove Python body — later with other Tensor subclasses
- [ ] wrap up → ChargedTensor
