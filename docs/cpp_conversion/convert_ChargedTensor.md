# Conversion of ChargedTensor

## metadata

- original python name: `ChargedTensor`
- original python file: `cyten/tensors/_tensors.py`
- original python module: `cyten.tensors._tensors`
- declaration in C++ header file: `include/cyten/tensors/charged_tensor.h`
- definition in C++ file: `src/tensors/charged_tensor.cpp`
- pybind11 binding: `pybind/tensors/py_charged_tensor.cpp`
- trampoline: not needed (no library subclasses of ChargedTensor)
- first line of docstring: Tensors which are not symmetric, but carry a well defined charge.

## Module context

1–7. Labels / Tensor / Symmetric / Diagonal / Identity / Mask — done (monkey-patch deferred)
8. **`ChargedTensor`** — this conversion
9. Next: free-function batches

Stay on branch **`convert_tensors`**. Defer monkey-patch until all Tensor subclasses are converted (skill rule).

## Design notes

- Inherit `Tensor`; `using Ptr = std::shared_ptr<ChargedTensor>`.
- Members: `SymmetricTensor::Ptr invariant_part`, `std::optional<BlockBackend::BlockPtr> charged_state` (or nullable BlockPtr), `Space::Ptr charge_leg`.
- Static `constexpr` / string `_CHARGE_LEG_LABEL = "!"`.
- Ctor: `(invariant_part, charged_state)` — charge leg = `domain.factors[0]` of invariant part; ChargedTensor domain drops that factor.
- `supports_symmetry` → `symmetry->has_symmetric_braid()`.
- Factories: `from_block_func`, `from_dense_block`, `from_dense_block_single_sector` (NotImplemented like Python), `from_invariant_part`, `from_two_charge_legs`, `from_zero`, hdf5.
- Helpers: `_parse_inv_domain`, `_parse_inv_labels`.
- Overrides: `test_sanity`, `as_dtype`, `as_SymmetricTensor`, `copy`, `_get_item`, `move_to_device`, `_repr_header_lines`, `set_label`/`set_labels`, `to_backend`, `to_dense_block`, `to_dense_block_single_sector`.
- Free functions still Python (`squeeze_legs`, `tdot`, `bend_legs`, `combine_legs`) — call via `py::module_::import("cyten.tensors._tensors")` where needed.
- Update toml: `ChargedTensor = "ChargedTensor::Ptr"`.
- Python `set_labels` bug (`*self._CHARGE_LEG_LABEL`) → C++ append single charge label.
- Python hdf5 looks incomplete (`self.data`) — save/load `invariant_part` + `charged_state` sensibly.
- Preserve OPTIMIZE/TODO; full docstrings in bindings.
- Override `dagger`/`hc` properties in bindings (Tensor delegates to Python free `dagger` which will not see C++ ChargedTensor until monkey-patch / free-fn update).

## Dependencies

- Done: `Tensor`, `SymmetricTensor`, backends, spaces, `Dtype`
- Still Python: free ops (`squeeze_legs`, `tdot`, `bend_legs`, `combine_legs`, `dagger`, …)

## TODO list for conversion

- [x] initial setup (branch `convert_tensors`, listed)
- [x] planning (this file)
- [ ] generate / improve declaration
- [ ] generate / improve definitions
- [ ] pybind11 bindings (no trampoline)
- [ ] monkey-patch — deferred
- [ ] pytest
- [ ] remove Python body — later with other Tensor subclasses
- [ ] wrap up → free functions
