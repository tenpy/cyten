# Typed tensor C++ API (drop leftover `py::object`)

## metadata

- status: **done**
- related: [convert_tensors.md](convert_tensors.md) §17
- prerequisite: [convert_tensor_backend_cleanup.md](convert_tensor_backend_cleanup.md) (backend virtuals already typed)
- headers: `include/cyten/tensors/{tensor,symmetric_tensor,diagonal_tensor,mask,charged_tensor,helpers,constructors,ops_elementwise,ops_algebra,ops_legs,decompositions}.h`
- spaces: `include/cyten/symmetries/spaces.h` (`TensorProduct::factors`)

## Goal

Replace leftover `py::object` tensor / space / leg arguments and returns on the Layer 4 C++ API
with typed `shared_ptr`s (`Tensor::Ptr`, `Leg::Ptr`, …), so `libcyten` headers are usable without
duck-typed Python objects.

Flexible Python input formats (sequences of spaces, nested label lists, numpy arrays, `*args`)
stay at the **binding layer** only. Same split as the backend cleanup: no `py::object` on the C++
library surface unless it is inherently Python.

Class monkey-patch is already done (`_tensors.py` is a `_core` re-export).

## Type mapping

Use the **most specific** pointer the signature documents. Prefer `CPtr` for read-only inputs.
Existing aliases live in [forward_declare.h](../../include/cyten/tensors/forward_declare.h) and on
each class (`Tensor::Ptr`, `Space::Ptr`, …).

| Current `py::object` | C++ type |
| --- | --- |
| Tensor / any subclass | `Tensor(C)Ptr` (or `SymmetricTensor(C)Ptr` / `DiagonalTensor(C)Ptr` / `Mask(C)Ptr` / `ChargedTensor(C)Ptr` when the contract is that specific) |
| `(co)domain` already a product | `TensorProduct::Ptr` |
| single tensor leg (ElementarySpace or LegPipe) | `Leg::Ptr` |
| labels | `LegLabels` |
| dense / diag / mask blocks | `BlockBackend::BlockPtr` |
| scalars (`item`, `norm`, `inner`, `entropy`, …) | `BlockBackend::Scalar` |
| leg index or label | `LegRef` = `std::variant<int64, std::string>` |
| optional lists (`levels`, `pipes`, `algorithm`, `sort`) | `std::optional<std::vector<…>>` / `std::optional<std::string>` |
| per-leg braid levels | `LevelsSpec` = `std::vector<std::optional<int64>>` |
| `bend_right` | `BendRight` = `std::variant<bool, std::vector<std::optional<bool>>>` |
| `pipe_dualities` | `PipeDualities` = `std::variant<bool, std::vector<bool>>` |

**Python-only — keep `py::object` / `py::function` / `py::array`:**

- HDF5 saver/loader/`h5gr`
- `py::function` + `func_kwargs` (`from_block_func`, `_elementwise_unary`, `_binary_operand` callbacks)
- numpy conveniences: `to_numpy`, `diagonal_as_numpy`, `as_numpy_mask`, `numpy_dtype`
- `np_random` on `Mask::from_random`

## Prerequisite: `TensorProduct::factors` as `Leg::Ptr`

`TensorProduct` stored `std::vector<py::object> factors` until spaces monkey-patch. That forced
`Tensor::legs()`, `get_leg`, `_as_codomain_leg`, and `ChargedTensor::charge_leg` to return
`py::object`.

`:class:`Leg`` is the common base for a single tensor leg (`ElementarySpace` or `LegPipe`,
including `AbelianLegPipe`). Public factors are always legs:

```cpp
std::vector<Leg::Ptr> factors;
explicit TensorProduct(std::vector<Leg::Ptr> factors, ...);

// on Tensor
[[nodiscard]] std::vector<Leg::Ptr> legs() const;
[[nodiscard]] Leg::Ptr get_leg(...) const;
```

Nested `TensorProduct` as a factor was never a stored (co)domain factor. It was an internal trick:

- `from_partial_products` built a temporary product whose `factors` were input `TensorProduct`s,
  to fuse already-computed sector decompositions, then the **result** stored flattened legs.
- `insert_multiply` did the same with `factors = [self, other]`.

Those now fuse sector decompositions of `Space`s / `TensorProduct`s **without** assigning non-`Leg`
objects into `factors`. `insert_multiply` / `left_multiply` / `right_multiply` take `Leg::Ptr`.
`operator[]` returns `Leg::Ptr`.

Space-only operations on a pipe go through `Leg::as_Space()` / `dynamic_pointer_cast<Space>`.

## Pattern: drop py-object ctors from C++ classes

Most classes had a typed ctor plus a py-object ctor. The py-object overloads are deleted from
headers and `.cpp`. `_init_parse_args` / `_init_parse_labels` are typed helpers
(`TensorProduct::Ptr` + `LegLabels`). Sequence-of-spaces and nested-label parsing live in pybind
lambdas.

Other tightened returns:

- `Tensor::as_SymmetricTensor` → `SymmetricTensor::Ptr` (no longer `py::object` “until
  SymmetricTensor exists”)
- `SymmetricTensor::diagonal` / `DiagonalTensor::diagonal` → `DiagonalTensor::Ptr`
- `ChargedTensor::charge_leg` → `Leg::Ptr`
- `from_invariant_part` / `from_two_charge_legs` →
  `std::variant<ChargedTensor::Ptr, BlockBackend::Scalar>` when the result may be a number

## Free functions

Public headers (`constructors.h`, `ops_elementwise.h`, `ops_algebra.h`, `ops_legs.h`,
`decompositions.h`) take and return Ptrs / Scalars. Examples:

```cpp
TensorPtr dagger(TensorCPtr tensor);
std::variant<TensorPtr, BlockBackend::Scalar> compose(TensorCPtr tensor1, TensorCPtr tensor2, ...);
BlockBackend::Scalar inner(TensorCPtr A, TensorCPtr B, bool do_dagger = true);
TensorPtr apply_mask(TensorCPtr tensor, MaskCPtr mask, LegRef leg);
std::string get_same_device(std::vector<TensorCPtr> const&, std::string const& = ...);
std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr> svd(TensorCPtr tensor, ...);
```

**Number | tensor overloads:** elementwise (`angle`, `sqrt`, …) have `DiagonalTensor::Ptr` and
`BlockBackend::Scalar` overloads. Bindings dispatch on `py::isinstance`. Number-only paths that
never need to be called from C++ (`entropy` of a numpy array, `is_scalar` of a Python number) stay
in pybind.

`eye` returns `TensorPtr` (`Identity` vs `SymmetricTensor` chosen by `diagonal`).
`tensor_from_grid` takes `std::vector<std::vector<TensorPtr>>`.
`partial_trace` / `compose` / `tdot` return `std::variant<TensorPtr, Scalar>` because a full
contraction is a scalar.

Implementation is wrap-first: existing `.attr` bodies keep working as `*_py` internals; typed
public wrappers convert Ptr ↔ `py::object` and call them. Call sites that already have Ptrs
(`combine_legs` inside `exp` / decompositions / `_decomposition_prepare`) use the typed API
directly.

## Binding-layer notes

Most `m.def("foo", &foo)` keep working once types are `shared_ptr` of bound classes. Nested
labels, `None` domain, numpy blocks, dict `levels` / `bend_right`, and `*args` use thin lambdas.

Pitfalls that showed up in pytest:

- **Unsequenced argument evaluation** — never `foo(use(p), std::move(p))`. Convert first, then
  move.
- **Mask bool-dtype scalar multiply** — do not convert Python numbers with `as_scalar(obj,
  tensor->dtype)` (Mask is bool, so `2.0` became `True`). Use the one-arg block-backend
  `as_scalar`.
- **Explicit `None` + Tensor** — pybind11 rejects `scalar_multiply(None, tensor)` as incompatible
  arguments even with `py::object`. Thin Python wrappers in `cyten/tensors/_tensors.py` raise
  `TypeError` for `None`.
- **Python trampoline subclasses** (`DummyTensor`) cannot `cast<TensorCPtr>()` (smart_holder /
  `enable_shared_from_this`). `is_scalar` bindings duck-type via attributes instead of casting.
- **`entropy`** C++ returns `Scalar`; Python tests expect a numpy/Python float → binding calls
  `.to_numpy()`.
- **Defaults** on `*_py` live on forward declarations only (not also on definitions).

## Out of this pass

- HDF5
- Removing pybind includes from public headers entirely (this pass only removes **untyped
  tensor/space/leg objects**)
- `py::function` block-func factories and elementwise operator callbacks
- numpy-facing methods

## Checklist

- [x] Type `TensorProduct::factors` as `vector<Leg::Ptr>`; rework sector fusion
- [x] Type Tensor core (`as_SymmetricTensor`, `legs` / `get_leg`, drop py-object ctor)
- [x] Type subclass factories (except HDF5 and `py::function`)
- [x] Type helper signatures
- [x] Type constructors / elementwise / algebra / legs / decompositions
- [x] Update `convert_tensors.md` §17; not-slow pytest green
