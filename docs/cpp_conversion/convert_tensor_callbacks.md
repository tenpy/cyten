# Typed tensor callbacks (`py::function` → `std::function`)

## metadata

- status: **done**
- related: [convert_tensors.md](convert_tensors.md) §18
- prerequisite: [convert_tensor_typed_api.md](convert_tensor_typed_api.md) (batch 17; tensor args already typed)
- headers: `include/cyten/backends/tensor_backend.h`,
  `include/cyten/tensors/{symmetric_tensor,diagonal_tensor,charged_tensor,mask}.h`
- wrap helpers: `pybind/tensors/py_callbacks.hpp`

## Goal

Replace leftover `py::function` / `func_kwargs` on tensor factories, elementwise operators, and
the matching `TensorBackend` virtuals with `std::function` over `BlockBackend` blocks/scalars.
C++ can build tensors from block factories and apply elementwise ops without `py::function`.

Python-flexible calling conventions (`func_kwargs`, `shape_kw`, numpy `BlockLike` returns,
`operator.and_`) stay at the **binding layer** only. Same split as batch 17.

HDF5 and numpy-facing methods stay `py::object` / `py::array`.

## Type mapping

Aliases live next to the backend surface in `tensor_backend.h` (includes `block_backend.h` and
`Sector`). Precedent: `SectorMapFn` in `spaces.h`.

```cpp
using BlockUnaryFn = std::function<BlockBackend::BlockPtr(BlockBackend::BlockPtr const&)>;
using BlockBinaryFn = std::function<BlockBackend::BlockPtr(
    BlockBackend::BlockPtr const&, BlockBackend::BlockPtr const&)>;
using BlockFactoryFn = std::function<BlockBackend::BlockPtr(std::vector<int64> const& shape)>;
using SectorBlockFactoryFn = std::function<BlockBackend::BlockPtr(
    std::vector<int64> const& shape, Sector const& coupled)>;
using BlockToScalarFn = std::function<BlockBackend::Scalar(BlockBackend::BlockPtr const&)>;
using ScalarReduceFn = std::function<BlockBackend::Scalar(
    std::vector<BlockBackend::Scalar> const&)>;
using DtypeMapFn = std::function<Dtype(Dtype)>;
```

| Previous | C++ type |
| --- | --- |
| `from_block_func(py::function, …, func_kwargs, shape_kw)` | `from_block_func(BlockFactoryFn, …)` — drop `func_kwargs` / `shape_kw` |
| `from_sector_block_func(py::function, …, func_kwargs)` | `from_sector_block_func(SectorBlockFactoryFn, …)` |
| `_elementwise_unary(py::function, func_kwargs, …)` | `_elementwise_unary(BlockUnaryFn, …)` |
| `_elementwise_binary(py::object other, py::function, …)` | `_elementwise_binary(DiagonalTensorCPtr, BlockBinaryFn, …)` |
| `_binary_operand(py::object other, py::function, …)` | overloads: `Scalar` and `DiagonalTensorCPtr` (Mask: `bool` and `MaskCPtr`); return `Ptr` |
| `mask_unary_operand` / `_unary_operand` | `BlockUnaryFn` |
| `mask_binary_operand` | `BlockBinaryFn` |
| `reduce_DiagonalTensor(block_func, func)` | `BlockToScalarFn` + `ScalarReduceFn` |
| `act_block_diagonal_square_matrix(block_method, dtype_map)` | `BlockUnaryFn` + `std::optional<DtypeMapFn>` |
| backend `diagonal_elementwise_*` `py::dict func_kwargs` | drop kwargs; capture in the `std::function` |

**Python-only — keep in pybind:**

- `func_kwargs`, `shape_kw`
- converting `BlockLike` (numpy) → `BlockPtr` via `block_backend->as_block`
- dunder `NotImplemented` for unknown `other` types
- wrapping `operator.and_` / `operator.invert` / etc. into the typedefs (Mask has no Block bitwise ops yet)

Do **not** bind C++ factories/virtuals as raw `std::function` and rely on pybind’s automatic
conversion: Python `from_block_func` may return numpy arrays, and kwargs/`shape_kw` are not
expressible as `std::function`.

```mermaid
flowchart LR
  pyFunc["Python callable plus kwargs"]
  wrap["pybind wrap helper"]
  stdFn["std::function BlockPtr"]
  cppApi["C++ factory / backend virtual"]
  pyFunc --> wrap --> stdFn --> cppApi
```

## Binding wrap helpers

`pybind/tensors/py_callbacks.hpp` (same idea as `sector_map_from_python`):

- `block_factory_from_python` — `func(shape, **kwargs)` / `func(**{shape_kw: shape, **kwargs})`
- `sector_block_factory_from_python` — `func(shape, coupled, **kwargs)`
- `block_unary_from_python` / `block_binary_from_python`
- `adapt_block_bool_unary` / `adapt_block_bool_binary` — numpy bool arrays in, `BlockPtr` out

Public Python signatures of `from_block_func` / `_elementwise_unary` / dunders stay unchanged.
Shape is passed to Python as a **tuple**, not a list.

## TensorBackend virtuals

Changed in `tensor_backend.h`, all three backends, trampolines, and backend pybind:

- `from_sector_block_func` / `diagonal_from_sector_block_func` → `SectorBlockFactoryFn`; call
  `func(shape, coupled)` with `std::vector<int64>` + `Sector`
- `diagonal_elementwise_unary` / `_binary` → `BlockUnaryFn` / `BlockBinaryFn`; drop `func_kwargs`
- `mask_unary_operand` / `mask_binary_operand` → same
- `reduce_DiagonalTensor` → `BlockToScalarFn` + `ScalarReduceFn` (fold a `std::vector<Scalar>`)
- `act_block_diagonal_square_matrix` → `BlockUnaryFn` + `std::optional<DtypeMapFn>`

**Trampoline:** `PYBIND11_OVERRIDE` cannot pass `std::function` to a Python override. Wrap with
`py::cpp_function` first.

Backend pybind methods that Python calls keep taking `py::function` and wrap in the lambda.

Internal C++ callers that previously built `py::cpp_function` are real lambdas:

- `from_random_normal` / `from_random_uniform` / `from_eye` → `block_backend->random_*` / `ones_block`
- `DiagonalTensor::max` / `min` → `bb->max` / `bb->min`
- `ops_elementwise.cpp` (`angle`, `exp`, …) → `bb->angle` / `bb->matrix_exp` / etc.

## Tensor factories

`SymmetricTensor` / `DiagonalTensor` / `ChargedTensor`:

```cpp
static Ptr from_block_func(BlockFactoryFn func, TensorProduct::Ptr codomain, ...);
static Ptr from_sector_block_func(SectorBlockFactoryFn func, TensorProduct::Ptr codomain, ...);
```

`from_block_func` wraps `BlockFactoryFn` as `SectorBlockFactoryFn` that **ignores** `coupled`, then
`backend->from_sector_block_func`. No `as_block` / kwargs inside the C++ factory.

`ChargedTensor::from_block_func` forwards to `SymmetricTensor::from_block_func`.
Identity’s unsupported `from_block_func` stubs stay as-is.

## Elementwise / binary operand

On `DiagonalTensor` / `Identity` / `Mask`:

- `_elementwise_unary(BlockUnaryFn, bool maps_zero_to_zero)`
- `_elementwise_binary(DiagonalTensorCPtr other, BlockBinaryFn, bool partial_zero_is_zero)`
- `_binary_operand`: C++ overloads that **throw** on bad types; return `Ptr`. Drop
  `return_NotImplemented`.
  - Diagonal: `(Scalar, BlockBinaryFn, operand, right)` and `(DiagonalTensorCPtr, …)`
  - Mask: `(bool, BlockBinaryFn, …)` and `(MaskCPtr, …)`
- `_unary_operand(BlockUnaryFn)`
- `elementwise_almost_equal(DiagonalTensorCPtr other, …)`

Pybind dunders keep today’s Python protocol: unknown `other` → `NotImplemented`; else convert and
call the typed overload. Diagonal dunders use Block operators (`*`, `+`, `/`, `<`, …) instead of
`operator.mul`. Mask dunders still wrap `operator.and_` through the numpy adapter.

`ops_algebra.cpp` wrap-first `scalar_multiply_py` / `linear_combination_py` call the typed members
instead of `.attr("_elementwise_unary")` / `.attr("_binary_operand")`.

## Pitfalls

- **Unsequenced `std::move`** — convert, then move.
- **Trampoline `std::function` → Python** — wrap with `py::cpp_function`.
- **Do not auto-cast Python `std::function`** for factories — `BlockLike` vs `BlockPtr`.
- **Mask bool-dtype** — do not convert scalars with `as_scalar(obj, mask->dtype)`.
- **DummyTensor / smart_holder** — dunder bindings must not `cast<TensorCPtr>()` on Python
  trampoline subclasses (same as `is_scalar`).
- **`NotImplemented`** lives only in pybind; C++ throws `invalid_argument`.

## Out of this pass

- HDF5
- numpy-facing methods (`to_numpy`, `diagonal_as_numpy`, …)
- `from_tree_pairs(py::object trees)`, `from_grid` cell type
- Adding Block bitwise operators (Mask still uses the numpy adapter in pybind)
- Removing pybind includes from public headers entirely

## Checklist

- [x] Add callback typedefs; type `TensorBackend` virtuals + 3 backends + trampolines
- [x] Type `from_block_func` / `from_sector_block_func`; wrap helpers; C++ `from_eye` / `from_random_*`
- [x] Type `_elementwise_*`, `_binary_operand`, Mask operands; pybind `NotImplemented`; algebra / elementwise call sites
- [x] Document as convert_tensors batch 18; not-slow pytest green
