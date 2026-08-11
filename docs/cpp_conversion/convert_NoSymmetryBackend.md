# Conversion of NoSymmetryBackend

## Status

**Done / monkey-patched** on branch `convert_backends`. Exported as `cyten._core.NoSymmetryBackend`. Imported in `cyten/backends/no_symmetry.py` from `_core`.

Layer overview: [convert_backends.md](convert_backends.md). Abstract base: [convert_TensorBackend.md](convert_TensorBackend.md).

## Metadata

| Field | Value |
| --- | --- |
| original python name | `NoSymmetryBackend` |
| original python file | `cyten/backends/no_symmetry.py` |
| original python module | `cyten.backends.no_symmetry` |
| declaration | `include/cyten/backends/no_symmetry.h` |
| definition | `src/backends/no_symmetry.cpp` |
| pybind11 binding | `pybind/backends/py_no_symmetry.cpp` |
| trampoline | none (no further Python subclasses expected; concrete leaf) |
| first line of docstring | Abstract base class for backends that do not enforce any symmetry. |

## Design notes

### Data / Block

- Nested `NoSymmetryBackend::BlockData : TensorBackend::Data` holding `BlockBackend::BlockPtr block`.
- Helpers: `wrap(BlockPtr) -> DataPtr`, `unwrap(DataPtr) -> BlockPtr` (throws if wrong type), `block_from_tensor(py::object) -> BlockPtr` via `tensor.attr("data").cast<BlockPtr>()`.
- Python tensors still store a `Block` in `.data`. Read with `cast<BlockPtr>`; return `DataPtr` via `wrap()` for the abstract C++ API.
- For Python bindings of methods that return `Data`, prefer returning the unwrapped `BlockPtr` (custom lambdas / caster) so the Python surface stays Block-based.
- `item` / `data_item` bindings wrap a Python `Block` into `DataPtr` before calling C++ (`py_data`).

### Types (match `TensorBackend` overrides)

- Tensor args → `py::object`.
- `Data` / `DiagonalData` / `MaskData` → `DataPtr`.
- Callables → `py::function` / `py::dict` kwargs.
- Use base `make_pipe` (no override).
- `supports_symmetry`: true only for product `no_symmetry` (single `NoSymmetry` factor).
- `can_decompose_tensors = true`; `DataCls` = block backend’s `BlockCls` (via pybind type).

### Implementation style

- Most methods are one-liners delegating to `block_backend->*`.
- Access tensor attributes via `py::object` `.attr(...)`.
- Match C++23 style of `tensor_backend.cpp`.

## Dependencies

- Done: `TensorBackend`, `BlockBackend`, spaces, `Symmetry` / `NoSymmetry`, `FusionTree`.
- Still Python: tensor classes (Layer 4) — reach via `py::object`.

## TODO checklist

- [x] initial setup (on `convert_backends`; list_python_names; read Python source)
- [x] planning (this file)
- [x] generate the declaration draft (`gen_cpp_declaration`)
- [x] improve and fix the declaration draft (namespaces, `BlockData`, exact override signatures)
- [x] generate the C++ definitions (`gen_cpp_definition` + CMake)
- [x] improve and fix the definition drafts (implement all methods; compile)
- [x] generate pybind11 bindings (+ register in CMake / `_core` / header)
- [x] fix bindings (return Block where Data is exposed to Python; factory shared_ptr ctor)
- [x] trampoline — skip (leaf class)
- [x] monkey-patch via `no_symmetry.py`
- [x] run python tests with pytest
- [x] remove original python class body — module re-exports from `_core`
