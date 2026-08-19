# Conversion of planar.py

## metadata

- original python file: `cyten/tensors/planar.py`
- original python module: `cyten.tensors.planar`
- C++ declaration header: `include/cyten/tensors/planar.h`
- C++ definitions: `src/tensors/planar.cpp`
- pybind11 bindings: `pybind/tensors/py_planar.cpp`
- first line of docstring: Provides useful classes and functions for dealing with planar (systems of) tensors.

## objects

C++ declaration order is not Python source order (`ContractionTree` is used by `PlanarDiagram` before it is defined):

| Symbol | Kind | Notes |
| --- | --- | --- |
| `_as_valid_name`, `_is_charge_leg_label`, `_expected_labels`, `_assert_cyclic_labels`, `_split_tensor_text` | helpers | not bound |
| `_combine_placeholder_charge_legs`, `_wrap_open_charge_legs`, `_finalize_charge_legs` | helpers | not bound |
| `TensorPlaceholder` | class | subclass of C++ `LabelledLegs`; `dims` are `std::vector<BigOPolynomial>` |
| `ContractionTreeNode` / `ContractionTree` | classes | child `shared_ptr`, parent `weak_ptr`; `fuse` is in-place |
| `PlanarDiagram` | class | string parsers; `evaluate`; `optimize_order` greedy-only |
| `PlanarLinearOperator` | class | C++ + trampoline; thin Python wrapper kept |
| `parse_leg_bipartition`, `_planar_contraction_helper` | functions | helper used by public API |
| `planar_*` / `horizontal_factorization` | functions | public API |

## design notes

- Stay on the current branch (`convert_sparse`); no new branch.
- Dual `Tensor` / `TensorPlaceholder` overloads (`evaluate`, `_do_contractions`, `planar_contraction`, `planar_partial_trace`): paired C++ overloads; pybind dispatches on type.
- `evaluate` / `planar_contraction` / `planar_partial_trace` may return a scalar (`compose` / `partial_trace` already return `std::variant<TensorPtr, BlockBackend::Scalar>`).
- Diagram `definition`: `std::vector<std::tuple<std::string, std::string, std::optional<std::string>, std::string>>`. Flexible `str | dict | nested tuples` parsing is in pybind.
- `optimize_order('optimal')` stays `NotImplementedError`. Greedy currently falls back to `"definition"`.
- Preserve original Python docstrings on all public pybind11 bindings.
- Python `assert` sites that tests catch (`verify_diagram`, `parse_leg_bipartition`) raise `AssertionError` via `PyErr_SetString`, not `ValueError`.
- `PlanarDiagram::tensor_names()` returns `std::vector<std::string> const&` so range constructors do not bind iterators from two temporaries.
- `PlanarLinearOperator` keeps a thin Python subclass of the C++ type. Subclasses are documented to store `op_diagram` / `matvec_diagram` as *class* variables and pass `self.op_diagram` into `__init__`; pybind `def_readwrite` data descriptors would shadow those class variables and crash on an uninitialized holder. Bindings use `py::dynamic_attr()` and do not expose those two fields as C++ properties.
- `ContractionTreeNode::copy` sets children’s parent to the *new* node (Python passed `parent=self` of the original).
- Empty `TensorPlaceholder.dims` uses `BigOPolynomial::from_str("None")` per label, matching Python `from_str(None)` via `str(None)`.
- `product_of([])` is the empty polynomial.
- Insertion order of diagram tensors is `PlanarDiagram::tensor_names_`; lookup is `std::map`.
- `OPEN_LEG_SYMBOL in definition` is checked on the full definition string (Python quirk).
- Monkey-patch: `cyten/tensors/planar.py` re-exports converted names from `_core`. `partial_trace` is re-exported from `_tensors` because tests use `ct.planar.partial_trace`.

Register sources in CMake, `pybind/_core.cpp`, and `include/cyten/tensors.h`.

## Dependencies

- Done: `LabelledLegs`, `Tensor` hierarchy, `LinearOperator` (+ trampoline), `BigOPolynomial`, tensor free functions (`compose`, `partial_trace`, `combine_legs`, `permute_legs`, decompositions).
- `duplicate_entries`: still Python (`cyten.tools.misc`); inline a small C++ helper for `parse_leg_bipartition`.

## TODO list for conversion

- [x] initial setup (stay on `convert_sparse`; names listed; `test_planar.py` green)
- [x] planning (this file)
- [x] generate the declaration draft
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions
- [x] improve and fix the definition drafts
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline (`PlanarLinearOperator`)
- [x] monkey-patch the python binding into the Python library
- [x] run python tests with pytest (`test_planar.py`, then full suite)
- [x] remove original python code for the converted objects (kept thin `PlanarLinearOperator` wrapper)
- [ ] wrap up (commit remaining changes; merge back into `main_cpp`)
