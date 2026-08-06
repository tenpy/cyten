# Conversion of Leg

## metadata

- original python name: `Leg`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration in C++ header file: `include/cyten/symmetries/spaces.h`
- definition in C++ file: `src/symmetries/spaces.cpp`
- pybind11 binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyLeg` in `pybind/symmetries/py_trampolines.hpp` (required — Python subclasses remain)
- first line of docstring: Common base class for a single leg of a tensor.

## Module context (`spaces.py`)

Convert objects in dependency order (this file is step 1):

1. **`Leg`** — abstract base (this conversion).
2. **`Space`** — abstract base (parallel hierarchy; `ElementarySpace` inherits both).
3. **`LegPipe`** — subclass of `Leg`.
4. **`ElementarySpace`** — subclass of `Space` and `Leg` (multiple inheritance).
5. **`TensorProduct`** — subclass of `Space`.
6. **`AbelianLegPipe`** — subclass of `LegPipe` and `ElementarySpace`.
7. Free functions: `swap_gate`, `twist_gate`, helpers (`_unique_sorted_sectors`, …).

Keep original Python `Leg` (and all subclasses) until subclasses are converted. Export C++ `Leg` from `_core` for further Layer 2 work; **defer monkey-patch** if Python `LegPipe` / `ElementarySpace(Space, Leg)` inheritance via trampoline is unsafe (same pitfall as `BaseSymmetry`).

## Design notes

- Hold `Symmetry::Ptr symmetry` (same pattern as `FusionTree`).
- `dim` is `float64` (Python `int | float`; non-droppable symmetries may have non-integer qdim).
- `is_dual: bool`.
- `_basis_perm` / `_inverse_basis_perm`: **protected** `std::optional<std::vector<int64>>`. Getters return identity `arange(dim)` when empty; raise `SymmetryError` if `!symmetry->can_be_dropped()`.
- Pure virtual: `as_Space()`, `dual()`, `is_trivial()`, `operator==`.
- `as_Space` / `as_ElementarySpace`: `py::object` until those types exist in C++.
- `flat_legs` / `flat_spaces`: `std::vector<Leg::Ptr>`.
- No combined `set_basis_perm(basis, inverse)`: Python only ever sets one side via property setters. C++ has `set_basis_perm(optional)` and `set_inverse_basis_perm(optional)` (each derives the other). Pipes override both to forbid changes.
- `apply_basis_perm`: NumPy (`py::array`) for now.
- `ascii_arrow`: default throws; subclasses override (pipes / ElementarySpace).
- Trampoline **required** (`LegPipe`, `ElementarySpace`, `AbelianLegPipe` still Python).
- `enable_shared_from_this` + `py::smart_holder` like other Layer 2 bases.
- Helper `inverse_permutation`: small C++ helper in `spaces.cpp` or `tools` (Python `cyten.tools.misc` not yet in C++).

## Dependencies (already in C++)

- `Symmetry` / `BaseSymmetry`, `SymmetryError`, Sector types (indirect), pybind NumPy arrays

## Dependencies (still Python)

- `Space`, `ElementarySpace`, `LegPipe`, `AbelianLegPipe`
- `UNSPECIFIED`, `inverse_permutation` (tools)

## Related notes

- [convert_Spaces.md](convert_Spaces.md) — `BlockBackend::apply_basis_perm` still takes `std::vector<py::object>` legs; later switch to C++ `Leg` / `Space`.

## TODO list for conversion

- [x] initial setup (clean tree, `Leg` in `list_python_names`, pytest `test_spaces.py` 867 passed; branch `convert_Leg` from `convert_trees`)
- [x] planning (this file)
- [x] generate the declaration draft (`gen_cpp_declaration` → `include/cyten/symmetries/spaces.h`)
- [x] improve and fix the declaration draft (namespace, types, C++23 / pre-commit; protected perms + separate setters)
- [x] generate the C++ definitions (`gen_cpp_definition`; add to `src/CMakeLists.txt`)
- [x] improve and fix the definition drafts (CHECKME/FIXME; compile + ctest)
- [x] generate pybind11 bindings (`gen_pyb11_binding` → `py_spaces.cpp`; register in `py_symmetries.cpp` / `pybind/CMakeLists.txt`)
- [x] generate pybind11 trampoline (`PyLeg`)
- [ ] monkey-patch — **deferred** until subclasses converted; exported as `cyten._core.Leg`
- [x] run python tests (`test_spaces.py` 867 passed; still using Python `Leg`)
- [ ] remove original python `Leg` class body (only after subclasses converted)
- [ ] wrap up (then convert `Space` / `LegPipe`)
