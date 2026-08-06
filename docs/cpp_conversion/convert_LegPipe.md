# Conversion of LegPipe

## metadata

- original python name: `LegPipe`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration in C++ header file: `include/cyten/symmetries/spaces.h`
- definition in C++ file: `src/symmetries/spaces.cpp`
- pybind11 binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyLegPipe` in `pybind/symmetries/py_trampolines.hpp` (required — `AbelianLegPipe` subclasses)
- first line of docstring: A group of legs, i.e. resulting from combine_legs.

## Module context

1. `Leg` — done (monkey-patch deferred)
2. `Space` — done (monkey-patch deferred)
3. **`LegPipe`** — this conversion (subclass of `Leg`)
4. Next: `ElementarySpace`, `TensorProduct`, `AbelianLegPipe`, …

## Design notes

- Inherits `Leg`; holds `std::vector<Leg::Ptr> legs`, `int64 num_legs`, `bool combine_cstyle`.
- Ctor: product of dims; combined `basis_perm` via Python `combine_permutations` when any child has a custom perm (add `Leg::has_custom_basis_perm()`).
- `as_Space()` → Python `TensorProduct` until that class is converted.
- `dual()` → new `LegPipe` with reversed dual legs and flipped `is_dual` / `combine_cstyle`.
- `set_basis_perm` / `set_inverse_basis_perm` → throw `TypeError`.
- `ascii_arrow()` → `"║"`.
- `operator==`: LegPipe vs LegPipe; `is_abelian_leg_pipe()` virtual (false here; AbelianLegPipe later).
- Sequence protocol in bindings: `__getitem__` / `__iter__` / `__len__`.
- `repr(show_symmetry, one_line)` using `get_config()` print options.
- Trampoline required; monkey-patch deferred until `AbelianLegPipe` converted.

## TODO

- [x] initial setup
- [x] planning
- [ ] declaration draft → improve
- [ ] definitions → improve
- [ ] bindings + trampoline
- [ ] monkey-patch deferred
- [ ] pytest
- [ ] wrap up
