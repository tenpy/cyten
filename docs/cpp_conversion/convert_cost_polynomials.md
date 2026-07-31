# Conversion of cost_polynomials (`BigOMonomial`, `BigOPolynomial`)

## metadata

- original python names: `BigOMonomial`, `BigOPolynomial` (module `cost_polynomials`)
- original python file: `cyten/tools/cost_polynomials.py`
- original python module: `cyten.tools.cost_polynomials`
- declaration in C++ header file: `include/cyten/tools/cost_polynomials.h` (also pulled in by `include/cyten/tools.h`)
- definition in C++ file: `src/tools/cost_polynomials.cpp`
- pybind11 binding: `pybind/tools/py_cost_polynomials.cpp`
- first line of docstring (BigOMonomial): A symbolic representation of an algorithmic cost as a monomial.
- first line of docstring (BigOPolynomial): A symbolic representation of an algorithmic cost as a monomial.

Public Python export today: `cyten.tools.BigOPolynomial` (via `__init__.py`).
`BigOMonomial` is used internally and by `BigOPolynomial`; bind both for a faithful API.
No dedicated unit tests; covered indirectly by `tests/python_tests/test_planar.py`.
Trampoline: not needed (no subclasses / virtual overrides in the Python library).

## Dependencies / design notes

- Layer 0 tools; no cyten C++ deps beyond existing `tools.h` / `cyten.h`.
- Types: `dict[str, int]` → `std::map<std::string, int64>`; `terms` → `std::set<BigOMonomial>` (unique monomials; `BigOMonomial` has `operator<`).
- `BigOMonomial + BigOMonomial` returns a `BigOPolynomial`; multiplication stays monomial.
- `BigOPolynomial.from_str` / arithmetic accept `str` and `BigOMonomial` like Python.
- Planar calls `BigOPolynomial.prod(*polys)` unbound on the class — keep that callable from Python.
- `relations` in `is_negligible` / `simplify_terms` is unused (`NotImplementedError`); keep the same.
- Convert **BigOMonomial first**, then **BigOPolynomial**.

## TODO list for conversion

- [x] initial setup (clean tree, branch `convert_cost_polynomials`, planar/tools pytest green)
- [x] planning (this file)
- [x] generate the declaration draft (`gen_cpp_declaration` for both classes into `include/cyten/tools.h`)
- [x] improve and fix the declaration draft (namespace, types, C++23 / pre-commit)
- [x] generate the C++ definitions (`gen_cpp_definition` into `src/tools.cpp`)
- [x] improve and fix the definition drafts (CHECKME/FIXME, clang-tidy, rebuild, ctest)
- [x] generate pybind11 bindings (`gen_pyb11_binding` into `pybind/py_tools.cpp`)
- [x] fix bindings; recompile
- [x] trampoline: skip (no virtual overrides / subclasses)
- [x] monkey-patch: `from .._core import BigOMonomial, BigOPolynomial` in `cost_polynomials.py`
- [x] run python tests (planar first, then full suite)
- [x] remove original python class bodies; keep `_core` import
- [ ] wrap up — commit remaining changes (`cyten/tools/cost_polynomials.py`, docs) and merge `convert_cost_polynomials` into `main` / `main_cpp`
