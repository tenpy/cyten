# Conversion of get_backend (backend_factory)

## Status

**C++ hybrid factory done** — exported as `cyten._core.get_backend`. Not monkey-patched into Python yet.

- `no_symmetry` path constructs C++ `NoSymmetryBackend`
- `abelian` / `fusion_tree` still construct the Python classes (until those backends are converted)
- Cache keyed by `(tensor_backend, block_backend)` strings

## Metadata

| Field | Value |
| --- | --- |
| original python name | `get_backend` |
| original python file | `cyten/backends/backend_factory.py` |
| original python module | `cyten.backends` |
| declaration | `include/cyten/backends/backend_factory.h` |
| definition | `src/backends/backend_factory.cpp` |
| pybind11 binding | `pybind/backends/py_backend_factory.cpp` |
| trampoline | no |
| first line of docstring | Get an instance of an appropriate backend. |

## TODO

- [x] declaration / definition / bindings (hybrid)
- [ ] Switch abelian / fusion_tree paths to C++ once those backends exist
- [ ] monkey-patch via `backend_factory.py`
- [ ] pytest
