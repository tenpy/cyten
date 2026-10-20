# Conversion of get_backend (backend_factory)

## Status

**Done / monkey-patched** — exported as `lasyten._core.get_backend`, imported in `lasyten/backends/backend_factory.py`.

- `no_symmetry` / `abelian` / `fusion_tree` paths construct C++ backends
- Cache keyed by `(tensor_backend, block_backend)` strings on `lasyten._core._tensor_backend_cache`

## Metadata

| Field | Value |
| --- | --- |
| original python name | `get_backend` |
| original python file | `lasyten/backends/backend_factory.py` |
| original python module | `lasyten.backends` |
| declaration | `include/lasyten/backends/backend_factory.h` |
| definition | `src/backends/backend_factory.cpp` |
| pybind11 binding | `pybind/backends/py_backend_factory.cpp` |
| trampoline | no |
| first line of docstring | Get an instance of an appropriate backend. |

## TODO

- [x] declaration / definition / bindings (hybrid)
- [x] Switch abelian / fusion_tree paths to C++ once those backends exist
- [x] monkey-patch via `backend_factory.py`
- [x] pytest
