# Conversion of krylov_based.py

## metadata

- original python file: `cyten/tensors/krylov_based.py`
- original python module: `cyten.tensors.krylov_based`
- C++ declaration header: `include/cyten/tensors/krylov_based.h`
- C++ definitions: `src/tensors/krylov_based.cpp`
- pybind11 binding: `pybind/tensors/py_krylov_based.cpp`
- first line of docstring: Krylov-based algorithms for tensors

## objects

- `_to_number` / `_abs_number` — C++ helpers (not bound)
- `KrylovBased` — abstract base (trampoline for Python subclassing)
- `GMRES`
- `Arnoldi` (subclass of KrylovBased)
- `ArnoldiEvolution` (subclass of Arnoldi)
- `LanczosGroundState` (subclass of KrylovBased)
- `LanczosEvolution` (subclass of LanczosGroundState)
- `lanczos` — thin wrapper
- `lanczos_arpack` — **keep in Python** (depends on `HermitianNumpyArrayLinearOperator`)

## design notes

- Stay on the current branch (`convert_sparse`); no new branch.
- Small Krylov Hessenberg / tridiagonal matrices stay as dense C++ buffers; `eig` / `eigh` / `exp` of the small projected matrix go through NumPy (N_max is typically ≤ 20).
- Vector arithmetic uses `VectorLike` (`clone`, `scaled`, `axpy`, `inner`, `norm`).
- Options dict is parsed in the constructor (same keys/defaults as Python) and kept as `py::dict` for the public `options` attribute.
- Preserve original Python docstrings on all public pybind11 bindings.

## TODO list for conversion

- [x] initial setup
- [x] planning
- [x] generate the declaration draft
- [x] improve and fix the declaration draft
- [x] generate the C++ definitions
- [x] improve and fix the definition drafts
- [x] generate pybind11 bindings
- [x] generate pybind11 trampoline (KrylovBased)
- [x] monkey-patch the python binding into the Python library
- [x] run python tests with pytest (`test_krylov_based.py`, `test_direct_sum.py`)
- [x] remove original python code for the converted objects (kept `lanczos_arpack`)
- [ ] wrap up (commit remaining changes; merge back into `main_cpp`)
