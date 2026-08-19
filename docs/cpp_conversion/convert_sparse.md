# Conversion of sparse.py (`LinearOperator` stack)

## metadata

- original python file: `cyten/tensors/sparse.py`
- original python module: `cyten.tensors.sparse`
- C++ declaration header: `include/cyten/tensors/sparse.h`
- C++ definitions: `src/tensors/sparse.cpp`
- pybind11 bindings: `pybind/tensors/py_sparse.cpp`
- first migrated objects:
  - `LinearOperator`
  - `TensorLinearOperator`
  - `LinearOperatorWrapper`
  - `SumLinearOperator`
  - `ShiftedLinearOperator`
  - `ProjectedLinearOperator`
  - `gram_schmidt`

## scope

- Keep `NumpyArrayLinearOperator` and `HermitianNumpyArrayLinearOperator` in Python for now.
- Keep scipy-based conversion helpers in Python.
- Monkey-patch the converted classes/functions from `_core` at the bottom of
  `cyten/tensors/sparse.py`.

## notes

- Conversion targets the updated `VectorLike` API (`Tensor` and `DirectSum` compatible).
- `TensorLinearOperator` now enforces two-leg contractible tensor input and removes the old TODO.
- `ProjectedLinearOperator.to_tensor()` is still intentionally `NotImplemented`, matching current
  Python behavior.
