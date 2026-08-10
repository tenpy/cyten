# Conversion of AbelianLegPipe

## metadata

- original python name: `AbelianLegPipe`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration: `include/cyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyAbelianLegPipe`
- first line of docstring: Special case of a LegPipe for abelian group symmetries.

## Design notes

### Diamond MI

```cpp
class AbelianLegPipe : public LegPipe, public ElementarySpace
```

Single virtual `Leg` / `LegOrSpace`. `prepare(...)` computes sectors/maps/`basis_perm`; delegated ctor runs `LegPipe` then `ElementarySpace` (second `init_leg` wins).

### Members

`sector_strides`, `fusion_outcomes_sort`, `block_ind_map_slices`, `block_ind_map` — C++ `make_stride` / grid helpers in `spaces.cpp`.

### Monkey-patch

Patching **only** `AbelianLegPipe` fails: Python `ElementarySpace` legs cannot cast to `ElementarySpace::Ptr`.

Full hierarchy monkey-patch (after the Python class bodies) works:

```python
from .._core import AbelianLegPipe, ElementarySpace, Leg, LegPipe, Space, TensorProduct
```

Binding tweaks needed for tests: expose `_basis_perm` / `_inverse_basis_perm`; `take_slice` accepts array-like via `py::array::ensure`.

`ascii_arrow` overridden on `AbelianLegPipe` to resolve LegPipe vs ElementarySpace ambiguity.

## TODO

- [x] setup / plan
- [x] declaration
- [x] definitions
- [x] bindings (+ trampoline)
- [x] build / pytest / parity
- [x] monkey-patch (full spaces hierarchy)
