# Conversion of TensorProduct

## metadata

- original python name: `TensorProduct`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration: `include/cyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyTensorProduct` in `pybind/symmetries/py_trampolines.hpp`
- first line of docstring: Represents a tensor product of Spaces, e.g. the (co-)domain of a tensor.

## Design notes

### Inheritance

Python/C++: `TensorProduct(Space)` only — no Leg MI. Override `dual_space`, `change_symmetry`, `drop_symmetry`, `operator==`.

### Heterogeneous `factors`

Python type is `list[Space | LegPipe]`. In practice: `ElementarySpace`, nested `TensorProduct`, `LegPipe` / `AbelianLegPipe`. Store as `std::vector<py::object>` so Python and C++ objects mix until monkey-patch; dispatch `flat_spaces` / `dual` / `change_symmetry` / `drop_symmetry` via attributes or typed casts when possible.

Implemented as a set of anonymous-namespace helpers in `spaces.cpp` (`factor_symmetry`, `factor_flat`, `factor_num_flat_legs`, `leg_as_space`, `factor_change_symmetry`, `factor_drop_symmetry`, `factor_repr`) that first try a typed C++ cast (`Space` / `LegPipe`) and fall back to Python attribute access. This keeps the implementation working for factors that are still pure-Python objects.

### Construction

`Space`'s constructor needs the symmetry, sector decomposition and multiplicities, all of which are derived from `factors`. Resolved with a private `static Prepared prepare(...)` helper plus a delegating private constructor `TensorProduct(std::vector<py::object>, Prepared)`.

### Sector computation (`_calc_sectors`)

Hard part: abelian path uses `make_grid` (Python `cyten.tools.misc`) + `multiple_fusion_broadcast`; non-abelian recurses with `fusion_outcomes` / `_n_symbol`. Flatten pipes via `flat_spaces` then `as_Space`. Reuse existing C++ `_sort_sectors` / `_unique_sorted_sectors` helpers already in `spaces.cpp`.

Done in `calc_sectors_of_spaces` (works on the flattened `Space::Ptr` list) with `calc_sectors_of_factors` as the entry point. The abelian branch replicates `make_grid` directly in C++ (no Python round-trip needed) and feeds the resulting index grid into `symmetry->multiple_fusion_broadcast`. Empty factor list yields the trivial sector with multiplicity 1.

### `drop_symmetry`

When dropping *all* symmetries, the trivial sector must be taken from the *remaining* symmetry (`no_symmetry`), not from the old one — otherwise `sector_ind_len` mismatches and `test_sanity` fails.

### Iterators

`iter_uncoupled` / `iter_forest_blocks` / `iter_tree_blocks` are generators. Prefer C++ methods that fill `std::vector` of result structs (or accept a callback); expose Python generators in bindings. Depend on C++ `fusion_trees` / `FusionTree` in `trees.h`.

Implemented as eager `std::vector<UncoupledItem | ForestBlockItem | TreeBlockItem>` returns; the bindings turn each vector into a Python iterator of tuples, so the Python-side signature is unchanged.

### Slices

Python `slice` → `IndexSlice` `{start, stop}`; bindings convert to `py::slice`.

### Trampoline

Only method-bound virtuals are overridden: `test_sanity`, `change_symmetry`, `drop_symmetry`, `as_ElementarySpace`, `operator__eq__`. `dual_space` is *not* trampolined since it is exposed as the `dual` property (see `convert_ElementarySpace.md`). `#include <pybind11/functional.h>` is required in the trampoline header for the `SectorMapFn` argument of `change_symmetry`.

### Monkey-patch

Deferred until `AbelianLegPipe` (and remaining free fns) if following the spaces module plan. Export `cyten._core.TensorProduct`.

## Fixed along the way

`LegPipe::repr` printed C++ booleans (`true` / `false`) instead of the Python spelling. `bool_repr` was moved into the earlier anonymous-namespace block of `spaces.cpp` so `LegPipe::repr` can use it.

## TODO

- [x] setup / plan
- [x] declaration
- [x] definitions
- [x] bindings (+ trampoline)
- [x] build / pytest
- [x] parity script vs. Python `TensorProduct` (`tmp/check_tensor_product.py`)
- [ ] monkey-patch deferred (until `AbelianLegPipe`); only `cyten._core.TensorProduct` is exported for now
