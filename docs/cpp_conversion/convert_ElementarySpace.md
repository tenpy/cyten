# Conversion of ElementarySpace

## metadata

- original python name: `ElementarySpace`
- original python file: `cyten/symmetries/spaces.py`
- original python module: `cyten.symmetries.spaces`
- declaration: `include/cyten/symmetries/spaces.h`
- definition: `src/symmetries/spaces.cpp`
- binding: `pybind/symmetries/py_spaces.cpp`
- trampoline: `PyElementarySpace` (required — `AbelianLegPipe` subclasses)
- first line of docstring: A Space that is defined as (the dual of) a direct sum of sectors.

## Design notes

### Multiple inheritance

Python: `ElementarySpace(Space, Leg)`. C++: `class ElementarySpace : public Space, public virtual Leg`.

`LegPipe` updated to `public virtual Leg` so future `AbelianLegPipe(LegPipe, ElementarySpace)` has a single `Leg` subobject.

### Conflicting `dual()` return types

`Space::dual` → `Space::Ptr` and `Leg::dual` → `Leg::Ptr` cannot both be overridden under one name. Resolve with intermediate hooks:

- `Space`: keep `dual()` as non-pure wrapper calling pure `dual_space()`
- `Leg`: keep `dual()` as non-pure wrapper calling pure `dual_leg()`
- `ElementarySpace`: implement both hooks to return the same `ElementarySpace::Ptr`

### Single `enable_shared_from_this`

`Leg` and `Space` each derived from `std::enable_shared_from_this`, which is ambiguous (and, worse,
gives two control-block pointers) in `ElementarySpace`. Both now derive `public virtual` from a new
empty base `LegOrSpace : std::enable_shared_from_this<LegOrSpace>`, so there is exactly one
subobject. `Leg::shared_leg()` / `Space::shared_space()` / `ElementarySpace::shared_es()` downcast it.

`Space::as_Space()` had to become non-virtual: `Leg::as_Space()` returns `py::object`, so the two
cannot be overridden under one name in `ElementarySpace`.

### Virtual base initialization with trampolines

`using ElementarySpace::ElementarySpace` does not let the trampoline initialize the *virtual* base
`Leg` (the most-derived class must do that, and inherited ctors don't). Fix: `Leg` got a protected
default ctor plus a protected `init_leg(...)`; every concrete ctor (`LegPipe`, `ElementarySpace`)
calls `init_leg(...)` in its body rather than in the mem-init list.

### Shared state

`Space` and `Leg` both store `symmetry` / `dim`; ctor initializes `Space` first, then `Leg` with `Space::dim`. `is_dual` lives on `Leg`. Extra member: `defining_sectors`.

### Factories

`from_basis`, `from_defining_sectors`, `from_sector_decomposition`, `from_null_space`, `from_trivial_sector`, `from_largest_common_subspace`, `from_independent_symmetries` — port helpers `_sort_sectors` into `spaces.cpp`. Prefer C++ `SectorArray` ops; call Python for `rank_data` / `format_like_list` where useful.

### Return types

`as_ElementarySpace` / `change_symmetry` / `drop_symmetry` on `Space`/`Leg` become able to return real `ElementarySpace` (`py::object` or `Ptr`). Update `Space::as_ElementarySpace` to construct C++ `ElementarySpace`.

### Never trampoline property-bound virtuals

Already noted for `BaseSymmetry`, but it bit us again: a virtual bound with
`def_property_readonly` must not appear in the trampoline. `PYBIND11_OVERRIDE` does
`getattr(self, name)`, finds the property *value*, and raises
`TypeError: Object of type 'list' is not an instance of 'function'`. This made `LegPipe.flat_legs`
(and `flat_spaces` / `num_flat_legs` / `ascii_arrow` / `is_trivial`) unusable for *every* pipe
created from Python, since all of them are `PyLegPipe` instances. Those overrides are now removed
from `PyLeg` / `PySpace` / `PyLegPipe` and were never added to `PyElementarySpace`.

Remaining known limitation: `Leg::dual_leg` / `Leg::is_trivial` / `Space::dual_space` are *pure*
virtual and property-bound, so the trampoline must keep an override to stay concrete. A Python
subclass of the abstract `Leg` / `Space` that defines `dual` as a property therefore still cannot
be called from C++. Concrete classes are unaffected; revisit if a pure-Python `Leg` is needed.

### Monkey-patch

Deferred until `AbelianLegPipe` is converted (MI). Export `cyten._core.ElementarySpace`.

## TODO

- [x] setup / plan
- [x] declaration (hooks for dual; virtual Leg)
- [x] definitions
- [x] bindings + trampoline
- [x] build / pytest (`test_spaces.py`: 867 passed; parity checked against the Python class with
      `tmp/check_elementary_space.py`)
- [ ] monkey-patch deferred
