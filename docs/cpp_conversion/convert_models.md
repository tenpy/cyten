# Conversion of cyten.models (Layer 5)

## metadata

- original python module: `cyten.models`
- branch: `convert_models`
- C++ umbrella header: `include/cyten/models.h`
- C++ sources: `src/models/{degrees_of_freedom,sites,couplings}.cpp`
- pybind11 bindings: `pybind/models/py_{degrees_of_freedom,sites,couplings}.cpp`

## scope

Convert (monkey-patch from `_core`):

| File | Objects |
| --- | --- |
| `degrees_of_freedom.py` | `Site`, `SpinDOF`, `OccupationDOF`, `BosonicDOF`, `FermionicDOF`, `ClockDOF`, `AnyonDOF`, `ALL_SPECIES` |
| `sites.py` | All concrete `*Site` classes |
| `couplings.py` | `Coupling`, `freeze`, factories |

**Deferred:** `tenpy_models.py` (mockup, not exported).

## design notes

- No `Hdf5Exportable` in C++ — `Site` does not inherit HDF5 base.
- Virtual `Site` base for `SpinHalfFermionSite` MI (`SpinDOF` + `FermionicDOF`).
- Dense operator tables stored as `py::array` with write-protected views (`as_immutable_array`).
- Trampolines: `Site` (`test_sanity`), `OccupationDOF` (abstract numpy getters).
- `Coupling._permuted` cache and `_levels` kept as private C++ members.

## tests

- `tests/python_tests/models/test_site.py`
- `tests/python_tests/models/test_couplings.py`
