# Conversion of SUN

## Status

In progress on branch `convert_SUN`.

## Metadata

| Field | Value |
| --- | --- |
| original python name | `SUN` |
| original python file | `cyten/symmetries/_symmetries.py` |
| declaration | `include/cyten/symmetries/sun.h` |
| definition | `src/symmetries/sun.cpp` |
| pybind11 binding | `pybind/symmetries/py_sun.cpp` |
| trampoline | none |
| first line of docstring | SU(N) group symmetry |

## Design notes

- Subclass C++ `Group` with `FusionStyle::general`.
- Hold `CGfile` / `Ffile` / `Rfile` as `py::object` (`h5py.File`); most symbol lookups stay as Python HDF5 attribute access from C++.
- Pure C++ for GT validation, `sector_dim`, `dual_sector`, `S_index_irrep_weight`.
- Tests skip without local HDF5 data files.

## TODO checklist

- [x] setup
- [ ] declaration / definitions / bindings / monkey-patch / pytest
- [ ] wrap up
