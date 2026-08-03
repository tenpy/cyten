# Conversion of SymmetryError / FusionStyle / BraidingStyle

## Status

**In progress.** Hand-written C++ (codegen fails on stdlib bases `Exception` / `IntEnum` with `KeyError`).

## Metadata

| Object | Original Python | C++ header | C++ source | Bindings |
| --- | --- | --- | --- | --- |
| `SymmetryError` | `cyten/symmetries/_symmetries.py` | `include/cyten/symmetries/exceptions.h` | n/a (header-only) | `pybind/symmetries/py_exceptions.cpp` |
| `BraidChiralityUnspecifiedError` | same | same | n/a | same |
| `FusionStyle` | same | `include/cyten/symmetries/styles.h` | n/a | `pybind/symmetries/py_styles.cpp` |
| `BraidingStyle` | same | same | n/a | same |

## Notes

- `pybind11_codegen.py gen_cpp_declaration --py-name SymmetryError` → `KeyError: 'Exception'`
- `... --py-name FusionStyle` / `BraidingStyle` → `KeyError: 'IntEnum'`
- Bind enums with `py::native_enum` and Python base `"enum.IntEnum"` so comparisons like `braiding_style <= BraidingStyle.fermionic` keep working.
- Exceptions: `std::runtime_error` subclasses, registered with `py::register_exception`.

## TODO checklist

- [x] initial setup / planning
- [ ] C++ declarations (hand-written)
- [ ] pybind11 bindings
- [ ] monkey-patch into `_symmetries.py`
- [ ] pytest `tests/python_tests/test_symmetries.py`
- [ ] remove original Python definitions once green
