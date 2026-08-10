# Conversion of remaining concrete symmetries

## Status

**Done** on branch `convert_remaining_symmetries`.

Converted and monkey-patched from `_core` (Python class bodies removed from `_symmetries.py`):

| Class | C++ / pybind |
| --- | --- |
| `FermionNumber` | `fermion_number.*` |
| `FermionParity` | `fermion_parity.*` |
| `ZNAnyonCategory` | `zn_anyon_category.*` |
| `ZNAnyonCategory2` | `zn_anyon_category2.*` |
| `QuantumDoubleZNAnyonCategory` | `quantum_double_zn_anyon_category.*` |
| `ToricCodeCategory` | `toric_code_category.*` |
| `FibonacciAnyonCategory` | `fibonacci_anyon_category.*` |
| `IsingAnyonCategory` | `ising_anyon_category.*` |
| `SU2_kAnyonCategory` | `su2_k_anyon_category.*` |
| `SU3_3AnyonCategory` | `su3_3_anyon_category.*` |

Shared helper: `include/cyten/symmetries/topo_ones.h`.

`tests/python_tests/test_symmetries.py`: **48 passed, 1 skipped**.

## Design notes

- All are `SymmetryFactor` leaves (except `ToricCodeCategory` → `QuantumDoubleZNAnyonCategory`).
- No trampolines (no Python subclasses of these leaves beyond ToricCode, which is also C++).
- `_default_c_symbol` uses `F(c, a, b, d, e, f)`, not `F(a, b, c, …)`.
- Ising `_r_symbol`: nontrivial when both charges are nonzero (`np.all(concat([a,b]))`), not only σ×σ.
- `ZNAnyonCategory2(N odd)` must raise `AssertionError` (tests use `pytest.raises(AssertionError)`).
- SU3_3 C-map uses C-fusion conditions `(a,b→e), (e,c→d), (a,c→f), (f,b→d)`.
