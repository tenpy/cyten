# FusionSymbol

Native owning dense array for topological symmetry data (F/R/C/B symbols, fusion
tensors, swap gates, Z isomorphisms, S-matrix).

## Layout

- Rank `1..4`, shape stored as `std::array<std::size_t,4>` with unused axes `= 1`
- Dtype: **`Float64` or `Complex128` only**
- Contiguous C-order storage via `std::variant<vector<float64>, vector<complex128>>`

## Dtype rules

- Abelian / SU(2) / real topological data → `Float64` (including trivial ones formerly
  stored as NumPy `intp`)
- Symmetries with `has_complex_topological_data` (or genuinely complex entries) →
  `Complex128`
- Hot-path fusion-tree amplitudes may **read** entries as `complex128` without changing
  storage dtype
- Do **not** force all symbols to complex: that would promote real SU(2) tensors via
  `Dtype.common(tensor, fusion_tensor_dtype)`

## Related APIs

- `batch_sector_dim` / `batch_qdim` return `std::vector<int64>` / `std::vector<float64>`
  (not `FusionSymbol`)
- `block_from_fusion_symbol` / `fusion_symbol_from_block` bridge to `BlockBackend::Block`
- SUN CG→F/R contractions run on `Block`, then convert back to `FusionSymbol`
- pybind caster: `FusionSymbol` ↔ `numpy.ndarray` (transparent to Python)

## Files

- `include/cyten/symmetries/fusion_symbol.h`
- `src/symmetries/fusion_symbol.cpp`
- `include/cyten/symmetries/topo_ones.h` (thin wrappers around `FusionSymbol::one_*`)
