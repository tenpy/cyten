#pragma once

/// Domain helpers on Sector / SectorArray (lexsort, unique, concat, …).
/// Matches the former NumPy idioms used in spaces/trees/backends.

#include "sector.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

/// Lexicographic compare of rows matching ``np.lexsort(sectors.T)``:
/// last column is the primary key, first column the least significant.
[[nodiscard]] int sector_row_cmp_lexsort(SectorArray const& a,
                                         std::size_t i,
                                         SectorArray const& b,
                                         std::size_t j) noexcept;

/// Indices that sort rows like ``np.lexsort(sectors.T)``.
[[nodiscard]] std::vector<std::size_t> lexsort_indices(SectorArray const& sectors);

/// ``(sorted_sectors, permutation)`` with ``sorted = sectors[perm]``.
[[nodiscard]] std::pair<SectorArray, std::vector<std::size_t>> sorted_sectors(
  SectorArray const& sectors);

/// Like :func:`find_row_differences` in Python.
[[nodiscard]] std::vector<std::size_t> find_row_differences(SectorArray const& sectors,
                                                            bool include_len = false);

/// Sort then merge duplicate rows; optionally aggregate multiplicities (sum).
/// Returns ``(unique_sorted_sectors, multiplicities, perm)``.
[[nodiscard]] std::tuple<SectorArray, std::vector<std::int64_t>, std::vector<std::size_t>>
unique_sorted_sectors(SectorArray const& unsorted_sectors,
                      std::vector<std::int64_t> const& unsorted_multiplicities);

/// Index of the first row equal to ``sector``, or nullopt.
[[nodiscard]] std::optional<std::size_t> row_where(SectorArray const& sectors,
                                                   Sector const& sector);

[[nodiscard]] bool rows_equal(SectorArray const& a, SectorArray const& b) noexcept;

/// Single-row SectorArray (old ``sector[None, :]``).
[[nodiscard]] SectorArray sector_array_from_sector(Sector const& sector);

[[nodiscard]] SectorArray concat_sector_arrays(SectorArray const& a, SectorArray const& b);

[[nodiscard]] SectorArray repeat_row(Sector const& sector, std::size_t n);

/// Select rows by index list (fancy indexing).
[[nodiscard]] SectorArray take_rows(SectorArray const& sectors,
                                    std::span<const std::size_t> indices);

[[nodiscard]] SectorArray take_rows_bool(SectorArray const& sectors, std::span<const bool> mask);

/// Slice ``sectors[start:stop]`` (stop exclusive; negative not supported).
[[nodiscard]] SectorArray slice_rows(SectorArray const& sectors,
                                     std::size_t start,
                                     std::size_t stop);

/// Strict/non-strict merge of two lex-sorted SectorArrays (``np.lexsort`` order).
/// Yields pairs via callback ``(i, j)``; use -1 for missing side in noncommon variant.
void iter_common_sorted_arrays(SectorArray const& a,
                               SectorArray const& b,
                               bool a_strict,
                               bool b_strict,
                               std::function<void(std::ptrdiff_t, std::ptrdiff_t)> const& yield);

} // namespace cyten
