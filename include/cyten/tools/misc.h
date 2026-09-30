#pragma once

#include <cyten/cyten.h>
#include <cyten/symmetries/sector.h>

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cyten {

/// Strides for a C-style (last index fastest) or F-style (first index fastest) array of `shape`.
[[nodiscard]] std::vector<int64> make_stride(std::vector<int64> const& shape, bool cstyle = true);

/// All multi-indices into `shape` as rows. C-style varies the last index fastest.
[[nodiscard]] std::vector<std::vector<int64>> make_grid(std::vector<int64> const& shape,
                                                        bool cstyle = true);

/// Combined flat permutation from per-axis permutations (see Python ``combine_permutations``).
[[nodiscard]] std::vector<int64> combine_permutations(std::vector<std::vector<int64>> const& perms,
                                                      bool cstyle = true);

[[nodiscard]] bool is_permutation(std::vector<int64> const& perm);

/// Inverse of a permutation of ``range(len(perm))`` such that ``inv[perm[j]] == j``.
[[nodiscard]] std::vector<int64> inverse_permutation(std::vector<int64> const& perm);

/// Ranks of `a` (``argsort(argsort(a))``). Stable ranks preserve appearance order for ties.
[[nodiscard]] std::vector<int64> rank_data(std::vector<int64> const& a, bool stable = true);

/// ``logical_and(good1, good2)`` if any entry remains true; else warn and return `good1`.
[[nodiscard]] std::vector<bool> combine_constraints(std::vector<bool> const& good1,
                                                    std::vector<bool> const& good2,
                                                    char const* warn);

/// Lookup table from each sector to the indices where it appears in `sectors`.
[[nodiscard]] std::unordered_map<Sector, std::vector<int64>> list_to_dict_list(
  SectorArray const& sectors);

/// Duplicate values in `seq`, excluding those in `ignore`.
template<typename T>
[[nodiscard]] std::unordered_set<T>
duplicate_entries(std::vector<T> const& seq, std::vector<T> const& ignore = {})
{
    std::unordered_set<T> ignore_set(ignore.begin(), ignore.end());
    std::unordered_set<T> dups;
    for (std::size_t i = 0; i < seq.size(); ++i) {
        T const& ele = seq[i];
        if (ignore_set.contains(ele)) {
            continue;
        }
        for (std::size_t j = i + 1; j < seq.size(); ++j) {
            if (seq[j] == ele) {
                dups.insert(ele);
                break;
            }
        }
    }
    return dups;
}

/// Yield ``(i, j)`` with ``a[i] == b[j]``. Assumes strictly ascending 1D sequences.
void iter_common_sorted_1d(std::vector<int64> const& a,
                           std::vector<int64> const& b,
                           std::function<void(std::ptrdiff_t i, std::ptrdiff_t j)> const& yield);

// For 2D / sector rows prefer ``SectorArray::iter_common_sorted`` or
// ``BlockInds::iter_common_sorted``.

} // namespace cyten
