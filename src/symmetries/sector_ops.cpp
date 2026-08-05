#include <cyten/symmetries/sector_ops.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cyten {

int
sector_row_cmp_lexsort(SectorArray const& a,
                       std::size_t i,
                       SectorArray const& b,
                       std::size_t j) noexcept
{
    // Match np.lexsort(sectors.T): compare last column first.
    auto const qa = a.sector_ind_len;
    auto const qb = b.sector_ind_len;
    auto const n = qa < qb ? qa : qb;
    auto const* ra = a.data.data() + i * static_cast<std::size_t>(qa);
    auto const* rb = b.data.data() + j * static_cast<std::size_t>(qb);
    for (std::uint8_t k = 0; k < n; ++k) {
        auto const ik = static_cast<std::uint8_t>(n - 1 - k);
        if (ra[ik] < rb[ik]) {
            return -1;
        }
        if (ra[ik] > rb[ik]) {
            return 1;
        }
    }
    if (qa < qb) {
        return -1;
    }
    if (qa > qb) {
        return 1;
    }
    return 0;
}

std::vector<std::size_t>
lexsort_indices(SectorArray const& sectors)
{
    std::vector<std::size_t> perm(sectors.num_sectors);
    std::iota(perm.begin(), perm.end(), std::size_t{ 0 });
    std::stable_sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
        return sector_row_cmp_lexsort(sectors, i, sectors, j) < 0;
    });
    return perm;
}

std::pair<SectorArray, std::vector<std::size_t>>
sorted_sectors(SectorArray const& sectors)
{
    auto perm = lexsort_indices(sectors);
    return { take_rows(sectors, perm), std::move(perm) };
}

std::vector<std::size_t>
find_row_differences(SectorArray const& sectors, bool include_len)
{
    auto const n = sectors.num_sectors;
    std::vector<std::size_t> out;
    out.reserve(n + 1);
    if (n == 0) {
        if (include_len) {
            out.push_back(0);
        }
        return out;
    }
    out.push_back(0);
    for (std::size_t i = 1; i < n; ++i) {
        if (sectors[i - 1] != sectors[i]) {
            out.push_back(i);
        }
    }
    if (include_len) {
        out.push_back(n);
    }
    return out;
}

std::tuple<SectorArray, std::vector<std::int64_t>, std::vector<std::size_t>>
unique_sorted_sectors(SectorArray const& unsorted_sectors,
                      std::vector<std::int64_t> const& unsorted_multiplicities)
{
    if (unsorted_multiplicities.size() != unsorted_sectors.num_sectors) {
        throw std::invalid_argument("unique_sorted_sectors: multiplicities length mismatch");
    }
    auto [sectors, perm] = sorted_sectors(unsorted_sectors);
    std::vector<std::int64_t> mults(sectors.num_sectors);
    for (std::size_t i = 0; i < perm.size(); ++i) {
        mults[i] = unsorted_multiplicities[perm[i]];
    }

    auto const diffs = find_row_differences(sectors, /*include_len=*/true);
    // diffs includes 0 and num_sectors; unique rows at diffs[0], diffs[1], ... diffs[m-2]
    auto const n_unique = diffs.size() >= 1 ? diffs.size() - 1 : 0;
    SectorArray unique(n_unique, sectors.sector_ind_len);
    std::vector<std::int64_t> unique_mults(n_unique);
    for (std::size_t u = 0; u < n_unique; ++u) {
        auto const start = diffs[u];
        auto const stop = diffs[u + 1];
        unique.set(u, sectors[start]);
        std::int64_t sum = 0;
        for (std::size_t i = start; i < stop; ++i) {
            sum += mults[i];
        }
        unique_mults[u] = sum;
    }
    return { std::move(unique), std::move(unique_mults), std::move(perm) };
}

std::optional<std::size_t>
row_where(SectorArray const& sectors, Sector const& sector)
{
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        if (sectors[i] == sector) {
            return i;
        }
    }
    return std::nullopt;
}

bool
rows_equal(SectorArray const& a, SectorArray const& b) noexcept
{
    return a == b;
}

SectorArray
sector_array_from_sector(Sector const& sector)
{
    SectorArray out(1, sector.len());
    out.set(0, sector);
    return out;
}

SectorArray
concat_sector_arrays(SectorArray const& a, SectorArray const& b)
{
    if (a.num_sectors == 0) {
        return b;
    }
    if (b.num_sectors == 0) {
        return a;
    }
    if (a.sector_ind_len != b.sector_ind_len) {
        throw std::invalid_argument("concat_sector_arrays: sector_ind_len mismatch");
    }
    SectorArray out(a.num_sectors + b.num_sectors, a.sector_ind_len);
    std::copy(a.data.begin(), a.data.end(), out.data.begin());
    std::copy(
      b.data.begin(), b.data.end(), out.data.begin() + static_cast<std::ptrdiff_t>(a.data.size()));
    return out;
}

SectorArray
repeat_row(Sector const& sector, std::size_t n)
{
    SectorArray out(n, sector.len());
    for (std::size_t i = 0; i < n; ++i) {
        out.set(i, sector);
    }
    return out;
}

SectorArray
take_rows(SectorArray const& sectors, std::span<const std::size_t> indices)
{
    SectorArray out(indices.size(), sectors.sector_ind_len);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto const idx = indices[i];
        if (idx >= sectors.num_sectors) {
            throw std::out_of_range("take_rows: index out of range");
        }
        out.set(i, sectors[idx]);
    }
    return out;
}

SectorArray
take_rows_bool(SectorArray const& sectors, std::span<const bool> mask)
{
    if (mask.size() != sectors.num_sectors) {
        throw std::invalid_argument("take_rows_bool: mask length mismatch");
    }
    std::size_t count = 0;
    for (bool m : mask) {
        count += static_cast<std::size_t>(m);
    }
    SectorArray out(count, sectors.sector_ind_len);
    std::size_t o = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            out.set(o++, sectors[i]);
        }
    }
    return out;
}

SectorArray
slice_rows(SectorArray const& sectors, std::size_t start, std::size_t stop)
{
    if (start > stop || stop > sectors.num_sectors) {
        throw std::out_of_range("slice_rows: invalid range");
    }
    std::vector<std::size_t> idx(stop - start);
    std::iota(idx.begin(), idx.end(), start);
    return take_rows(sectors, idx);
}

void
iter_common_sorted_arrays(SectorArray const& a,
                          SectorArray const& b,
                          bool a_strict,
                          bool b_strict,
                          std::function<void(std::ptrdiff_t, std::ptrdiff_t)> const& yield)
{
    if ((!a_strict) && (!b_strict)) {
        throw std::invalid_argument(
          "iter_common_sorted_arrays: one array must be strictly sorted");
    }
    if (a.sector_ind_len != b.sector_ind_len) {
        throw std::invalid_argument("iter_common_sorted_arrays: sector_ind_len mismatch");
    }
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.num_sectors && j < b.num_sectors) {
        int const cmp = sector_row_cmp_lexsort(a, i, b, j);
        if (cmp < 0) {
            ++i;
        } else if (cmp > 0) {
            ++j;
        } else {
            yield(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j));
            if (b_strict) {
                ++i;
            }
            if (a_strict) {
                ++j;
            }
        }
    }
}

} // namespace cyten
