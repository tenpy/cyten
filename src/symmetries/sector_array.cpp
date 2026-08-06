#include <cyten/symmetries/sector.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace cyten {

SectorArray::SectorArray(std::size_t n, std::uint8_t sector_ind_len_)
  : Base(n, Sector::zeros(sector_ind_len_))
  , sector_ind_len_(sector_ind_len_)
{
    if (sector_ind_len_ > max_sector_ind_len) {
        throw std::invalid_argument("SectorArray sector_ind_len exceeds max_sector_ind_len");
    }
}

SectorArray::SectorArray(std::vector<Sector> sectors)
  : Base(std::move(sectors))
{
    if (Base::empty()) {
        sector_ind_len_ = 0;
        return;
    }
    sector_ind_len_ = front().len();
    if (sector_ind_len_ > max_sector_ind_len) {
        throw std::invalid_argument("SectorArray sector_ind_len exceeds max_sector_ind_len");
    }
    for (Sector const& s : *this) {
        if (s.len() != sector_ind_len_) {
            throw std::invalid_argument("SectorArray: inconsistent sector lengths");
        }
    }
}

SectorArray
SectorArray::empty(std::uint8_t sector_ind_len_)
{
    return SectorArray(0, sector_ind_len_);
}

SectorArray
SectorArray::from_sector(Sector const& sector)
{
    SectorArray out(1, sector.len());
    out[0] = sector;
    return out;
}

SectorArray
SectorArray::repeat(Sector const& sector, std::size_t n)
{
    SectorArray out(n, sector.len());
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = sector;
    }
    return out;
}

void
SectorArray::check_sector_len(Sector const& s) const
{
    if (s.len() != sector_ind_len_) {
        throw std::invalid_argument("SectorArray: sector length mismatch");
    }
}

void
SectorArray::resize(std::size_t n)
{
    Base::resize(n, Sector::zeros(sector_ind_len_));
}

void
SectorArray::resize(std::size_t n, Sector const& fill)
{
    check_sector_len(fill);
    Base::resize(n, fill);
}

void
SectorArray::push_back(Sector const& s)
{
    check_sector_len(s);
    Base::push_back(s);
}

void
SectorArray::push_back(Sector&& s)
{
    check_sector_len(s);
    Base::push_back(std::move(s));
}

int
SectorArray::cmp_lexsort(SectorArray const& a,
                         std::size_t i,
                         SectorArray const& b,
                         std::size_t j) noexcept
{
    // Match np.lexsort(sectors.T): compare last column first.
    auto const qa = a.sector_ind_len_;
    auto const qb = b.sector_ind_len_;
    auto const n = qa < qb ? qa : qb;
    Sector const& sa = a[i];
    Sector const& sb = b[j];
    for (std::uint8_t k = 0; k < n; ++k) {
        auto const ik = static_cast<std::uint8_t>(n - 1 - k);
        if (sa[ik] < sb[ik]) {
            return -1;
        }
        if (sa[ik] > sb[ik]) {
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
SectorArray::lexsort_indices() const
{
    std::vector<std::size_t> perm(size());
    std::iota(perm.begin(), perm.end(), std::size_t{ 0 });
    std::stable_sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
        return cmp_lexsort(*this, i, *this, j) < 0;
    });
    return perm;
}

std::pair<SectorArray, std::vector<std::size_t>>
SectorArray::sorted() const
{
    auto perm = lexsort_indices();
    return { take(perm), std::move(perm) };
}

std::vector<std::size_t>
SectorArray::find_row_differences(bool include_len) const
{
    auto const n = size();
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
        if ((*this)[i - 1] != (*this)[i]) {
            out.push_back(i);
        }
    }
    if (include_len) {
        out.push_back(n);
    }
    return out;
}

std::tuple<SectorArray, std::vector<std::int64_t>, std::vector<std::size_t>>
SectorArray::unique_sorted(std::vector<std::int64_t> const& multiplicities) const
{
    if (multiplicities.size() != size()) {
        throw std::invalid_argument("SectorArray::unique_sorted: multiplicities length mismatch");
    }
    auto [sectors, perm] = sorted();
    std::vector<std::int64_t> mults(sectors.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        mults[i] = multiplicities[perm[i]];
    }

    auto const diffs = sectors.find_row_differences(/*include_len=*/true);
    auto const n_unique = diffs.size() >= 1 ? diffs.size() - 1 : 0;
    SectorArray unique(n_unique, sectors.sector_ind_len_);
    std::vector<std::int64_t> unique_mults(n_unique);
    for (std::size_t u = 0; u < n_unique; ++u) {
        auto const start = diffs[u];
        auto const stop = diffs[u + 1];
        unique[u] = sectors[start];
        std::int64_t sum = 0;
        for (std::size_t i = start; i < stop; ++i) {
            sum += mults[i];
        }
        unique_mults[u] = sum;
    }
    return { std::move(unique), std::move(unique_mults), std::move(perm) };
}

std::optional<std::size_t>
SectorArray::row_where(Sector const& sector) const
{
    for (std::size_t i = 0; i < size(); ++i) {
        if ((*this)[i] == sector) {
            return i;
        }
    }
    return std::nullopt;
}

SectorArray
SectorArray::concat(SectorArray const& other) const
{
    if (size() == 0) {
        if (other.size() != 0 || sector_ind_len_ == other.sector_ind_len_) {
            return other;
        }
        throw std::invalid_argument("SectorArray::concat: sector_ind_len mismatch");
    }
    if (other.size() == 0) {
        return *this;
    }
    if (sector_ind_len_ != other.sector_ind_len_) {
        throw std::invalid_argument("SectorArray::concat: sector_ind_len mismatch");
    }
    auto const n_a = size();
    auto const n_b = other.size();
    SectorArray out(n_a + n_b, sector_ind_len_);
    for (std::size_t i = 0; i < n_a; ++i) {
        out[i] = (*this)[i];
    }
    for (std::size_t i = 0; i < n_b; ++i) {
        out[n_a + i] = other[i];
    }
    return out;
}

SectorArray
SectorArray::take(std::span<const std::size_t> indices) const
{
    SectorArray out(indices.size(), sector_ind_len_);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto const idx = indices[i];
        if (idx >= size()) {
            throw std::out_of_range("SectorArray::take: index out of range");
        }
        out[i] = (*this)[idx];
    }
    return out;
}

SectorArray
SectorArray::take_mask(std::vector<bool> const& mask) const
{
    if (mask.size() != size()) {
        throw std::invalid_argument("SectorArray::take_mask: mask length mismatch");
    }
    std::size_t count = 0;
    for (bool m : mask) {
        count += static_cast<std::size_t>(m);
    }
    SectorArray out(count, sector_ind_len_);
    std::size_t o = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            out[o++] = (*this)[i];
        }
    }
    return out;
}

SectorArray
SectorArray::slice(std::size_t start, std::size_t stop) const
{
    if (start > stop || stop > size()) {
        throw std::out_of_range("SectorArray::slice: invalid range");
    }
    std::vector<std::size_t> idx(stop - start);
    std::iota(idx.begin(), idx.end(), start);
    return take(idx);
}

void
SectorArray::iter_common_sorted(SectorArray const& a,
                                SectorArray const& b,
                                bool a_strict,
                                bool b_strict,
                                std::function<void(std::ptrdiff_t, std::ptrdiff_t)> const& yield)
{
    if ((!a_strict) && (!b_strict)) {
        throw std::invalid_argument(
          "SectorArray::iter_common_sorted: one array must be strictly sorted");
    }
    if (a.sector_ind_len_ != b.sector_ind_len_) {
        throw std::invalid_argument("SectorArray::iter_common_sorted: sector_ind_len mismatch");
    }
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() && j < b.size()) {
        int const cmp = cmp_lexsort(a, i, b, j);
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
