#include <cyten/tools/misc.h>

#include <cyten/tools/warn.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace cyten {

std::vector<int64>
make_stride(std::vector<int64> const& shape, bool cstyle)
{
    auto const L = shape.size();
    std::vector<int64> res(L, 1);
    int64 stride = 1;
    if (cstyle) {
        for (std::size_t a = L; a-- > 1;) {
            stride *= shape[a];
            res[a - 1] = stride;
        }
    } else {
        for (std::size_t a = 0; a + 1 < L; ++a) {
            stride *= shape[a];
            res[a + 1] = stride;
        }
    }
    return res;
}

std::vector<std::vector<int64>>
make_grid(std::vector<int64> const& shape, bool cstyle)
{
    int64 n = 1;
    for (auto s : shape) {
        n *= s;
    }
    auto const strides = make_stride(shape, cstyle);
    std::vector<std::vector<int64>> grid(static_cast<std::size_t>(n),
                                         std::vector<int64>(shape.size()));
    for (int64 m = 0; m < n; ++m) {
        for (std::size_t i = 0; i < shape.size(); ++i) {
            grid[static_cast<std::size_t>(m)][i] = (m / strides[i]) % shape[i];
        }
    }
    return grid;
}

bool
is_permutation(std::vector<int64> const& perm)
{
    std::vector<int64> sorted = perm;
    std::ranges::sort(sorted);
    for (std::size_t i = 0; i < sorted.size(); ++i) {
        if (sorted[i] != static_cast<int64>(i)) {
            return false;
        }
    }
    return true;
}

std::vector<int64>
combine_permutations(std::vector<std::vector<int64>> const& perms, bool cstyle)
{
    for (auto const& p : perms) {
        if (!is_permutation(p)) {
            throw std::invalid_argument("combine_permutations: not a permutation");
        }
    }
    std::vector<int64> shape;
    shape.reserve(perms.size());
    for (auto const& p : perms) {
        shape.push_back(static_cast<int64>(p.size()));
    }
    auto const strides = make_stride(shape, cstyle);
    int64 n = 1;
    for (auto s : shape) {
        n *= s;
    }
    std::vector<int64> result(static_cast<std::size_t>(n));
    for (int64 m = 0; m < n; ++m) {
        int64 val = 0;
        for (std::size_t i = 0; i < perms.size(); ++i) {
            auto const idx = static_cast<std::size_t>((m / strides[i]) % shape[i]);
            val += perms[i][idx] * strides[i];
        }
        result[static_cast<std::size_t>(m)] = val;
    }
    return result;
}

std::vector<int64>
inverse_permutation(std::vector<int64> const& perm)
{
    std::vector<int64> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        auto const idx = perm[i];
        if (idx < 0 || static_cast<std::size_t>(idx) >= perm.size()) {
            throw std::invalid_argument("inverse_permutation: index out of range");
        }
        inv[static_cast<std::size_t>(idx)] = static_cast<int64>(i);
    }
    return inv;
}

std::vector<int64>
rank_data(std::vector<int64> const& a, bool stable)
{
    std::vector<std::size_t> order(a.size());
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    auto cmp = [&a](std::size_t i, std::size_t j) { return a[i] < a[j]; };
    if (stable) {
        std::ranges::stable_sort(order, cmp);
    } else {
        std::ranges::sort(order, cmp);
    }
    std::vector<int64> ranks(a.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        ranks[order[i]] = static_cast<int64>(i);
    }
    return ranks;
}

std::vector<bool>
combine_constraints(std::vector<bool> const& good1,
                    std::vector<bool> const& good2,
                    char const* warn_msg)
{
    if (good1.size() != good2.size()) {
        throw std::invalid_argument("combine_constraints: shape mismatch");
    }
    std::vector<bool> res(good1.size());
    bool any = false;
    for (std::size_t i = 0; i < good1.size(); ++i) {
        res[i] = good1[i] && good2[i];
        any = any || res[i];
    }
    if (any) {
        return res;
    }
    warn(std::string("truncation: can't satisfy constraint for ") + warn_msg, /*stack_level=*/3);
    return good1;
}

std::unordered_map<Sector, std::vector<int64>>
list_to_dict_list(SectorArray const& sectors)
{
    std::unordered_map<Sector, std::vector<int64>> d;
    d.reserve(sectors.size());
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        d[sectors[i]].push_back(static_cast<int64>(i));
    }
    return d;
}

void
iter_common_sorted_1d(std::vector<int64> const& a,
                      std::vector<int64> const& b,
                      std::function<void(std::ptrdiff_t, std::ptrdiff_t)> const& yield)
{
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] < b[j]) {
            ++i;
        } else if (b[j] < a[i]) {
            ++j;
        } else {
            yield(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j));
            ++i;
            ++j;
        }
    }
}

} // namespace cyten
