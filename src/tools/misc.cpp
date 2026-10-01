#include <cyten/tools/misc.h>

#include <cyten/tools/warn.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

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

namespace {

[[nodiscard]] std::vector<std::string>
split_lines(std::string s)
{
    // expand tabs
    std::string expanded;
    expanded.reserve(s.size());
    for (char c : s) {
        if (c == '\t') {
            expanded.append(8, ' ');
        } else {
            expanded.push_back(c);
        }
    }
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= expanded.size()) {
        auto pos = expanded.find('\n', start);
        if (pos == std::string::npos) {
            lines.push_back(expanded.substr(start));
            break;
        }
        lines.push_back(expanded.substr(start, pos - start));
        start = pos + 1;
    }
    return lines;
}

/// Unicode codepoint count (matches Python ``len`` on ``str``), not byte length.
[[nodiscard]] std::size_t
utf8_len(std::string_view s)
{
    std::size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) {
            ++n;
        }
    }
    return n;
}

/// Pad `line` with spaces to `width` codepoints (Python ``f'{line:{align}{width}}'``).
[[nodiscard]] std::string
pad_utf8(std::string const& line, std::size_t width, char halign)
{
    auto const len = utf8_len(line);
    if (len >= width) {
        return line;
    }
    auto const pad = width - len;
    if (halign == 'r') {
        return std::string(pad, ' ') + line;
    }
    if (halign == 'c') {
        auto const left = pad / 2;
        return std::string(left, ' ') + line + std::string(pad - left, ' ');
    }
    return line + std::string(pad, ' ');
}

} // namespace

std::string
vert_join(std::vector<std::string> const& strlist,
          char valign,
          char halign,
          std::string const& delim)
{
    std::vector<std::vector<std::string>> cols;
    cols.reserve(strlist.size());
    std::vector<std::size_t> numlines;
    std::vector<std::size_t> widths;
    std::size_t totallines = 0;
    for (auto const& s : strlist) {
        auto lines = split_lines(s);
        std::size_t w = 0;
        for (auto const& l : lines) {
            // Match Python ``len``: column width is in Unicode codepoints, not bytes.
            // Box-drawing chars in ascii diagrams are multi-byte UTF-8.
            w = std::max(w, utf8_len(l));
        }
        totallines = std::max(totallines, lines.size());
        numlines.push_back(lines.size());
        widths.push_back(w);
        cols.push_back(std::move(lines));
    }

    std::vector<std::vector<std::string>> res(totallines,
                                              std::vector<std::string>(strlist.size()));
    for (std::size_t j = 0; j < strlist.size(); ++j) {
        for (std::size_t i = 0; i < totallines; ++i) {
            res[i][j] = std::string(widths[j], ' ');
        }
        std::size_t voffset = 0;
        if (valign == 'b') {
            voffset = totallines - numlines[j];
        } else if (valign == 'c') {
            voffset = (totallines - numlines[j]) / 2;
        } else if (valign != 't') {
            throw std::invalid_argument("vert_join: invalid valign");
        }
        for (std::size_t i = 0; i < cols[j].size(); ++i) {
            res[i + voffset][j] = pad_utf8(cols[j][i], widths[j], halign);
        }
    }

    std::string out;
    for (std::size_t i = 0; i < totallines; ++i) {
        if (i > 0) {
            out += '\n';
        }
        for (std::size_t j = 0; j < strlist.size(); ++j) {
            if (j > 0) {
                out += delim;
            }
            out += res[i][j];
        }
    }
    return out;
}

} // namespace cyten
