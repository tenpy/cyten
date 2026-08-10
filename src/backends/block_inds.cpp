#include <cyten/backends/block_inds.h>

#include <cyten/backends/block_inds_numpy.h>

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace cyten {

namespace {

void
check_row_len(std::size_t got, std::size_t expect, char const* ctx)
{
    if (got != expect) {
        throw std::invalid_argument(std::string(ctx) + ": row length mismatch");
    }
}

} // namespace

void
BlockInds::check_shape() const
{
    if (data_.size() != nrows_ * ncols_) {
        throw std::invalid_argument("BlockInds: data size does not match nrows * ncols");
    }
}

BlockInds::BlockInds(std::size_t nrows, std::size_t ncols)
  : data_(nrows * ncols, int64{ 0 })
  , nrows_(nrows)
  , ncols_(ncols)
{
}

BlockInds::BlockInds(std::vector<int64> data, std::size_t nrows, std::size_t ncols)
  : data_(std::move(data))
  , nrows_(nrows)
  , ncols_(ncols)
{
    check_shape();
}

BlockInds
BlockInds::empty(std::size_t ncols)
{
    return BlockInds(0, ncols);
}

BlockInds
BlockInds::zeros(std::size_t nrows, std::size_t ncols)
{
    return BlockInds(nrows, ncols);
}

BlockInds
BlockInds::from_row(std::span<const int64> row)
{
    BlockInds out(1, row.size());
    std::copy(row.begin(), row.end(), out.data_.begin());
    return out;
}

BlockInds
BlockInds::from_rows(std::vector<std::vector<int64>> const& rows)
{
    if (rows.empty()) {
        return BlockInds{};
    }
    auto const ncols = rows[0].size();
    BlockInds out(rows.size(), ncols);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        if (rows[i].size() != ncols) {
            throw std::invalid_argument("BlockInds::from_rows: inconsistent row lengths");
        }
        std::copy(rows[i].begin(),
                  rows[i].end(),
                  out.data_.begin() + static_cast<std::ptrdiff_t>(i * ncols));
    }
    return out;
}

BlockInds
BlockInds::arange_diag(std::size_t n, std::size_t n_cols)
{
    BlockInds out(n, n_cols);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n_cols; ++j) {
            out(i, j) = static_cast<int64>(i);
        }
    }
    return out;
}

bool
BlockInds::is_lexsorted() const
{
    for (std::size_t i = 1; i < nrows_; ++i) {
        if (cmp_lexsort(*this, i - 1, *this, i) > 0) {
            return false;
        }
    }
    return true;
}

bool
BlockInds::all_ge(int64 value) const
{
    for (int64 v : data_) {
        if (v < value) {
            return false;
        }
    }
    return true;
}

bool
BlockInds::all_lt_per_column(std::span<const int64> maxes) const
{
    if (maxes.size() != ncols_) {
        throw std::invalid_argument("BlockInds::all_lt_per_column: maxes length mismatch");
    }
    for (std::size_t i = 0; i < nrows_; ++i) {
        for (std::size_t j = 0; j < ncols_; ++j) {
            if ((*this)(i, j) >= maxes[j]) {
                return false;
            }
        }
    }
    return true;
}

bool
BlockInds::columns_equal(std::size_t c0, std::size_t c1) const
{
    if (c0 >= ncols_ || c1 >= ncols_) {
        throw std::out_of_range("BlockInds::columns_equal: column out of range");
    }
    for (std::size_t i = 0; i < nrows_; ++i) {
        if ((*this)(i, c0) != (*this)(i, c1)) {
            return false;
        }
    }
    return true;
}

BlockInds
BlockInds::take_columns_i64(std::span<const int64> col_perm) const
{
    std::vector<std::size_t> cols(col_perm.size());
    for (std::size_t j = 0; j < col_perm.size(); ++j) {
        auto c = col_perm[j];
        if (c < 0) {
            c += static_cast<int64>(ncols_);
        }
        if (c < 0 || static_cast<std::size_t>(c) >= ncols_) {
            throw std::out_of_range("BlockInds::take_columns_i64: column out of range");
        }
        cols[j] = static_cast<std::size_t>(c);
    }
    return take_columns(cols);
}

void
BlockInds::set_column(std::size_t col, std::span<const int64> values)
{
    if (col >= ncols_) {
        throw std::out_of_range("BlockInds::set_column: column out of range");
    }
    if (values.size() != nrows_) {
        throw std::invalid_argument("BlockInds::set_column: values length mismatch");
    }
    for (std::size_t i = 0; i < nrows_; ++i) {
        (*this)(i, col) = values[i];
    }
}

void
BlockInds::assign_columns(std::size_t dest_col0, BlockInds const& src)
{
    if (src.nrows_ != nrows_) {
        throw std::invalid_argument("BlockInds::assign_columns: nrows mismatch");
    }
    if (dest_col0 + src.ncols_ > ncols_) {
        throw std::out_of_range("BlockInds::assign_columns: columns out of range");
    }
    for (std::size_t i = 0; i < nrows_; ++i) {
        for (std::size_t j = 0; j < src.ncols_; ++j) {
            (*this)(i, dest_col0 + j) = src(i, j);
        }
    }
}

BlockInds
BlockInds::column_stack(std::vector<std::span<const int64>> const& cols)
{
    if (cols.empty()) {
        return BlockInds::empty(0);
    }
    auto const nrows = cols[0].size();
    for (auto const& c : cols) {
        if (c.size() != nrows) {
            throw std::invalid_argument("BlockInds::column_stack: column length mismatch");
        }
    }
    BlockInds out(nrows, cols.size());
    for (std::size_t j = 0; j < cols.size(); ++j) {
        for (std::size_t i = 0; i < nrows; ++i) {
            out(i, j) = cols[j][i];
        }
    }
    return out;
}

BlockInds
BlockInds::hstack(std::vector<BlockInds> const& parts)
{
    if (parts.empty()) {
        return BlockInds{};
    }
    auto const nrows = parts[0].nrows_;
    std::size_t ncols = 0;
    for (auto const& p : parts) {
        if (p.nrows_ != nrows) {
            throw std::invalid_argument("BlockInds::hstack: nrows mismatch");
        }
        ncols += p.ncols_;
    }
    BlockInds out(nrows, ncols);
    std::size_t col0 = 0;
    for (auto const& p : parts) {
        for (std::size_t i = 0; i < nrows; ++i) {
            for (std::size_t j = 0; j < p.ncols_; ++j) {
                out(i, col0 + j) = p(i, j);
            }
        }
        col0 += p.ncols_;
    }
    return out;
}

void
BlockInds::resize(std::size_t nrows)
{
    data_.resize(nrows * ncols_, int64{ 0 });
    nrows_ = nrows;
}

void
BlockInds::push_back_row(std::span<const int64> row)
{
    check_row_len(row.size(), ncols_, "BlockInds::push_back_row");
    data_.insert(data_.end(), row.begin(), row.end());
    ++nrows_;
}

int
BlockInds::cmp_lexsort(BlockInds const& a,
                       std::size_t i,
                       BlockInds const& b,
                       std::size_t j) noexcept
{
    // Match np.lexsort(arr.T): compare last column first.
    auto const na = a.ncols_;
    auto const nb = b.ncols_;
    auto const n = na < nb ? na : nb;
    for (std::size_t k = 0; k < n; ++k) {
        auto const ik = n - 1 - k;
        auto const va = a(i, ik);
        auto const vb = b(j, ik);
        if (va < vb) {
            return -1;
        }
        if (va > vb) {
            return 1;
        }
    }
    if (na < nb) {
        return -1;
    }
    if (na > nb) {
        return 1;
    }
    return 0;
}

std::vector<std::size_t>
BlockInds::lexsort_indices() const
{
    std::vector<std::size_t> perm(nrows_);
    std::iota(perm.begin(), perm.end(), std::size_t{ 0 });
    std::stable_sort(perm.begin(), perm.end(), [&](std::size_t i, std::size_t j) {
        return cmp_lexsort(*this, i, *this, j) < 0;
    });
    return perm;
}

std::pair<BlockInds, std::vector<std::size_t>>
BlockInds::sorted() const
{
    auto perm = lexsort_indices();
    return { take(perm), std::move(perm) };
}

std::vector<std::size_t>
BlockInds::find_row_differences(bool include_len) const
{
    std::vector<std::size_t> out;
    out.reserve(nrows_ + 1);
    if (nrows_ == 0) {
        if (include_len) {
            out.push_back(0);
        }
        return out;
    }
    out.push_back(0);
    for (std::size_t i = 1; i < nrows_; ++i) {
        bool diff = false;
        for (std::size_t j = 0; j < ncols_; ++j) {
            if ((*this)(i - 1, j) != (*this)(i, j)) {
                diff = true;
                break;
            }
        }
        if (diff) {
            out.push_back(i);
        }
    }
    if (include_len) {
        out.push_back(nrows_);
    }
    return out;
}

std::optional<std::size_t>
BlockInds::row_where(std::span<const int64> query) const
{
    check_row_len(query.size(), ncols_, "BlockInds::row_where");
    for (std::size_t i = 0; i < nrows_; ++i) {
        bool match = true;
        for (std::size_t j = 0; j < ncols_; ++j) {
            if ((*this)(i, j) != query[j]) {
                match = false;
                break;
            }
        }
        if (match) {
            return i;
        }
    }
    return std::nullopt;
}

BlockInds
BlockInds::concat(BlockInds const& other) const
{
    if (nrows_ == 0) {
        if (other.nrows_ != 0 || ncols_ == other.ncols_) {
            return other;
        }
        throw std::invalid_argument("BlockInds::concat: ncols mismatch");
    }
    if (other.nrows_ == 0) {
        return *this;
    }
    if (ncols_ != other.ncols_) {
        throw std::invalid_argument("BlockInds::concat: ncols mismatch");
    }
    BlockInds out(nrows_ + other.nrows_, ncols_);
    std::copy(data_.begin(), data_.end(), out.data_.begin());
    std::copy(other.data_.begin(),
              other.data_.end(),
              out.data_.begin() + static_cast<std::ptrdiff_t>(data_.size()));
    return out;
}

BlockInds
BlockInds::take(std::span<const std::size_t> indices) const
{
    BlockInds out(indices.size(), ncols_);
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto const idx = indices[i];
        if (idx >= nrows_) {
            throw std::out_of_range("BlockInds::take: index out of range");
        }
        auto const* src = data_.data() + idx * ncols_;
        std::copy(src, src + ncols_, out.data_.data() + i * ncols_);
    }
    return out;
}

BlockInds
BlockInds::take_i64(std::span<const int64> indices) const
{
    std::vector<std::size_t> idx(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        auto v = indices[i];
        if (v < 0) {
            throw std::out_of_range("BlockInds::take_i64: negative index");
        }
        idx[i] = static_cast<std::size_t>(v);
    }
    return take(idx);
}

BlockInds
BlockInds::take_mask(std::vector<bool> const& mask) const
{
    if (mask.size() != nrows_) {
        throw std::invalid_argument("BlockInds::take_mask: mask length mismatch");
    }
    std::size_t count = 0;
    for (bool m : mask) {
        count += static_cast<std::size_t>(m);
    }
    BlockInds out(count, ncols_);
    std::size_t o = 0;
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) {
            auto const* src = data_.data() + i * ncols_;
            std::copy(src, src + ncols_, out.data_.data() + o * ncols_);
            ++o;
        }
    }
    return out;
}

BlockInds
BlockInds::slice(std::size_t start, std::size_t stop) const
{
    if (stop > nrows_) {
        stop = nrows_;
    }
    if (start >= stop) {
        return BlockInds::empty(ncols_);
    }
    std::vector<std::size_t> idx(stop - start);
    std::iota(idx.begin(), idx.end(), start);
    return take(idx);
}

BlockInds
BlockInds::take_columns(std::span<const std::size_t> col_perm) const
{
    for (auto c : col_perm) {
        if (c >= ncols_) {
            throw std::out_of_range("BlockInds::take_columns: column out of range");
        }
    }
    BlockInds out(nrows_, col_perm.size());
    for (std::size_t i = 0; i < nrows_; ++i) {
        for (std::size_t j = 0; j < col_perm.size(); ++j) {
            out(i, j) = (*this)(i, col_perm[j]);
        }
    }
    return out;
}

BlockInds
BlockInds::reverse_columns() const
{
    if (ncols_ == 0) {
        return *this;
    }
    std::vector<std::size_t> perm(ncols_);
    for (std::size_t j = 0; j < ncols_; ++j) {
        perm[j] = ncols_ - 1 - j;
    }
    return take_columns(perm);
}

BlockInds
BlockInds::insert_column(std::size_t col, int64 fill) const
{
    if (col > ncols_) {
        throw std::out_of_range("BlockInds::insert_column: column out of range");
    }
    BlockInds out(nrows_, ncols_ + 1);
    for (std::size_t i = 0; i < nrows_; ++i) {
        for (std::size_t j = 0; j < col; ++j) {
            out(i, j) = (*this)(i, j);
        }
        out(i, col) = fill;
        for (std::size_t j = col; j < ncols_; ++j) {
            out(i, j + 1) = (*this)(i, j);
        }
    }
    return out;
}

BlockInds
BlockInds::delete_columns(std::vector<bool> const& keep_mask) const
{
    if (keep_mask.size() != ncols_) {
        throw std::invalid_argument("BlockInds::delete_columns: mask length mismatch");
    }
    std::vector<std::size_t> keep;
    keep.reserve(ncols_);
    for (std::size_t j = 0; j < ncols_; ++j) {
        if (keep_mask[j]) {
            keep.push_back(j);
        }
    }
    return take_columns(keep);
}

std::pair<BlockInds, BlockInds>
BlockInds::hsplit(std::size_t at_col) const
{
    if (at_col > ncols_) {
        throw std::out_of_range("BlockInds::hsplit: split point out of range");
    }
    std::vector<std::size_t> left(at_col);
    std::iota(left.begin(), left.end(), std::size_t{ 0 });
    std::vector<std::size_t> right(ncols_ - at_col);
    std::iota(right.begin(), right.end(), at_col);
    return { take_columns(left), take_columns(right) };
}

BlockInds
BlockInds::repeat_columns(std::size_t times) const
{
    if (times == 0) {
        return BlockInds::empty(0);
    }
    if (times == 1) {
        return *this;
    }
    BlockInds out(nrows_, ncols_ * times);
    for (std::size_t i = 0; i < nrows_; ++i) {
        for (std::size_t t = 0; t < times; ++t) {
            for (std::size_t j = 0; j < ncols_; ++j) {
                out(i, t * ncols_ + j) = (*this)(i, j);
            }
        }
    }
    return out;
}

std::vector<int64>
BlockInds::column(std::size_t j) const
{
    if (j >= ncols_) {
        throw std::out_of_range("BlockInds::column: column out of range");
    }
    std::vector<int64> out(nrows_);
    for (std::size_t i = 0; i < nrows_; ++i) {
        out[i] = (*this)(i, j);
    }
    return out;
}

std::vector<int64>
BlockInds::pack(std::span<const int64> strides) const
{
    if (strides.size() != ncols_) {
        throw std::invalid_argument("BlockInds::pack: strides length mismatch");
    }
    std::vector<int64> out(nrows_, 0);
    for (std::size_t i = 0; i < nrows_; ++i) {
        int64 sum = 0;
        for (std::size_t j = 0; j < ncols_; ++j) {
            sum += (*this)(i, j) * strides[j];
        }
        out[i] = sum;
    }
    return out;
}

std::size_t
BlockInds::searchsorted_column(std::size_t col, int64 value) const
{
    if (col >= ncols_) {
        throw std::out_of_range("BlockInds::searchsorted_column: column out of range");
    }
    std::size_t lo = 0;
    std::size_t hi = nrows_;
    while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2;
        if ((*this)(mid, col) < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

void
BlockInds::iter_common_sorted(BlockInds const& a,
                              BlockInds const& b,
                              bool a_strict,
                              bool b_strict,
                              std::function<void(std::ptrdiff_t, std::ptrdiff_t)> const& yield)
{
    if ((!a_strict) && (!b_strict)) {
        throw std::invalid_argument(
          "BlockInds::iter_common_sorted: one array must be strictly sorted");
    }
    if (a.ncols_ != b.ncols_) {
        throw std::invalid_argument("BlockInds::iter_common_sorted: ncols mismatch");
    }
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.nrows_ && j < b.nrows_) {
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

void
BlockInds::iter_common_noncommon_sorted(
  BlockInds const& a,
  BlockInds const& b,
  std::function<void(std::optional<std::ptrdiff_t>, std::optional<std::ptrdiff_t>)> const& yield)
{
    if (a.ncols_ != b.ncols_) {
        throw std::invalid_argument("BlockInds::iter_common_noncommon_sorted: ncols mismatch");
    }
    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.nrows_ && j < b.nrows_) {
        int const cmp = cmp_lexsort(a, i, b, j);
        if (cmp < 0) {
            yield(static_cast<std::ptrdiff_t>(i), std::nullopt);
            ++i;
        } else if (cmp > 0) {
            yield(std::nullopt, static_cast<std::ptrdiff_t>(j));
            ++j;
        } else {
            yield(static_cast<std::ptrdiff_t>(i), static_cast<std::ptrdiff_t>(j));
            ++i;
            ++j;
        }
    }
    for (; i < a.nrows_; ++i) {
        yield(static_cast<std::ptrdiff_t>(i), std::nullopt);
    }
    for (; j < b.nrows_; ++j) {
        yield(std::nullopt, static_cast<std::ptrdiff_t>(j));
    }
}

void
BlockInds::save_hdf5(py::object hdf5_saver, py::object /*h5gr*/, std::string const& subpath) const
{
    hdf5_saver.attr("save")(block_inds_to_numpy(*this), subpath + "values");
}

BlockInds
BlockInds::from_hdf5(py::object hdf5_loader, py::object /*h5gr*/, std::string const& subpath)
{
    return block_inds_from_numpy(hdf5_loader.attr("load")(subpath + "values"));
}

} // namespace cyten
