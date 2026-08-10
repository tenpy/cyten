#pragma once

/// Owning 2D int64 array for abelian ``block_inds``: shape ``(nrows, ncols)``.
///
/// Layout is a flat row-major ``std::vector<int64>``. ``ncols`` is remembered even when
/// ``nrows == 0`` (empty tensors) or when ``ncols == 0`` (0-leg scalars with one block).
/// Lex order matches ``np.lexsort(arr.T)`` (last column primary).

#include <cyten/cyten.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace cyten {

class BlockInds
{
  public:
    BlockInds() = default;

    /// ``n`` rows of ``ncols`` zeros.
    BlockInds(std::size_t nrows, std::size_t ncols);

    /// Own existing flat row-major data of size ``nrows * ncols``.
    BlockInds(std::vector<int64> data, std::size_t nrows, std::size_t ncols);

    /// Empty array that still remembers ``ncols``.
    static BlockInds empty(std::size_t ncols);

    /// Alias for ``BlockInds(nrows, ncols)`` (all zeros).
    static BlockInds zeros(std::size_t nrows, std::size_t ncols);

    /// Single row from a span of length ``ncols``.
    static BlockInds from_row(std::span<const int64> row);

    /// Stack 1D columns of equal length into shape ``(n, n_cols)``.
    static BlockInds column_stack(std::vector<std::span<const int64>> const& cols);

    /// Horizontal concatenation; all inputs must share ``nrows``.
    static BlockInds hstack(std::vector<BlockInds> const& parts);

    [[nodiscard]] std::size_t nrows() const noexcept { return nrows_; }
    [[nodiscard]] std::size_t ncols() const noexcept { return ncols_; }
    [[nodiscard]] std::size_t size() const noexcept { return nrows_; } // rows; NumPy-like len()
    [[nodiscard]] bool empty() const noexcept { return nrows_ == 0; }

    [[nodiscard]] std::vector<int64> const& data() const noexcept { return data_; }
    [[nodiscard]] std::vector<int64>& data() noexcept { return data_; }

    [[nodiscard]] int64 const& operator()(std::size_t i, std::size_t j) const
    {
        return data_[i * ncols_ + j];
    }
    [[nodiscard]] int64& operator()(std::size_t i, std::size_t j) { return data_[i * ncols_ + j]; }

    [[nodiscard]] std::span<const int64> row(std::size_t i) const
    {
        return std::span<const int64>(data_.data() + i * ncols_, ncols_);
    }
    [[nodiscard]] std::span<int64> row(std::size_t i)
    {
        return std::span<int64>(data_.data() + i * ncols_, ncols_);
    }

    void resize(std::size_t nrows);
    void push_back_row(std::span<const int64> row);

    [[nodiscard]] std::vector<std::size_t> lexsort_indices() const;
    [[nodiscard]] std::pair<BlockInds, std::vector<std::size_t>> sorted() const;

    [[nodiscard]] std::vector<std::size_t> find_row_differences(bool include_len = false) const;

    [[nodiscard]] std::optional<std::size_t> row_where(std::span<const int64> query) const;

    [[nodiscard]] BlockInds concat(BlockInds const& other) const; // vstack
    [[nodiscard]] BlockInds take(std::span<const std::size_t> indices) const;
    [[nodiscard]] BlockInds take_i64(std::span<const int64> indices) const;
    [[nodiscard]] BlockInds take_mask(std::vector<bool> const& mask) const;
    [[nodiscard]] BlockInds slice(std::size_t start, std::size_t stop) const;

    [[nodiscard]] BlockInds take_columns(std::span<const std::size_t> col_perm) const;
    [[nodiscard]] BlockInds reverse_columns() const;
    [[nodiscard]] BlockInds insert_column(std::size_t col, int64 fill = 0) const;
    [[nodiscard]] BlockInds delete_columns(std::vector<bool> const& keep_mask) const;
    [[nodiscard]] std::pair<BlockInds, BlockInds> hsplit(std::size_t at_col) const;
    [[nodiscard]] BlockInds repeat_columns(std::size_t times) const;

    /// Column ``j`` as a contiguous vector copy.
    [[nodiscard]] std::vector<int64> column(std::size_t j) const;

    /// ``(bi * strides).sum(axis=1)`` with ``strides.size() == ncols``.
    [[nodiscard]] std::vector<int64> pack(std::span<const int64> strides) const;

    /// ``np.searchsorted`` on a single column (assumes that column is sorted ascending).
    [[nodiscard]] std::size_t searchsorted_column(std::size_t col, int64 value) const;

    /// Merge walk of two lex-sorted BlockInds (``np.lexsort`` order).
    static void iter_common_sorted(
      BlockInds const& a,
      BlockInds const& b,
      bool a_strict,
      bool b_strict,
      std::function<void(std::ptrdiff_t, std::ptrdiff_t)> const& yield);

    /// Like Python ``iter_common_noncommon_sorted_arrays``; ``nullopt`` means absent side.
    static void iter_common_noncommon_sorted(
      BlockInds const& a,
      BlockInds const& b,
      std::function<void(std::optional<std::ptrdiff_t>, std::optional<std::ptrdiff_t>)> const&
        yield);

    friend bool operator==(BlockInds const& a, BlockInds const& b) noexcept
    {
        return a.nrows_ == b.nrows_ && a.ncols_ == b.ncols_ && a.data_ == b.data_;
    }
    friend bool operator!=(BlockInds const& a, BlockInds const& b) noexcept { return !(a == b); }

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    static BlockInds from_hdf5(py::object hdf5_loader,
                               py::object h5gr,
                               std::string const& subpath);

  private:
    std::vector<int64> data_;
    std::size_t nrows_ = 0;
    std::size_t ncols_ = 0;

    void check_shape() const;
    [[nodiscard]] static int cmp_lexsort(BlockInds const& a,
                                         std::size_t i,
                                         BlockInds const& b,
                                         std::size_t j) noexcept;
};

} // namespace cyten
