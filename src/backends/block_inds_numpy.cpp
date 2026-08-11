#include <cyten/backends/block_inds_numpy.h>

#include <stdexcept>

namespace cyten {

namespace {

template<typename T>
bool
load_block_inds_from_buffer(py::buffer_info const& info, BlockInds& out)
{
    if (info.ndim == 1) {
        // Treat a 1D vector as a single row (e.g. get_block_num query).
        auto const ncols = static_cast<std::size_t>(info.shape[0]);
        auto const* ptr = static_cast<T const*>(info.ptr);
        auto const stride0 = info.strides[0] / static_cast<ssize_t>(sizeof(T));
        out = BlockInds(1, ncols);
        for (std::size_t j = 0; j < ncols; ++j) {
            out(0, j) = static_cast<int64>(ptr[static_cast<ssize_t>(j) * stride0]);
        }
        return true;
    }
    if (info.ndim != 2) {
        return false;
    }
    auto const nrows = static_cast<std::size_t>(info.shape[0]);
    auto const ncols = static_cast<std::size_t>(info.shape[1]);
    auto const* ptr = static_cast<T const*>(info.ptr);
    auto const stride0 = info.strides[0] / static_cast<ssize_t>(sizeof(T));
    auto const stride1 = info.strides[1] / static_cast<ssize_t>(sizeof(T));
    out = BlockInds(nrows, ncols);
    for (std::size_t i = 0; i < nrows; ++i) {
        for (std::size_t j = 0; j < ncols; ++j) {
            out(i, j) = static_cast<int64>(
              ptr[static_cast<ssize_t>(i) * stride0 + static_cast<ssize_t>(j) * stride1]);
        }
    }
    return true;
}

} // namespace

py::array_t<int64>
block_inds_to_numpy(BlockInds const& src)
{
    py::array_t<int64> arr(
      { static_cast<py::ssize_t>(src.nrows()), static_cast<py::ssize_t>(src.ncols()) });
    auto r = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < src.nrows(); ++i) {
        for (std::size_t j = 0; j < src.ncols(); ++j) {
            r(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) = src(i, j);
        }
    }
    return arr;
}

BlockInds
block_inds_from_numpy(py::handle src)
{
    BlockInds out;
    py::array arr = py::array::ensure(src);
    if (!arr) {
        throw std::invalid_argument("block_inds_from_numpy: expected array-like");
    }
    auto const info = arr.request();
    bool ok = false;
    if (info.item_type_is_equivalent_to<std::int16_t>()) {
        ok = load_block_inds_from_buffer<std::int16_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int32_t>()) {
        ok = load_block_inds_from_buffer<std::int32_t>(info, out);
    } else if (info.item_type_is_equivalent_to<std::int64_t>()) {
        ok = load_block_inds_from_buffer<std::int64_t>(info, out);
    } else {
        auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(src);
        if (casted) {
            ok = load_block_inds_from_buffer<int64>(casted.request(), out);
        }
    }
    if (!ok) {
        throw std::invalid_argument("block_inds_from_numpy: expected 1D or 2D integer array");
    }
    return out;
}

} // namespace cyten
