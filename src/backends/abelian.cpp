#include <cyten/backends/abelian.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

namespace {

py::module_
numpy()
{
    return py::module_::import("numpy");
}

/// Reorder rows of a 2D int64 array according to ``perm`` (1D index array).
py::array_t<int64>
take_rows(py::array_t<int64> const& arr, py::array const& perm)
{
    return numpy()
      .attr("take")(arr, perm, py::arg("axis") = 0)
      .cast<py::array_t<int64>>();
}

} // namespace

AbelianBackendData::AbelianBackendData(Dtype dtype_,
                                       std::string device_,
                                       std::vector<BlockBackend::BlockPtr> blocks_,
                                       py::array_t<int64> block_inds_,
                                       bool is_sorted)
  : dtype(dtype_)
  , device(std::move(device_))
  , blocks(std::move(blocks_))
  , block_inds(std::move(block_inds_))
{
    if (block_inds.ndim() != 2) {
        throw std::invalid_argument("AbelianBackendData: block_inds must be 2D");
    }
    if (static_cast<std::size_t>(block_inds.shape(0)) != blocks.size()) {
        throw std::invalid_argument(
          "AbelianBackendData: len(blocks) must equal block_inds.shape[0]");
    }

    if (!is_sorted) {
        auto np = numpy();
        // ``np.lexsort(block_inds.T)`` — last row is primary key (numpy convention).
        auto perm = np.attr("lexsort")(block_inds.attr("T")).cast<py::array_t<int64>>();
        block_inds = take_rows(block_inds, perm);

        auto perm_buf = perm.unchecked<1>();
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(blocks.size());
        for (py::ssize_t i = 0; i < perm_buf.shape(0); ++i) {
            sorted_blocks.push_back(blocks[static_cast<std::size_t>(perm_buf(i))]);
        }
        blocks = std::move(sorted_blocks);
    }
}

std::optional<int64>
AbelianBackendData::get_block_num(py::array_t<int64> query) const
{
    // OPTIMIZE use sorted for lookup?
    auto np = numpy();
    py::object equal = np.attr("all")(np.attr("equal")(block_inds, query), py::arg("axis") = 1);
    py::array match = np.attr("argwhere")(equal)
                        .attr("__getitem__")(py::make_tuple(py::ellipsis(), 0))
                        .cast<py::array>();
    if (match.size() == 0) {
        return std::nullopt;
    }
    return match.cast<py::array_t<int64>>().at(0);
}

BlockBackend::BlockPtr
AbelianBackendData::get_block(py::array_t<int64> query) const
{
    auto block_num = get_block_num(query);
    if (!block_num.has_value()) {
        return nullptr;
    }
    return blocks[static_cast<std::size_t>(*block_num)];
}

void
AbelianBackendData::save_hdf5(py::object hdf5_saver,
                              py::object /*h5gr*/,
                              std::string const& subpath) const
{
    hdf5_saver.attr("save")(block_inds, subpath + "block_inds");
    hdf5_saver.attr("save")(blocks, subpath + "blocks");
    hdf5_saver.attr("save")(dtype::to_numpy_dtype(dtype), subpath + "dtype");
    hdf5_saver.attr("save")(device, subpath + "device");
}

AbelianBackendData::Ptr
AbelianBackendData::from_hdf5(py::object hdf5_loader,
                              py::object h5gr,
                              std::string const& subpath)
{
    auto block_inds =
      hdf5_loader.attr("load")(subpath + "block_inds").cast<py::array_t<int64>>();
    auto blocks =
      hdf5_loader.attr("load")(subpath + "blocks").cast<std::vector<BlockBackend::BlockPtr>>();
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    py::object dt = hdf5_loader.attr("load")(subpath + "dtype");
    Dtype dtype = dtype::from_numpy_dtype(dt);

    // Already sorted when saved; skip lexsort.
    auto obj = std::make_shared<AbelianBackendData>(
      dtype, std::move(device), std::move(blocks), std::move(block_inds), /*is_sorted=*/true);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
