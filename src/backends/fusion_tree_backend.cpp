#include <cyten/backends/fusion_tree_backend.h>

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

/// Reorder rows of a 2D array according to ``perm`` (1D index array).
py::array
take_rows(py::array const& arr, py::array const& perm)
{
    return numpy().attr("take")(arr, perm, py::arg("axis") = 0).cast<py::array>();
}

} // namespace

FusionTreeData::FusionTreeData(py::array block_inds_,
                               std::vector<BlockBackend::BlockPtr> blocks_,
                               Dtype dtype_,
                               std::string device_,
                               bool is_sorted)
  : block_inds(std::move(block_inds_))
  , blocks(std::move(blocks_))
  , dtype(dtype_)
  , device(std::move(device_))
{
    if (block_inds.ndim() != 2) {
        throw std::invalid_argument("FusionTreeData: block_inds must be 2D");
    }
    if (block_inds.shape(1) != 2) {
        throw std::invalid_argument("FusionTreeData: block_inds must have shape (N, 2)");
    }
    if (static_cast<std::size_t>(block_inds.shape(0)) != blocks.size()) {
        throw std::invalid_argument(
          "FusionTreeData: len(blocks) must equal block_inds.shape[0]");
    }

    if (!is_sorted) {
        auto np = numpy();
        // ``np.lexsort(block_inds.T)`` — last column is primary key (numpy convention).
        auto perm = np.attr("lexsort")(block_inds.attr("T")).cast<py::array>();
        block_inds = take_rows(block_inds, perm);

        auto perm_i64 =
          py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(perm);
        auto perm_buf = perm_i64.unchecked<1>();
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(blocks.size());
        for (py::ssize_t i = 0; i < perm_buf.shape(0); ++i) {
            sorted_blocks.push_back(blocks[static_cast<std::size_t>(perm_buf(i))]);
        }
        blocks = std::move(sorted_blocks);
    }
}

std::optional<int64>
FusionTreeData::block_ind_from_coupled(Sector coupled, TensorProduct::Ptr domain) const
{
    auto domain_sector_ind = domain->sector_decomposition_where(coupled);
    if (!domain_sector_ind.has_value()) {
        return std::nullopt;
    }
    return block_ind_from_domain_sector_ind(*domain_sector_ind);
}

std::optional<int64>
FusionTreeData::block_ind_from_domain_sector_ind(int64 domain_sector_ind) const
{
    auto np = numpy();
    // Column 1 only — that axis is sorted by construction (lexsort last-key-first).
    py::array col1 = block_inds[py::make_tuple(py::ellipsis(), 1)].cast<py::array>();
    int64 ind = np.attr("searchsorted")(col1, domain_sector_ind).cast<int64>();

    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(block_inds);
    auto buf = bi.unchecked<2>();
    int64 n_rows = static_cast<int64>(buf.shape(0));

    if (ind >= n_rows || buf(ind, 1) != domain_sector_ind) {
        return std::nullopt;
    }
    if (ind + 1 < n_rows && buf(ind + 1, 1) == domain_sector_ind) {
        throw std::runtime_error(
          "FusionTreeData: duplicate domain sector index in block_inds[:, 1]");
    }
    return ind;
}

void
FusionTreeData::discard_zero_blocks(std::shared_ptr<BlockBackend> backend, float64 eps)
{
    std::vector<int64> keep;
    keep.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if ((backend->norm(blocks[i]) >= eps).as_bool()) {
            keep.push_back(static_cast<int64>(i));
        }
    }

    std::vector<BlockBackend::BlockPtr> kept_blocks;
    kept_blocks.reserve(keep.size());
    for (int64 i : keep) {
        kept_blocks.push_back(blocks[static_cast<std::size_t>(i)]);
    }
    blocks = std::move(kept_blocks);
    block_inds = take_rows(block_inds, py::cast(keep));
}

void
FusionTreeData::save_hdf5(py::object hdf5_saver, py::object /*h5gr*/, std::string subpath) const
{
    hdf5_saver.attr("save")(block_inds, subpath + "block_inds");
    hdf5_saver.attr("save")(blocks, subpath + "blocks");
    hdf5_saver.attr("save")(dtype, subpath + "dtype");
    hdf5_saver.attr("save")(device, subpath + "device");
}

FusionTreeData::Ptr
FusionTreeData::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string subpath)
{
    auto block_inds = hdf5_loader.attr("load")(subpath + "block_inds").cast<py::array>();
    auto blocks =
      hdf5_loader.attr("load")(subpath + "blocks").cast<std::vector<BlockBackend::BlockPtr>>();
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    auto dtype = hdf5_loader.attr("load")(subpath + "dtype").cast<Dtype>();

    // Already sorted when saved; skip lexsort.
    auto obj = std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, std::move(device), /*is_sorted=*/true);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
