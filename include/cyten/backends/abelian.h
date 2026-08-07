#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

class AbelianBackend; // deferred — convert after AbelianBackendData

/// Data stored in a Tensor for :class:`AbelianBackend`.
///
/// The :attr:`block_inds` can be visualized as follows::
///
///     |           ---- codomain ---->  <--- domain ----
///     |
///     |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    b |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    l |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    o |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    c |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    k |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |    s |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
///     |      v
///
/// Attributes
/// ----------
/// dtype : Dtype
///     The dtype of the data
/// device : str
///     The device on which the blocks are currently stored.
///     We currently only support tensors which have all blocks on a single device.
///     Should be the device returned by :func:`BlockBackend.as_device`.
/// blocks : list of block
///     A list of blocks containing the actual entries of the tensor.
///     Leg order is ``[*codomain, *reversed(domain()]``, like ``Tensor.legs``.
/// block_inds : 2D ndarray
///     A 2D array of positive integers with shape (len(blocks), num_legs).
///     The block `blocks[n]` belongs to the `block_inds[n, m]`-th sector of ``leg``.
///     By convention, ``np.lexsort(block_inds.T)`` is sorted.
class AbelianBackendData : public TensorBackend::Data
{
  public:
    using Ptr = std::shared_ptr<AbelianBackendData>;
    using CPtr = std::shared_ptr<const AbelianBackendData>;

    Dtype dtype;
    std::string device;
    std::vector<BlockBackend::BlockPtr> blocks;
    py::array_t<int64> block_inds;

    /// Construct data. If ``is_sorted`` is false, permute ``blocks`` / ``block_inds``
    /// according to ``np.lexsort(block_inds.T)``.
    AbelianBackendData(Dtype dtype,
                       std::string device,
                       std::vector<BlockBackend::BlockPtr> blocks,
                       py::array_t<int64> block_inds,
                       bool is_sorted = false);

    ~AbelianBackendData() override = default;

    /// Return the index ``n`` of the block which matches ``block_inds``,
    /// i.e. such that ``all(self.block_inds[n, :] == block_inds)``.
    /// Return ``nullopt`` if no such ``n`` exists.
    std::optional<int64> get_block_num(py::array_t<int64> block_inds) const;

    /// Get the block at given block indices, or ``nullptr`` if none exists.
    BlockBackend::BlockPtr get_block(py::array_t<int64> block_inds) const;

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
