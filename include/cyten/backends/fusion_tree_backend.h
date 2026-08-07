#pragma once

#include <cyten/backends/tensor_backend.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Data stored in a Tensor for :class:`FusionTreeBackend`.
///
/// Attributes
/// ----------
/// block_inds : 2D array
///     Indices that specify the coupled sectors of the non-zero blocks.
///     ``block_inds[n] == [i, j]`` indicates that the coupled sector for ``blocks[n]`` is given by
///     ``tensor.codomain.sector_decomposition[i] == coupled == tensor.domain.sector_decomposition[j]``.
/// blocks : list of 2D Block
///     The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
/// dtype : Dtype
///     The dtype of the tensor (and of the `blocks`).
/// device : str
///     The device on which the blocks are currently stored.
///     We currently only support tensors which have all blocks on a single device.
///     Should be the device returned by :func:`BlockBackend.as_device`.
///
/// If ``is_sorted`` is ``False`` (default) in the constructor, we permute `blocks` and
/// `block_inds` according to ``np.lexsort(block_inds.T)``. If ``True``, we assume they are
/// sorted *without* checking.
class FusionTreeData : public TensorBackend::Data
{
  public:
    using Ptr = std::shared_ptr<FusionTreeData>;
    using CPtr = std::shared_ptr<const FusionTreeData>;

    py::array block_inds;
    std::vector<BlockBackend::BlockPtr> blocks;
    Dtype dtype;
    std::string device;

    FusionTreeData(py::array block_inds,
                   std::vector<BlockBackend::BlockPtr> blocks,
                   Dtype dtype,
                   std::string device,
                   bool is_sorted = false);

    /// Return `ind` such that ``blocks[ind]`` is associated with the `coupled` sector.
    ///
    /// This is such that ``domain.sector_decomposition[block_inds[res][1]] == coupled``.
    ///
    /// Note: we use the domain (and not the codomain), since only the :attr:`block_inds[:, 1]`
    /// are sorted.
    [[nodiscard]] std::optional<int64> block_ind_from_coupled(Sector coupled,
                                                              TensorProduct::Ptr domain) const;

    /// Return `ind` such that ``block_inds[ind, 1] == domain_sector_ind``.
    ///
    /// Note: we use the domain (and not the codomain), since only the :attr:`block_inds[:, 1]`
    /// are sorted.
    [[nodiscard]] std::optional<int64> block_ind_from_domain_sector_ind(
      int64 domain_sector_ind) const;

    /// Discard blocks whose norm is below the threshold `eps`.
    void discard_zero_blocks(std::shared_ptr<BlockBackend> backend, float64 eps);

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string subpath) const;

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string subpath);
};

} // namespace cyten
