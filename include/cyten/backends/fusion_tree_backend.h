#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/trees.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
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

/// A backend based on fusion trees.
///
/// Notes
/// -----
/// Data is :class:`FusionTreeData` (coupled-sector ``block_inds`` + forest blocks).
class FusionTreeBackend : public TensorBackend
{
  public:
    using Ptr = std::shared_ptr<FusionTreeBackend>;
    using CPtr = std::shared_ptr<const FusionTreeBackend>;

    /// Numerical threshold for discarding near-zero blocks after topological moves.
    float64 eps = 5.0e-14;

    /// Wrap ``FusionTreeData`` as abstract ``DataPtr``.
    static DataPtr wrap(FusionTreeData::Ptr d);

    /// Unwrap ``DataPtr`` to ``FusionTreeData``; throws if wrong type or null.
    static FusionTreeData::Ptr unwrap(DataPtr d);

    /// Read ``tensor.data`` as ``FusionTreeData``.
    static FusionTreeData::Ptr data_from_tensor(py::object tensor);

    explicit FusionTreeBackend(std::shared_ptr<BlockBackend> block_backend,
                               float64 eps = 5.0e-14);
    ~FusionTreeBackend() override = default;

    void test_tensor_sanity(py::object a, bool is_diagonal) override;
    void test_mask_sanity(py::object a) override;

    DataPtr act_block_diagonal_square_matrix(py::object a,
                                             py::function block_method,
                                             py::object dtype_map) override;

    DataPtr add_trivial_leg(py::object a,
                            int64 legs_pos,
                            bool add_to_domain,
                            int64 co_domain_pos,
                            TensorProduct::Ptr new_codomain,
                            TensorProduct::Ptr new_domain) override;

    bool almost_equal(py::object a, py::object b, float64 rtol, float64 atol) override;

    DataPtr apply_mask_to_DiagonalTensor(py::object tensor, py::object mask) override;

    DataPtr combine_legs(py::object tensor,
                         std::vector<std::vector<int64>> leg_idcs_combine,
                         std::vector<LegPipe::Ptr> pipes,
                         TensorProduct::Ptr new_codomain,
                         TensorProduct::Ptr new_domain) override;

    DataPtr compose(py::object a, py::object b) override;

    DataPtr copy_data(py::object a, std::optional<std::string> device = std::nullopt) override;

    DataPtr dagger(py::object a) override;

    BlockBackend::Scalar data_item(DataPtr a) override;

    bool diagonal_all(py::object a) override;

    bool diagonal_any(py::object a) override;

    DataPtr diagonal_elementwise_binary(py::object a,
                                        py::object b,
                                        py::function func,
                                        py::dict func_kwargs,
                                        bool partial_zero_is_zero) override;

    DataPtr diagonal_elementwise_unary(py::object a,
                                       py::function func,
                                       py::dict func_kwargs,
                                       bool maps_zero_to_zero) override;

    DataPtr diagonal_from_block(BlockBackend::BlockPtr a,
                                TensorProduct::Ptr co_domain,
                                float64 tol) override;

    DataPtr diagonal_from_sector_block_func(py::function func,
                                            TensorProduct::Ptr co_domain) override;

    DataPtr diagonal_tensor_from_full_tensor(py::object a,
                                             std::optional<float64> tol = 1e-12) override;

    BlockBackend::Scalar diagonal_tensor_trace_full(py::object a) override;

    BlockBackend::BlockPtr diagonal_tensor_to_block(py::object a) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> diagonal_to_mask(py::object tens) override;

    std::tuple<Space::Ptr, DataPtr> diagonal_transpose(py::object tens) override;

    std::tuple<DataPtr, DataPtr, ElementarySpace::Ptr>
    eigh(py::object a, bool new_leg_dual, std::optional<std::string> sort = std::nullopt) override;

    DataPtr eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device) override;

    DataPtr from_dense_block(BlockBackend::BlockPtr a,
                             TensorProduct::Ptr codomain,
                             TensorProduct::Ptr domain,
                             float64 tol) override;

    DataPtr from_dense_block_trivial_sector(BlockBackend::BlockPtr block, Space::Ptr leg) override;

    DataPtr from_grid(std::vector<std::vector<py::object>> grid,
                      TensorProduct::Ptr new_codomain,
                      TensorProduct::Ptr new_domain,
                      std::vector<std::vector<int64>> left_mult_slices,
                      std::vector<std::vector<int64>> right_mult_slices,
                      Dtype dtype,
                      std::string device) override;

    DataPtr from_random_normal(TensorProduct::Ptr codomain,
                               TensorProduct::Ptr domain,
                               float64 sigma,
                               Dtype dtype,
                               std::string device) override;

    DataPtr from_sector_block_func(py::function func,
                                   TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain) override;

    DataPtr from_tree_pairs(
      std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      Dtype dtype,
      std::string device) override;

    DataPtr full_data_from_diagonal_tensor(py::object a) override;

    DataPtr full_data_from_mask(py::object a, Dtype dtype) override;

    std::string get_device_from_data(DataPtr a) override;

    Dtype get_dtype_from_data(DataPtr a) override;

    BlockBackend::Scalar get_element(py::object a, std::vector<int64> idcs) override;

    BlockBackend::Scalar get_element_diagonal(py::object a, int64 idx) override;

    BlockBackend::Scalar get_element_mask(py::object a, std::vector<int64> idcs) override;

    BlockBackend::Scalar inner(py::object a, py::object b, bool do_dagger) override;

    DataPtr inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                    Space::Ptr space,
                                                    ElementarySpace::Ptr charge_leg) override;

    BlockBackend::BlockPtr inv_part_to_dense_block_single_sector(py::object tensor) override;

    DataPtr linear_combination(BlockBackend::Scalar a,
                               py::object v,
                               BlockBackend::Scalar b,
                               py::object w) override;

    std::tuple<DataPtr, DataPtr> lq(py::object tensor, TensorProduct::Ptr new_co_domain) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_binary_operand(py::object mask1,
                                                                  py::object mask2,
                                                                  py::function func) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_large_leg(py::object tensor, py::object mask, int64 leg_idx) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_small_leg(py::object tensor, py::object mask, int64 leg_idx) override;

    DataPtr mask_dagger(py::object mask) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_from_block(BlockBackend::BlockPtr a,
                                                              Space::Ptr large_leg) override;

    BlockBackend::BlockPtr mask_to_block(py::object a) override;

    DataPtr mask_to_diagonal(py::object a, Dtype dtype) override;

    std::tuple<Space::Ptr, Space::Ptr, DataPtr> mask_transpose(py::object tens) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_unary_operand(py::object mask,
                                                                 py::function func) override;

    DataPtr move_to_device(py::object a, std::string device) override;

    DataPtr mul(BlockBackend::Scalar a, py::object b) override;

    BlockBackend::Scalar norm(py::object a) override;

    DataPtr outer(py::object a, py::object b) override;

    DataPtr partial_compose(py::object a,
                            py::object b,
                            int64 a_first_leg,
                            TensorProduct::Ptr new_codomain,
                            TensorProduct::Ptr new_domain) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> partial_trace(
      py::object tensor,
      std::vector<std::pair<int64, int64>> pairs,
      std::optional<std::vector<int64>> levels) override;

    DataPtr permute_legs(py::object a,
                         std::vector<int64> codomain_idcs,
                         std::vector<int64> domain_idcs,
                         TensorProduct::Ptr new_codomain,
                         TensorProduct::Ptr new_domain,
                         bool mixes_codomain_domain,
                         std::vector<std::optional<int64>> levels,
                         std::vector<std::optional<bool>> bend_right) override;

    std::tuple<DataPtr, DataPtr> qr(py::object a, TensorProduct::Ptr new_co_domain) override;

    BlockBackend::Scalar reduce_DiagonalTensor(py::object tensor,
                                               py::function block_func,
                                               py::function func) override;

    DataPtr scale_axis(py::object a, py::object b, int64 leg) override;

    DataPtr split_legs(py::object a,
                       std::vector<int64> leg_idcs,
                       TensorProduct::Ptr new_codomain,
                       TensorProduct::Ptr new_domain) override;

    DataPtr squeeze_legs(py::object a, std::vector<int64> idcs) override;

    bool supports_symmetry(Symmetry::Ptr symmetry) override;

    std::tuple<DataPtr, DataPtr, DataPtr> svd(py::object a,
                                              TensorProduct::Ptr new_co_domain,
                                              std::optional<std::string> algorithm) override;

    py::object state_tensor_product(BlockBackend::BlockPtr state1,
                                    BlockBackend::BlockPtr state2,
                                    LegPipe::Ptr pipe) override;

    DataPtr to_block_backend(DataPtr data,
                             std::shared_ptr<BlockBackend> block_backend,
                             std::optional<Dtype> dtype = std::nullopt,
                             std::optional<std::string> device = std::nullopt) override;

    BlockBackend::BlockPtr to_dense_block(py::object a) override;

    BlockBackend::BlockPtr to_dense_block_trivial_sector(py::object tensor) override;

    DataPtr to_dtype(py::object a, Dtype dtype) override;

    BlockBackend::Scalar trace_full(py::object a,
                                    std::vector<int64> idcs1,
                                    std::vector<int64> idcs2) override;

    std::tuple<DataPtr, ElementarySpace::Ptr, float64, float64> truncate_singular_values(
      py::object S,
      std::optional<int64> chi_max,
      int64 chi_min,
      float64 degeneracy_tol,
      float64 trunc_cut,
      float64 svd_min,
      bool minimize_error = true) override;

    DataPtr zero_data(TensorProduct::Ptr codomain,
                      TensorProduct::Ptr domain,
                      Dtype dtype,
                      std::string device,
                      bool all_blocks = false) override;

    DataPtr zero_diagonal_data(TensorProduct::Ptr co_domain,
                               Dtype dtype,
                               std::string device) override;

    DataPtr zero_mask_data(Space::Ptr large_leg, std::string device) override;

    /// Apply a sequence of braid/bend/twist instructions (used by :meth:`permute_legs`).
    DataPtr apply_instructions(py::object tensor,
                               py::object instructions,
                               std::vector<int64> codomain_idcs,
                               std::vector<int64> domain_idcs,
                               TensorProduct::Ptr new_codomain,
                               TensorProduct::Ptr new_domain,
                               bool mixes_codomain_domain);

  private:
    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    _mask_contract(py::object tensor, py::object mask, int64 leg_idx, bool large_leg);

    /// Helper for :meth:`to_dense_block` — contribution of one forest block.
    std::tuple<BlockBackend::BlockPtr, int64, int64> _get_forest_block_contribution(
      BlockBackend::BlockPtr block,
      Symmetry::Ptr sym,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      Sector coupled,
      py::object a_sectors,
      py::object b_sectors,
      std::vector<int64> a_dims,
      std::vector<int64> b_dims,
      int64 tree_block_width,
      int64 tree_block_height,
      int64 i1_init,
      int64 i2_init,
      std::vector<int64> m_mults,
      std::vector<int64> n_mults,
      Dtype dtype) const;

    /// Helper for :meth:`from_dense_block` — accumulate one forest block into ``block``.
    void _add_forest_block_entries(BlockBackend::BlockPtr block,
                                   BlockBackend::BlockPtr entries,
                                   Symmetry::Ptr sym,
                                   TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain,
                                   Sector coupled,
                                   float64 dim_c,
                                   py::object a_sectors,
                                   py::object b_sectors,
                                   std::vector<int64> a_dims,
                                   std::vector<int64> b_dims,
                                   int64 tree_block_width,
                                   int64 tree_block_height,
                                   int64 i1_init,
                                   int64 i2_init,
                                   std::vector<int64> m_mults,
                                   std::vector<int64> n_mults) const;
};

} // namespace cyten
