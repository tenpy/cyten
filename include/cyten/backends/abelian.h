#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
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

/// Charge-allowed block index combinations for ``codomain`` / ``domain``, lexsorted.
py::array_t<int64> valid_block_inds(TensorProduct::Ptr codomain, TensorProduct::Ptr domain);

/// Backend for Abelian group symmetries.
///
/// Notes
/// -----
/// The data stored for the various tensor classes defined in ``cyten.tensors`` is::
///
///     - ``SymmetricTensor``:
///         An ``AbelianBackendData`` instance whose blocks have as many axes as the tensor has
///         legs.
///
///     - ``DiagonalTensor`` :
///         An ``AbelianBackendData`` instance whose blocks have only a single axis.
///
///     - ``Mask`` :
///         An ``AbelianBackendData`` instance whose blocks have only a single axis and bool
///         values.
class AbelianBackend : public TensorBackend
{
  public:
    using Ptr = std::shared_ptr<AbelianBackend>;
    using CPtr = std::shared_ptr<const AbelianBackend>;

    /// Wrap ``AbelianBackendData`` as abstract ``DataPtr``.
    static DataPtr wrap(AbelianBackendData::Ptr d);

    /// Unwrap ``DataPtr`` to ``AbelianBackendData``; throws if wrong type or null.
    static AbelianBackendData::Ptr unwrap(DataPtr d);

    /// Read ``tensor.data`` as ``AbelianBackendData``.
    static AbelianBackendData::Ptr data_from_tensor(py::object tensor);

    explicit AbelianBackend(std::shared_ptr<BlockBackend> block_backend);
    ~AbelianBackend() override = default;

    void test_tensor_sanity(py::object a, bool is_diagonal) override;
    void test_mask_sanity(py::object a) override;

    LegPipe::Ptr make_pipe(std::vector<Leg::Ptr> legs,
                           bool is_dual,
                           LegPipe::Ptr pipe = nullptr) override;

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

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string subpath) override;

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string subpath);

    /// Map incoming multi-leg block indices through a pipe ``block_ind_map``.
    py::array_t<int64> leg_pipe_map_incoming_block_inds(
      AbelianLegPipe const& pipe,
      py::array_t<int64> incoming_block_inds) const;

  private:
    DataPtr _compose_worker(py::object a, py::object b);
    DataPtr _compose_no_contraction(py::object a, py::object b);
    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> _mask_contract(py::object tensor,
                                                                               py::object mask,
                                                                               int64 leg_idx,
                                                                               bool large_leg);
};

} // namespace cyten
