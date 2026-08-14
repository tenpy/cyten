#pragma once

#include <cyten/backends/tensor_backend.h>

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

/// Backend for tensors that do not enforce any symmetry.
///
/// Notes
/// -----
/// The data stored for the various tensor classes defined in ``cyten.tensors`` is::
///
///     - ``SymmetricTensor``:
///         A single Block with as many axes as there a legs on the tensor.
///         Same leg order as ``Tensor.legs``, i.e. ``[*codomain, *reversed(domain)]``.
///
///     - ``DiagonalTensor`` :
///         A single 1D Block. The diagonal of the corresponding 2D block of a ``Tensor``.
///
///     - ``Mask``:
///         The bool values indicate which indices of the large leg are kept for the small leg.
class NoSymmetryBackend : public TensorBackend
{
  public:
    using Ptr = std::shared_ptr<NoSymmetryBackend>;
    using CPtr = std::shared_ptr<const NoSymmetryBackend>;

    /// Thin ``Data`` wrapper around a single dense block (Python stores the Block directly).
    class BlockData : public TensorBackend::Data
    {
      public:
        using Ptr = std::shared_ptr<BlockData>;
        using CPtr = std::shared_ptr<const BlockData>;

        BlockBackend::BlockPtr block;

        explicit BlockData(BlockBackend::BlockPtr b);
    };

    /// Wrap a Block as abstract ``DataPtr``.
    static DataPtr wrap(BlockBackend::BlockPtr b);

    /// Unwrap ``DataPtr`` to Block; throws if not ``BlockData`` (or null).
    static BlockBackend::BlockPtr unwrap(DataPtr d);

    /// Read ``tensor.data`` as a Block (Python still stores Block on tensors).
    static BlockBackend::BlockPtr block_from_tensor(TensorCPtr tensor);

    explicit NoSymmetryBackend(std::shared_ptr<BlockBackend> block_backend);
    ~NoSymmetryBackend() override = default;

    void test_tensor_sanity(TensorCPtr a, bool is_diagonal) override;
    void test_mask_sanity(MaskCPtr a) override;

    DataPtr act_block_diagonal_square_matrix(
      SymmetricTensorCPtr a,
      BlockUnaryFn block_method,
      std::optional<DtypeMapFn> dtype_map = std::nullopt) override;

    DataPtr add_trivial_leg(TensorCPtr a,
                            int64 legs_pos,
                            bool add_to_domain,
                            int64 co_domain_pos,
                            TensorProduct::Ptr new_codomain,
                            TensorProduct::Ptr new_domain) override;

    bool almost_equal(TensorCPtr a, TensorCPtr b, float64 rtol, float64 atol) override;

    DataPtr apply_mask_to_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask) override;

    DataPtr combine_legs(TensorCPtr tensor,
                         std::vector<std::vector<int64>> leg_idcs_combine,
                         std::vector<LegPipe::Ptr> pipes,
                         TensorProduct::Ptr new_codomain,
                         TensorProduct::Ptr new_domain) override;

    DataPtr compose(SymmetricTensorCPtr a, SymmetricTensorCPtr b) override;

    DataPtr copy_data(TensorCPtr a, std::optional<std::string> device = std::nullopt) override;

    DataPtr dagger(TensorCPtr a) override;

    BlockBackend::Scalar data_item(DataPtr a) override;

    bool diagonal_all(DiagonalTensorCPtr a) override;

    bool diagonal_any(DiagonalTensorCPtr a) override;

    DataPtr diagonal_elementwise_binary(DiagonalTensorCPtr a,
                                        DiagonalTensorCPtr b,
                                        BlockBinaryFn func,
                                        bool partial_zero_is_zero) override;

    DataPtr diagonal_elementwise_unary(DiagonalTensorCPtr a,
                                       BlockUnaryFn func,
                                       bool maps_zero_to_zero) override;

    DataPtr diagonal_from_block(BlockBackend::BlockPtr a,
                                TensorProduct::Ptr co_domain,
                                float64 tol) override;

    DataPtr diagonal_from_sector_block_func(SectorBlockFactoryFn func,
                                            TensorProduct::Ptr co_domain) override;

    DataPtr diagonal_tensor_from_full_tensor(SymmetricTensorCPtr a,
                                             std::optional<float64> tol = 1e-12) override;

    BlockBackend::Scalar diagonal_tensor_trace_full(DiagonalTensorCPtr a) override;

    BlockBackend::BlockPtr diagonal_tensor_to_block(DiagonalTensorCPtr a) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> diagonal_to_mask(DiagonalTensorCPtr tens) override;

    std::tuple<Space::Ptr, DataPtr> diagonal_transpose(DiagonalTensorCPtr tens) override;

    std::tuple<DataPtr, DataPtr, ElementarySpace::Ptr> eigh(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) override;

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

    /// Generate tensor data from a function ``func(shape, coupled) -> Block``.
    DataPtr from_sector_block_func(SectorBlockFactoryFn func,
                                   TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain) override;

    DataPtr from_tree_pairs(
      std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      Dtype dtype,
      std::string device) override;

    DataPtr full_data_from_diagonal_tensor(DiagonalTensorCPtr a) override;

    DataPtr full_data_from_mask(MaskCPtr a, Dtype dtype) override;

    std::string get_device_from_data(DataPtr a) override;

    Dtype get_dtype_from_data(DataPtr a) override;

    BlockBackend::Scalar get_element(SymmetricTensorCPtr a, std::vector<int64> idcs) override;

    BlockBackend::Scalar get_element_diagonal(DiagonalTensorCPtr a, int64 idx) override;

    BlockBackend::Scalar get_element_mask(MaskCPtr a, std::vector<int64> idcs) override;

    BlockBackend::Scalar inner(SymmetricTensorCPtr a,
                               SymmetricTensorCPtr b,
                               bool do_dagger) override;

    DataPtr inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                    Space::Ptr space,
                                                    ElementarySpace::Ptr charge_leg) override;

    BlockBackend::BlockPtr inv_part_to_dense_block_single_sector(
      SymmetricTensorCPtr tensor) override;

    DataPtr linear_combination(BlockBackend::Scalar a,
                               TensorCPtr v,
                               BlockBackend::Scalar b,
                               TensorCPtr w) override;

    std::tuple<DataPtr, DataPtr> lq(SymmetricTensorCPtr tensor,
                                    TensorProduct::Ptr new_co_domain) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_binary_operand(MaskCPtr mask1,
                                                                  MaskCPtr mask2,
                                                                  BlockBinaryFn func) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_large_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_small_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) override;

    DataPtr mask_dagger(MaskCPtr mask) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_from_block(BlockBackend::BlockPtr a,
                                                              Space::Ptr large_leg) override;

    BlockBackend::BlockPtr mask_to_block(MaskCPtr a) override;

    DataPtr mask_to_diagonal(MaskCPtr a, Dtype dtype) override;

    std::tuple<Space::Ptr, Space::Ptr, DataPtr> mask_transpose(MaskCPtr tens) override;

    std::tuple<DataPtr, ElementarySpace::Ptr> mask_unary_operand(MaskCPtr mask,
                                                                 BlockUnaryFn func) override;

    DataPtr move_to_device(TensorCPtr a, std::string device) override;

    DataPtr mul(BlockBackend::Scalar a, TensorCPtr b) override;

    BlockBackend::Scalar norm(TensorCPtr a) override;

    DataPtr outer(SymmetricTensorCPtr a, SymmetricTensorCPtr b) override;

    DataPtr partial_compose(SymmetricTensorCPtr a,
                            SymmetricTensorCPtr b,
                            int64 a_first_leg,
                            TensorProduct::Ptr new_codomain,
                            TensorProduct::Ptr new_domain) override;

    std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> partial_trace(
      SymmetricTensorCPtr tensor,
      std::vector<std::pair<int64, int64>> pairs,
      std::vector<std::optional<int64>> levels) override;

    DataPtr permute_legs(TensorCPtr a,
                         std::vector<int64> codomain_idcs,
                         std::vector<int64> domain_idcs,
                         TensorProduct::Ptr new_codomain,
                         TensorProduct::Ptr new_domain,
                         bool mixes_codomain_domain,
                         std::vector<std::optional<int64>> levels,
                         std::vector<std::optional<bool>> bend_right) override;

    std::tuple<DataPtr, DataPtr> qr(SymmetricTensorCPtr a,
                                    TensorProduct::Ptr new_co_domain) override;

    BlockBackend::Scalar reduce_DiagonalTensor(DiagonalTensorCPtr tensor,
                                               BlockToScalarFn block_func,
                                               ScalarReduceFn func) override;

    DataPtr scale_axis(TensorCPtr a, DiagonalTensorCPtr b, int64 leg) override;

    DataPtr split_legs(TensorCPtr a,
                       std::vector<int64> leg_idcs,
                       TensorProduct::Ptr new_codomain,
                       TensorProduct::Ptr new_domain) override;

    DataPtr squeeze_legs(TensorCPtr a, std::vector<int64> idcs) override;

    bool supports_symmetry(Symmetry::Ptr symmetry) override;

    std::tuple<DataPtr, DataPtr, DataPtr> svd(SymmetricTensorCPtr a,
                                              TensorProduct::Ptr new_co_domain,
                                              std::optional<std::string> algorithm) override;

    py::object state_tensor_product(BlockBackend::BlockPtr state1,
                                    BlockBackend::BlockPtr state2,
                                    LegPipe::Ptr pipe) override;

    DataPtr to_block_backend(DataPtr data,
                             std::shared_ptr<BlockBackend> block_backend,
                             std::optional<Dtype> dtype = std::nullopt,
                             std::optional<std::string> device = std::nullopt) override;

    BlockBackend::BlockPtr to_dense_block(TensorCPtr a) override;

    BlockBackend::BlockPtr to_dense_block_trivial_sector(TensorCPtr tensor) override;

    DataPtr to_dtype(TensorCPtr a, Dtype dtype) override;

    BlockBackend::Scalar trace_full(SymmetricTensorCPtr a,
                                    std::vector<int64> idcs1,
                                    std::vector<int64> idcs2) override;

    std::tuple<DataPtr, ElementarySpace::Ptr, float64, float64> truncate_singular_values(
      DiagonalTensorCPtr S,
      std::optional<int64> chi_max,
      int64 chi_min,
      float64 degeneracy_tol,
      float64 trunc_cut,
      std::optional<float64> svd_min,
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
};

} // namespace cyten
