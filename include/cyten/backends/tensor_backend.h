#pragma once

#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/symmetry.h>
#include <cyten/symmetries/trees.h>
#include <cyten/tensors/forward_declare.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

/// Abstract base class for tensor-backends.
///
/// A backends implements functions that act on tensors.
/// We abstract two separate concepts for a backend.
/// There is a block backend, that abstracts what the numerical data format (numpy array,
/// torch Tensor, CUDA tensor, ...) is and a tensor-backend that abstracts how block-sparse
/// structures that arise from symmetries are accounted for.
///
/// A tensor backend has a the :attr:`block_backend` as an attribute and can call its functions
/// to operate on blocks. This allows the tensor backend to be agnostic of the details of these
/// blocks.
///
/// Tensor args are typed C++ pointers (see tensors/forward_declare.h); backend headers must not
/// include complete tensor headers.
class TensorBackend : public std::enable_shared_from_this<TensorBackend>
{
  public:
    using Ptr = std::shared_ptr<TensorBackend>;
    using CPtr = std::shared_ptr<const TensorBackend>;

    /// Backend-specific payload stored on a tensor (except symmetry data on legs).
    /// Concrete backends subclass this (or wrap a :class:`BlockBackend::Block`).
    class Data : public std::enable_shared_from_this<Data>
    {
      public:
        using Ptr = std::shared_ptr<Data>;
        using CPtr = std::shared_ptr<const Data>;

        virtual ~Data() = default;
    };

    using DataPtr = Data::Ptr;
    using DataCPtr = Data::CPtr;

    /// Python ``DataCls`` used by sanity checks (``isinstance(a.data, DataCls)``).
    /// Set by concrete backends (e.g. block class or pybind type object).
    py::object DataCls;

    /// If the decompositions (SVD, QR, EIGH, ...) can operate on many-leg tensors,
    /// or require legs to be combined first.
    bool can_decompose_tensors = false;

    std::shared_ptr<BlockBackend> block_backend;

    explicit TensorBackend(std::shared_ptr<BlockBackend> block_backend);
    virtual ~TensorBackend() = default;

    virtual std::string __repr__() const;
    virtual std::string __str__() const;

    /// Convert tensor to a python scalar.
    ///
    /// Assumes that tensor is a scalar (i.e. has only one entry).
    BlockBackend::Scalar item(TensorCPtr a);

    /// Called as part of :meth:`cyten.Tensor.test_sanity`.
    ///
    /// Perform sanity checks on the ``a.data``, and possibly additional backend-specific checks
    /// of the tensor.
    virtual void test_tensor_sanity(TensorCPtr a, bool is_diagonal);

    virtual void test_mask_sanity(MaskCPtr a);

    /// Make a pipe *of the appropriate type* for :meth:`combine_legs`.
    ///
    /// If `pipe` is given, try to return it if suitable.
    virtual LegPipe::Ptr make_pipe(std::vector<Leg::Ptr> legs,
                                   bool is_dual,
                                   LegPipe::Ptr pipe = nullptr);

    /// Apply functions like exp() and log() on a (square) block-diagonal `a`.
    virtual DataPtr act_block_diagonal_square_matrix(SymmetricTensorCPtr a,
                                                     py::function block_method,
                                                     py::object dtype_map) = 0;

    virtual DataPtr add_trivial_leg(TensorCPtr a,
                                    int64 legs_pos,
                                    bool add_to_domain,
                                    int64 co_domain_pos,
                                    TensorProduct::Ptr new_codomain,
                                    TensorProduct::Ptr new_domain) = 0;

    virtual bool almost_equal(TensorCPtr a, TensorCPtr b, float64 rtol, float64 atol) = 0;

    virtual DataPtr apply_mask_to_DiagonalTensor(DiagonalTensorCPtr tensor, MaskCPtr mask) = 0;

    /// Implementation of :func:`cyten.tensors.combine_legs`.
    virtual DataPtr combine_legs(TensorCPtr tensor,
                                 std::vector<std::vector<int64>> leg_idcs_combine,
                                 std::vector<LegPipe::Ptr> pipes,
                                 TensorProduct::Ptr new_codomain,
                                 TensorProduct::Ptr new_domain) = 0;

    /// Assumes ``a.domain == b.codomain`` and performs contraction over those legs.
    virtual DataPtr compose(SymmetricTensorCPtr a, SymmetricTensorCPtr b) = 0;

    /// Return a copy.
    virtual DataPtr copy_data(TensorCPtr a, std::optional<std::string> device = std::nullopt) = 0;

    virtual DataPtr dagger(TensorCPtr a) = 0;

    /// Assumes that data is a scalar (as defined in tensors.is_scalar).
    virtual BlockBackend::Scalar data_item(DataPtr a) = 0;

    /// Assumes a boolean DiagonalTensor. If all entries are True.
    virtual bool diagonal_all(DiagonalTensorCPtr a) = 0;

    /// Assumes a boolean DiagonalTensor. If any entry is True.
    virtual bool diagonal_any(DiagonalTensorCPtr a) = 0;

    /// Return a modified copy of the data, resulting from applying an elementwise function.
    virtual DataPtr diagonal_elementwise_binary(DiagonalTensorCPtr a,
                                                DiagonalTensorCPtr b,
                                                py::function func,
                                                py::dict func_kwargs,
                                                bool partial_zero_is_zero) = 0;

    /// Return a modified copy of the data, resulting from applying an elementwise function.
    virtual DataPtr diagonal_elementwise_unary(DiagonalTensorCPtr a,
                                               py::function func,
                                               py::dict func_kwargs,
                                               bool maps_zero_to_zero) = 0;

    /// The DiagonalData from a 1D block in *internal* basis order.
    virtual DataPtr diagonal_from_block(BlockBackend::BlockPtr a,
                                        TensorProduct::Ptr co_domain,
                                        float64 tol) = 0;

    /// Generate diagonal data from a function.
    virtual DataPtr diagonal_from_sector_block_func(py::function func,
                                                    TensorProduct::Ptr co_domain) = 0;

    /// Get the DiagonalData corresponding to a tensor with two legs.
    virtual DataPtr diagonal_tensor_from_full_tensor(SymmetricTensorCPtr a,
                                                     std::optional<float64> tol = 1e-12) = 0;

    virtual BlockBackend::Scalar diagonal_tensor_trace_full(DiagonalTensorCPtr a) = 0;

    /// Forget about symmetry structure and convert to a single 1D block.
    virtual BlockBackend::BlockPtr diagonal_tensor_to_block(DiagonalTensorCPtr a) = 0;

    /// Convert a DiagonalTensor to a Mask. Returns ``mask_data, small_leg``.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> diagonal_to_mask(
      DiagonalTensorCPtr tens) = 0;

    /// Transpose a diagonal tensor. Also return the new leg ``tens.leg.dual``.
    virtual std::tuple<Space::Ptr, DataPtr> diagonal_transpose(DiagonalTensorCPtr tens) = 0;

    /// Eigenvalue decomposition of a hermitian tensor.
    virtual std::tuple<DataPtr, DataPtr, ElementarySpace::Ptr> eigh(
      SymmetricTensorCPtr a,
      bool new_leg_dual,
      std::optional<std::string> sort = std::nullopt) = 0;

    /// Data for :meth:`SymmetricTensor.eye`.
    virtual DataPtr eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device) = 0;

    /// Convert a dense block to the data for a symmetric tensor.
    virtual DataPtr from_dense_block(BlockBackend::BlockPtr a,
                                     TensorProduct::Ptr codomain,
                                     TensorProduct::Ptr domain,
                                     float64 tol) = 0;

    /// Data of a single-leg `Tensor` from the *part of* the coefficients in the trivial sector.
    virtual DataPtr from_dense_block_trivial_sector(BlockBackend::BlockPtr block,
                                                    Space::Ptr leg) = 0;

    /// Data from a grid of tensors.
    virtual DataPtr from_grid(std::vector<std::vector<py::object>> grid,
                              TensorProduct::Ptr new_codomain,
                              TensorProduct::Ptr new_domain,
                              std::vector<std::vector<int64>> left_mult_slices,
                              std::vector<std::vector<int64>> right_mult_slices,
                              Dtype dtype,
                              std::string device) = 0;

    virtual DataPtr from_random_normal(TensorProduct::Ptr codomain,
                                       TensorProduct::Ptr domain,
                                       float64 sigma,
                                       Dtype dtype,
                                       std::string device) = 0;

    /// Generate tensor data from a function.
    virtual DataPtr from_sector_block_func(py::function func,
                                           TensorProduct::Ptr codomain,
                                           TensorProduct::Ptr domain) = 0;

    /// Compute the data for :meth:`SymmetricTensor.from_tree_pairs`.
    virtual DataPtr from_tree_pairs(
      std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
      TensorProduct::Ptr codomain,
      TensorProduct::Ptr domain,
      Dtype dtype,
      std::string device) = 0;

    virtual DataPtr full_data_from_diagonal_tensor(DiagonalTensorCPtr a) = 0;

    /// May assume that the mask is a projection.
    virtual DataPtr full_data_from_mask(MaskCPtr a, Dtype dtype) = 0;

    /// Extract the device from the data object.
    virtual std::string get_device_from_data(DataPtr a) = 0;

    virtual Dtype get_dtype_from_data(DataPtr a) = 0;

    /// Get a single scalar element from a tensor.
    virtual BlockBackend::Scalar get_element(SymmetricTensorCPtr a, std::vector<int64> idcs) = 0;

    /// Get a single scalar element from a diagonal tensor.
    virtual BlockBackend::Scalar get_element_diagonal(DiagonalTensorCPtr a, int64 idx) = 0;

    /// Get a single scalar element from a mask.
    virtual BlockBackend::Scalar get_element_mask(MaskCPtr a, std::vector<int64> idcs) = 0;

    /// tensors.inner on SymmetricTensors.
    virtual BlockBackend::Scalar inner(SymmetricTensorCPtr a,
                                       SymmetricTensorCPtr b,
                                       bool do_dagger) = 0;

    /// Data for the invariant part used in ChargedTensor.from_dense_block_single_sector.
    virtual DataPtr inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                            Space::Ptr space,
                                                            ElementarySpace::Ptr charge_leg) = 0;

    /// Inverse of inv_part_from_dense_block_single_sector.
    virtual BlockBackend::BlockPtr inv_part_to_dense_block_single_sector(
      SymmetricTensorCPtr tensor) = 0;

    /// Form the linear combinations ``a * v + b * w``.
    virtual DataPtr linear_combination(BlockBackend::Scalar a,
                                       TensorCPtr v,
                                       BlockBackend::Scalar b,
                                       TensorCPtr w) = 0;

    virtual std::tuple<DataPtr, DataPtr> lq(SymmetricTensorCPtr tensor,
                                            TensorProduct::Ptr new_co_domain) = 0;

    /// Elementwise binary function acting on two masks.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> mask_binary_operand(MaskCPtr mask1,
                                                                          MaskCPtr mask2,
                                                                          py::function func) = 0;

    /// Contraction with the large leg of a Mask.
    virtual std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_large_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) = 0;

    /// Contraction with the small leg of a Mask.
    virtual std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
    mask_contract_small_leg(TensorCPtr tensor, MaskCPtr mask, int64 leg_idx) = 0;

    virtual DataPtr mask_dagger(MaskCPtr mask) = 0;

    /// Data for a *projection* Mask, and the resulting small leg, from a 1D block.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> mask_from_block(BlockBackend::BlockPtr a,
                                                                      Space::Ptr large_leg) = 0;

    /// As a block of the large_leg, in *internal* basis order.
    virtual BlockBackend::BlockPtr mask_to_block(MaskCPtr a) = 0;

    virtual DataPtr mask_to_diagonal(MaskCPtr a, Dtype dtype) = 0;

    /// Transpose a mask. Also return the new ``space_in`` and ``space_out``.
    virtual std::tuple<Space::Ptr, Space::Ptr, DataPtr> mask_transpose(MaskCPtr tens) = 0;

    /// Elementwise function acting on a mask.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr> mask_unary_operand(MaskCPtr mask,
                                                                         py::function func) = 0;

    /// Move tensor to a given device.
    virtual DataPtr move_to_device(TensorCPtr a, std::string device) = 0;

    virtual DataPtr mul(BlockBackend::Scalar a, TensorCPtr b) = 0;

    /// Norm of a tensor. order has already been parsed and is a number.
    virtual BlockBackend::Scalar norm(TensorCPtr a) = 0;

    /// Form the outer product, or tensor product of maps.
    virtual DataPtr outer(SymmetricTensorCPtr a, SymmetricTensorCPtr b) = 0;

    /// Contract the codomain (domain) of `b` with the a part of the domain (codomain) of `a`.
    virtual DataPtr partial_compose(SymmetricTensorCPtr a,
                                    SymmetricTensorCPtr b,
                                    int64 a_first_leg,
                                    TensorProduct::Ptr new_codomain,
                                    TensorProduct::Ptr new_domain) = 0;

    /// Perform an arbitrary number of traces. Pairs are converted to leg idcs.
    /// Returns ``data, codomain, domain``.
    virtual std::tuple<DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> partial_trace(
      SymmetricTensorCPtr tensor,
      std::vector<std::pair<int64, int64>> pairs,
      std::vector<std::optional<int64>> levels) = 0;

    /// Permute legs on the tensors.
    virtual DataPtr permute_legs(TensorCPtr a,
                                 std::vector<int64> codomain_idcs,
                                 std::vector<int64> domain_idcs,
                                 TensorProduct::Ptr new_codomain,
                                 TensorProduct::Ptr new_domain,
                                 bool mixes_codomain_domain,
                                 std::vector<std::optional<int64>> levels,
                                 std::vector<std::optional<bool>> bend_right) = 0;

    /// Perform a QR decomposition.
    virtual std::tuple<DataPtr, DataPtr> qr(SymmetricTensorCPtr a,
                                            TensorProduct::Ptr new_co_domain) = 0;

    /// Reduce a diagonal tensor to a single number.
    virtual BlockBackend::Scalar reduce_DiagonalTensor(DiagonalTensorCPtr tensor,
                                                       py::function block_func,
                                                       py::function func) = 0;

    /// Scale axis ``leg`` of ``a`` with ``b``.
    virtual DataPtr scale_axis(TensorCPtr a, DiagonalTensorCPtr b, int64 leg) = 0;

    /// Split (multiple) product space legs.
    virtual DataPtr split_legs(TensorCPtr a,
                               std::vector<int64> leg_idcs,
                               TensorProduct::Ptr new_codomain,
                               TensorProduct::Ptr new_domain) = 0;

    /// Assume the legs at given indices are trivial and get rid of them.
    virtual DataPtr squeeze_legs(TensorCPtr a, std::vector<int64> idcs) = 0;

    virtual bool supports_symmetry(Symmetry::Ptr symmetry) = 0;

    virtual std::tuple<DataPtr, DataPtr, DataPtr> svd(SymmetricTensorCPtr a,
                                                      TensorProduct::Ptr new_co_domain,
                                                      std::optional<std::string> algorithm) = 0;

    /// TODO clearly define what this should do in tensors.py first!
    virtual py::object state_tensor_product(BlockBackend::BlockPtr state1,
                                            BlockBackend::BlockPtr state2,
                                            LegPipe::Ptr pipe) = 0;

    virtual DataPtr to_block_backend(DataPtr data,
                                     std::shared_ptr<BlockBackend> block_backend,
                                     std::optional<Dtype> dtype = std::nullopt,
                                     std::optional<std::string> device = std::nullopt) = 0;

    /// Forget about symmetry structure and convert to a single block.
    virtual BlockBackend::BlockPtr to_dense_block(TensorCPtr a) = 0;

    /// Single-leg tensor to the *part of* the coefficients in the trivial sector.
    virtual BlockBackend::BlockPtr to_dense_block_trivial_sector(TensorCPtr tensor) = 0;

    /// Cast to given dtype. No copy if already has dtype.
    virtual DataPtr to_dtype(TensorCPtr a, Dtype dtype) = 0;

    virtual BlockBackend::Scalar trace_full(SymmetricTensorCPtr a,
                                            std::vector<int64> idcs1,
                                            std::vector<int64> idcs2) = 0;

    /// Implementation of :func:`cyten.tensors.truncate_singular_values`.
    virtual std::tuple<DataPtr, ElementarySpace::Ptr, float64, float64> truncate_singular_values(
      DiagonalTensorCPtr S,
      std::optional<int64> chi_max,
      int64 chi_min,
      float64 degeneracy_tol,
      float64 trunc_cut,
      std::optional<float64> svd_min,
      bool minimize_error = true) = 0;

    /// Helper function for :meth:`truncate_singular_values`.
    std::tuple<py::array, float64, float64> _truncate_singular_values_selection(
      py::array S,
      py::object qdims,
      std::optional<int64> chi_max,
      int64 chi_min,
      float64 degeneracy_tol,
      float64 trunc_cut,
      std::optional<float64> svd_min,
      bool minimize_error = true);

    /// Data for a zero tensor.
    virtual DataPtr zero_data(TensorProduct::Ptr codomain,
                              TensorProduct::Ptr domain,
                              Dtype dtype,
                              std::string device,
                              bool all_blocks = false) = 0;

    virtual DataPtr zero_diagonal_data(TensorProduct::Ptr co_domain,
                                       Dtype dtype,
                                       std::string device) = 0;

    virtual DataPtr zero_mask_data(Space::Ptr large_leg, std::string device) = 0;

    /// If the Tensor is comprised of real numbers.
    ///
    /// Complex numbers with small or zero imaginary part still cause a `False` return.
    virtual bool is_real(TensorCPtr a);

    virtual void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string subpath);

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string subpath);
};

/// The conventional order of legs: ``[*codomain.factors, *reversed(domain.factors)]``.
///
/// Factors are stored as ``py::object`` on :class:`TensorProduct` (same as the C++ spaces API).
std::vector<py::object> conventional_leg_order(TensorProduct::Ptr codomain,
                                               TensorProduct::Ptr domain);

/// Overload accepting a tensor-like object with ``.codomain`` / ``.domain`` attributes.
std::vector<py::object> conventional_leg_order(py::object tensor_or_codomain,
                                               py::object domain = py::none());

std::vector<py::object> conventional_leg_order(TensorCPtr tensor);

/// If the given objects have the same backend, return it. Raise otherwise.
TensorBackend::Ptr get_same_backend(const std::vector<py::object>& objs,
                                    std::string error_msg = "Incompatible backends.");

TensorBackend::Ptr get_same_backend(const std::vector<TensorCPtr>& objs,
                                    std::string error_msg = "Incompatible backends.");

} // namespace cyten
