#pragma once

#include <complex>
#include <cyten/block_backend/block.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/block_backend/scalar.h>
#include <cyten/cyten.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cyten {

// CHECKME: the following was appended by .cursor/skills/pybind11-codegen/pybind11_codegen.py
// gen_cpp_declaration --py-name BlockBackend --header-file include/cyten/block_backend_draft.h
/// Abstract base class that defines the operation on dense blocks.
class BlockBackend
{
  public:
    std::string default_device;
    std::vector<std::string> svd_algorithms; // first is default

  public:
    explicit BlockBackend(std::string default_device);
    virtual ~BlockBackend() = default;

    /// Name of the backend class for __repr__ / __str__ (e.g. "NumpyBlockBackend").
    virtual std::string get_backend_name() const;

    /// The absolute value of a complex number, elementwise.
    virtual BlockPtr abs(BlockCPtr const& a) = 0;
    /// Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block
    BlockPtr apply_basis_perm(BlockCPtr const& block,
                              std::vector<py::object> const& legs,
                              bool inv = false);
    /// Apply permutations to every axis of a dense block
    virtual BlockPtr apply_leg_permutations(BlockCPtr const& block,
                                            std::vector<py::array_t<cyten_int>> const& perms) = 0;
    /// Convert objects to blocks.
    virtual BlockPtr as_block(py::object a,
                              std::optional<Dtype> dtype = std::nullopt,
                              std::optional<std::string> device = std::nullopt) = 0;
    /// Convert input string to unambiguous device name.
    virtual std::string as_device(std::optional<std::string> device) = 0;
    /// Return the indices (one per axis) of the largest entry (by magnitude) of the block
    virtual std::vector<cyten_int> abs_argmax(BlockCPtr const& block) = 0;
    virtual BlockPtr add_axis(BlockCPtr const& a, int pos) = 0;
    /// Require a boolean block. If all of its entries are True
    virtual bool block_all(BlockCPtr const& a) = 0;
    virtual bool allclose(BlockCPtr const& a,
                          BlockCPtr const& b,
                          double rtol = 1e-5,      // NOLINT(readability-magic-numbers)
                          double atol = 1e-8) = 0; // NOLINT(readability-magic-numbers)
    /// The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise.
    virtual BlockPtr angle(BlockCPtr const& a) = 0;
    /// Require a boolean block. If any of its entries are True
    virtual bool block_any(BlockCPtr const& a) = 0;
    /// Apply a mask (1D boolean block) to a block, slicing/projecting that axis
    virtual BlockPtr apply_mask(BlockCPtr const& block, BlockCPtr const& mask, int ax) = 0;
    /// Return the permutation that would sort a block along one axis.
    BlockPtr argsort(BlockCPtr const& block,
                     std::optional<std::string> sort = std::nullopt,
                     int axis = 0);
    /// Like :meth:`block_argsort` but can assume real valued block, and sort ascending
    virtual BlockPtr _argsort(BlockCPtr const& block, int axis) = 0;
    /// Combine each group of legs in `leg_idcs_combine` into a single leg.
    BlockPtr combine_legs(BlockCPtr const& a,
                          std::vector<std::vector<int>> const& leg_idcs_combine,
                          std::vector<bool> const& cstyles);
    BlockPtr combine_legs(BlockCPtr const& a,
                          std::vector<std::vector<int>> const& leg_idcs_combine,
                          bool cstyles = true);
    /// Complex conjugate of a block
    virtual BlockPtr conj(BlockCPtr const& a) = 0;
    /// Create a new, independent block with the same data
    virtual BlockPtr copy_block(BlockCPtr const& a,
                                std::optional<std::string> device = std::nullopt) = 0;
    /// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
    virtual BlockPtr cutoff_inverse(BlockCPtr const& a, double cutoff) = 0;
    /// Permute axes to reverse order and elementwise conj.
    BlockPtr dagger(BlockCPtr const& a);
    virtual Dtype get_dtype(BlockCPtr const& a) = 0;
    /// Eigenvalue decomposition of a 2D hermitian block.
    virtual std::tuple<BlockPtr, BlockPtr> eigh(
      BlockCPtr const& block,
      std::optional<std::string> sort = std::nullopt) = 0;
    /// Eigenvalues of a 2D hermitian block.
    virtual BlockPtr eigvalsh(BlockCPtr const& block,
                              std::optional<std::string> sort = std::nullopt) = 0;
    virtual BlockPtr enlarge_leg(BlockCPtr const& block, BlockCPtr const& mask, int axis) = 0;
    /// The *elementwise* exponential.
    virtual BlockPtr exp(BlockCPtr const& a) = 0;
    /// Return a 2D square block that has the 1D ``diag`` on the diagonal
    virtual BlockPtr block_from_diagonal(BlockCPtr const& diag) = 0;
    /// Convert a mask to a full block.
    virtual BlockPtr block_from_mask(BlockCPtr const& mask, Dtype dtype) = 0;
    virtual BlockPtr block_from_numpy(py::array const& a,
                                      std::optional<Dtype> dtype = std::nullopt,
                                      std::optional<std::string> device = std::nullopt) = 0;
    virtual std::string get_device(BlockCPtr const& a) = 0;
    /// Get the diagonal of a 2D block as a 1D block
    virtual BlockPtr get_diagonal(BlockCPtr const& a,
                                  std::optional<double> tol = std::nullopt) = 0;
    /// The imaginary part of a complex number, elementwise.
    virtual BlockPtr imag(BlockCPtr const& a) = 0;
    /// Dense block version of tensors.inner.
    std::complex<cyten_float> inner(BlockCPtr const& a, BlockCPtr const& b, bool do_dagger);
    /// If the block is comprised of real numbers.
    bool is_real(BlockCPtr const& a);
    /// Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python
    /// float or complex
    virtual py::object item(BlockCPtr const& a) = 0;
    /// The kronecker product.
    virtual BlockPtr kron(BlockCPtr const& a, BlockCPtr const& b) = 0;
    virtual BlockPtr linear_combination(Scalar a_coef,
                                        BlockCPtr const& v,
                                        Scalar b_coef,
                                        BlockCPtr const& w) = 0;
    /// The *elementwise* natural logarithm.
    virtual BlockPtr log(BlockCPtr const& a) = 0;
    virtual cyten_float max(BlockCPtr const& a) = 0;
    virtual cyten_float max_abs(BlockCPtr const& a) = 0;
    virtual cyten_float min(BlockCPtr const& a) = 0;
    virtual BlockPtr mul(py::object a, BlockCPtr const& b) = 0;
    /// The p-norm vector-norm of a block.
    virtual cyten_float norm(BlockCPtr const& a,
                             double order = 2,
                             std::optional<int> axis = std::nullopt) = 0;
    /// Outer product of blocks.
    virtual BlockPtr outer(BlockCPtr const& a, BlockCPtr const& b) = 0;
    virtual BlockPtr permute_axes(BlockCPtr const& a, std::vector<int> const& permutation) = 0;
    /// For a matrix `a` with two combined multi-indices, permute the sub-indices.
    BlockPtr permute_combined_matrix(BlockCPtr const& block,
                                     std::vector<cyten_int> const& dims1,
                                     std::vector<int> const& idcs1,
                                     std::vector<cyten_int> const& dims2,
                                     std::vector<int> const& idcs2);
    /// For a matrix `a` with a single combined multi-index, permute sub-indices.
    BlockPtr permute_combined_idx(BlockCPtr const& block,
                                  int axis,
                                  std::vector<cyten_int> const& dims,
                                  std::vector<int> const& idcs);
    virtual BlockPtr random_normal(std::vector<cyten_int> const& dims,
                                   Dtype dtype,
                                   double sigma,
                                   std::optional<std::string> device = std::nullopt) = 0;
    virtual BlockPtr random_uniform(std::vector<cyten_int> const& dims,
                                    Dtype dtype,
                                    std::optional<std::string> device = std::nullopt) = 0;
    /// The real part of a complex number, elementwise.
    virtual BlockPtr real(BlockCPtr const& a) = 0;
    /// If a block is close to its real part, return the real part.
    virtual BlockPtr real_if_close(BlockCPtr const& a, double tol) = 0;
    /// Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.
    virtual BlockPtr tile(BlockCPtr const& a, int repeats) = 0;
    virtual std::vector<std::string> _block_repr_lines(BlockCPtr const& a,
                                                       std::string const& indent,
                                                       int max_width,
                                                       int max_lines) = 0;
    virtual BlockPtr reshape(BlockCPtr const& a, std::vector<cyten_int> const& shape) = 0;
    /// Multiply block with the factors (a 1D block), along a given axis.
    virtual BlockPtr scale_axis(BlockCPtr const& block, BlockCPtr const& factors, int axis) = 0;
    virtual std::vector<cyten_int> get_shape(BlockCPtr const& a) = 0;
    /// Split legs into groups of legs with specified dimensions.
    BlockPtr split_legs(BlockCPtr const& a,
                        std::vector<int> const& idcs,
                        std::vector<std::vector<cyten_int>> const& dims,
                        std::vector<bool> const& cstyles);
    BlockPtr split_legs(BlockCPtr const& a,
                        std::vector<int> const& idcs,
                        std::vector<std::vector<cyten_int>> const& dims,
                        bool cstyles = true);
    /// The elementwise square root
    virtual BlockPtr sqrt(BlockCPtr const& a) = 0;
    virtual BlockPtr squeeze_axes(BlockCPtr const& a, std::vector<int> const& idcs) = 0;
    /// Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.
    virtual BlockPtr stable_log(BlockCPtr const& block, double cutoff) = 0;
    /// The sum over a single axis.
    virtual BlockPtr sum(BlockCPtr const& a, int ax) = 0;
    /// The sum of all entries of the block.
    virtual std::complex<cyten_float> sum_all(BlockCPtr const& a) = 0;
    virtual BlockPtr multiply_blocks(BlockCPtr const& a, BlockCPtr const& b) = 0; // elementwise
    virtual BlockPtr tdot(BlockCPtr const& a,
                          BlockCPtr const& b,
                          std::vector<int> const& idcs_a,
                          std::vector<int> const& idcs_b) = 0;
    /// Version of ``tensors.outer`` on blocks.
    BlockPtr tensor_outer(BlockCPtr const& a, BlockCPtr const& b, int K);
    virtual BlockPtr to_dtype(BlockCPtr const& a, Dtype dtype) = 0;
    virtual py::object to_numpy(BlockCPtr const& a,
                                std::optional<py::object> numpy_dtype = std::nullopt) = 0;
    virtual std::complex<cyten_float> trace_full(BlockCPtr const& a) = 0;
    virtual BlockPtr trace_partial(BlockCPtr const& a,
                                   std::vector<int> const& idcs1,
                                   std::vector<int> const& idcs2,
                                   std::vector<int> const& remaining_idcs) = 0;
    /// The identity matrix, reshaped to a block.
    BlockPtr eye_block(std::vector<cyten_int> const& legs,
                       Dtype dtype,
                       std::optional<std::string> device = std::nullopt);
    /// The ``dim x dim`` identity matrix
    virtual BlockPtr eye_matrix(int dim,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    virtual py::object get_block_element(BlockCPtr const& a,
                                         std::vector<cyten_int> const& idcs) = 0;
    /// Get an element of a mask.
    virtual bool get_block_mask_element(BlockCPtr const& a,
                                        cyten_int large_leg_idx,
                                        cyten_int small_leg_idx,
                                        cyten_int sum_block = 0) = 0;
    /// As in numpy.dot, both a and b might be matrix or vector.
    virtual BlockPtr matrix_dot(BlockCPtr const& a, BlockCPtr const& b) = 0;
    virtual BlockPtr matrix_exp(BlockCPtr const& matrix) = 0;
    virtual BlockPtr matrix_log(BlockCPtr const& matrix) = 0;
    std::tuple<BlockPtr, BlockPtr> matrix_lq(BlockCPtr const& a, bool full);
    /// QR decomposition of a 2D block
    virtual std::tuple<BlockPtr, BlockPtr> matrix_qr(BlockCPtr const& a, bool full) = 0;
    /// Internal version of :meth:`matrix_svd`, to be implemented by subclasses.
    virtual std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      BlockCPtr const& a,
      std::optional<std::string> algorithm = std::nullopt) = 0;
    virtual BlockPtr ones_block(std::vector<cyten_int> const& shape,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    /// Wait for asynchronous processes (if any) to finish
    void synchronize();
    /// Assert block type and optional shape/dtype/device. Throws std::runtime_error if any check
    /// fails.
    void test_block_sanity(BlockCPtr const& block,
                           std::optional<std::vector<cyten_int>> expect_shape = std::nullopt,
                           std::optional<Dtype> expect_dtype = std::nullopt,
                           std::optional<std::string> expect_device = std::nullopt);
    virtual BlockPtr zeros(std::vector<cyten_int> const& shape,
                           Dtype dtype,
                           std::optional<std::string> device = std::nullopt) = 0;

    /// Save backend state to HDF5.
    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath);
    /// Load backend from HDF5.
    static std::shared_ptr<BlockBackend> from_hdf5(py::object hdf5_loader,
                                                   py::object h5gr,
                                                   std::string const& subpath);

  protected:
    /// Return true if block is of the backend's block type. Used by test_block_sanity.
    virtual bool is_correct_block_type(BlockCPtr const& block) const = 0;
};

} // namespace cyten
