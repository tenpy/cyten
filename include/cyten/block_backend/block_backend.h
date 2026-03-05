#pragma once

#include <complex>
#include <cyten/block_backend/dtypes.h>
#include <cyten/block_backend/scalar.h>
#include <cyten/cyten.h>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

// CHECKME: the following was appended by .cursor/skills/pybind11-codegen/pybind11_codegen.py
// gen_cpp_declaration --py-name BlockBackend --header-file include/cyten/block_backend_draft.h
/// Abstract base class that defines the operation on dense blocks.
class BlockBackend
{
  public:
    /// Abstract base class for dense blocks. Subclassed per backend (e.g.
    /// NumpyBlockBackend::Block). Access to elements should be done exclusively through the
    /// BlockBackend.
    class Block
    {
      public:
        // subclasses should have constructor from numpy array
        // explicit Block(std::shared_ptr<py::array> arr);

        virtual ~Block() = default;

        /// convert to numpy array, might be copy or (immutable) view
        virtual py::array to_numpy() const = 0;

        /// Shape of the block (one size per axis).
        virtual std::vector<int64> shape() const = 0;

        /// Dtype of the block entries.
        virtual Dtype dtype() const = 0;

        /// Device string (e.g. "cpu", "cuda:0").
        virtual std::string device() const = 0;
    };
    using BlockPtr = std::shared_ptr<Block>;
    using BlockCPtr = std::shared_ptr<const Block>;

  public:
    std::string default_device;
    std::vector<std::string> svd_algorithms; // first is default

  public:
    explicit BlockBackend(std::string default_device);
    virtual ~BlockBackend() = default;

    /// Name of the backend class for __repr__ / __str__ (e.g. "NumpyBlockBackend").
    virtual std::string get_backend_name() const;

    /// The absolute value of a complex number, elementwise.
    virtual BlockPtr abs(const BlockCPtr& a) = 0;
    /// Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block
    BlockPtr apply_basis_perm(const BlockCPtr& block,
                              const std::vector<py::object>& legs,
                              bool inv = false);
    /// Apply permutations to every axis of a dense block
    virtual BlockPtr apply_leg_permutations(const BlockCPtr& block,
                                            const std::vector<py::array_t<int64>>& perms) = 0;
    /// Convert objects to blocks.
    virtual BlockPtr as_block(py::object a,
                              std::optional<Dtype> dtype = std::nullopt,
                              std::optional<std::string> device = std::nullopt) = 0;
    /// Convert input string to unambiguous device name.
    virtual std::string as_device(std::optional<std::string> device) = 0;
    /// Return the indices (one per axis) of the largest entry (by magnitude) of the block
    virtual std::vector<int64> abs_argmax(const BlockCPtr& block) = 0;
    virtual BlockPtr add_axis(const BlockCPtr& a, int64 pos) = 0;
    /// Require a boolean block. If all of its entries are True
    virtual bool block_all(const BlockCPtr& a) = 0;
    virtual bool allclose(const BlockCPtr& a,
                          const BlockCPtr& b,
                          float64 rtol = 1e-5,      // NOLINT(readability-magic-numbers)
                          float64 atol = 1e-8) = 0; // NOLINT(readability-magic-numbers)
    /// The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise.
    virtual BlockPtr angle(const BlockCPtr& a) = 0;
    /// Require a boolean block. If any of its entries are True
    virtual bool block_any(const BlockCPtr& a) = 0;
    /// Apply a mask (1D boolean block) to a block, slicing/projecting that axis
    virtual BlockPtr apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax) = 0;
    /// Return the permutation that would sort a block along one axis.
    BlockPtr argsort(const BlockCPtr& block,
                     std::optional<std::string> sort = std::nullopt,
                     int64 axis = 0);
    /// Like :meth:`block_argsort` but can assume real valued block, and sort ascending
    virtual BlockPtr _argsort(const BlockCPtr& block, int64 axis) = 0;
    /// Combine each group of legs in `leg_idcs_combine` into a single leg.
    BlockPtr combine_legs(const BlockCPtr& a,
                          const std::vector<std::vector<int64>>& leg_idcs_combine,
                          const std::vector<bool>& cstyles);
    BlockPtr combine_legs(const BlockCPtr& a,
                          const std::vector<std::vector<int64>>& leg_idcs_combine,
                          bool cstyles = true);
    /// Complex conjugate of a block
    virtual BlockPtr conj(const BlockCPtr& a) = 0;
    /// Create a new, independent block with the same data
    virtual BlockPtr copy_block(const BlockCPtr& a,
                                std::optional<std::string> device = std::nullopt) = 0;
    /// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
    virtual BlockPtr cutoff_inverse(const BlockCPtr& a, float64 cutoff) = 0;
    /// Permute axes to reverse order and elementwise conj.
    BlockPtr dagger(const BlockCPtr& a);
    virtual Dtype get_dtype(const BlockCPtr& a) = 0;
    /// Eigenvalue decomposition of a 2D hermitian block.
    virtual std::tuple<BlockPtr, BlockPtr> eigh(
      const BlockCPtr& block,
      std::optional<std::string> sort = std::nullopt) = 0;
    /// Eigenvalues of a 2D hermitian block.
    virtual BlockPtr eigvalsh(const BlockCPtr& block,
                              std::optional<std::string> sort = std::nullopt) = 0;
    virtual BlockPtr enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis) = 0;
    /// The *elementwise* exponential.
    virtual BlockPtr exp(const BlockCPtr& a) = 0;
    /// Return a 2D square block that has the 1D ``diag`` on the diagonal
    virtual BlockPtr block_from_diagonal(const BlockCPtr& diag) = 0;
    /// Convert a mask to a full block.
    virtual BlockPtr block_from_mask(const BlockCPtr& mask, Dtype dtype) = 0;
    virtual BlockPtr block_from_numpy(const py::array& a,
                                      std::optional<Dtype> dtype = std::nullopt,
                                      std::optional<std::string> device = std::nullopt) = 0;
    virtual std::string get_device(const BlockCPtr& a) = 0;
    /// Get the diagonal of a 2D block as a 1D block
    virtual BlockPtr get_diagonal(const BlockCPtr& a,
                                  std::optional<float64> tol = std::nullopt) = 0;
    /// The imaginary part of a complex number, elementwise.
    virtual BlockPtr imag(const BlockCPtr& a) = 0;
    /// Dense block version of tensors.inner.
    complex128 inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger);
    /// If the block is comprised of real numbers.
    bool is_real(const BlockCPtr& a);
    /// Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python
    /// float or complex
    virtual py::object item(const BlockCPtr& a) = 0;
    /// The kronecker product.
    virtual BlockPtr kron(const BlockCPtr& a, const BlockCPtr& b) = 0;
    virtual BlockPtr linear_combination(Scalar a_coef,
                                        const BlockCPtr& v,
                                        Scalar b_coef,
                                        const BlockCPtr& w) = 0;
    /// The *elementwise* natural logarithm.
    virtual BlockPtr log(const BlockCPtr& a) = 0;
    virtual float64 max(const BlockCPtr& a) = 0;
    virtual float64 max_abs(const BlockCPtr& a) = 0;
    virtual float64 min(const BlockCPtr& a) = 0;
    virtual BlockPtr mul(py::object a, const BlockCPtr& b) = 0;
    /// The p-norm vector-norm of a block.
    virtual float64 norm(const BlockCPtr& a,
                         float64 order = 2,
                         std::optional<int64> axis = std::nullopt) = 0;
    /// Outer product of blocks.
    virtual BlockPtr outer(const BlockCPtr& a, const BlockCPtr& b) = 0;
    virtual BlockPtr permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation) = 0;
    /// For a matrix `a` with two combined multi-indices, permute the sub-indices.
    BlockPtr permute_combined_matrix(const BlockCPtr& block,
                                     const std::vector<int64>& dims1,
                                     const std::vector<int64>& idcs1,
                                     const std::vector<int64>& dims2,
                                     const std::vector<int64>& idcs2);
    /// For a matrix `a` with a single combined multi-index, permute sub-indices.
    BlockPtr permute_combined_idx(const BlockCPtr& block,
                                  int64 axis,
                                  const std::vector<int64>& dims,
                                  const std::vector<int64>& idcs);
    virtual BlockPtr random_normal(const std::vector<int64>& dims,
                                   Dtype dtype,
                                   float64 sigma,
                                   std::optional<std::string> device = std::nullopt) = 0;
    virtual BlockPtr random_uniform(const std::vector<int64>& dims,
                                    Dtype dtype,
                                    std::optional<std::string> device = std::nullopt) = 0;
    /// The real part of a complex number, elementwise.
    virtual BlockPtr real(const BlockCPtr& a) = 0;
    /// If a block is close to its real part, return the real part.
    virtual BlockPtr real_if_close(const BlockCPtr& a, float64 tol) = 0;
    /// Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.
    virtual BlockPtr tile(const BlockCPtr& a, int64 repeats) = 0;
    virtual std::vector<std::string> _block_repr_lines(const BlockCPtr& a,
                                                       const std::string& indent,
                                                       int64 max_width,
                                                       int64 max_lines) = 0;
    virtual BlockPtr reshape(const BlockCPtr& a, const std::vector<int64>& shape) = 0;
    /// Multiply block with the factors (a 1D block), along a given axis.
    virtual BlockPtr scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis) = 0;
    virtual std::vector<int64> get_shape(const BlockCPtr& a) = 0;
    /// Split legs into groups of legs with specified dimensions.
    BlockPtr split_legs(const BlockCPtr& a,
                        const std::vector<int64>& idcs,
                        const std::vector<std::vector<int64>>& dims,
                        const std::vector<bool>& cstyles);
    BlockPtr split_legs(const BlockCPtr& a,
                        const std::vector<int64>& idcs,
                        const std::vector<std::vector<int64>>& dims,
                        bool cstyles = true);
    /// The elementwise square root
    virtual BlockPtr sqrt(const BlockCPtr& a) = 0;
    virtual BlockPtr squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs) = 0;
    /// Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.
    virtual BlockPtr stable_log(const BlockCPtr& block, float64 cutoff) = 0;
    /// The sum over a single axis.
    virtual BlockPtr sum(const BlockCPtr& a, int64 ax) = 0;
    /// The sum of all entries of the block.
    virtual complex128 sum_all(const BlockCPtr& a) = 0;
    virtual BlockPtr multiply_blocks(const BlockCPtr& a, const BlockCPtr& b) = 0; // elementwise
    virtual BlockPtr tdot(const BlockCPtr& a,
                          const BlockCPtr& b,
                          const std::vector<int64>& idcs_a,
                          const std::vector<int64>& idcs_b) = 0;
    /// Version of ``tensors.outer`` on blocks.
    BlockPtr tensor_outer(const BlockCPtr& a, const BlockCPtr& b, int64 K);
    virtual BlockPtr to_dtype(const BlockCPtr& a, Dtype dtype) = 0;
    virtual py::object to_numpy(const BlockCPtr& a,
                                std::optional<py::object> numpy_dtype = std::nullopt) = 0;
    virtual complex128 trace_full(const BlockCPtr& a) = 0;
    virtual BlockPtr trace_partial(const BlockCPtr& a,
                                   const std::vector<int64>& idcs1,
                                   const std::vector<int64>& idcs2,
                                   const std::vector<int64>& remaining_idcs) = 0;
    /// The identity matrix, reshaped to a block.
    BlockPtr eye_block(const std::vector<int64>& legs,
                       Dtype dtype,
                       std::optional<std::string> device = std::nullopt);
    /// The ``dim x dim`` identity matrix
    virtual BlockPtr eye_matrix(int64 dim,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    virtual py::object get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs) = 0;
    /// Get an element of a mask.
    virtual bool get_block_mask_element(const BlockCPtr& a,
                                        int64 large_leg_idx,
                                        int64 small_leg_idx,
                                        int64 sum_block = 0) = 0;
    /// As in numpy.dot, both a and b might be matrix or vector.
    virtual BlockPtr matrix_dot(const BlockCPtr& a, const BlockCPtr& b) = 0;
    virtual BlockPtr matrix_exp(const BlockCPtr& matrix) = 0;
    virtual BlockPtr matrix_log(const BlockCPtr& matrix) = 0;
    std::tuple<BlockPtr, BlockPtr> matrix_lq(const BlockCPtr& a, bool full);
    /// QR decomposition of a 2D block
    virtual std::tuple<BlockPtr, BlockPtr> matrix_qr(const BlockCPtr& a, bool full) = 0;
    /// Internal version of :meth:`matrix_svd`, to be implemented by subclasses.
    virtual std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      const BlockCPtr& a,
      std::optional<std::string> algorithm = std::nullopt) = 0;
    virtual BlockPtr ones_block(const std::vector<int64>& shape,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    /// Wait for asynchronous processes (if any) to finish
    void synchronize();
    /// Assert block type and optional shape/dtype/device. Throws std::runtime_error if any check
    /// fails.
    void test_block_sanity(const BlockCPtr& block,
                           std::optional<std::vector<int64>> expect_shape = std::nullopt,
                           std::optional<Dtype> expect_dtype = std::nullopt,
                           std::optional<std::string> expect_device = std::nullopt);
    virtual BlockPtr zeros(const std::vector<int64>& shape,
                           Dtype dtype,
                           std::optional<std::string> device = std::nullopt) = 0;

    /// Save backend state to HDF5.
    void save_hdf5(py::object hdf5_saver, py::object h5gr, const std::string& subpath);
    /// Load backend from HDF5.
    static std::shared_ptr<BlockBackend> from_hdf5(py::object hdf5_loader,
                                                   py::object h5gr,
                                                   const std::string& subpath);

  protected:
    /// Return true if block is of the backend's block type. Used by test_block_sanity.
    virtual bool is_correct_block_type(const BlockCPtr& block) const = 0;
};

using BlockPtr = std::shared_ptr<BlockBackend::Block>;
using BlockCPtr = std::shared_ptr<const BlockBackend::Block>;

} // namespace cyten
