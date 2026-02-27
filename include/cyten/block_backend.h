#pragma once

#include <complex>
#include <cyten/block.h>
#include <cyten/cyten.h>
#include <cyten/dtypes.h>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

// CHECKME: the following was appended by .cursor/skills/pybind11-codegen/pybind11_codegen.py
// gen_cpp_declaration --py-name BlockBackend --header-file include/cyten/block_backend_draft.h
/// Abstract base class that defines the operation on dense blocks.
class BlockBackend
{
  public:
    static TYPEOF_BlockCls BlockCls;

  public:
    BlockBackend(std::string default_device);
    virtual ~BlockBackend() = default;
    str __repr__();
    str __str__();
    /// The absolute value of a complex number, elementwise.
    virtual Block abs(Block a) = 0;
    /// Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block
    Block apply_basis_perm(Block block, list_Space_ legs, bool inv = false);
    /// Apply permutations to every axis of a dense block
    Block apply_leg_permutations(Block block, list_np_ndarray_ perms);
    /// Convert objects to blocks.
    virtual Block_tuple_Block_Dtype_ as_block(TYPEOF_a a,
                                              Dtype dtype = py::none(),
                                              bool return_dtype = false,
                                              std::string device = py::none()) {
        pure
    };
    /// Convert input string to unambiguous device name.
    virtual std::string as_device(std::string device) = 0;
    /// Return the indices (one per axis) of the largest entry (by magnitude) of the block
    virtual list_int_ abs_argmax(Block block) = 0;
    virtual Block add_axis(Block a, cyten_int pos) = 0;
    /// Require a boolean block. If all of its entries are True
    virtual bool block_all(TYPEOF_a a) = 0;
    virtual bool allclose(Block a,
                          Block b,
                          cyten_float rtol = 1e-05,
                          cyten_float atol = 1e-08) = 0;
    /// The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise.
    virtual Block angle(Block a) = 0;
    /// Require a boolean block. If any of its entries are True
    virtual bool block_any(TYPEOF_a a) = 0;
    /// Apply a mask (1D boolean block) to a block, slicing/projecting that axis
    Block apply_mask(Block block, Block mask, cyten_int ax);
    /// Return the permutation that would sort a block along one axis.
    Block argsort(Block block, std::string sort = py::none(), cyten_int axis = 0);
    /// Like :meth:`block_argsort` but can assume real valued block, and sort ascending
    virtual Block _argsort(Block block, cyten_int axis) = 0;
    /// Combine each group of legs in `leg_idcs_combine` into a single leg.
    Block combine_legs(Block a, list_list_int__ leg_idcs_combine, bool_list_bool_ cstyles = true);
    /// Complex conjugate of a block
    virtual Block conj(Block a) = 0;
    /// Create a new, independent block with the same data
    virtual Block copy_block(Block a, std::string device = py::none()) = 0;
    /// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
    Block cutoff_inverse(Block a, cyten_float cutoff);
    /// Permute axes to reverse order and elementwise conj.
    Block dagger(Block a);
    virtual Dtype get_dtype(Block a) = 0;
    /// Eigenvalue decomposition of a 2D hermitian block.
    virtual tuple_Block_Block_ eigh(Block block, std::string sort = py::none()) = 0;
    /// Eigenvalues of a 2D hermitian block.
    virtual Block eigvalsh(Block block, std::string sort = py::none()) = 0;
    virtual Block enlarge_leg(Block block, Block mask, cyten_int axis);
    /// The *elementwise* exponential.
    virtual Block exp(Block a) = 0;
    /// Return a 2D square block that has the 1D ``diag`` on the diagonal
    virtual Block block_from_diagonal(Block diag) = 0;
    /// Convert a mask to a full block.
    virtual Block block_from_mask(Block mask, Dtype dtype) = 0;
    virtual Block block_from_numpy(np_NDArray a,
                                   Dtype dtype = py::none(),
                                   std::string device = py::none()) = 0;
    virtual std::string get_device(Block a) = 0;
    /// Get the diagonal of a 2D block as a 1D block
    virtual Block get_diagonal(Block a, cyten_float tol) = 0;
    /// The imaginary part of a complex number, elementwise.
    virtual Block imag(Block a) = 0;
    /// Dense block version of tensors.inner.
    virtual cyten_float inner(Block a, Block b, bool do_dagger);
    /// If the block is comprised of real numbers.
    bool is_real(Block a);
    /// Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python
    /// float or complex
    virtual cyten_float item(Block a) = 0;
    /// The kronecker product.
    virtual Block kron(Block a, Block b) = 0;
    Block linear_combination(TYPEOF_a a, Block v, TYPEOF_b b, Block w);
    /// The *elementwise* natural logarithm.
    virtual Block log(Block a) = 0;
    virtual cyten_float max(Block a) = 0;
    virtual cyten_float max_abs(Block a) = 0;
    virtual cyten_float min(Block a) = 0;
    Block mul(cyten_float a, Block b);
    /// The p-norm vector-norm of a block.
    virtual cyten_float norm(Block a, cyten_int order = 2, cyten_int axis = py::none()) = 0;
    /// Outer product of blocks.
    virtual Block outer(Block a, Block b) = 0;
    virtual Block permute_axes(Block a, list_int_ permutation) = 0;
    /// For a matrix `a` with two combined multi-indices, permute the sub-indices.
    Block permute_combined_matrix(Block block,
                                  Sequence_int_ dims1,
                                  Sequence_int_ idcs1,
                                  Sequence_int_ dims2,
                                  Sequence_int_ idcs2) {
        pure
    };
    /// For a matrix `a` with a single combined multi-index, permute sub-indices.
    Block permute_combined_idx(Block block,
                               cyten_int axis,
                               Sequence_int_ dims,
                               Sequence_int_ idcs);
    virtual Block random_normal(list_int_ dims,
                                Dtype dtype,
                                cyten_float sigma,
                                std::string device = py::none()) = 0;
    virtual Block random_uniform(list_int_ dims, Dtype dtype, std::string device = py::none()) = 0;
    /// The real part of a complex number, elementwise.
    virtual Block real(Block a) = 0;
    /// If a block is close to its real part, return the real part.
    virtual Block real_if_close(Block a, cyten_float tol) = 0;
    /// Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.
    virtual Block tile(Block a, cyten_int repeats) = 0;
    virtual list_str_ _block_repr_lines(Block a,
                                        std::string indent,
                                        cyten_int max_width,
                                        cyten_int max_lines) = 0;
    virtual Block reshape(Block a, tuple_int_ shape) = 0;
    /// Multiply block with the factors (a 1D block), along a given axis.
    Block scale_axis(Block block, Block factors, cyten_int axis);
    virtual tuple_int_ get_shape(Block a) = 0;
    /// Split legs into groups of legs with specified dimensions.
    Block split_legs(Block a,
                     list_int_ idcs,
                     list_list_int__ dims,
                     bool_list_bool_ cstyles = true);
    /// The elementwise square root
    virtual Block sqrt(Block a) = 0;
    virtual Block squeeze_axes(Block a, list_int_ idcs) = 0;
    /// Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.
    virtual Block stable_log(Block block, cyten_float cutoff) = 0;
    /// The sum over a single axis.
    virtual Block sum(Block a, cyten_int ax) = 0;
    /// The sum of all entries of the block.
    virtual cyten_float sum_all(Block a) = 0;
    virtual Block tdot(Block a, Block b, list_int_ idcs_a, list_int_ idcs_b) = 0;
    /// Version of ``tensors.outer`` on blocks.
    Block tensor_outer(Block a, Block b, cyten_int K);
    virtual Block to_dtype(Block a, Dtype dtype) = 0;
    np_NDArray to_numpy(Block a, TYPEOF_numpy_dtype numpy_dtype = py::none());
    virtual cyten_float trace_full(Block a) = 0;
    virtual Block trace_partial(Block a,
                                list_int_ idcs1,
                                list_int_ idcs2,
                                list_int_ remaining_idcs) = 0;
    /// The identity matrix, reshaped to a block.
    Block eye_block(list_int_ legs, Dtype dtype, std::string device = py::none());
    /// The ``dim x dim`` identity matrix
    virtual Block eye_matrix(cyten_int dim, Dtype dtype, std::string device = py::none()) = 0;
    virtual cyten_complex get_block_element(Block a, list_int_ idcs) = 0;
    /// Get an element of a mask.
    bool get_block_mask_element(Block a,
                                cyten_int large_leg_idx,
                                cyten_int small_leg_idx,
                                cyten_int sum_block = 0);
    /// As in numpy.dot, both a and b might be matrix or vector.
    virtual Block matrix_dot(Block a, Block b) = 0;
    virtual Block matrix_exp(Block matrix) = 0;
    virtual Block matrix_log(Block matrix) = 0;
    tuple_Block_Block_ matrix_lq(Block a, bool full);
    /// QR decomposition of a 2D block
    virtual tuple_Block_Block_ matrix_qr(Block a, bool full) = 0;
    /// Internal version of :meth:`matrix_svd`, to be implemented by subclasses.
    virtual tuple_Block_Block_Block_ matrix_svd(Block a, std::string algorithm) = 0;
    virtual Block ones_block(list_int_ shape, Dtype dtype, std::string device = py::none()) = 0;
    /// Wait for asynchronous processes (if any) to finish
    virtual TYPEOF_return synchronize();
    TYPEOF_return test_block_sanity(TYPEOF_block block,
                                    tuple_int______None expect_shape = py::none(),
                                    Dtype_None expect_dtype = py::none(),
                                    std::string expect_device = py::none()) {
        pure
    };
    virtual Block zeros(list_int_ shape, Dtype dtype, std::string device = py::none()) = 0;
    TYPEOF_return save_hdf5(TYPEOF_hdf5_saver hdf5_saver,
                            TYPEOF_h5gr h5gr,
                            TYPEOF_subpath subpath);
    TYPEOF_return from_hdf5(TYPEOF_hdf5_loader hdf5_loader,
                            TYPEOF_h5gr h5gr,
                            TYPEOF_subpath subpath);
};

} // namespace cyten
