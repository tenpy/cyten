#pragma once

#include <cyten/block.h>
#include <cyten/block_backend.h>
#include <memory>

namespace cyten {

/// Block that holds a numpy array in a py::object.
class NumpyBlock : public Block
{
  public:
    explicit NumpyBlock(py::object arr);
    std::vector<cyten_int> shape() const override;
    Dtype dtype() const override;
    std::string device() const override;
    py::object operator[](std::vector<cyten_int> const& idcs) const override;
    py::object const& array() const { return arr_; }

  private:
    py::object arr_;
};

// CHECKME: the following was appended by .cursor/skills/pybind11-codegen/pybind11_codegen.py
// gen_cpp_declaration --py-name NumpyBlockBackend --header-file
// include/cyten/numpy_block_backend.h
/// A block backend using numpy.
class NumpyBlockBackend : public BlockBackend
{
  public:
    static TYPEOF_BlockCls BlockCls;
    static TYPEOF_svd_algorithms svd_algorithms;
    static TYPEOF_cyten_dtype_map cyten_dtype_map;
    static TYPEOF_backend_dtype_map backend_dtype_map;

  public:
    NumpyBlockBackend();
    virtual ~NumpyBlockBackend() = default;
    virtual Block abs(Block a) override;
    virtual Block as_block(TYPEOF_a a,
                           Dtype dtype = py::none(),
                           bool return_dtype = false,
                           std::string device = py::none()) override;
    virtual std::string as_device(std::string device) override;
    virtual Block add_axis(Block a, cyten_int pos) override;
    virtual list_int_ abs_argmax(Block block) override;
    virtual bool block_all(TYPEOF_a a) override;
    virtual bool allclose(Block a,
                          Block b,
                          cyten_float rtol = 1e-05,
                          cyten_float atol = 1e-08) override;
    virtual Block angle(Block a) override;
    virtual bool block_any(TYPEOF_a a) override;
    virtual Block apply_mask(Block block, Block mask, cyten_int ax) override;
    virtual Block _argsort(Block block, cyten_int axis) override;
    virtual Block conj(Block a) override;
    virtual Block copy_block(Block a, std::string device = py::none()) override;
    /// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
    virtual Block cutoff_inverse(Block a, cyten_float cutoff) override;
    virtual Dtype get_dtype(Block a) override;
    virtual tuple_Block_Block_ eigh(Block block, std::string sort = py::none()) override;
    virtual Block eigvalsh(Block block, std::string sort = py::none()) override;
    virtual Block enlarge_leg(Block block, Block mask, cyten_int axis) override;
    virtual Block exp(Block a) override;
    virtual Block block_from_diagonal(Block diag) override;
    virtual Block block_from_mask(Block mask, Dtype dtype) override;
    virtual Block block_from_numpy(np_NDArray a,
                                   Dtype dtype = py::none(),
                                   std::string device = py::none()) override;
    virtual std::string get_device(Block a) override;
    virtual Block get_diagonal(Block a, cyten_float tol) override;
    virtual Block imag(Block a) override;
    virtual cyten_float inner(Block a, Block b, bool do_dagger) override;
    virtual cyten_float item(Block a) override;
    virtual Block kron(Block a, Block b) override;
    virtual Block log(Block a) override;
    virtual cyten_float max(Block a) override;
    virtual cyten_float max_abs(Block a) override;
    virtual cyten_float min(Block a) override;
    virtual cyten_float norm(Block a, cyten_int order = 2, cyten_int axis = py::none()) override;
    virtual Block outer(Block a, Block b) override;
    virtual Block permute_axes(Block a, list_int_ permutation) override;
    virtual Block random_normal(list_int_ dims,
                                Dtype dtype,
                                cyten_float sigma,
                                std::string device = py::none()) override;
    virtual Block random_uniform(list_int_ dims,
                                 Dtype dtype,
                                 std::string device = py::none()) override;
    virtual Block real(Block a) override;
    virtual Block real_if_close(Block a, cyten_float tol) override;
    virtual Block tile(Block a, cyten_int repeats, cyten_int axis = py::none()) override;
    virtual list_str_ _block_repr_lines(Block a,
                                        std::string indent,
                                        cyten_int max_width,
                                        cyten_int max_lines) override;
    virtual Block reshape(Block a, tuple_int_ shape) override;
    virtual tuple_int_ get_shape(Block a) override;
    virtual Block sqrt(Block a) override;
    virtual Block squeeze_axes(Block a, list_int_ idcs) override;
    virtual Block stable_log(Block block, cyten_float cutoff) override;
    virtual Block sum(Block a, cyten_int ax) override;
    virtual cyten_float sum_all(Block a) override;
    virtual Block tdot(Block a, Block b, list_int_ idcs_a, list_int_ idcs_b) override;
    virtual Block to_dtype(Block a, Dtype dtype) override;
    virtual cyten_float trace_full(Block a) override;
    virtual Block trace_partial(Block a,
                                list_int_ idcs1,
                                list_int_ idcs2,
                                list_int_ remaining) override;
    virtual Block eye_matrix(cyten_int dim, Dtype dtype, std::string device = py::none()) override;
    virtual cyten_complex get_block_element(Block a, list_int_ idcs) override;
    virtual Block matrix_dot(Block a, Block b) override;
    virtual Block matrix_exp(Block matrix) override;
    virtual Block matrix_log(Block matrix) override;
    virtual tuple_Block_Block_ matrix_qr(Block a, bool full) override;
    virtual tuple_Block_Block_Block_ matrix_svd(Block a, std::string algorithm) override;
    virtual Block ones_block(list_int_ shape,
                             Dtype dtype,
                             std::string device = py::none()) override;
    virtual Block zeros(list_int_ shape, Dtype dtype, std::string device = py::none()) override;
};

} // namespace cyten
