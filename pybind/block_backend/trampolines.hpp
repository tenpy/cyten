#pragma once

#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/numpy.h>
#include <pybind11/pybind11.h>

namespace cyten {

/// @brief pybind11 trampoline class for Block in Python
template<class BlockBase = BlockBackend::Block>
class PyBlock
  : public BlockBase
  , py::trampoline_self_life_support
{
  public:
    using BlockBase::BlockBase; // inherit constructors

    std::vector<int64> shape() const override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(std::vector<int64>), BlockBase, shape);
    }
    Dtype dtype() const override { PYBIND11_OVERRIDE_PURE(Dtype, BlockBase, dtype); }
    std::string device() const override { PYBIND11_OVERRIDE_PURE(std::string, BlockBase, device); }
}; // trampoline class PyBlock

/// @brief pybind11 trampoline class for BlockBackend in Python
template<class BlockBackendBase = BlockBackend>
class PyBlockBackend
  : public BlockBackendBase
  , py::trampoline_self_life_support
{
  public:
    using BlockBackendBase::BlockBackendBase; // inherit constructors

    BlockPtr abs(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, abs, a);
    }
    BlockPtr apply_leg_permutations(const BlockCPtr& block,
                                    const std::vector<py::array_t<int64>>& perms) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, apply_leg_permutations, block, perms);
    }
    BlockPtr as_block(py::object a,
                      std::optional<Dtype> dtype,
                      std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, as_block, a, dtype, device);
    }
    std::string as_device(std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(std::string, BlockBackendBase, as_device, device);
    }
    std::vector<int64> abs_argmax(const BlockCPtr& block) override
    {
        PYBIND11_OVERRIDE_PURE(std::vector<int64>, BlockBackendBase, abs_argmax, block);
    }
    BlockPtr add_axis(const BlockCPtr& a, int64 pos) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, add_axis, a, pos);
    }
    bool block_all(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackendBase, block_all, a);
    }
    bool allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol) override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackendBase, allclose, a, b, rtol, atol);
    }
    BlockPtr angle(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, angle, a);
    }
    bool block_any(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackendBase, block_any, a);
    }
    BlockPtr apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, apply_mask, block, mask, ax);
    }
    BlockPtr _argsort(const BlockCPtr& block, int64 axis) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, _argsort, block, axis);
    }
    BlockPtr conj(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, conj, a);
    }
    BlockPtr copy_block(const BlockCPtr& a, std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, copy_block, a, device);
    }
    BlockPtr cutoff_inverse(const BlockCPtr& a, float64 cutoff) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, cutoff_inverse, a, cutoff);
    }
    Dtype get_dtype(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(Dtype, BlockBackendBase, get_dtype, a);
    }
    std::tuple<BlockPtr, BlockPtr> eigh(const BlockCPtr& block,
                                        std::optional<std::string> sort) override
    {
        PYBIND11_OVERRIDE_PURE(
          PYBIND11_TYPE(std::tuple<BlockPtr, BlockPtr>), BlockBackendBase, eigh, block, sort);
    }
    BlockPtr eigvalsh(const BlockCPtr& block, std::optional<std::string> sort) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, eigvalsh, block, sort);
    }
    BlockPtr enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, enlarge_leg, block, mask, axis);
    }
    BlockPtr exp(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, exp, a);
    }
    BlockPtr block_from_diagonal(const BlockCPtr& diag) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, block_from_diagonal, diag);
    }
    BlockPtr block_from_mask(const BlockCPtr& mask, Dtype dtype) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, block_from_mask, mask, dtype);
    }
    BlockPtr block_from_numpy(const py::array& a,
                              std::optional<Dtype> dtype,
                              std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, block_from_numpy, a, dtype, device);
    }
    std::string get_device(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(std::string, BlockBackendBase, get_device, a);
    }
    BlockPtr get_diagonal(const BlockCPtr& a, std::optional<float64> tol) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, get_diagonal, a, tol);
    }
    BlockPtr imag(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, imag, a);
    }
    py::object item(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(py::object, BlockBackendBase, item, a);
    }
    BlockPtr kron(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, kron, a, b);
    }
    BlockPtr linear_combination(Scalar a_coef,
                                const BlockCPtr& v,
                                Scalar b_coef,
                                const BlockCPtr& w) override
    {
        PYBIND11_OVERRIDE_PURE(
          BlockPtr, BlockBackendBase, linear_combination, a_coef, v, b_coef, w);
    }
    BlockPtr log(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, log, a);
    }
    float64 max(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackendBase, max, a);
    }
    float64 max_abs(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackendBase, max_abs, a);
    }
    float64 min(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackendBase, min, a);
    }
    BlockPtr mul(py::object a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, mul, a, b);
    }
    float64 norm(const BlockCPtr& a, float64 order, std::optional<int64> axis) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackendBase, norm, a, order, axis);
    }
    BlockPtr outer(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, outer, a, b);
    }
    BlockPtr permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, permute_axes, a, permutation);
    }
    BlockPtr random_normal(const std::vector<int64>& dims,
                           Dtype dtype,
                           float64 sigma,
                           std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(
          BlockPtr, BlockBackendBase, random_normal, dims, dtype, sigma, device);
    }
    BlockPtr random_uniform(const std::vector<int64>& dims,
                            Dtype dtype,
                            std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, random_uniform, dims, dtype, device);
    }
    BlockPtr real(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, real, a);
    }
    BlockPtr real_if_close(const BlockCPtr& a, float64 tol) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, real_if_close, a, tol);
    }
    BlockPtr tile(const BlockCPtr& a, int64 repeats) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, tile, a, repeats);
    }
    std::vector<std::string> _block_repr_lines(const BlockCPtr& a,
                                               const std::string& indent,
                                               int64 max_width,
                                               int64 max_lines) override
    {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>,
                               BlockBackendBase,
                               _block_repr_lines,
                               a,
                               indent,
                               max_width,
                               max_lines);
    }
    BlockPtr reshape(const BlockCPtr& a, const std::vector<int64>& shape) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, reshape, a, shape);
    }
    BlockPtr scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, scale_axis, block, factors, axis);
    }
    std::vector<int64> get_shape(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(std::vector<int64>, BlockBackendBase, get_shape, a);
    }
    BlockPtr sqrt(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, sqrt, a);
    }
    BlockPtr squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, squeeze_axes, a, idcs);
    }
    BlockPtr stable_log(const BlockCPtr& block, float64 cutoff) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, stable_log, block, cutoff);
    }
    BlockPtr sum(const BlockCPtr& a, int64 ax) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, sum, a, ax);
    }
    complex128 sum_all(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(complex128), BlockBackendBase, sum_all, a);
    }
    BlockPtr tdot(const BlockCPtr& a,
                  const BlockCPtr& b,
                  const std::vector<int64>& idcs_a,
                  const std::vector<int64>& idcs_b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, tdot, a, b, idcs_a, idcs_b);
    }
    BlockPtr to_dtype(const BlockCPtr& a, Dtype dtype) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, to_dtype, a, dtype);
    }
    py::object to_numpy(const BlockCPtr& a, std::optional<py::object> numpy_dtype) override
    {
        PYBIND11_OVERRIDE_PURE(py::object, BlockBackendBase, to_numpy, a, numpy_dtype);
    }
    complex128 trace_full(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(complex128), BlockBackendBase, trace_full, a);
    }
    BlockPtr trace_partial(const BlockCPtr& a,
                           const std::vector<int64>& idcs1,
                           const std::vector<int64>& idcs2,
                           const std::vector<int64>& remaining_idcs) override
    {
        PYBIND11_OVERRIDE_PURE(
          BlockPtr, BlockBackendBase, trace_partial, a, idcs1, idcs2, remaining_idcs);
    }
    BlockPtr eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, eye_matrix, dim, dtype, device);
    }
    py::object get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs) override
    {
        PYBIND11_OVERRIDE_PURE(py::object, BlockBackendBase, get_block_element, a, idcs);
    }
    bool get_block_mask_element(const BlockCPtr& a,
                                int64 large_leg_idx,
                                int64 small_leg_idx,
                                int64 sum_block) override
    {
        PYBIND11_OVERRIDE_PURE(bool,
                               BlockBackendBase,
                               get_block_mask_element,
                               a,
                               large_leg_idx,
                               small_leg_idx,
                               sum_block);
    }
    BlockPtr matrix_dot(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, matrix_dot, a, b);
    }
    BlockPtr matrix_exp(const BlockCPtr& matrix) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, matrix_exp, matrix);
    }
    BlockPtr matrix_log(const BlockCPtr& matrix) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, matrix_log, matrix);
    }
    std::tuple<BlockPtr, BlockPtr> matrix_qr(const BlockCPtr& a, bool full) override
    {
        PYBIND11_OVERRIDE_PURE(
          PYBIND11_TYPE(std::tuple<BlockPtr, BlockPtr>), BlockBackendBase, matrix_qr, a, full);
    }
    std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      const BlockCPtr& a,
      std::optional<std::string> algorithm) override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(std::tuple<BlockPtr, BlockPtr, BlockPtr>),
                               BlockBackendBase,
                               matrix_svd,
                               a,
                               algorithm);
    }
    BlockPtr ones_block(const std::vector<int64>& shape,
                        Dtype dtype,
                        std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, ones_block, shape, dtype, device);
    }
    BlockPtr zeros(const std::vector<int64>& shape,
                   Dtype dtype,
                   std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, zeros, shape, dtype, device);
    }
    std::string get_backend_name() const override
    {
        PYBIND11_OVERRIDE(std::string, BlockBackendBase, get_backend_name);
    }
    BlockPtr multiply_blocks(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackendBase, multiply_blocks, a, b);
    }
    bool is_correct_block_type(const BlockCPtr& block) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackendBase, is_correct_block_type, block);
    }
}; // trampoline class PyBlockBackend

// TODO: trampoline classes for Block

} // namespace cyten
