#pragma once

#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/numpy.h>
#include <pybind11/pybind11.h>

namespace cyten {

/// @brief pybind11 trampoline class for Block in Python
class PyBlock
  : public BlockBackend::Block
  , py::trampoline_self_life_support
{
  public:
    using BlockBackend::Block::Block; // inherit constructors

    BlockBackend* get_backend() const override
    {
        PYBIND11_OVERRIDE_PURE(BlockBackend*, BlockBackend::Block, get_backend);
    }
    std::vector<int64> shape() const override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(std::vector<int64>), BlockBackend::Block, shape);
    }
    int64 ndim() const override { PYBIND11_OVERRIDE(int64, BlockBackend::Block, ndim); }
    Dtype dtype() const override { PYBIND11_OVERRIDE_PURE(Dtype, BlockBackend::Block, dtype); }
    const std::string& device() const override
    {
        PYBIND11_OVERRIDE_PURE(const std::string&, BlockBackend::Block, device);
    }
    py::array to_numpy() const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, BlockBackend::Block, to_numpy);
    }
    BlockCPtr get_item(py::object key) const override
    {
        PYBIND11_OVERRIDE_PURE(BlockCPtr, BlockBackend::Block, get_item, key);
    }
    BlockPtr get_item(py::object key) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend::Block, get_item, key);
    }
    void set_item(py::object key, py::object value) override
    {
        PYBIND11_OVERRIDE_PURE(void, BlockBackend::Block, set_item, key, value);
    }
    py::array to_numpy(Dtype dtype) const override
    {
        PYBIND11_OVERRIDE(py::array, BlockBackend::Block, to_numpy, dtype);
    }
}; // trampoline class PyBlock

/// @brief pybind11 trampoline class for BlockBackend in Python
class PyBlockBackend
  : public BlockBackend
  , py::trampoline_self_life_support
{
  public:
    using BlockBackend::BlockBackend; // inherit constructors
    PyBlockBackend(BlockBackend&& base)
      : BlockBackend(std::move(base))
    {
    }

    BlockPtr abs(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, abs, a);
    }
    BlockPtr apply_leg_permutations(const BlockCPtr& block,
                                    const std::vector<py::array_t<int64>>& perms) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, apply_leg_permutations, block, perms);
    }
    BlockPtr as_block(py::object a,
                      std::optional<Dtype> dtype,
                      std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, as_block, a, dtype, device);
    }
    std::string as_device(std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(std::string, BlockBackend, as_device, device);
    }
    std::vector<int64> abs_argmax(const BlockCPtr& block) override
    {
        PYBIND11_OVERRIDE_PURE(std::vector<int64>, BlockBackend, abs_argmax, block);
    }
    BlockPtr add_axis(const BlockCPtr& a, int64 pos) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, add_axis, a, pos);
    }
    bool block_all(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackend, block_all, a);
    }
    bool allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol) override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackend, allclose, a, b, rtol, atol);
    }
    BlockPtr angle(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, angle, a);
    }
    bool block_any(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackend, block_any, a);
    }
    BlockPtr apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, apply_mask, block, mask, ax);
    }
    BlockPtr _argsort(const BlockCPtr& block, int64 axis) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, _argsort, block, axis);
    }
    BlockPtr conj(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, conj, a);
    }
    BlockPtr copy_block(const BlockCPtr& a, std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, copy_block, a, device);
    }
    BlockPtr cutoff_inverse(const BlockCPtr& a, float64 cutoff) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, cutoff_inverse, a, cutoff);
    }
    std::tuple<BlockPtr, BlockPtr> eigh(const BlockCPtr& block,
                                        std::optional<std::string> sort) override
    {
        PYBIND11_OVERRIDE_PURE(
          PYBIND11_TYPE(std::tuple<BlockPtr, BlockPtr>), BlockBackend, eigh, block, sort);
    }
    BlockPtr eigvalsh(const BlockCPtr& block, std::optional<std::string> sort) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, eigvalsh, block, sort);
    }
    BlockPtr enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, enlarge_leg, block, mask, axis);
    }
    BlockPtr exp(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, exp, a);
    }
    BlockPtr block_from_diagonal(const BlockCPtr& diag) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, block_from_diagonal, diag);
    }
    BlockPtr block_from_mask(const BlockCPtr& mask, Dtype dtype) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, block_from_mask, mask, dtype);
    }
    BlockPtr block_from_numpy(const py::array& a,
                              std::optional<Dtype> dtype,
                              std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, block_from_numpy, a, dtype, device);
    }
    BlockPtr get_diagonal(const BlockCPtr& a, std::optional<float64> tol) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, get_diagonal, a, tol);
    }
    BlockPtr imag(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, imag, a);
    }
    py::object item(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(py::object, BlockBackend, item, a);
    }
    BlockPtr kron(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, kron, a, b);
    }
    BlockPtr linear_combination(Scalar a_coef,
                                const BlockCPtr& v,
                                Scalar b_coef,
                                const BlockCPtr& w) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, linear_combination, a_coef, v, b_coef, w);
    }
    BlockPtr log(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, log, a);
    }
    float64 max(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackend, max, a);
    }
    float64 max_abs(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackend, max_abs, a);
    }
    float64 min(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackend, min, a);
    }
    BlockPtr mul(py::object a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, mul, a, b);
    }
    float64 norm(const BlockCPtr& a, float64 order, std::optional<int64> axis) override
    {
        PYBIND11_OVERRIDE_PURE(float64, BlockBackend, norm, a, order, axis);
    }
    BlockPtr outer(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, outer, a, b);
    }
    BlockPtr permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, permute_axes, a, permutation);
    }
    BlockPtr random_normal(const std::vector<int64>& dims,
                           Dtype dtype,
                           float64 sigma,
                           std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, random_normal, dims, dtype, sigma, device);
    }
    BlockPtr random_uniform(const std::vector<int64>& dims,
                            Dtype dtype,
                            std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, random_uniform, dims, dtype, device);
    }
    BlockPtr real(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, real, a);
    }
    BlockPtr real_if_close(const BlockCPtr& a, float64 tol) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, real_if_close, a, tol);
    }
    BlockPtr tile(const BlockCPtr& a, int64 repeats) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, tile, a, repeats);
    }
    std::vector<std::string> _block_repr_lines(const BlockCPtr& a,
                                               const std::string& indent,
                                               int64 max_width,
                                               int64 max_lines) override
    {
        PYBIND11_OVERRIDE_PURE(std::vector<std::string>,
                               BlockBackend,
                               _block_repr_lines,
                               a,
                               indent,
                               max_width,
                               max_lines);
    }
    BlockPtr reshape(const BlockCPtr& a, const std::vector<int64>& shape) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, reshape, a, shape);
    }
    BlockPtr scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, scale_axis, block, factors, axis);
    }
    BlockPtr sqrt(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, sqrt, a);
    }
    BlockPtr squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, squeeze_axes, a, idcs);
    }
    BlockPtr stable_log(const BlockCPtr& block, float64 cutoff) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, stable_log, block, cutoff);
    }
    BlockPtr sum(const BlockCPtr& a, int64 ax) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, sum, a, ax);
    }
    complex128 sum_all(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(complex128), BlockBackend, sum_all, a);
    }
    BlockPtr tdot(const BlockCPtr& a,
                  const BlockCPtr& b,
                  const std::vector<int64>& idcs_a,
                  const std::vector<int64>& idcs_b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, tdot, a, b, idcs_a, idcs_b);
    }
    BlockPtr to_dtype(const BlockCPtr& a, Dtype dtype) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, to_dtype, a, dtype);
    }
    complex128 trace_full(const BlockCPtr& a) override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(complex128), BlockBackend, trace_full, a);
    }
    BlockPtr trace_partial(const BlockCPtr& a,
                           const std::vector<int64>& idcs1,
                           const std::vector<int64>& idcs2,
                           const std::vector<int64>& remaining_idcs) override
    {
        PYBIND11_OVERRIDE_PURE(
          BlockPtr, BlockBackend, trace_partial, a, idcs1, idcs2, remaining_idcs);
    }
    BlockPtr eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, eye_matrix, dim, dtype, device);
    }
    py::object get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs) override
    {
        PYBIND11_OVERRIDE_PURE(py::object, BlockBackend, get_block_element, a, idcs);
    }
    bool get_block_mask_element(const BlockCPtr& a,
                                int64 large_leg_idx,
                                int64 small_leg_idx,
                                int64 sum_block) override
    {
        PYBIND11_OVERRIDE(
          bool, BlockBackend, get_block_mask_element, a, large_leg_idx, small_leg_idx, sum_block);
    }
    BlockPtr matrix_dot(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, matrix_dot, a, b);
    }
    BlockPtr matrix_exp(const BlockCPtr& matrix) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, matrix_exp, matrix);
    }
    BlockPtr matrix_log(const BlockCPtr& matrix) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, matrix_log, matrix);
    }
    std::tuple<BlockPtr, BlockPtr> matrix_qr(const BlockCPtr& a, bool full) override
    {
        PYBIND11_OVERRIDE_PURE(
          PYBIND11_TYPE(std::tuple<BlockPtr, BlockPtr>), BlockBackend, matrix_qr, a, full);
    }
    std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      const BlockCPtr& a,
      std::optional<std::string> algorithm) override
    {
        PYBIND11_OVERRIDE_PURE(PYBIND11_TYPE(std::tuple<BlockPtr, BlockPtr, BlockPtr>),
                               BlockBackend,
                               matrix_svd,
                               a,
                               algorithm);
    }
    const std::vector<std::string>& possible_svd_algorithms() const override
    {
        PYBIND11_OVERRIDE_PURE(
          const std::vector<std::string>&, BlockBackend, possible_svd_algorithms);
    }
    BlockPtr ones_block(const std::vector<int64>& shape,
                        Dtype dtype,
                        std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, ones_block, shape, dtype, device);
    }
    BlockPtr zeros(const std::vector<int64>& shape,
                   Dtype dtype,
                   std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, zeros, shape, dtype, device);
    }
    std::string get_backend_name() const override
    {
        PYBIND11_OVERRIDE(std::string, BlockBackend, get_backend_name);
    }
    BlockPtr multiply_blocks(const BlockCPtr& a, const BlockCPtr& b) override
    {
        PYBIND11_OVERRIDE_PURE(BlockPtr, BlockBackend, multiply_blocks, a, b);
    }
    bool is_correct_block_type(const BlockCPtr& block) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, BlockBackend, is_correct_block_type, block);
    }
}; // trampoline class PyBlockBackend

// TODO: trampoline classes for Block

} // namespace cyten
