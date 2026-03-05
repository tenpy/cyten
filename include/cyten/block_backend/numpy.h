#pragma once

#include <cyten/block_backend/block_backend.h>
#include <memory>

namespace cyten {

/// A block backend using numpy.
class NumpyBlockBackend : public BlockBackend
{
  public:
    /// Block that holds a numpy array in a py::array.
    class PYBIND11_EXPORT Block : public BlockBackend::Block
    {
      public:
        explicit Block(py::array arr);
        virtual ~Block() = default;

        std::vector<cyten_int> shape() const override;
        Dtype dtype() const override;
        std::string device() const override;
        py::array to_numpy() const override { return arr_; }

      protected:
        py::array arr_;
    };

  public:
    NumpyBlockBackend();

    static std::shared_ptr<NumpyBlockBackend> load_hdf5(py::object hdf5_loader,
                                                        py::object h5gr,
                                                        std::string const& subpath);

    std::string get_backend_name() const override;

    BlockPtr apply_leg_permutations(BlockCPtr const& block,
                                    std::vector<py::array_t<cyten_int>> const& perms) override;
    BlockPtr as_block(py::object a,
                      std::optional<Dtype> dtype,
                      std::optional<std::string> device) override;
    std::string as_device(std::optional<std::string> device) override;
    std::vector<cyten_int> abs_argmax(BlockCPtr const& block) override;
    BlockPtr abs(BlockCPtr const& a) override;
    BlockPtr add_axis(BlockCPtr const& a, int pos) override;
    bool block_all(BlockCPtr const& a) override;
    bool allclose(BlockCPtr const& a, BlockCPtr const& b, double rtol, double atol) override;
    BlockPtr angle(BlockCPtr const& a) override;
    bool block_any(BlockCPtr const& a) override;
    BlockPtr apply_mask(BlockCPtr const& block, BlockCPtr const& mask, int ax) override;
    BlockPtr _argsort(BlockCPtr const& block, int axis) override;
    BlockPtr conj(BlockCPtr const& a) override;
    BlockPtr copy_block(BlockCPtr const& a, std::optional<std::string> device) override;
    /// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
    BlockPtr cutoff_inverse(BlockCPtr const& a, double cutoff) override;
    Dtype get_dtype(BlockCPtr const& a) override;
    std::tuple<BlockPtr, BlockPtr> eigh(BlockCPtr const& block,
                                        std::optional<std::string> sort) override;
    BlockPtr eigvalsh(BlockCPtr const& block, std::optional<std::string> sort) override;
    BlockPtr enlarge_leg(BlockCPtr const& block, BlockCPtr const& mask, int axis) override;
    BlockPtr exp(BlockCPtr const& a) override;
    BlockPtr block_from_diagonal(BlockCPtr const& diag) override;
    BlockPtr block_from_mask(BlockCPtr const& mask, Dtype dtype) override;
    BlockPtr block_from_numpy(py::array const& a,
                              std::optional<Dtype> dtype,
                              std::optional<std::string> device) override;
    std::string get_device(BlockCPtr const& a) override;
    BlockPtr get_diagonal(BlockCPtr const& a, std::optional<double> tol) override;
    bool get_block_mask_element(BlockCPtr const& a,
                                cyten_int large_leg_idx,
                                cyten_int small_leg_idx,
                                cyten_int sum_block) override;
    BlockPtr imag(BlockCPtr const& a) override;
    py::object item(BlockCPtr const& a) override;
    BlockPtr kron(BlockCPtr const& a, BlockCPtr const& b) override;
    BlockPtr linear_combination(Scalar a_coef,
                                BlockCPtr const& v,
                                Scalar b_coef,
                                BlockCPtr const& w) override;
    BlockPtr log(BlockCPtr const& a) override;
    double max(BlockCPtr const& a) override;
    double max_abs(BlockCPtr const& a) override;
    double min(BlockCPtr const& a) override;
    BlockPtr mul(py::object a, BlockCPtr const& b) override;
    double norm(BlockCPtr const& a, double order, std::optional<int> axis) override;
    BlockPtr outer(BlockCPtr const& a, BlockCPtr const& b) override;
    BlockPtr permute_axes(BlockCPtr const& a, std::vector<int> const& permutation) override;
    BlockPtr random_normal(std::vector<cyten_int> const& dims,
                           Dtype dtype,
                           double sigma,
                           std::optional<std::string> device) override;
    BlockPtr random_uniform(std::vector<cyten_int> const& dims,
                            Dtype dtype,
                            std::optional<std::string> device) override;
    BlockPtr real(BlockCPtr const& a) override;
    BlockPtr real_if_close(BlockCPtr const& a, double tol) override;
    BlockPtr scale_axis(BlockCPtr const& block, BlockCPtr const& factors, int axis) override;
    BlockPtr tile(BlockCPtr const& a, int repeats) override;
    std::vector<std::string> _block_repr_lines(BlockCPtr const& a,
                                               std::string const& indent,
                                               int max_width,
                                               int max_lines) override;
    BlockPtr reshape(BlockCPtr const& a, std::vector<cyten_int> const& shape) override;
    std::vector<cyten_int> get_shape(BlockCPtr const& a) override;
    BlockPtr sqrt(BlockCPtr const& a) override;
    BlockPtr squeeze_axes(BlockCPtr const& a, std::vector<int> const& idcs) override;
    BlockPtr stable_log(BlockCPtr const& block, double cutoff) override;
    BlockPtr sum(BlockCPtr const& a, int ax) override;
    std::complex<cyten_float> sum_all(BlockCPtr const& a) override;
    BlockPtr multiply_blocks(BlockCPtr const& a, BlockCPtr const& b) override;
    BlockPtr tdot(BlockCPtr const& a,
                  BlockCPtr const& b,
                  std::vector<int> const& idcs_a,
                  std::vector<int> const& idcs_b) override;
    BlockPtr to_dtype(BlockCPtr const& a, Dtype dtype) override;
    py::object to_numpy(BlockCPtr const& a, std::optional<py::object> numpy_dtype) override;
    std::complex<cyten_float> trace_full(BlockCPtr const& a) override;
    BlockPtr trace_partial(BlockCPtr const& a,
                           std::vector<int> const& idcs1,
                           std::vector<int> const& idcs2,
                           std::vector<int> const& remaining_idcs) override;
    BlockPtr eye_matrix(int dim, Dtype dtype, std::optional<std::string> device) override;
    py::object get_block_element(BlockCPtr const& a, std::vector<cyten_int> const& idcs) override;
    BlockPtr matrix_dot(BlockCPtr const& a, BlockCPtr const& b) override;
    BlockPtr matrix_exp(BlockCPtr const& matrix) override;
    BlockPtr matrix_log(BlockCPtr const& matrix) override;
    std::tuple<BlockPtr, BlockPtr> matrix_qr(BlockCPtr const& a, bool full) override;
    std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      BlockCPtr const& a,
      std::optional<std::string> algorithm) override;
    BlockPtr ones_block(std::vector<cyten_int> const& shape,
                        Dtype dtype,
                        std::optional<std::string> device) override;
    BlockPtr zeros(std::vector<cyten_int> const& shape,
                   Dtype dtype,
                   std::optional<std::string> device) override;

  protected:
    bool is_correct_block_type(BlockCPtr const& block) const override;

  private:
    static NumpyBlockBackend::Block const* ptr(BlockCPtr const& b);
    static py::object obj(BlockCPtr const& b);
    static BlockPtr wrap(py::object arr);
};

} // namespace cyten
