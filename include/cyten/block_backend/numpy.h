#pragma once

#include <cyten/block_backend/block_backend.h>
#include <memory>
#include <pybind11/pytypes.h>

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

        BlockBackend* get_backend() const override;

        std::vector<int64> shape() const override;
        Dtype dtype() const override;
        const std::string& device() const override;
        py::array to_numpy() const override { return arr_; }
        py::array to_numpy(Dtype dtype) const override;

        BlockPtr get_item(py::object key) override;
        py::object _item_key_cast_Blocks_to_numpy(py::object key) const;
        BlockCPtr get_item(py::object key) const override;
        void set_item(py::object key, py::object value) override;

        complex128 _item_as_complex128() const override;
        int64 _item_as_int64() const override;

      protected:
        py::array arr_;
    };

  private:
    std::shared_ptr<Scalar> as_scalar(py::array value);

  public:
    std::shared_ptr<Scalar> as_scalar(complex128 value, Dtype dtype) override;
    std::shared_ptr<Scalar> as_scalar(py::object value, Dtype dtype) override;
    std::shared_ptr<Scalar> as_scalar(bool b) override;
    std::shared_ptr<Scalar> as_scalar(float32 x) override;
    std::shared_ptr<Scalar> as_scalar(float64 x) override;
    std::shared_ptr<Scalar> as_scalar(complex64 z) override;
    std::shared_ptr<Scalar> as_scalar(complex128 z) override;

  public:
    /// Get the backend instance for the given device (nearly-singleton per device).
    static NumpyBlockBackend* from_factory(const std::string& device = "cpu");
    /// Get a shared_ptr to the backend (e.g. for load_hdf5 or Python reference).
    static std::shared_ptr<NumpyBlockBackend> from_factory_shared(
      const std::string& device = "cpu");

  protected:
    explicit NumpyBlockBackend();

  public:
    static std::shared_ptr<NumpyBlockBackend> load_hdf5(py::object hdf5_loader,
                                                        py::object h5gr,
                                                        const std::string& subpath);

    std::string get_backend_name() const override;

    BlockPtr apply_leg_permutations(const BlockCPtr& block,
                                    const std::vector<py::array_t<int64>>& perms) override;
    BlockPtr as_block(py::object a,
                      std::optional<Dtype> dtype,
                      std::optional<std::string> device) override;
    std::string as_device(std::optional<std::string> device) override;
    std::vector<int64> abs_argmax(const BlockCPtr& block) override;
    BlockPtr abs(const BlockCPtr& a) override;
    BlockPtr add_axis(const BlockCPtr& a, int64 pos) override;
    bool all(const BlockCPtr& a) override;
    bool allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol) override;
    BlockPtr angle(const BlockCPtr& a) override;
    bool any(const BlockCPtr& a) override;
    BlockPtr apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax) override;
    BlockPtr _argsort(const BlockCPtr& block, int64 axis) override;
    BlockPtr conj(const BlockCPtr& a) override;
    BlockPtr copy_block(const BlockCPtr& a, std::optional<std::string> device) override;
    /// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
    BlockPtr cutoff_inverse(const BlockCPtr& a, float64 cutoff) override;
    std::tuple<BlockPtr, BlockPtr> eigh(const BlockCPtr& block,
                                        std::optional<std::string> sort) override;
    BlockPtr eigvalsh(const BlockCPtr& block, std::optional<std::string> sort) override;
    BlockPtr enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis) override;
    BlockPtr exp(const BlockCPtr& a) override;
    BlockPtr block_from_diagonal(const BlockCPtr& diag) override;
    BlockPtr block_from_mask(const BlockCPtr& mask, Dtype dtype) override;
    BlockPtr block_from_numpy(const py::array& a,
                              std::optional<Dtype> dtype,
                              std::optional<std::string> device) override;
    BlockPtr get_diagonal(const BlockCPtr& a, std::optional<float64> tol) override;
    BlockPtr imag(const BlockCPtr& a) override;
    Scalar inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger) override;
    Scalar item(const BlockCPtr& a) override;
    BlockPtr kron(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr linear_combination(const Scalar& a_coef,
                                const BlockCPtr& v,
                                const Scalar& b_coef,
                                const BlockCPtr& w) override;
    BlockPtr log(const BlockCPtr& a) override;
    Scalar max(const BlockCPtr& a) override;
    Scalar max_abs(const BlockCPtr& a) override;
    Scalar min(const BlockCPtr& a) override;
    BlockPtr mul(const Scalar& a, const BlockCPtr& b) override;
    Scalar norm(const BlockCPtr& a, float64 order, std::optional<int64> axis) override;
    BlockPtr outer(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation) override;
    BlockPtr random_normal(const std::vector<int64>& dims,
                           Dtype dtype,
                           float64 sigma,
                           std::optional<std::string> device) override;
    BlockPtr random_uniform(const std::vector<int64>& dims,
                            Dtype dtype,
                            std::optional<std::string> device) override;
    BlockPtr real(const BlockCPtr& a) override;
    BlockPtr real_if_close(const BlockCPtr& a, float64 tol) override;
    BlockPtr scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis) override;
    BlockPtr tile(const BlockCPtr& a, int64 repeats) override;
    std::vector<std::string> _block_repr_lines(const BlockCPtr& a,
                                               const std::string& indent,
                                               int64 max_width,
                                               int64 max_lines) override;
    BlockPtr reshape(const BlockCPtr& a, const std::vector<int64>& shape) override;
    BlockPtr sqrt(const BlockCPtr& a) override;
    BlockPtr squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs) override;
    BlockPtr stable_log(const BlockCPtr& block, float64 cutoff) override;
    BlockPtr sum(const BlockCPtr& a, int64 ax) override;
    Scalar sum_all(const BlockCPtr& a) override;
    BlockPtr multiply_blocks(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr tdot(const BlockCPtr& a,
                  const BlockCPtr& b,
                  const std::vector<int64>& idcs_a,
                  const std::vector<int64>& idcs_b) override;
    BlockPtr to_dtype(const BlockCPtr& a, Dtype dtype) override;
    Scalar trace_full(const BlockCPtr& a) override;
    BlockPtr trace_partial(const BlockCPtr& a,
                           const std::vector<int64>& idcs1,
                           const std::vector<int64>& idcs2,
                           const std::vector<int64>& remaining_idcs) override;
    BlockPtr eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device) override;
    Scalar get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs) override;
    BlockPtr matrix_dot(const BlockCPtr& a, const BlockCPtr& b) override;
    BlockPtr matrix_exp(const BlockCPtr& matrix) override;
    BlockPtr matrix_log(const BlockCPtr& matrix) override;
    std::tuple<BlockPtr, BlockPtr> matrix_qr(const BlockCPtr& a, bool full) override;
    std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      const BlockCPtr& a,
      std::optional<std::string> algorithm) override;
    const std::vector<std::string>& possible_svd_algorithms() const override;
    BlockPtr ones_block(const std::vector<int64>& shape,
                        Dtype dtype,
                        std::optional<std::string> device) override;
    BlockPtr zeros(const std::vector<int64>& shape,
                   Dtype dtype,
                   std::optional<std::string> device) override;

  protected:
    bool is_correct_block_type(const BlockCPtr& block) const override;

  private:
    static const NumpyBlockBackend::Block* ptr(const BlockCPtr& b);
    static py::object obj(const BlockCPtr& b);
    static BlockPtr wrap(py::object arr);
};

} // namespace cyten
