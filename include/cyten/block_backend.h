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

/// Abstract base class for block backends. Operations on dense blocks; subclass per backend.
class BlockBackend
{
  public:
    explicit BlockBackend(std::string default_device);
    virtual ~BlockBackend() = default;

    /// Name of the backend class for __repr__ / __str__ (e.g. "NumpyBlockBackend").
    virtual std::string get_backend_name() const;

    std::string default_device;
    std::vector<std::string> svd_algorithms; // first is default

    /// Apply basis_perm (or inverse_basis_perm) of each leg on the corresponding axis.
    /// \param legs Python Space/leg objects with .basis_perm and .inverse_basis_perm attributes.
    BlockPtr apply_basis_perm(BlockCPtr const& block,
                              std::vector<py::object> const& legs,
                              bool inv = false);

    // ----- virtual: each backend implements (no default) -----
    virtual BlockPtr apply_leg_permutations(BlockCPtr const& block,
                                            std::vector<py::array_t<cyten_int>> const& perms) = 0;

    // ----- abstract (pure virtual) -----
    /// Returns (block, dtype). Second is set only when return_dtype is true.
    virtual std::pair<BlockPtr, std::optional<Dtype>> as_block(
      py::object a,
      std::optional<Dtype> dtype = std::nullopt,
      bool return_dtype = false,
      std::optional<std::string> device = std::nullopt) = 0;
    virtual std::string as_device(std::optional<std::string> device) = 0;
    virtual std::vector<cyten_int> abs_argmax(BlockCPtr const& block) = 0;
    virtual BlockPtr abs(BlockCPtr const& a) = 0;
    virtual BlockPtr add_axis(BlockCPtr const& a, int pos) = 0;
    virtual bool block_all(BlockCPtr const& a) = 0;
    virtual bool allclose(BlockCPtr const& a,
                          BlockCPtr const& b,
                          double rtol = 1e-5,
                          double atol = 1e-8) = 0;
    virtual BlockPtr angle(BlockCPtr const& a) = 0;
    virtual bool block_any(BlockCPtr const& a) = 0;
    virtual BlockPtr _argsort(BlockCPtr const& block, int axis) = 0;
    virtual BlockPtr conj(BlockCPtr const& a) = 0;
    virtual BlockPtr copy_block(BlockCPtr const& a,
                                std::optional<std::string> device = std::nullopt) = 0;
    virtual Dtype get_dtype(BlockCPtr const& a) = 0;
    virtual std::tuple<BlockPtr, BlockPtr> eigh(
      BlockCPtr const& block,
      std::optional<std::string> sort = std::nullopt) = 0;
    virtual BlockPtr eigvalsh(BlockCPtr const& block,
                              std::optional<std::string> sort = std::nullopt) = 0;
    virtual BlockPtr exp(BlockCPtr const& a) = 0;
    virtual BlockPtr block_from_diagonal(BlockCPtr const& diag) = 0;
    virtual BlockPtr block_from_mask(BlockCPtr const& mask, Dtype dtype) = 0;
    virtual BlockPtr block_from_numpy(py::array const& a,
                                      std::optional<Dtype> dtype = std::nullopt,
                                      std::optional<std::string> device = std::nullopt) = 0;
    virtual std::string get_device(BlockCPtr const& a) = 0;
    virtual BlockPtr get_diagonal(BlockCPtr const& a,
                                  std::optional<double> tol = std::nullopt) = 0;
    virtual BlockPtr imag(BlockCPtr const& a) = 0;
    virtual py::object item(BlockCPtr const& a) = 0;
    virtual BlockPtr kron(BlockCPtr const& a, BlockCPtr const& b) = 0;
    virtual BlockPtr log(BlockCPtr const& a) = 0;
    virtual double max(BlockCPtr const& a) = 0;
    virtual double max_abs(BlockCPtr const& a) = 0;
    virtual double min(BlockCPtr const& a) = 0;
    virtual double norm(BlockCPtr const& a,
                        double order = 2,
                        std::optional<int> axis = std::nullopt) = 0;
    virtual BlockPtr outer(BlockCPtr const& a, BlockCPtr const& b) = 0;
    virtual BlockPtr permute_axes(BlockCPtr const& a, std::vector<int> const& permutation) = 0;
    virtual BlockPtr random_normal(std::vector<cyten_int> const& dims,
                                   Dtype dtype,
                                   double sigma,
                                   std::optional<std::string> device = std::nullopt) = 0;
    virtual BlockPtr random_uniform(std::vector<cyten_int> const& dims,
                                    Dtype dtype,
                                    std::optional<std::string> device = std::nullopt) = 0;
    virtual BlockPtr real(BlockCPtr const& a) = 0;
    virtual BlockPtr real_if_close(BlockCPtr const& a, double tol) = 0;
    virtual BlockPtr tile(BlockCPtr const& a, int repeats) = 0;
    virtual std::vector<std::string> _block_repr_lines(BlockCPtr const& a,
                                                       std::string const& indent,
                                                       int max_width,
                                                       int max_lines) = 0;
    virtual BlockPtr reshape(BlockCPtr const& a, std::vector<cyten_int> const& shape) = 0;
    virtual std::vector<cyten_int> get_shape(BlockCPtr const& a) = 0;
    virtual BlockPtr sqrt(BlockCPtr const& a) = 0;
    virtual BlockPtr squeeze_axes(BlockCPtr const& a, std::vector<int> const& idcs) = 0;
    virtual BlockPtr stable_log(BlockCPtr const& block, double cutoff) = 0;
    virtual BlockPtr sum(BlockCPtr const& a, int ax) = 0;
    virtual std::complex<cyten_float> sum_all(BlockCPtr const& a) = 0;
    virtual BlockPtr multiply_blocks(BlockCPtr const& a, BlockCPtr const& b) = 0; // elementwise
    virtual BlockPtr tdot(BlockCPtr const& a,
                          BlockCPtr const& b,
                          std::vector<int> const& idcs_a,
                          std::vector<int> const& idcs_b) = 0;
    virtual BlockPtr to_dtype(BlockCPtr const& a, Dtype dtype) = 0;
    virtual std::complex<cyten_float> trace_full(BlockCPtr const& a) = 0;
    virtual BlockPtr trace_partial(BlockCPtr const& a,
                                   std::vector<int> const& idcs1,
                                   std::vector<int> const& idcs2,
                                   std::vector<int> const& remaining_idcs) = 0;
    virtual BlockPtr eye_matrix(int dim,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    virtual py::object get_block_element(BlockCPtr const& a,
                                         std::vector<cyten_int> const& idcs) = 0;
    virtual bool get_block_mask_element(BlockCPtr const& a,
                                        cyten_int large_leg_idx,
                                        cyten_int small_leg_idx,
                                        cyten_int sum_block = 0) = 0;
    virtual BlockPtr matrix_dot(BlockCPtr const& a, BlockCPtr const& b) = 0;
    virtual BlockPtr matrix_exp(BlockCPtr const& matrix) = 0;
    virtual BlockPtr matrix_log(BlockCPtr const& matrix) = 0;
    virtual std::tuple<BlockPtr, BlockPtr> matrix_qr(BlockCPtr const& a, bool full) = 0;
    virtual std::tuple<BlockPtr, BlockPtr, BlockPtr> matrix_svd(
      BlockCPtr const& a,
      std::optional<std::string> algorithm = std::nullopt) = 0;
    virtual BlockPtr ones_block(std::vector<cyten_int> const& shape,
                                Dtype dtype,
                                std::optional<std::string> device = std::nullopt) = 0;
    virtual BlockPtr zeros(std::vector<cyten_int> const& shape,
                           Dtype dtype,
                           std::optional<std::string> device = std::nullopt) = 0;

    // ----- virtual with no default (indexing is backend-specific) -----
    virtual BlockPtr apply_mask(BlockCPtr const& block, BlockCPtr const& mask, int ax) = 0;

    // ----- default implementations (non-virtual or virtual with body) -----
    BlockPtr argsort(BlockCPtr const& block,
                     std::optional<std::string> sort = std::nullopt,
                     int axis = 0);
    BlockPtr combine_legs(BlockCPtr const& a,
                          std::vector<std::vector<int>> const& leg_idcs_combine,
                          std::vector<bool> const& cstyles);
    BlockPtr combine_legs(BlockCPtr const& a,
                          std::vector<std::vector<int>> const& leg_idcs_combine,
                          bool cstyles = true);
    virtual BlockPtr cutoff_inverse(BlockCPtr const& a, double cutoff) = 0;
    BlockPtr dagger(BlockCPtr const& a);
    virtual BlockPtr enlarge_leg(BlockCPtr const& block, BlockCPtr const& mask, int axis) = 0;
    bool is_real(BlockCPtr const& a);
    virtual BlockPtr linear_combination(py::object a_coef,
                                        BlockCPtr const& v,
                                        py::object b_coef,
                                        BlockCPtr const& w) = 0;
    virtual BlockPtr mul(py::object a, BlockCPtr const& b) = 0;
    BlockPtr permute_combined_matrix(BlockCPtr const& block,
                                     std::vector<cyten_int> const& dims1,
                                     std::vector<int> const& idcs1,
                                     std::vector<cyten_int> const& dims2,
                                     std::vector<int> const& idcs2);
    BlockPtr permute_combined_idx(BlockCPtr const& block,
                                  int axis,
                                  std::vector<cyten_int> const& dims,
                                  std::vector<int> const& idcs);
    virtual BlockPtr scale_axis(BlockCPtr const& block, BlockCPtr const& factors, int axis) = 0;
    BlockPtr split_legs(BlockCPtr const& a,
                        std::vector<int> const& idcs,
                        std::vector<std::vector<cyten_int>> const& dims,
                        std::vector<bool> const& cstyles);
    BlockPtr split_legs(BlockCPtr const& a,
                        std::vector<int> const& idcs,
                        std::vector<std::vector<cyten_int>> const& dims,
                        bool cstyles = true);
    BlockPtr tensor_outer(BlockCPtr const& a, BlockCPtr const& b, int K);
    virtual py::object to_numpy(BlockCPtr const& a,
                                std::optional<py::object> numpy_dtype = std::nullopt) = 0;
    BlockPtr eye_block(std::vector<cyten_int> const& legs,
                       Dtype dtype,
                       std::optional<std::string> device = std::nullopt);
    std::tuple<BlockPtr, BlockPtr> matrix_lq(BlockCPtr const& a, bool full);
    void synchronize();

    std::complex<cyten_float> inner(
      BlockCPtr const& a,
      BlockCPtr const& b,
      bool do_dagger); // uses multiply_blocks, conj, permute_axes, sum_all
};

} // namespace cyten
