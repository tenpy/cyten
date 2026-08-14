#pragma once

#include <cyten/tensors/diagonal_tensor.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

/// A boolean mask that can be used to project or enlarge a leg.
///
/// Masks come in two versions: projections and inclusions. A projection Mask has a single leg, the
/// :attr:`large_leg` in its domain and maps it to a single leg, the :attr:`small_leg` in the
/// codomain. An inclusion Mask is the dagger of this projection Mask and maps from the small leg
/// in the domain to the large leg in the codomain.
class Mask : public Tensor
{
  public:
    using Ptr = std::shared_ptr<Mask>;
    using CPtr = std::shared_ptr<const Mask>;

    /// Float / complex dtypes are forbidden (bool only). Matches Python ``_forbidden_dtypes``.
    static std::vector<Dtype> _forbidden_dtypes;

    bool is_projection = true;
    TensorBackend::DataPtr data;

    Mask(TensorBackend::DataPtr data,
         Space::Ptr space_in,
         Space::Ptr space_out,
         bool is_projection,
         TensorBackend::Ptr backend,
         Symmetry::Ptr symmetry,
         LegLabels labels,
         std::string device);

    ~Mask() override = default;

    [[nodiscard]] std::vector<Dtype> const& forbidden_dtypes() const override;

    void test_sanity() const override;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    /// The large leg (domain for projection, codomain for inclusion).
    [[nodiscard]] ElementarySpace::Ptr large_leg() const;

    /// The small leg (codomain for projection, domain for inclusion).
    [[nodiscard]] ElementarySpace::Ptr small_leg() const;

    // --- factories ---

    /// The identity map as a Mask, i.e. the mask that keeps all states and discards none.
    [[nodiscard]] static Ptr from_eye(Space::Ptr leg,
                                      bool is_projection = true,
                                      TensorBackend::Ptr backend = nullptr,
                                      std::optional<LegLabels> labels = std::nullopt,
                                      std::optional<std::string> device = std::nullopt);

    /// Create a projection Mask from a boolean block.
    [[nodiscard]] static Ptr from_block_mask(BlockBackend::BlockPtr block_mask,
                                             Space::Ptr large_leg,
                                             TensorBackend::Ptr backend = nullptr,
                                             std::optional<LegLabels> labels = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    /// Create a projection Mask from a boolean DiagonalTensor.
    [[nodiscard]] static Ptr from_DiagonalTensor(DiagonalTensorCPtr diag);

    /// Create a projection Mask from the indices that are kept.
    [[nodiscard]] static Ptr from_indices(py::object indices,
                                          Space::Ptr large_leg,
                                          TensorBackend::Ptr backend = nullptr,
                                          std::optional<LegLabels> labels = std::nullopt,
                                          std::optional<std::string> device = std::nullopt);

    /// Create a random projection Mask. ``np_random`` stays Python-only.
    [[nodiscard]] static Ptr from_random(Space::Ptr large_leg,
                                         Space::Ptr small_leg = nullptr,
                                         TensorBackend::Ptr backend = nullptr,
                                         float64 p_keep = 0.5,
                                         int64 min_keep = 0,
                                         std::optional<LegLabels> labels = std::nullopt,
                                         std::optional<std::string> device = std::nullopt,
                                         py::object np_random = py::none());

    /// The zero projection Mask, that discards all states and keeps none.
    [[nodiscard]] static Ptr from_zero(Space::Ptr large_leg,
                                       TensorBackend::Ptr backend = nullptr,
                                       std::optional<LegLabels> labels = std::nullopt,
                                       std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_hdf5(py::object hdf5_loader,
                                       py::object h5gr,
                                       std::string const& subpath);

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    // --- Tensor overrides ---

    [[nodiscard]] Tensor::Ptr as_dtype(Dtype dtype) override;

    [[nodiscard]] SymmetricTensorPtr as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    /// Like :meth:`as_SymmetricTensor`, with an explicit result dtype (Python Mask API).
    [[nodiscard]] SymmetricTensorPtr as_SymmetricTensor(bool guarantee_copy,
                                                        std::optional<std::string> warning,
                                                        Dtype dtype);

    [[nodiscard]] Tensor::Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) override;

    [[nodiscard]] Tensor::Ptr dagger() const override;

    [[nodiscard]] BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override;

    void move_to_device(std::string device) override;

    [[nodiscard]] Tensor::Ptr to_backend(
      TensorBackend::Ptr backend,
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt) override;

    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) override;

    /// Override of :meth:`Tensor::to_numpy` (non-virtual on base; Mask has its own dense layout).
    [[nodiscard]] py::array to_numpy(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      py::object numpy_dtype = py::none(),
      bool understood_braiding = false);

    // --- Mask-specific API ---

    [[nodiscard]] DiagonalTensor::Ptr as_DiagonalTensor(Dtype dtype = Dtype::Complex128);

    [[nodiscard]] BlockBackend::BlockPtr as_block_mask();

    [[nodiscard]] py::array as_numpy_mask();

    [[nodiscard]] bool all() const;

    [[nodiscard]] bool any() const;

    /// Alias for :meth:`orthogonal_complement`.
    [[nodiscard]] Ptr logical_not();

    /// The "opposite" Mask, that keeps exactly what self discards and vv.
    [[nodiscard]] Ptr orthogonal_complement();

    /// Utility for binary boolean ops (``&``, ``|``, ``^``, ``==``, ``!=``).
    [[nodiscard]] Ptr _binary_operand(bool other, BlockBinaryFn func, std::string const& operand);

    [[nodiscard]] Ptr _binary_operand(MaskCPtr other,
                                      BlockBinaryFn func,
                                      std::string const& operand);

    [[nodiscard]] Ptr _unary_operand(BlockUnaryFn func);
};

} // namespace cyten
