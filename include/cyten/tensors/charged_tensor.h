#pragma once

#include <cyten/tensors/symmetric_tensor.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

/// Tensors which are not symmetric, but carry a well defined charge.
///
/// This captures two related but slightly different concepts. In both cases, the main component
/// is an invariant part (:class:`SymmetricTensor`) with an additional hidden charge leg.
class ChargedTensor : public Tensor
{
  public:
    using Ptr = std::shared_ptr<ChargedTensor>;
    using CPtr = std::shared_ptr<const ChargedTensor>;

    /// Canonical label for the charge leg on the invariant part.
    static constexpr char const* _CHARGE_LEG_LABEL = "!";

    SymmetricTensor::Ptr invariant_part;
    /// ``nullptr`` means unspecified charged state (Python ``None``).
    BlockBackend::BlockPtr charged_state;
    Space::Ptr charge_leg;

    /// Construct from invariant part and optional charged state.
    ChargedTensor(py::object invariant_part, py::object charged_state = py::none());

    ChargedTensor(SymmetricTensor::Ptr invariant_part,
                  BlockBackend::BlockPtr charged_state = nullptr);

    ~ChargedTensor() override = default;

    void test_sanity() const override;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    // --- helpers ---

    /// Build the domain of the invariant part from the ChargedTensor domain + charge.
    ///
    /// Returns ``(inv_domain, charge_leg)``.
    [[nodiscard]] static std::tuple<TensorProduct::Ptr, Space::Ptr> _parse_inv_domain(
      TensorProduct::Ptr domain,
      py::object charge);

    /// Like :meth:`Tensor::_init_parse_labels`, also returning invariant-part labels.
    [[nodiscard]] static std::tuple<LegLabels, LegLabels> _parse_inv_labels(
      py::object labels,
      TensorProduct::Ptr const& codomain,
      TensorProduct::Ptr const& domain);

    /// If the :class:`ChargedTensor` concept is well defined for the `symmetry`.
    [[nodiscard]] static bool supports_symmetry(Symmetry::Ptr const& symmetry);

    // --- factories ---

    [[nodiscard]] static Ptr from_block_func(py::function func,
                                             py::object charge,
                                             py::object codomain,
                                             py::object domain = py::none(),
                                             py::object charged_state = py::none(),
                                             TensorBackend::Ptr backend = nullptr,
                                             py::object labels = py::none(),
                                             py::object func_kwargs = py::none(),
                                             std::optional<std::string> shape_kw = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_dense_block(py::object block,
                                              py::object codomain,
                                              py::object domain = py::none(),
                                              py::object charge = py::none(),
                                              TensorBackend::Ptr backend = nullptr,
                                              py::object labels = py::none(),
                                              std::optional<Dtype> dtype = std::nullopt,
                                              std::optional<std::string> device = std::nullopt,
                                              float64 tol = 1e-6,
                                              bool understood_braiding = false);

    /// Not implemented (matches Python).
    [[nodiscard]] static Ptr from_dense_block_single_sector(
      py::object vector,
      py::object space,
      Sector sector,
      TensorBackend::Ptr backend = nullptr,
      std::optional<std::string> label = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    /// Like constructor, but if ``invariant_part`` has only one leg, return a scalar (Python
    /// number / BlockBackend::Scalar as ``py::object``) when ``charged_state`` is given.
    [[nodiscard]] static py::object from_invariant_part(py::object invariant_part,
                                                        py::object charged_state = py::none());

    [[nodiscard]] static py::object from_two_charge_legs(py::object invariant_part,
                                                        py::object state1 = py::none(),
                                                        py::object state2 = py::none());

    [[nodiscard]] static Ptr from_zero(py::object codomain,
                                       py::object domain = py::none(),
                                       py::object charge = py::none(),
                                       py::object charged_state = py::none(),
                                       TensorBackend::Ptr backend = nullptr,
                                       py::object labels = py::none(),
                                       Dtype dtype = Dtype::Complex128,
                                       std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_hdf5(py::object hdf5_loader,
                                       py::object h5gr,
                                       std::string const& subpath);

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    // --- Tensor overrides ---

    [[nodiscard]] Tensor::Ptr as_dtype(Dtype dtype) override;

    [[nodiscard]] py::object as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    [[nodiscard]] Tensor::Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) override;

    [[nodiscard]] Tensor::Ptr dagger() const override;

    [[nodiscard]] BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override;

    void move_to_device(std::string device) override;

    [[nodiscard]] std::vector<std::string> _repr_header_lines(std::string const& indent,
                                                              bool use_symm_str = false) const override;

    LabelledLegs& set_label(int64 pos, LegLabel label) override;
    Tensor& set_labels(LegLabels labels) override;

    [[nodiscard]] Tensor::Ptr to_backend(TensorBackend::Ptr backend,
                                         std::optional<Dtype> dtype = std::nullopt,
                                         std::optional<std::string> device = std::nullopt) override;

    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) override;

    /// Return the components associated with a single sector.
    [[nodiscard]] BlockBackend::BlockPtr to_dense_block_single_sector();

    [[nodiscard]] py::object as_py_object();
    [[nodiscard]] py::object as_py_object() const;
};

} // namespace cyten
