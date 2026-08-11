#pragma once

#include <cyten/tensors/symmetric_tensor.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

/// Special case of a :class:`SymmetricTensor` that is diagonal in the computational basis.
///
/// The domain and codomain of a diagonal tensor are the same and consist of a single leg.
class DiagonalTensor : public SymmetricTensor
{
  public:
    using Ptr = std::shared_ptr<DiagonalTensor>;
    using CPtr = std::shared_ptr<const DiagonalTensor>;

    /// Empty — bool dtype is allowed for diagonal tensors (Python ``_forbidden_dtypes = []``).
    static std::vector<Dtype> _forbidden_dtypes;

    /// Construct from flexible Python-style inputs.
    DiagonalTensor(TensorBackend::DataPtr data,
                   py::object leg,
                   TensorBackend::Ptr backend = nullptr,
                   py::object labels = py::none());

    /// Construct from already-parsed C++ inputs.
    DiagonalTensor(TensorBackend::DataPtr data,
                   Space::Ptr leg,
                   TensorBackend::Ptr backend,
                   Symmetry::Ptr symmetry,
                   LegLabels labels);

    ~DiagonalTensor() override = default;

    [[nodiscard]] std::vector<Dtype> const& forbidden_dtypes() const override;

    void test_sanity() const override;
    void verify_dtype() const override;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    /// The single space that makes up the domain and codomain.
    [[nodiscard]] Space::Ptr leg() const;

    // --- factories ---

    [[nodiscard]] static Ptr from_block_func(py::function func,
                                             py::object leg,
                                             TensorBackend::Ptr backend = nullptr,
                                             py::object labels = py::none(),
                                             py::object func_kwargs = py::none(),
                                             std::optional<std::string> shape_kw = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_dense_block(py::object block,
                                              py::object leg,
                                              TensorBackend::Ptr backend = nullptr,
                                              py::object labels = py::none(),
                                              std::optional<Dtype> dtype = std::nullopt,
                                              float64 tol = 1e-6,
                                              std::optional<std::string> device = std::nullopt,
                                              bool understood_braiding = false);

    [[nodiscard]] static Ptr from_diag_block(py::object diag,
                                             py::object leg,
                                             TensorBackend::Ptr backend = nullptr,
                                             py::object labels = py::none(),
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt,
                                             float64 tol = 1e-6);

    [[nodiscard]] static Ptr from_eye(py::object leg,
                                      TensorBackend::Ptr backend = nullptr,
                                      py::object labels = py::none(),
                                      Dtype dtype = Dtype::Float64,
                                      std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_random_normal(py::object leg,
                                                py::object mean = py::none(),
                                                float64 sigma = 1.0,
                                                TensorBackend::Ptr backend = nullptr,
                                                py::object labels = py::none(),
                                                Dtype dtype = Dtype::Complex128,
                                                std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_random_uniform(py::object leg,
                                                 TensorBackend::Ptr backend = nullptr,
                                                 py::object labels = py::none(),
                                                 Dtype dtype = Dtype::Complex128,
                                                 std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_sector_block_func(
      py::function func,
      py::object leg,
      TensorBackend::Ptr backend = nullptr,
      py::object labels = py::none(),
      py::object func_kwargs = py::none(),
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_tensor(py::object tens, std::optional<float64> tol = 1e-12);

    [[nodiscard]] static Ptr from_zero(py::object leg,
                                       TensorBackend::Ptr backend = nullptr,
                                       py::object labels = py::none(),
                                       Dtype dtype = Dtype::Complex128,
                                       std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_hdf5(py::object hdf5_loader,
                                       py::object h5gr,
                                       std::string const& subpath);

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    // --- Tensor / SymmetricTensor overrides ---

    [[nodiscard]] Tensor::Ptr as_dtype(Dtype dtype) override;

    [[nodiscard]] py::object as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    [[nodiscard]] Tensor::Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) override;

    [[nodiscard]] virtual py::object diagonal(bool check_offdiagonal = false) const;

    [[nodiscard]] BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override;

    void move_to_device(std::string device) override;

    [[nodiscard]] Tensor::Ptr to_backend(TensorBackend::Ptr backend,
                                         std::optional<Dtype> dtype = std::nullopt,
                                         std::optional<std::string> device = std::nullopt) override;

    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) override;

    // --- Diagonal-specific API ---

    [[nodiscard]] virtual Ptr as_DiagonalTensor(bool guarantee_copy = false,
                                                std::optional<std::string> warning = std::nullopt);

    [[nodiscard]] virtual BlockBackend::BlockPtr diagonal_as_block(
      std::optional<Dtype> dtype = std::nullopt);

    [[nodiscard]] virtual py::array diagonal_as_numpy(py::object numpy_dtype = py::none());

    [[nodiscard]] virtual Ptr elementwise_almost_equal(py::object other,
                                                       float64 rtol = 1e-5,
                                                       float64 atol = 1e-8);

    [[nodiscard]] virtual Ptr _elementwise_unary(py::function func,
                                                 py::object func_kwargs = py::none(),
                                                 bool maps_zero_to_zero = false);

    [[nodiscard]] virtual Ptr _elementwise_binary(py::object other,
                                                  py::function func,
                                                  py::object func_kwargs = py::none(),
                                                  bool partial_zero_is_zero = false);

    /// Common implementation for the binary dunder methods ``__mul__`` etc.
    [[nodiscard]] virtual py::object _binary_operand(py::object other,
                                                     py::function func,
                                                     std::string const& operand,
                                                     bool return_NotImplemented = false,
                                                     bool right = false);

    [[nodiscard]] virtual bool all() const;
    [[nodiscard]] virtual bool any() const;

    [[nodiscard]] virtual BlockBackend::Scalar max() const;
    [[nodiscard]] virtual BlockBackend::Scalar min() const;

    [[nodiscard]] virtual Ptr abs() const;

  protected:
    [[nodiscard]] py::object as_py_object() override;
    [[nodiscard]] py::object as_py_object() const override;
};

/// Special case of a :class:`DiagonalTensor` that is exactly the identity map.
class Identity : public DiagonalTensor
{
  public:
    using Ptr = std::shared_ptr<Identity>;
    using CPtr = std::shared_ptr<const Identity>;

    /// Construct from flexible Python-style inputs.
    explicit Identity(py::object leg,
                      TensorBackend::Ptr backend = nullptr,
                      std::optional<Dtype> dtype = std::nullopt,
                      std::optional<std::string> device = std::nullopt,
                      py::object labels = py::none());

    /// Construct from already-parsed C++ inputs.
    Identity(Space::Ptr leg,
             TensorBackend::Ptr backend,
             Symmetry::Ptr symmetry,
             LegLabels labels,
             Dtype dtype,
             std::string device);

    ~Identity() override = default;

    void test_sanity() const override;

    [[nodiscard]] std::string class_name() const override;

    // Unsupported factories (TypeError in Python)
    static void unsupported_factory(char const* name);

    [[nodiscard]] static Ptr from_eye(py::object leg,
                                      TensorBackend::Ptr backend = nullptr,
                                      py::object labels = py::none(),
                                      Dtype dtype = Dtype::Float64,
                                      std::optional<std::string> device = std::nullopt);

    [[nodiscard]] Tensor::Ptr as_dtype(Dtype dtype) override;

    [[nodiscard]] py::object as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    [[nodiscard]] DiagonalTensor::Ptr as_DiagonalTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) override;

    [[nodiscard]] py::object _binary_operand(py::object other,
                                             py::function func,
                                             std::string const& operand,
                                             bool return_NotImplemented = false,
                                             bool right = false) override;

    [[nodiscard]] Tensor::Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) override;

    [[nodiscard]] py::object diagonal(bool check_offdiagonal = false) const override;

    [[nodiscard]] BlockBackend::BlockPtr diagonal_as_block(
      std::optional<Dtype> dtype = std::nullopt) override;

    [[nodiscard]] py::array diagonal_as_numpy(py::object numpy_dtype = py::none()) override;

    [[nodiscard]] DiagonalTensor::Ptr elementwise_almost_equal(py::object other,
                                                               float64 rtol = 1e-5,
                                                               float64 atol = 1e-8) override;

    [[nodiscard]] DiagonalTensor::Ptr _elementwise_unary(py::function func,
                                                         py::object func_kwargs = py::none(),
                                                         bool maps_zero_to_zero = false) override;

    [[nodiscard]] DiagonalTensor::Ptr _elementwise_binary(py::object other,
                                                          py::function func,
                                                          py::object func_kwargs = py::none(),
                                                          bool partial_zero_is_zero = false) override;

    [[nodiscard]] BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override;

    [[nodiscard]] bool all() const override;
    [[nodiscard]] bool any() const override;

    [[nodiscard]] BlockBackend::Scalar max() const override;
    [[nodiscard]] BlockBackend::Scalar min() const override;

    [[nodiscard]] DiagonalTensor::Ptr abs() const override;

    void move_to_device(std::string device) override;

    [[nodiscard]] Tensor::Ptr to_backend(TensorBackend::Ptr backend,
                                         std::optional<Dtype> dtype = std::nullopt,
                                         std::optional<std::string> device = std::nullopt) override;

    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) override;

  protected:
    [[nodiscard]] py::object as_py_object() override;
    [[nodiscard]] py::object as_py_object() const override;
};

} // namespace cyten
