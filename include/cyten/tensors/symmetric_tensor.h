#pragma once

#include <cyten/backends/abelian.h>
#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/backends/no_symmetry.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/sector.h>
#include <cyten/symmetries/trees.h>
#include <cyten/tensors/tensor.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyten {

/// A tensor that is symmetric, i.e. invariant under the symmetry.
///
/// .. note ::
///     The constructor is not particularly user friendly.
///     Consider using the various classmethods instead.
class SymmetricTensor : public Tensor
{
  public:
    using Ptr = std::shared_ptr<SymmetricTensor>;
    using CPtr = std::shared_ptr<const SymmetricTensor>;

    /// Backend-specific free parameters of tensors with the given symmetry.
    TensorBackend::DataPtr data;

    /// Construct from flexible Python-style inputs.
    SymmetricTensor(TensorBackend::DataPtr data,
                    py::object codomain,
                    py::object domain = py::none(),
                    TensorBackend::Ptr backend = nullptr,
                    py::object labels = py::none());

    /// Construct from already-parsed C++ inputs.
    SymmetricTensor(TensorBackend::DataPtr data,
                    TensorProduct::Ptr codomain,
                    TensorProduct::Ptr domain,
                    TensorBackend::Ptr backend,
                    Symmetry::Ptr symmetry,
                    LegLabels labels);

    ~SymmetricTensor() override = default;

    void test_sanity() const override;

    void verify_dtype() const;

    [[nodiscard]] std::string ascii_diagram_type_name() const override;
    [[nodiscard]] std::string class_name() const override;

    [[nodiscard]] static std::optional<Dtype> _parse_default_dtype(std::optional<Dtype> dtype,
                                                                  Symmetry::Ptr const& symmetry);

    // --- factories ---

    [[nodiscard]] static Ptr from_block_func(py::function func,
                                             py::object codomain,
                                             py::object domain = py::none(),
                                             TensorBackend::Ptr backend = nullptr,
                                             py::object labels = py::none(),
                                             py::object func_kwargs = py::none(),
                                             std::optional<std::string> shape_kw = std::nullopt,
                                             std::optional<Dtype> dtype = std::nullopt,
                                             std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_dense_block(py::object block,
                                              py::object codomain,
                                              py::object domain = py::none(),
                                              TensorBackend::Ptr backend = nullptr,
                                              py::object labels = py::none(),
                                              std::optional<Dtype> dtype = std::nullopt,
                                              std::optional<std::string> device = std::nullopt,
                                              float64 tol = 1e-6,
                                              bool understood_braiding = false);

    [[nodiscard]] static Ptr from_dense_block_trivial_sector(
      py::object vector,
      Space::Ptr space,
      TensorBackend::Ptr backend = nullptr,
      std::optional<std::string> device = std::nullopt,
      LegLabel label = std::nullopt);

    [[nodiscard]] static Ptr from_eye(py::object co_domain,
                                      TensorBackend::Ptr backend = nullptr,
                                      py::object labels = py::none(),
                                      Dtype dtype = Dtype::Complex128,
                                      std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_random_normal(py::object codomain,
                                                py::object domain = py::none(),
                                                py::object mean = py::none(),
                                                float64 sigma = 1.0,
                                                TensorBackend::Ptr backend = nullptr,
                                                py::object labels = py::none(),
                                                std::optional<Dtype> dtype = Dtype::Complex128,
                                                std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_random_uniform(py::object codomain,
                                                 py::object domain = py::none(),
                                                 TensorBackend::Ptr backend = nullptr,
                                                 py::object labels = py::none(),
                                                 Dtype dtype = Dtype::Complex128,
                                                 std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_sector_block_func(
      py::function func,
      py::object codomain,
      py::object domain = py::none(),
      TensorBackend::Ptr backend = nullptr,
      py::object labels = py::none(),
      py::object func_kwargs = py::none(),
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_sector_projection(
      py::object co_domain,
      Sector sector,
      TensorBackend::Ptr backend = nullptr,
      py::object labels = py::none(),
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_tree_pairs(
      std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
      py::object codomain,
      py::object domain = py::none(),
      TensorBackend::Ptr backend = nullptr,
      py::object labels = py::none(),
      std::optional<Dtype> dtype = std::nullopt,
      std::optional<std::string> device = std::nullopt);

    [[nodiscard]] static Ptr from_zero(py::object codomain,
                                       py::object domain = py::none(),
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

    [[nodiscard]] py::object diagonal(bool check_offdiagonal = false) const;

    [[nodiscard]] BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override;

    void move_to_device(std::string device) override;

    [[nodiscard]] Tensor::Ptr to_backend(TensorBackend::Ptr backend,
                                         std::optional<Dtype> dtype = std::nullopt,
                                         std::optional<std::string> device = std::nullopt) override;

    [[nodiscard]] BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) override;

    [[nodiscard]] BlockBackend::BlockPtr to_dense_block_trivial_sector() const;

  protected:
    /// ``py::cast`` of ``shared_from_this`` as SymmetricTensor (for backend APIs taking
    /// ``py::object``).
    [[nodiscard]] py::object as_py_object();
    [[nodiscard]] py::object as_py_object() const;
};

} // namespace cyten
