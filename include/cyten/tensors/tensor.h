#pragma once

#include <cyten/backends/backend_factory.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/config.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/symmetry.h>
#include <cyten/tensors/labels.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

/// Common base class for tensors.
///
/// The legs of the tensor (spaces of the domain or codomain) can be referred to either via
/// string labels (see :ref:`tensor_leg_labels` and the :attr:`labels` attribute) or via integer
/// positional indices. Both allow you to be ignorant of the distinction between domain and
/// codomain (see :ref:`tensors_as_maps`).
class Tensor
  : public LabelledLegs
  , public std::enable_shared_from_this<Tensor>
{
  public:
    using Ptr = std::shared_ptr<Tensor>;
    using CPtr = std::shared_ptr<const Tensor>;

    /// Dtypes rejected by :meth:`test_sanity` (Python ``_forbidden_dtypes``).
    static std::vector<Dtype> _forbidden_dtypes;

    /// Subclass-overridable forbidden dtypes (e.g. DiagonalTensor allows bool).
    [[nodiscard]] virtual std::vector<Dtype> const& forbidden_dtypes() const;

    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    TensorBackend::Ptr backend;
    Symmetry::Ptr symmetry;
    Dtype dtype = Dtype::Float64;
    std::string device;
    /// Dimension of each of the :attr:`legs` (codomain dims, then reversed domain dims).
    std::vector<float64> shape;

    /// Construct from flexible Python-style inputs (used by trampoline / bindings).
    Tensor(py::object codomain,
           py::object domain,
           TensorBackend::Ptr backend,
           py::object labels,
           Dtype dtype,
           std::string device);

    /// Construct from already-parsed C++ inputs (concrete subclasses).
    Tensor(TensorProduct::Ptr codomain,
           TensorProduct::Ptr domain,
           TensorBackend::Ptr backend,
           Symmetry::Ptr symmetry,
           LegLabels labels,
           Dtype dtype,
           std::string device);

    ~Tensor() override = default;

    Tensor(Tensor const&) = delete;
    Tensor& operator=(Tensor const&) = delete;
    Tensor(Tensor&&) = delete;
    Tensor& operator=(Tensor&&) = delete;

    /// Common input parsing for ``__init__`` methods of tensor classes.
    ///
    /// Also checks if they are compatible.
    ///
    /// Returns ``(codomain, domain, backend, symmetry)``.
    static std::tuple<TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, Symmetry::Ptr>
    _init_parse_args(py::object codomain, py::object domain, TensorBackend::Ptr backend);

    /// Parse the various allowed input formats for labels to the format of :attr:`labels`.
    ///
    /// Also supports a special case for input formats of endomorphisms (maps where domain
    /// and codomain coincide), where a flat list of labels for the codomain can be given,
    /// and the domain labels are auto-filled with the respective dual labels.
    static LegLabels _init_parse_labels(py::object labels,
                                        TensorProduct::Ptr const& codomain,
                                        TensorProduct::Ptr const& domain,
                                        bool is_endomorphism = false);

    /// Perform sanity checks.
    void test_sanity() const override;

    /// An ascii representation of the tensor.
    ///
    /// It shows the type, leg labels, leg dimensions and leg arrows.
    [[nodiscard]] virtual std::string ascii_diagram() const;

    /// Convert to a tensor of the given dtype on the same device.
    [[nodiscard]] virtual Ptr as_dtype(Dtype dtype) = 0;

    /// Convert to a :class:`SymmetricTensor`, if possible.
    ///
    /// Returns ``py::object`` until :class:`SymmetricTensor` exists in C++.
    [[nodiscard]] virtual py::object as_SymmetricTensor(
      bool guarantee_copy = false,
      std::optional<std::string> warning = std::nullopt) = 0;

    /// Copy the tensor.
    [[nodiscard]] virtual Ptr copy(bool deep = true,
                                   std::optional<std::string> device = std::nullopt,
                                   std::optional<Dtype> dtype = std::nullopt) = 0;

    /// Convert to a tensor with a different backend.
    [[nodiscard]] virtual Ptr to_backend(TensorBackend::Ptr backend,
                                         std::optional<Dtype> dtype = std::nullopt,
                                         std::optional<std::string> device = std::nullopt) = 0;

    /// Convert to a dense block of the backend, if possible.
    [[nodiscard]] virtual BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false) = 0;

    /// The labels that refer to legs in the codomain.
    [[nodiscard]] LegLabels codomain_labels() const;

    /// Hermitian adjoint. Calls free function :func:`dagger` once converted.
    [[nodiscard]] virtual Ptr dagger() const;

    /// The labels that refer to legs in the domain.
    [[nodiscard]] LegLabels domain_labels() const;

    /// If any of the legs is a pipe.
    [[nodiscard]] bool has_pipes() const;

    /// The :func:`dagger` (alias of :meth:`dagger`).
    [[nodiscard]] Ptr hc() const { return dagger(); }

    /// All legs of the tensor.
    ///
    /// These the spaces of the codomain, followed by the duals of the domain spaces
    /// *in reverse order*. Factors may be :class:`LegPipe` (not only :class:`Space`).
    [[nodiscard]] std::vector<py::object> legs() const;

    /// Move tensor to a given device, *in place*.
    virtual void move_to_device(std::string device) = 0;

    /// How many of the legs are in the codomain. See :ref:`tensors_as_maps`.
    [[nodiscard]] int64 num_codomain_legs() const;

    /// How many of the legs are in the domain. See :ref:`tensors_as_maps`.
    [[nodiscard]] int64 num_domain_legs() const;

    /// Number of flat legs in the codomain.
    [[nodiscard]] int64 num_codomain_flat_legs() const;

    /// Number of flat legs in the domain.
    [[nodiscard]] int64 num_domain_flat_legs() const;

    /// Total number of flat legs of self.
    [[nodiscard]] int64 num_flat_legs() const;

    /// The number of free parameters for the given legs.
    ///
    /// This is the dimension of the space of symmetry-preserving tensors with the given legs.
    [[nodiscard]] int64 num_parameters() const;

    /// The number of entries of a dense block representation of self.
    [[nodiscard]] int64 size() const;

    /// The :func:`transpose`. Calls free function once converted.
    [[nodiscard]] virtual Ptr T() const;

    /// Implementation of :meth:`__getitem__`.
    ///
    /// Can assume we have one non-negative integer index per leg.
    [[nodiscard]] virtual BlockBackend::Scalar _get_item(std::vector<int64> const& idx) = 0;

    /// Return the leg, as if it was moved to the codomain.
    ///
    /// May be a :class:`LegPipe` (not only a :class:`Space`).
    [[nodiscard]] py::object _as_codomain_leg(std::variant<int64, std::string> idx) const;

    /// Return the leg, as if it was moved to the domain.
    ///
    /// May be a :class:`LegPipe` (not only a :class:`Space`).
    [[nodiscard]] py::object _as_domain_leg(std::variant<int64, std::string> idx) const;

    /// Print :attr:`ascii_diagram` to stdout.
    void dbg() const;

    /// Parse a leg index or a leg label.
    ///
    /// Returns ``(in_domain, co_domain_idx, legs_idx)``.
    [[nodiscard]] std::tuple<bool, int64, int64> _parse_leg_idx(
      std::variant<int64, std::string> which_leg) const;

    [[nodiscard]] virtual std::vector<std::string> _repr_header_lines(std::string const& indent,
                                                                     bool use_symm_str = false) const;

    /// Basically ``self.legs[which_leg]``, but allows labels and multiple indices.
    ///
    /// May be a :class:`LegPipe` (not only a :class:`Space`).
    [[nodiscard]] py::object get_leg(std::variant<int64, std::string> which_leg) const;
    [[nodiscard]] std::vector<py::object> get_leg(
      std::vector<std::variant<int64, std::string>> const& which_legs) const;

    /// Get the specified leg from the domain or codomain.
    ///
    /// May be a :class:`LegPipe` (not only a :class:`Space`).
    [[nodiscard]] py::object get_leg_co_domain(std::variant<int64, std::string> which_leg) const;
    [[nodiscard]] std::vector<py::object> get_leg_co_domain(
      std::vector<std::variant<int64, std::string>> const& which_legs) const;

    /// Set the given labels, in-place. Return the modified instance.
    ///
    /// Accepts the flexible Python label formats via :meth:`_init_parse_labels`.
    Tensor& set_labels(py::object labels);
    Tensor& set_labels(LegLabels labels) override;

    /// Convert to a numpy array.
    [[nodiscard]] py::array to_numpy(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      py::object numpy_dtype = py::none(),
      bool understood_braiding = false);

    [[nodiscard]] virtual std::string __repr__() const;
    [[nodiscard]] virtual std::string __str__() const;

    /// Subclass short name for :meth:`ascii_diagram` (``Symm``, ``Diag``, …).
    [[nodiscard]] virtual std::string ascii_diagram_type_name() const;

    /// Python ``type(self).__name__`` for ``__repr__`` / ``__str__``.
    [[nodiscard]] virtual std::string class_name() const;
};

} // namespace cyten
