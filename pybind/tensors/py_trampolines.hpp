#pragma once

#include <cyten/tensors/tensor.h>

#include <pybind11/pybind11.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

/// pybind11 trampoline for :class:`Tensor` so Python subclasses can override virtuals.
class PyTensor
  : public Tensor
  , public py::trampoline_self_life_support
{
  public:
    using Tensor::Tensor;

    std::string ascii_diagram() const override
    {
        PYBIND11_OVERRIDE(std::string, Tensor, ascii_diagram);
    }

    Ptr as_dtype(Dtype dtype) override
    {
        PYBIND11_OVERRIDE_PURE(Ptr, Tensor, as_dtype, dtype);
    }

    py::object as_SymmetricTensor(bool guarantee_copy,
                                  std::optional<std::string> warning) override
    {
        PYBIND11_OVERRIDE_PURE(
          py::object, Tensor, as_SymmetricTensor, guarantee_copy, warning);
    }

    Ptr copy(bool deep, std::optional<std::string> device, std::optional<Dtype> dtype) override
    {
        PYBIND11_OVERRIDE_PURE(Ptr, Tensor, copy, deep, device, dtype);
    }

    Ptr to_backend(TensorBackend::Ptr backend,
                   std::optional<Dtype> dtype,
                   std::optional<std::string> device) override
    {
        PYBIND11_OVERRIDE_PURE(Ptr, Tensor, to_backend, backend, dtype, device);
    }

    BlockBackend::BlockPtr to_dense_block(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
      std::optional<Dtype> dtype,
      bool understood_braiding) override
    {
        PYBIND11_OVERRIDE_PURE(
          BlockBackend::BlockPtr, Tensor, to_dense_block, leg_order, dtype, understood_braiding);
    }

    void move_to_device(std::string device) override
    {
        PYBIND11_OVERRIDE_PURE(void, Tensor, move_to_device, device);
    }

    BlockBackend::Scalar _get_item(std::vector<int64> const& idx) override
    {
        PYBIND11_OVERRIDE_PURE(BlockBackend::Scalar, Tensor, _get_item, idx);
    }

    Ptr dagger() const override
    {
        PYBIND11_OVERRIDE(Ptr, Tensor, dagger);
    }

    Ptr T() const override
    {
        PYBIND11_OVERRIDE(Ptr, Tensor, T);
    }

    std::string ascii_diagram_type_name() const override
    {
        PYBIND11_OVERRIDE(std::string, Tensor, ascii_diagram_type_name);
    }

    std::string class_name() const override
    {
        PYBIND11_OVERRIDE(std::string, Tensor, class_name);
    }

    std::string __repr__() const override
    {
        PYBIND11_OVERRIDE(std::string, Tensor, __repr__);
    }

    std::string __str__() const override
    {
        PYBIND11_OVERRIDE(std::string, Tensor, __str__);
    }

    std::vector<std::string> _repr_header_lines(std::string const& indent,
                                                bool use_symm_str) const override
    {
        PYBIND11_OVERRIDE(
          std::vector<std::string>, Tensor, _repr_header_lines, indent, use_symm_str);
    }

    void test_sanity() const override
    {
        PYBIND11_OVERRIDE(void, Tensor, test_sanity);
    }

    Tensor& set_labels(LegLabels labels) override
    {
        PYBIND11_OVERRIDE(Tensor&, Tensor, set_labels, labels);
    }
};

} // namespace cyten
