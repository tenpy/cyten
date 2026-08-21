#include <cyten/tensors/direct_sum.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tensors/vector_like.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include <pybind11/operators.h>

#include "docstrings/tensors/vector_like.h"

#include <string>

namespace cyten {

namespace {

class PyVectorLike
  : public VectorLike
  , public py::trampoline_self_life_support
{
  public:
    using VectorLike::VectorLike;

    Ptr clone() const override { PYBIND11_OVERRIDE_PURE_NAME(Ptr, VectorLike, "copy", clone); }

    Dtype vector_dtype() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(Dtype, VectorLike, "dtype", vector_dtype);
    }

    std::string vector_device() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, VectorLike, "device", vector_device);
    }

    TensorBackend::Ptr vector_backend() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(TensorBackend::Ptr, VectorLike, "backend", vector_backend);
    }

    BlockBackend::Scalar vector_norm() const override
    {
        PYBIND11_OVERRIDE_PURE(BlockBackend::Scalar, VectorLike, vector_norm);
    }

    BlockBackend::Scalar vector_inner(CPtr other, bool do_dagger) const override
    {
        PYBIND11_OVERRIDE_PURE(BlockBackend::Scalar, VectorLike, vector_inner, other, do_dagger);
    }

    Ptr scaled(BlockBackend::Scalar const& a) const override
    {
        PYBIND11_OVERRIDE_PURE(Ptr, VectorLike, scaled, a);
    }

    Ptr axpy(BlockBackend::Scalar const& a, CPtr other) const override
    {
        PYBIND11_OVERRIDE_PURE(Ptr, VectorLike, axpy, a, other);
    }

    bool compatible_with(CPtr other) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, VectorLike, compatible_with, other);
    }
};

BlockBackend::Scalar
as_scalar_for(VectorLike const& vec, py::object value)
{
    return py::cast(vec.vector_backend()->block_backend)
      .attr("as_scalar")(value)
      .cast<BlockBackend::Scalar>();
}

py::object
py_cast_vector_like(VectorLike::Ptr p)
{
    if (!p) {
        return py::none();
    }
    if (auto t = std::dynamic_pointer_cast<Tensor>(p)) {
        return py::cast(std::move(t));
    }
    if (auto ds = std::dynamic_pointer_cast<DirectSum>(p)) {
        return py::cast(std::move(ds));
    }
    return py::cast(std::move(p));
}

} // namespace

void
bind_tensors_vector_like(py::module_& m)
{
    py::class_<VectorLike, PyVectorLike, py::smart_holder> cls(m, "VectorLike");
    cls.doc() = DOC(cyten, VectorLike);

    cls.attr("__array_ufunc__") = py::none();
    cls.def(py::init<>());

    cls.def(
      "copy",
      [](VectorLike const& self) { return py_cast_vector_like(self.clone()); },
      DOC(cyten, VectorLike, clone));
    cls.def_property_readonly(
      "dtype", &VectorLike::vector_dtype, DOC(cyten, VectorLike, vector_dtype));
    cls.def_property_readonly(
      "device", &VectorLike::vector_device, DOC(cyten, VectorLike, vector_device));
    cls.def_property_readonly(
      "backend", &VectorLike::vector_backend, DOC(cyten, VectorLike, vector_backend));
    cls.def("compatible_with",
            &VectorLike::compatible_with,
            py::arg("other"),
            DOC(cyten, VectorLike, compatible_with));
    cls.def(
      "scaled",
      [](VectorLike const& self, py::object a) {
          return py_cast_vector_like(self.scaled(as_scalar_for(self, a)));
      },
      py::arg("a"),
      DOC(cyten, VectorLike, scaled));
    cls.def(
      "axpy",
      [](VectorLike const& self, py::object a, VectorLike::CPtr other) {
          return py_cast_vector_like(self.axpy(as_scalar_for(self, a), std::move(other)));
      },
      py::arg("a"),
      py::arg("other"),
      DOC(cyten, VectorLike, axpy));

    cls
      .def(
        "__add__",
        [](VectorLike::Ptr self, VectorLike::CPtr other) {
            auto one = self->vector_backend()->block_backend->as_scalar(1.0);
            return py_cast_vector_like(self->axpy(one, std::move(other)));
        },
        py::is_operator())
      .def(
        "__sub__",
        [](VectorLike::Ptr self, VectorLike::CPtr other) {
            auto minus_one = self->vector_backend()->block_backend->as_scalar(-1.0);
            return py_cast_vector_like(other->axpy(minus_one, std::move(self)));
        },
        py::is_operator())
      .def(
        "__mul__",
        [](VectorLike::Ptr self, py::object other) {
            return py_cast_vector_like(self->scaled(as_scalar_for(*self, other)));
        },
        py::is_operator())
      .def(
        "__rmul__",
        [](VectorLike::Ptr self, py::object other) {
            return py_cast_vector_like(self->scaled(as_scalar_for(*self, other)));
        },
        py::is_operator())
      .def(
        "__truediv__",
        [](VectorLike::Ptr self, py::object other) {
            py::object inv;
            try {
                inv = py::float_(1.0) / other;
            } catch (py::error_already_set&) {
                throw py::value_error("VectorLike can only be divided by invertible scalars.");
            }
            return py_cast_vector_like(self->scaled(as_scalar_for(*self, inv)));
        },
        py::is_operator())
      .def(
        "__neg__",
        [](VectorLike::Ptr self) {
            auto minus_one = self->vector_backend()->block_backend->as_scalar(-1.0);
            return py_cast_vector_like(self->scaled(minus_one));
        },
        py::is_operator())
      .def("__pos__", [](VectorLike::Ptr self) { return self; });
}

} // namespace cyten
