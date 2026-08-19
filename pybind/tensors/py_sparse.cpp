#include <cyten/tensors/sparse.h>

#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tensors/tensor.h>

#include "../py_cyten_pybind11.h"

#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

namespace cyten {

namespace {

std::optional<LegLabels>
optional_labels(py::object labels)
{
    if (labels.is_none()) {
        return std::nullopt;
    }
    return labels.cast<LegLabels>();
}

class PyLinearOperator
  : public LinearOperator
  , public py::trampoline_self_life_support
{
  public:
    using LinearOperator::LinearOperator;

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override
    {
        PYBIND11_OVERRIDE_PURE(VectorLike::Ptr, LinearOperator, matvec, vec);
    }

    TensorPtr to_tensor(TensorBackend::Ptr backend) override
    {
        PYBIND11_OVERRIDE_PURE(TensorPtr, LinearOperator, to_tensor, backend);
    }

    LinearOperator::Ptr adjoint() override { PYBIND11_OVERRIDE(LinearOperator::Ptr, LinearOperator, adjoint); }
};

} // namespace

void
bind_tensors_sparse(py::module_& m)
{
    py::class_<LinearOperator, PyLinearOperator, py::smart_holder> linear_operator(m, "LinearOperator");
    linear_operator.doc() = R"pydoc(Base class for a linear operator acting on cyten vectors.)pydoc";
    linear_operator.attr("acts_on") = LinearOperator::acts_on;

    linear_operator
      .def(py::init<std::vector<Leg::Ptr>, Dtype, std::optional<LegLabels>>(),
           py::arg("vector_legs"),
           py::arg("dtype"),
           py::arg("vector_labels") = py::none())
      .def_readwrite("vector_legs", &LinearOperator::vector_legs)
      .def_property(
        "vector_labels",
        [](LinearOperator const& self) -> py::object {
            if (!self.vector_labels.has_value()) {
                return py::none();
            }
            return py::cast(*self.vector_labels);
        },
        [](LinearOperator& self, py::object labels) { self.vector_labels = optional_labels(labels); })
      .def_readwrite("dtype", &LinearOperator::dtype)
      .def("matvec", &LinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &LinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("to_matrix", &LinearOperator::to_matrix, py::arg("backend") = nullptr)
      .def("adjoint", &LinearOperator::adjoint);

    py::class_<TensorLinearOperator, LinearOperator, py::smart_holder>(m, "TensorLinearOperator")
      .def(py::init<SymmetricTensorPtr, std::variant<int64, std::string>>(),
           py::arg("tensor"),
           py::arg("which_leg") = -1)
      .def_readwrite("tensor", &TensorLinearOperator::tensor)
      .def_readwrite("which_leg", &TensorLinearOperator::which_leg)
      .def_readwrite("other_leg", &TensorLinearOperator::other_leg)
      .def("matvec", &TensorLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &TensorLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &TensorLinearOperator::adjoint);

    py::class_<LinearOperatorWrapper, LinearOperator, py::smart_holder>(m, "LinearOperatorWrapper")
      .def(py::init<LinearOperator::Ptr>(), py::arg("original_operator"))
      .def_readwrite("original_operator", &LinearOperatorWrapper::original_operator)
      .def("unwrapped", &LinearOperatorWrapper::unwrapped, py::arg("recursive") = true)
      .def("__getattr__",
           [](LinearOperatorWrapper const& self, std::string const& name) {
               return py::getattr(py::cast(self.original_operator), name.c_str());
           })
      .def("matvec", &LinearOperatorWrapper::matvec, py::arg("vec"))
      .def("to_tensor", &LinearOperatorWrapper::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &LinearOperatorWrapper::adjoint);

    py::class_<SumLinearOperator, LinearOperatorWrapper, py::smart_holder>(m, "SumLinearOperator")
      .def(py::init([](LinearOperator::Ptr original_operator, py::args more_ops) {
               std::vector<LinearOperator::Ptr> more;
               more.reserve(more_ops.size());
               for (auto item : more_ops) {
                   more.push_back(item.cast<LinearOperator::Ptr>());
               }
               return std::make_shared<SumLinearOperator>(std::move(original_operator), std::move(more));
           }),
           py::arg("original_operator"))
      .def_readwrite("more_operators", &SumLinearOperator::more_operators)
      .def("matvec", &SumLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &SumLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &SumLinearOperator::adjoint);

    py::class_<ShiftedLinearOperator, LinearOperatorWrapper, py::smart_holder>(m, "ShiftedLinearOperator")
      .def(py::init([](LinearOperator::Ptr original_operator, py::object shift) {
               return std::make_shared<ShiftedLinearOperator>(std::move(original_operator),
                                                              py::cast<complex128>(shift));
           }),
           py::arg("original_operator"),
           py::arg("shift"))
      .def_readwrite("shift", &ShiftedLinearOperator::shift)
      .def("matvec", &ShiftedLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &ShiftedLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &ShiftedLinearOperator::adjoint);

    py::class_<ProjectedLinearOperator, LinearOperatorWrapper, py::smart_holder>(m, "ProjectedLinearOperator")
      .def(py::init<LinearOperator::Ptr,
                    std::vector<VectorLike::Ptr>,
                    bool,
                    std::optional<complex128>>(),
           py::arg("original_operator"),
           py::arg("ortho_vecs"),
           py::arg("project_operator") = true,
           py::arg("penalty") = py::none())
      .def_readwrite("ortho_vecs", &ProjectedLinearOperator::ortho_vecs)
      .def_readwrite("project_operator", &ProjectedLinearOperator::project_operator)
      .def_property(
        "penalty",
        [](ProjectedLinearOperator const& self) -> py::object {
            if (!self.penalty.has_value()) {
                return py::none();
            }
            return py::cast(*self.penalty);
        },
        [](ProjectedLinearOperator& self, py::object value) {
            if (value.is_none()) {
                self.penalty = std::nullopt;
                return;
            }
            self.penalty = py::cast<complex128>(value);
        })
      .def("matvec", &ProjectedLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &ProjectedLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &ProjectedLinearOperator::adjoint);

    m.def("gram_schmidt",
          &gram_schmidt,
          py::arg("vecs"),
          py::arg("rcond") = kGramSchmidtDefaultRcond,
          R"pydoc(Gram-Schmidt orthonormalization of a list of vectors.)pydoc");
}

} // namespace cyten
