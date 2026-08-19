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

std::vector<LegLabel>
to_leg_labels(py::object maybe_labels)
{
    std::vector<LegLabel> out;
    for (auto item : maybe_labels) {
        if (item.is_none()) {
            out.push_back(std::nullopt);
        } else {
            out.push_back(item.cast<std::string>());
        }
    }
    return out;
}

std::optional<LegLabels>
optional_vector_labels_from_py(py::object labels)
{
    if (labels.is_none()) {
        return std::nullopt;
    }
    return LegLabels(to_leg_labels(labels));
}

class PyLinearOperatorAdapter : public LinearOperator
{
  public:
    py::object original;

    explicit PyLinearOperatorAdapter(py::object op)
      : LinearOperator(op.attr("vector_legs").cast<std::vector<Leg::Ptr>>(),
                       op.attr("dtype").cast<Dtype>(),
                       optional_vector_labels_from_py(op.attr("vector_labels")))
      , original(std::move(op))
    {
    }

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override
    {
        auto res = original.attr("matvec")(py::cast(vec));
        return res.cast<VectorLike::Ptr>();
    }

    TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override
    {
        py::object arg = backend ? py::cast(backend) : py::none();
        auto res = original.attr("to_tensor")(arg);
        return res.cast<TensorPtr>();
    }

    LinearOperator::Ptr adjoint() override;
};

LinearOperator::Ptr
wrap_linear_operator(py::object op)
{
    try {
        return op.cast<LinearOperator::Ptr>();
    } catch (py::cast_error const&) {
        return std::make_shared<PyLinearOperatorAdapter>(std::move(op));
    }
}

LinearOperator::Ptr
PyLinearOperatorAdapter::adjoint()
{
    auto res = original.attr("adjoint")();
    return wrap_linear_operator(std::move(res));
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

    LinearOperator::Ptr adjoint() override
    {
        PYBIND11_OVERRIDE(LinearOperator::Ptr, LinearOperator, adjoint);
    }
};

} // namespace

void
bind_tensors_sparse(py::module_& m)
{
    py::class_<LinearOperator, PyLinearOperator, py::smart_holder> linear_operator(
      m, "LinearOperator");
    linear_operator.doc() = R"pydoc(Base class for a linear operator acting on cyten tensors.

Attributes
----------
vector_legs : list of Space
    The legs of tensors that this operator can act on.
vector_labels : list of str or None
    Labels of the vectors that this operator can act on, or ``None``.
dtype : Dtype
    The dtype of a full representation of the operator
acts_on : list of str
    Labels of the state on which the operator can act. NB: Class attribute.
)pydoc";
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
        [](LinearOperator& self, py::object labels) {
            self.vector_labels = optional_labels(labels);
        })
      .def_readwrite("dtype", &LinearOperator::dtype)
      .def("matvec",
           &LinearOperator::matvec,
           py::arg("vec"),
           R"pydoc(Apply the linear operator to a "vector".

We consider as vectors all :class:`~cyten.tensors.VectorLike` objects, including
:class:`~cyten.tensors.Tensor` (any rank) and :class:`~cyten.tensors.DirectSum`.
The result of `matvec` must live in the same vector space as `vec`.
)pydoc")
      .def("to_tensor",
           &LinearOperator::to_tensor,
           py::arg("backend") = nullptr,
           R"pydoc(Compute a full tensor representation of the linear operator.

Returns
-------
A tensor `t` with ``2 * N`` legs ``[a1, a2, ..., aN, aN*, ..., a2*, a1*]``, where
``[a1, a2, ..., aN]`` are the legs of the vectors this operator acts on.
S.t. ``self.matvec(vec)`` is equivalent to ``tdot(t, vec, [N, ..., 2*N-1], [N-1,...,0])``.
)pydoc")
      .def("to_matrix",
           &LinearOperator::to_matrix,
           py::arg("backend") = nullptr,
           R"pydoc(The tensor representation of self, reshaped to a matrix.)pydoc")
      .def("adjoint",
           &LinearOperator::adjoint,
           R"pydoc(Return the hermitian conjugate operator.

If `self` is hermitian, subclasses *can* choose to implement this to define
the adjoint operator of `self` to be `self`.
)pydoc");

    auto tensor_linear_operator_cls =
      py::class_<TensorLinearOperator, LinearOperator, py::smart_holder>(m,
                                                                         "TensorLinearOperator");
    tensor_linear_operator_cls.doc() =
      R"pydoc(Linear operator defined by a two-leg tensor with contractible legs.

The matvec is defined by contracting one of the two legs of this tensor with the vector.
This class is effectively a thin wrapper around tensors that allows them to be used as inputs
for sparse linear algebra routines, such as lanczos.

Parameters
----------
tensor :
    The tensor that is contracted with the vector on matvec
which_leg : int or str
    Which leg of `tensor` is to be contracted on matvec
)pydoc";
    tensor_linear_operator_cls
      .def(py::init<SymmetricTensorPtr, std::variant<int64, std::string>>(),
           py::arg("tensor"),
           py::arg("which_leg") = -1)
      .def_readwrite("tensor", &TensorLinearOperator::tensor)
      .def_readwrite("which_leg", &TensorLinearOperator::which_leg)
      .def_readwrite("other_leg", &TensorLinearOperator::other_leg)
      .def("matvec", &TensorLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &TensorLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &TensorLinearOperator::adjoint);

    auto linear_operator_wrapper_cls =
      py::class_<LinearOperatorWrapper, LinearOperator, py::smart_holder>(m,
                                                                          "LinearOperatorWrapper");
    linear_operator_wrapper_cls.doc() =
      R"pydoc(Base class for wrapping around another :class:`LinearOperator`.

The wrapped operator is stored as :attr:`original_operator`.
Use :meth:`unwrapped` to recover the innermost operator.

.. warning ::
    If there are multiple levels of wrapping operators, the order might be critical to get
    correct results; e.g. :class:`ProjectedLinearOperator` needs to be the outer-most
    wrapper to produce correct results and/or be efficient.

Parameters
----------
original_operator : :class:`LinearOperator`
    The original operator implementing the `matvec`.
)pydoc";
    linear_operator_wrapper_cls.def(py::init<LinearOperator::Ptr>(), py::arg("original_operator"))
      .def_readwrite("original_operator", &LinearOperatorWrapper::original_operator)
      .def("unwrapped",
           &LinearOperatorWrapper::unwrapped,
           py::arg("recursive") = true,
           R"pydoc(Return the original :class:`LinearOperator`

By default, unwrapping is done recursively, such that the result is *not* a `LinearOperatorWrapper`.
)pydoc")
      .def("__getattr__",
           [](LinearOperatorWrapper const& self, std::string const& name) {
               return py::getattr(py::cast(self.original_operator), name.c_str());
           })
      .def("matvec", &LinearOperatorWrapper::matvec, py::arg("vec"))
      .def("to_tensor", &LinearOperatorWrapper::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &LinearOperatorWrapper::adjoint);

    auto sum_linear_operator_cls =
      py::class_<SumLinearOperator, LinearOperatorWrapper, py::smart_holder>(m,
                                                                             "SumLinearOperator");
    sum_linear_operator_cls.doc() = R"pydoc(The sum of multiple operators.)pydoc";
    sum_linear_operator_cls
      .def(py::init([](LinearOperator::Ptr original_operator, py::args more_ops) {
               std::vector<LinearOperator::Ptr> more;
               more.reserve(more_ops.size());
               for (auto item : more_ops) {
                   more.push_back(item.cast<LinearOperator::Ptr>());
               }
               return std::make_shared<SumLinearOperator>(std::move(original_operator),
                                                          std::move(more));
           }),
           py::arg("original_operator"))
      .def_readwrite("more_operators", &SumLinearOperator::more_operators)
      .def("matvec", &SumLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &SumLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &SumLinearOperator::adjoint);

    auto shifted_linear_operator_cls =
      py::class_<ShiftedLinearOperator, LinearOperatorWrapper, py::smart_holder>(
        m, "ShiftedLinearOperator");
    shifted_linear_operator_cls.doc() =
      R"pydoc(A shifted operator, i.e. ``original_operator + shift * identity``.

This can be useful e.g. for better Lanczos convergence.)pydoc";
    shifted_linear_operator_cls
      .def(py::init([](py::object original_operator, py::object shift) {
               return std::make_shared<ShiftedLinearOperator>(
                 wrap_linear_operator(std::move(original_operator)), py::cast<complex128>(shift));
           }),
           py::arg("original_operator"),
           py::arg("shift"))
      .def_readwrite("shift", &ShiftedLinearOperator::shift)
      .def("matvec", &ShiftedLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &ShiftedLinearOperator::to_tensor, py::arg("backend") = nullptr)
      .def("adjoint", &ShiftedLinearOperator::adjoint);

    auto projected_linear_operator_cls =
      py::class_<ProjectedLinearOperator, LinearOperatorWrapper, py::smart_holder>(
        m, "ProjectedLinearOperator");
    projected_linear_operator_cls.doc() =
      R"pydoc(Projected version ``P H P + penalty * (1 - P)`` of an original operator ``H``.

The projector ``P = 1 - sum_o |o> <o|`` is given in terms of a set :attr:`ortho_vecs` of vectors
``|o>``.

The result is that all vectors from the subspace spanned by the :attr:`ortho_vecs` are eigenvectors
with eigenvalue `penalty`, while the eigensystem in the "rest" (i.e. in the orthogonal complement
to that subspace) remains unchanged.

This can be used to exclude the :attr:`ortho_vecs` from extremal eigensolvers, i.e. to find
the extremal eigenvectors among those that are orthogonal to the :attr:`ortho_vecs`.
In previous versions of tenpy, this behavior was achieved by an argument called `orthogonal_to`.
If this is done, at least for krylov-based eigensolvers such as lanczos, the penalty should be chosen
such that the `ortho_vecs` are somewhere in the bulk of the spectrum.
This is because lanczos has best convergence for the extremal eigenvalues and we want to converge
the solutions well, not the `ortho_vecs`.
E.g. for a typical Hamiltonian with a spectrum symmetric around zero, ``project_operator=True``
and ``penalty=None`` shifts the `ortho_vecs` to eigenvalue zero, thus fulfilling this criterion.
However, for operators with e.g. strictly positive spectrum, this prescription might fail.

Parameters
----------
original_operator : :class:`LinearOperator`-like
    The original operator, denoted ``H`` in the summary above.
ortho_vecs : list of :class:`~cyten.tensors.Tensor`
    The list of vectors spanning the projected space.
    They need not be orthonormal, as Gram-Schmidt is performed on them explicitly.
project_operator: bool
    If False (True per default), the projection of the operator ``H -> P H P`` is skipped
    and ``H + penalty * (1 - P)`` is represented instead.
penalty : complex, optional
    See summary above. Defaults to ``None``, which is equivalent to ``0.``.
)pydoc";
    projected_linear_operator_cls
      .def(py::init([](py::object original_operator,
                       std::vector<VectorLike::Ptr> ortho_vecs,
                       bool project_operator,
                       std::optional<complex128> penalty) {
               return std::make_shared<ProjectedLinearOperator>(
                 wrap_linear_operator(std::move(original_operator)),
                 std::move(ortho_vecs),
                 project_operator,
                 penalty);
           }),
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
          R"pydoc(Gram-Schmidt orthonormalization of a list of vectors.

Parameters
----------
vecs : list of :class:`~cyten.tensors.VectorLike`
    The list of vectors to be orthogonalized. All must be mutually compatible.
rcond : float
    Vectors of ``norm < rcond`` (after projecting out previous vectors) are discarded.

Returns
-------
list of :class:`~cyten.tensors.VectorLike`
    A list of orthonormal vectors which span the same space as `vecs`.
)pydoc");
}

} // namespace cyten
