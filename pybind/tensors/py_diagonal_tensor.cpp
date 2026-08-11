#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/diagonal_tensor.h>

#include "py_trampolines.hpp"

#include "../py_cyten_pybind11.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

namespace {

std::optional<std::vector<std::variant<int64, std::string>>>
optional_leg_order(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<std::variant<int64, std::string>> out;
    for (auto item : to_iterable(obj)) {
        if (py::isinstance<py::str>(item)) {
            out.emplace_back(item.cast<std::string>());
        } else {
            out.emplace_back(item.cast<int64>());
        }
    }
    return out;
}

} // namespace

void
bind_tensors_diagonal_tensor(py::module_& m)
{
    py::class_<DiagonalTensor, SymmetricTensor, PyDiagonalTensor, py::smart_holder> cls(
      m, "DiagonalTensor");
    cls.doc() = R"pydoc(
Special case of a :class:`SymmetricTensor` that is diagonal in the computational basis.

The domain and codomain of a diagonal tensor are the same and consist of a single leg::

|        │
|      ┏━┷━┓
|      ┃ D ┃
|      ┗━┯━┛
|        │

A diagonal tensor then is a map that is a multiple of the identity on each sector of the leg,
i.e. it is given by :math:`\bigoplus_a \lambda_a \eye_a`, where the sum goes over sectors
:math:`a` of the `leg` :math:`V = \bigoplus_a a`.

This is the natural type e.g. for singular values or eigenvalue and allows
:ref:`elementwise <diagonal_elementwise>` operations.

Parameters
----------
data
    The numerical data ("free parameters") comprising the tensor. type is backend-specific
leg: Space
    The single leg in both the domain and codomain
backend : TensorBackend
    The backend of the tensor.
labels: list[list[str | None]] | list[str | None] | None
    Specify the labels for the legs.
    Can either give two lists, one for the codomain, one for the domain.
    Or a single flat list for all legs in the order of the :attr:`legs`,
    such that ``[codomain_labels, domain_labels]`` is equivalent
    to ``[*codomain_legs, *reversed(domain_legs)]``.

.. _diagonal_elementwise:

Elementwise Functions
---------------------
A bunch of "elementwise" functions can be defined for diagonal tensors.
If a function can be defined as a power series in ``D`` and ``D.hc``, its action can be achieved
by applying that power series to the diagonal elements individually.
E.g. :func:`complex_conj`, :func:`sqrt`, :func:`exp` etc.
)pydoc";

    cls.def(py::init<TensorBackend::DataPtr, py::object, TensorBackend::Ptr, py::object>(),
            py::arg("data"),
            py::arg("leg"),
            py::arg("backend") = nullptr,
            py::arg("labels") = py::none());

    cls.def_property(
      "data",
      [](DiagonalTensor& self) -> py::object {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              return py::cast(NoSymmetryBackend::unwrap(self.data));
          }
          return py::cast(self.data);
      },
      [](DiagonalTensor& self, py::object obj) {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              self.data = NoSymmetryBackend::wrap(obj.cast<BlockBackend::BlockPtr>());
          } else {
              self.data = obj.cast<TensorBackend::DataPtr>();
          }
      });

    cls.def_property_readonly("leg", &DiagonalTensor::leg, "Return the single space that makes up to domain and codomain.");

    cls.def("test_sanity", &DiagonalTensor::test_sanity, "Perform sanity checks.");
    cls.def("verify_dtype", &DiagonalTensor::verify_dtype);

    cls.def_static("from_block_func",
                   &DiagonalTensor::from_block_func,
                   py::arg("func"),
                   py::arg("leg"),
                   py::arg("backend") = nullptr,
                   py::arg("labels") = py::none(),
                   py::arg("func_kwargs") = py::none(),
                   py::arg("shape_kw") = py::none(),
                   py::arg("dtype") = py::none(),
                   py::arg("device") = py::none());

    cls.def_static("from_dense_block",
                   &DiagonalTensor::from_dense_block,
                   py::arg("block"),
                   py::arg("leg"),
                   py::arg("backend") = nullptr,
                   py::arg("labels") = py::none(),
                   py::arg("dtype") = py::none(),
                   py::arg("tol") = 1e-6,
                   py::arg("device") = py::none(),
                   py::arg("understood_braiding") = false);

    cls.def_static(
      "from_diag_block",
      &DiagonalTensor::from_diag_block,
      py::arg("diag"),
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("tol") = 1e-6,
      R"pydoc(
Convert a dense 1D block containing the diagonal entries to a DiagonalTensor.

Parameters
----------
diag: Block-like
    The diagonal entries as a backend-specific block or some data that can be converted
    using :meth:`BlockBackend.as_block`. This includes e.g. nested python iterables
    or numpy arrays.
leg, backend, labels
    Arguments for constructor of :class:`DiagonalTensor`.
dtype: Dtype
    If given, `diag` is converted to this dtype.

See Also
--------
diagonal_as_block, diagonal_as_numpy
    Inverse methods that recover the `diag` entries.
)pydoc");

    cls.def_static(
      "from_eye",
      &DiagonalTensor::from_eye,
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Float64,
      py::arg("device") = py::none(),
      R"pydoc(
The identity map as a DiagonalTensor.

Parameters
----------
leg, backend, labels
    Arguments for constructor of :class:`DiagonalTensor`.
dtype: Dtype
    The dtype for the entries.
)pydoc");

    cls.def_static(
      "from_random_normal",
      &DiagonalTensor::from_random_normal,
      py::arg("leg"),
      py::arg("mean") = py::none(),
      py::arg("sigma") = 1.0,
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      R"pydoc(
Generate a sample from the complex normal distribution.

The probability density is

.. math ::
    p(T) \propto \mathrm{exp}\left[
        \frac{1}{2 \sigma^2} \mathrm{Tr} (T - \mathtt{mean}) (T - \mathtt{mean})^\dagger
    \right]

Parameters
----------
leg, backend, labels
    Arguments for constructor of :class:`DiagonalTensor`.
mean: DiagonalTensor, optional
    The mean of the distribution. ``None`` is equivalent to zero mean.
sigma: float
    The standard deviation of the distribution
dtype: Dtype
    The dtype for the entries.
)pydoc");

    cls.def_static(
      "from_random_uniform",
      &DiagonalTensor::from_random_uniform,
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      R"pydoc(
Generate a tensor with uniformly random block-entries.

The block entries, i.e. the free parameters of the tensor are drawn independently and
uniformly. If dtype is a real type, they are drawn from [-1, 1], if it is complex, real and
imaginary part are drawn independently from [-1, 1].

.. note ::
    This is not a well defined probability distribution on the space of symmetric tensors,
    since the meaning of the uniformly drawn numbers depends on both the choice of the
    basis and on the backend.

Parameters
----------
leg, backend, labels
    Arguments for constructor of :class:`DiagonalTensor`.
dtype: Dtype
    The dtype for the entries.
)pydoc");

    cls.def_static("from_sector_block_func",
                   &DiagonalTensor::from_sector_block_func,
                   py::arg("func"),
                   py::arg("leg"),
                   py::arg("backend") = nullptr,
                   py::arg("labels") = py::none(),
                   py::arg("func_kwargs") = py::none(),
                   py::arg("dtype") = py::none(),
                   py::arg("device") = py::none());

    cls.def_static(
      "from_tensor",
      &DiagonalTensor::from_tensor,
      py::arg("tens"),
      py::arg("tol") = 1e-12,
      R"pydoc(
Create DiagonalTensor from a Tensor.

Parameters
----------
tens : :class:`Tensor`
    Must have exactly two legs. Its diagonal entries ``tens[i, i]`` are used.
tol : float | None
    Tolerance for checking if the `tens` is actually diagonal, in the sense that any
    "off-diagonal" free parameters that should vanish are smaller than this by magnitude.
    Set to ``None`` to disable the check.
)pydoc");

    cls.def_static(
      "from_zero",
      &DiagonalTensor::from_zero,
      py::arg("leg"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      R"pydoc(
A zero tensor.

Parameters
----------
leg, backend, labels
    Arguments for constructor of :class:`DiagonalTensor`.
dtype: Dtype
    The dtype for the entries.
device: str
    The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
    the block backend.
)pydoc");

    cls.def_static("from_hdf5",
                   &DiagonalTensor::from_hdf5,
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   "Import DiagonalTensor from hdf5");

    cls.def("as_dtype", &DiagonalTensor::as_dtype, py::arg("dtype"));
    cls.def("as_SymmetricTensor",
            &DiagonalTensor::as_SymmetricTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none());
    cls.def("as_DiagonalTensor",
            &DiagonalTensor::as_DiagonalTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none());
    cls.def("copy",
            &DiagonalTensor::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none());
    cls.def("diagonal", &DiagonalTensor::diagonal, py::arg("check_offdiagonal") = false);
    cls.def("diagonal_as_block", &DiagonalTensor::diagonal_as_block, py::arg("dtype") = py::none());
    cls.def("diagonal_as_numpy", &DiagonalTensor::diagonal_as_numpy, py::arg("numpy_dtype") = py::none());
    cls.def("elementwise_almost_equal",
            &DiagonalTensor::elementwise_almost_equal,
            py::arg("other"),
            py::arg("rtol") = 1e-5,
            py::arg("atol") = 1e-8);
    cls.def("_elementwise_unary",
            &DiagonalTensor::_elementwise_unary,
            py::arg("func"),
            py::arg("func_kwargs") = py::none(),
            py::arg("maps_zero_to_zero") = false);
    cls.def("_elementwise_binary",
            &DiagonalTensor::_elementwise_binary,
            py::arg("other"),
            py::arg("func"),
            py::arg("func_kwargs") = py::none(),
            py::arg("partial_zero_is_zero") = false);
    cls.def("_binary_operand",
            &DiagonalTensor::_binary_operand,
            py::arg("other"),
            py::arg("func"),
            py::arg("operand"),
            py::arg("return_NotImplemented") = false,
            py::arg("right") = false);
    cls.def("_get_item", &DiagonalTensor::_get_item, py::arg("idx"));
    cls.def("all", &DiagonalTensor::all, "For a bool dtype, if all values are True. Raises for other dtypes.");
    cls.def("any", &DiagonalTensor::any, "For a bool dtype, if any value is True. Raises for other dtypes.");
    cls.def("max", &DiagonalTensor::max);
    cls.def("min", &DiagonalTensor::min);
    cls.def("move_to_device", &DiagonalTensor::move_to_device, py::arg("device"));
    cls.def("to_backend",
            &DiagonalTensor::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none());
    cls.def(
      "to_dense_block",
      [](DiagonalTensor& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false);
    cls.def("save_hdf5",
            &DiagonalTensor::save_hdf5,
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            "Export DiagonalTensor to hdf5 such that it can be re-imported with from_hdf5");

    // Elementwise dunders
    cls.def("__abs__", &DiagonalTensor::abs);
    cls.def(
      "__bool__",
      [](DiagonalTensor& self) {
          auto tensors_mod = py::module_::import("cyten.tensors._tensors");
          if (self.dtype == Dtype::Bool && tensors_mod.attr("is_scalar")(self).cast<bool>()) {
              return tensors_mod.attr("item")(self).cast<bool>();
          }
          throw std::invalid_argument(
            "The truth value of a non-scalar DiagonalTensor is ambiguous. Use a.any() or a.all()");
      });

    auto bind_binop = [&](char const* dunder, char const* op, bool right = false) {
        std::string dunder_s = dunder;
        cls.def(
          dunder,
          [dunder_s, op, right](DiagonalTensor& self, py::object other) {
              auto tensors_mod = py::module_::import("cyten.tensors._tensors");
              if (py::isinstance(other, tensors_mod.attr("Tensor")) || py::isinstance<Tensor>(other)) {
                  if (dunder_s == "__add__" || dunder_s == "__radd__") {
                      return tensors_mod.attr("linear_combination")(1.0, py::cast(self), 1.0, other);
                  }
                  if (dunder_s == "__sub__") {
                      return tensors_mod.attr("linear_combination")(1.0, py::cast(self), -1.0, other);
                  }
                  if (dunder_s == "__rsub__") {
                      return tensors_mod.attr("linear_combination")(1.0, other, -1.0, py::cast(self));
                  }
              }
              py::object op_func = py::module_::import("operator").attr(op);
              return self._binary_operand(other, op_func, op, /*return_NotImplemented=*/true, right);
          },
          py::arg("other"));
    };
    bind_binop("__add__", "add");
    bind_binop("__radd__", "add", true);
    bind_binop("__sub__", "sub");
    bind_binop("__rsub__", "sub", true);
    bind_binop("__mul__", "mul");
    bind_binop("__rmul__", "mul", true);
    bind_binop("__truediv__", "truediv");
    bind_binop("__rtruediv__", "truediv", true);
    bind_binop("__pow__", "pow");
    bind_binop("__rpow__", "pow", true);
    bind_binop("__lt__", "lt");
    bind_binop("__le__", "le");
    bind_binop("__gt__", "gt");
    bind_binop("__ge__", "ge");

    // --- Identity ---
    py::class_<Identity, DiagonalTensor, py::smart_holder> id_cls(m, "Identity");
    id_cls.doc() = R"pydoc(
Special case of a :class:`DiagonalTensor` that is exactly the identity map on its leg.
)pydoc";

    id_cls.def(py::init<py::object, TensorBackend::Ptr, std::optional<Dtype>, std::optional<std::string>, py::object>(),
               py::arg("leg"),
               py::arg("backend") = nullptr,
               py::arg("dtype") = py::none(),
               py::arg("device") = py::none(),
               py::arg("labels") = py::none());

    id_cls.def("test_sanity", &Identity::test_sanity, "Perform sanity checks.");

    auto bind_unsupported = [&](char const* name) {
        id_cls.def_static(
          name,
          [name](py::args, py::kwargs) {
              Identity::unsupported_factory(name);
              return Identity::Ptr{};
          });
    };
    bind_unsupported("from_block_func");
    bind_unsupported("from_dense_block");
    bind_unsupported("from_diag_block");
    bind_unsupported("from_random_normal");
    bind_unsupported("from_random_uniform");
    bind_unsupported("from_sector_block_func");
    bind_unsupported("from_tensor");
    bind_unsupported("from_zero");

    id_cls.def_static("from_eye",
                      &Identity::from_eye,
                      py::arg("leg"),
                      py::arg("backend") = nullptr,
                      py::arg("labels") = py::none(),
                      py::arg("dtype") = Dtype::Float64,
                      py::arg("device") = py::none());

    id_cls.def("as_dtype", &Identity::as_dtype, py::arg("dtype"));
    id_cls.def("as_SymmetricTensor",
               &Identity::as_SymmetricTensor,
               py::arg("guarantee_copy") = false,
               py::arg("warning") = py::none());
    id_cls.def("as_DiagonalTensor",
               &Identity::as_DiagonalTensor,
               py::arg("guarantee_copy") = false,
               py::arg("warning") = py::none());
    id_cls.def("copy",
               &Identity::copy,
               py::arg("deep") = true,
               py::arg("device") = py::none(),
               py::arg("dtype") = py::none());
    id_cls.def("diagonal", &Identity::diagonal, py::arg("check_offdiagonal") = false);
    id_cls.def("diagonal_as_block", &Identity::diagonal_as_block, py::arg("dtype") = py::none());
    id_cls.def("diagonal_as_numpy", &Identity::diagonal_as_numpy, py::arg("numpy_dtype") = py::none());
    id_cls.def("elementwise_almost_equal",
               &Identity::elementwise_almost_equal,
               py::arg("other"),
               py::arg("rtol") = 1e-5,
               py::arg("atol") = 1e-8);
    id_cls.def("_elementwise_unary",
               &Identity::_elementwise_unary,
               py::arg("func"),
               py::arg("func_kwargs") = py::none(),
               py::arg("maps_zero_to_zero") = false);
    id_cls.def("_elementwise_binary",
               &Identity::_elementwise_binary,
               py::arg("other"),
               py::arg("func"),
               py::arg("func_kwargs") = py::none(),
               py::arg("partial_zero_is_zero") = false);
    id_cls.def("_binary_operand",
               &Identity::_binary_operand,
               py::arg("other"),
               py::arg("func"),
               py::arg("operand"),
               py::arg("return_NotImplemented") = false,
               py::arg("right") = false);
    id_cls.def("_get_item", &Identity::_get_item, py::arg("idx"));
    id_cls.def("all", &Identity::all);
    id_cls.def("any", &Identity::any);
    id_cls.def("max", &Identity::max);
    id_cls.def("min", &Identity::min);
    id_cls.def("move_to_device", &Identity::move_to_device, py::arg("device"));
    id_cls.def("to_backend",
               &Identity::to_backend,
               py::arg("backend"),
               py::arg("dtype") = py::none(),
               py::arg("device") = py::none());
    id_cls.def(
      "to_dense_block",
      [](Identity& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false);

    id_cls.def("__abs__", &Identity::abs);
    id_cls.def("__bool__", [](Identity& self) {
        auto tensors_mod = py::module_::import("cyten.tensors._tensors");
        if (self.dtype == Dtype::Bool && tensors_mod.attr("is_scalar")(self).cast<bool>()) {
            return true;
        }
        throw std::invalid_argument(
          "The truth value of a non-scalar DiagonalTensor is ambiguous. Use a.any() or a.all()");
    });
}

} // namespace cyten
