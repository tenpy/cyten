#include <cyten/tensors/tensor.h>

#include "py_trampolines.hpp"

#include "../py_cyten_pybind11.h"

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include <cmath>
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

std::variant<int64, std::string>
as_leg_ref(py::handle obj)
{
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    return obj.cast<int64>();
}

} // namespace

void
bind_tensors_tensor(py::module_& m)
{
    py::class_<Tensor, LabelledLegs, PyTensor, py::smart_holder> cls(m, "Tensor");
    cls.doc() = R"pydoc(
Common base class for tensors.

TODO elaborate

The legs of the tensor (spaces of the domain or codomain) can be referred to either via
string labels (see :ref:`tensor_leg_labels` and the :attr:`labels` attribute) or via integer
positional indices. Both allow you to be ignorant of the distinction between domain and codomain
(see :ref:`tensors_as_maps`). For the integer indices, we refer to the position of a given legs
in the :attr:`Tensor.legs`. E.g. if ``codomain == [V, W, Z]`` and ``domain == [X, Y]``,
we have ``legs == [V, W, Z, Y.dual, X.dual]`` and indices ``1`` and ``-4`` both refer to the
``W`` leg in the codomain, while indices ``3`` and ``-2`` both refer to the ``X`` leg in the
domain. Graphically, the leg indices are arranged as follows::

|      11  10   9   8   7   6
|      ┏┷━━━┷━━━┷━━━┷━━━┷━━━┷┓
|      ┃          T          ┃
|      ┗┯━━━┯━━━┯━━━┯━━━┯━━━┯┛
|       0   1   2   3   4   5

A similar graphical representation is available as :attr:`Tensor.ascii_diagram` and can be
printed to stdout using :meth:`Tensor.dbg`.

Attributes
----------
codomain, domain : TensorProduct
    The domain and codomain of the tensor. See also :attr:`legs` and :ref:`tensors_as_maps`.
backend : TensorBackend
    The backend of the tensor.
symmetry : Symmetry
    The symmetry of the tensor.
num_legs : int
    The total number of legs in the domain and codomain.
dtype : Dtype
    The dtype of tensor entries. Note that a real dtype does not necessarily imply that
    the result of :meth:`to_dense_block` is real.
shape: tuple of int
    The dimension of each of the :attr:`legs`.
)pydoc";

    cls.def(py::init<py::object, py::object, TensorBackend::Ptr, py::object, Dtype, std::string>(),
            py::arg("codomain"),
            py::arg("domain"),
            py::arg("backend"),
            py::arg("labels"),
            py::arg("dtype"),
            py::arg("device"));

    cls.def_readwrite("codomain", &Tensor::codomain)
      .def_readwrite("domain", &Tensor::domain)
      .def_readwrite("backend", &Tensor::backend)
      .def_readwrite("symmetry", &Tensor::symmetry)
      .def_readwrite("dtype", &Tensor::dtype)
      .def_readwrite("device", &Tensor::device)
      .def_property(
        "shape",
        [](Tensor const& self) {
            // Match Space.dim / Leg.dim: whole-number dims as int (np.zeros etc.).
            py::list out;
            for (float64 d : self.shape) {
                if (std::isfinite(d) && std::floor(d) == d) {
                    out.append(py::int_(static_cast<long long>(d)));
                } else {
                    out.append(py::float_(d));
                }
            }
            return py::tuple(out);
        },
        [](Tensor& self, py::object shape_obj) {
            self.shape = shape_obj.cast<std::vector<float64>>();
        });

    cls.def_static("_init_parse_args",
                   &Tensor::_init_parse_args,
                   py::arg("codomain"),
                   py::arg("domain"),
                   py::arg("backend"),
                   R"pydoc(
Common input parsing for ``__init__`` methods of tensor classes.

Also checks if they are compatible.

Returns
-------
codomain, domain: TensorProduct
    The codomain and domain, converted to :class:`TensorProduct` if needed.
backend: TensorBackend
    The given backend, or the default backend compatible with `symmetry`.
symmetry: Symmetry
    The symmetry of the domain and codomain
)pydoc");

    cls.def_static("_init_parse_labels",
                   &Tensor::_init_parse_labels,
                   py::arg("labels"),
                   py::arg("codomain"),
                   py::arg("domain"),
                   py::arg("is_endomorphism") = false,
                   R"pydoc(
Parse the various allowed input formats for labels to the format of :attr:`labels`.

Also supports a special case for input formats of endomorphisms (maps where domain
and codomain coincide), where a flat list of labels for the codomain can be given,
and the domain labels are auto-filled with the respective dual labels.
)pydoc");

    cls.def("test_sanity", &Tensor::test_sanity, R"pydoc(Perform sanity checks.)pydoc");

    cls.def_property_readonly("ascii_diagram",
                              &Tensor::ascii_diagram,
                              R"pydoc(
An ascii representation of the tensor.

It shows the type, leg labels, leg dimensions and leg arrows.

Examples
--------
Consider the following example::

    |     123   123   132   123
    |       ^     v     v     ^
    |       a     b     c     d
    |   ┏━━━┷━━━━━┷━━━━━┷━━━━━┷━━━┓
    |   ┃          TEXT           ┃
    |   ┗┯━━━━━┯━━━━━┯━━━━━┯━━━━━┯┛
    |    i     h     g     f     e
    |    ^     v     ^     ^     v
    |   42   777    11     2     3

)pydoc");

    cls.def("as_dtype",
            &Tensor::as_dtype,
            py::arg("dtype"),
            R"pydoc(
Convert to a tensor of the given dtype on the same device.

Parameters
----------
dtype: Dtype
    The dtype of the result.
)pydoc");

    cls.def(
      "as_SymmetricTensor",
      [](Tensor& self, bool guarantee_copy, py::object warning) {
          std::optional<std::string> w;
          if (!warning.is_none()) {
              w = warning.cast<std::string>();
          }
          return self.as_SymmetricTensor(guarantee_copy, w);
      },
      py::arg("guarantee_copy") = false,
      py::arg("warning") = py::none(),
      R"pydoc(
Convert to a :class:`SymmetricTensor`, if possible.

Parameters
----------
guarantee_copy : bool
    If already a SymmetricTensor, we do *not* make a copy by default.
    Set this flag to ``True`` to guarantee a copy.
warning : str, optional
    If given, and if the conversion is non-trivial (i.e. if it was not already a
    SymmetricTensor to begin with), a warning with this text is issued.
)pydoc");

    cls.def(
      "copy",
      [](Tensor& self, bool deep, py::object device, py::object dtype) {
          std::optional<std::string> dev;
          std::optional<Dtype> dt;
          if (!device.is_none()) {
              dev = device.cast<std::string>();
          }
          if (!dtype.is_none()) {
              dt = dtype.cast<Dtype>();
          }
          return self.copy(deep, dev, dt);
      },
      py::arg("deep") = true,
      py::arg("device") = py::none(),
      py::arg("dtype") = py::none(),
      R"pydoc(
Copy the tensor.

Parameters
----------
deep: bool
    If the copy should be deep. A shallow copy is a new instance with the same data.
device: str, optional
    The device for the result. Per default, use the same device as `self`.
dtype: Dtype, optional
    The dtype of the result. Per default, use the same dtype as `self`.
)pydoc");

    cls.def(
      "to_backend",
      [](Tensor& self, TensorBackend::Ptr backend, py::object dtype, py::object device) {
          std::optional<Dtype> dt;
          std::optional<std::string> dev;
          if (!dtype.is_none()) {
              dt = dtype.cast<Dtype>();
          }
          if (!device.is_none()) {
              dev = device.cast<std::string>();
          }
          return self.to_backend(std::move(backend), dt, dev);
      },
      py::arg("backend"),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
Convert to a tensor with a different backend.

Parameters
----------
backend: TensorBackend
    The backend of the result.
dtype: Dtype, optional
    The dtype of the result. Per default, use the same dtype as `self`.
device: str, optional
    The device for the result. Per default, use the same device as `self`.
)pydoc");

    cls.def(
      "to_dense_block",
      [](Tensor& self, py::object leg_order, py::object dtype, bool understood_braiding) {
          std::optional<Dtype> dt;
          if (!dtype.is_none()) {
              dt = dtype.cast<Dtype>();
          }
          return self.to_dense_block(optional_leg_order(leg_order), dt, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false,
      R"pydoc(
Convert to a dense block of the backend, if possible.

This corresponds to "forgetting" the symmetry structure and is only possible if the
symmetry :attr:`Symmetry.can_be_dropped`.
The result is a backend-specific block, e.g. a numpy array if the block backend is a
:class:`NumpyBlockBackend` or a torch Tensor if the backend is a :class:`TorchBlockBackend`.

Parameters
----------
leg_order: list of (int | str), optional
    If given, the leg of the resulting block are permuted to match this leg order.
dtype: Dtype, optional
    If given, the result is converted to this dtype. Per default it has the :attr:`dtype`
    of the tensor.
understood_braiding : bool
    For symmetries with non-trivial (but symmetric) braiding, e.g. fermions, the resulting
    dense block does no longer capture the braiding statistics correctly. This means that
    :func:`permute_legs` is not consistently reproduced by e.g. ``numpy.transpose`` on
    the dense block representation. Permuting its legs would require e.g. explicit swap
    gates. When using the result, special care needs to be taken regarding the leg order.
    To avoid this pitfall, we raise an error by default. Set this flag to ``True`` to
    disable the error. It is then your responsibility to take care of leg orders and braids.
    See :mod:`cyten.testing.swap_gate_numpy` for manipulations on these dense blocks.
)pydoc");

    cls.def_property_readonly("codomain_labels",
                              &Tensor::codomain_labels,
                              R"pydoc(The labels that refer to legs in the codomain.)pydoc");
    cls.def_property_readonly(
      "dagger",
      [](py::object self) {
          return py::module_::import("cyten.tensors._tensors").attr("dagger")(self);
      },
      R"pydoc(
      The hermitian conjugate tensor, a.k.a the dagger of a tensor.
      
      For a tensor with one leg each in (co-)domain (i.e. a matrix), this coincides with
      the hermitian conjugate matrix :math:`(M^\dagger)_{i,j} = \bar{M}_{j, i}` .
      For a tensor ``A: W -> V`` the dagger is a map ``dagger(A): V -> W``.
      Graphically::
      
          |          e   d             a   b   c
          |          │   │             │   │   │
          |       ┏━━┷━━━┷━━┓         ┏┷━━━┷━━━┷┓
          |       ┃    A    ┃         ┃dagger(A)┃
          |       ┗┯━━━┯━━━┯┛         ┗━━┯━━━┯━━┛
          |        │   │   │             │   │
          |        a   b   c             e   d
      
      Where ``a, b, c, d, e`` denote the legs in to (co-)domain.
      
      Returns
      -------
      The hermitian conjugate tensor. Its legs and labels are::
      
          dagger(A).codomain == A.domain
          dagger(A).domain == A.codomain
          dagger(A).legs == [leg.dual for leg in reversed(A.legs)]
          dagger(A).labels == [_dual_leg_label(l) for l in reversed(A.labels)]
      
      Note that the resulting :attr:`Tensor.legs` only depend on the input :attr:`Tensor.legs`, not
      on their bipartition into domain and codomain.
      For labels, we toggle a duality marker, i.e. if ``A.labels == ['a', 'b', 'c', 'd*', 'e*']``,
      then ``dagger(A).labels == ['e', 'd', 'c*', 'b*','a*']``.
      )pydoc");
    cls.def_property_readonly("domain_labels",
                              &Tensor::domain_labels,
                              R"pydoc(The labels that refer to legs in the domain.)pydoc");
    cls.def_property_readonly("has_pipes",
                              &Tensor::has_pipes,
                              R"pydoc(If any of the legs is a pipe)pydoc");
    cls.def_property_readonly(
      "hc",
      [](py::object self) {
          return py::module_::import("cyten.tensors._tensors").attr("dagger")(self);
      },
      R"pydoc(The :func:`dagger`)pydoc");
    cls.def_property_readonly(
      "legs",
      &Tensor::legs,
      R"pydoc(
All legs of the tensor.

These the spaces of the codomain, followed by the duals of the domain spaces
*in reverse order*.
If we permute all legs to the codomain, we would get these spaces, i.e.::

    tensor.legs == tensor.permute_legs(codomain=range(tensor.num_legs)).codomain.spaces

See :ref:`tensors_as_maps`.
)pydoc");

    cls.def("move_to_device",
            &Tensor::move_to_device,
            py::arg("device"),
            R"pydoc(Move tensor to a given device, *in place*.)pydoc");

    cls.def_property_readonly(
      "num_codomain_legs",
      &Tensor::num_codomain_legs,
      R"pydoc(How many of the legs are in the codomain. See :ref:`tensors_as_maps`.)pydoc");
    cls.def_property_readonly(
      "num_domain_legs",
      &Tensor::num_domain_legs,
      R"pydoc(How many of the legs are in the domain. See :ref:`tensors_as_maps`.)pydoc");
    cls.def_property_readonly("num_codomain_flat_legs",
                              &Tensor::num_codomain_flat_legs,
                              R"pydoc(Number of flat legs in the codomain.)pydoc");
    cls.def_property_readonly("num_domain_flat_legs",
                              &Tensor::num_domain_flat_legs,
                              R"pydoc(Number of flat legs in the domain.)pydoc");
    cls.def_property_readonly("num_flat_legs",
                              &Tensor::num_flat_legs,
                              R"pydoc(Total number of flat legs of self.)pydoc");
    cls.def_property_readonly(
      "num_parameters",
      &Tensor::num_parameters,
      R"pydoc(
The number of free parameters for the given legs.

This is the dimension of the space of symmetry-preserving tensors with the given legs.
)pydoc");
    cls.def_property_readonly(
      "size",
      &Tensor::size,
      R"pydoc(
The number of entries of a dense block representation of self.

This is only defined if ``self.symmetry.can_be_dropped``.
In that case, it is the number of entries of :func:`to_dense_block`.
)pydoc");
    cls.def_property_readonly(
      "T",
      [](py::object self) {
          return py::module_::import("cyten.tensors._tensors").attr("transpose")(self);
      },
      R"pydoc(The :func:`transpose`.)pydoc");

    cls.def(
      "__getitem__",
      [](Tensor& self, py::object idx) {
          if (!self.symmetry->can_be_dropped()) {
              throw SymmetryError(std::format(
                "Can not access elements for tensor with symmetry {}", self.symmetry->repr()));
          }
          auto it = to_iterable(idx);
          std::vector<int64> parsed;
          for (auto item : it) {
              parsed.push_back(item.cast<int64>());
          }
          if (static_cast<int64>(parsed.size()) != self.num_legs) {
              throw py::index_error(std::format(
                "Expected {} indices (one per leg). Got {}", self.num_legs, parsed.size()));
          }
          for (std::size_t i = 0; i < parsed.size(); ++i) {
              parsed[i] = to_valid_idx(parsed[i], static_cast<int64>(self.shape[i]));
          }
          return self._get_item(parsed);
      },
      py::arg("idx"));

    cls.def("__setitem__",
            [](Tensor&, py::object, py::object) {
                throw py::type_error("Tensors do not support item assignment.");
            })
      .def("__eq__",
           [](Tensor& self, py::object) -> bool {
               throw py::type_error(
                 std::format("{} does not support == comparison. Use cyten.almost_equal instead.",
                             self.class_name()));
           })
      .def("__complex__",
           [](Tensor&) {
               throw py::type_error(
                 "complex() of a tensor is not defined. Use cyten.item() instead.");
           })
      .def("__float__", [](Tensor&) {
          throw py::type_error("float() of a tensor is not defined. Use cyten.item() instead.");
      });

    // Arithmetic / composition: defer to Python free functions until those are converted.
    cls.def("__add__",
            [](py::object self, py::object other) {
                return py::module_::import("cyten.tensors._tensors")
                  .attr("linear_combination")(1.0, self, 1.0, other);
            })
      .def("__sub__",
           [](py::object self, py::object other) {
               return py::module_::import("cyten.tensors._tensors")
                 .attr("linear_combination")(1.0, self, -1.0, other);
           })
      .def("__mul__",
           [](py::object self, py::object other) {
               return py::module_::import("cyten.tensors._tensors")
                 .attr("scalar_multiply")(other, self);
           })
      .def("__rmul__",
           [](py::object self, py::object other) {
               return py::module_::import("cyten.tensors._tensors")
                 .attr("scalar_multiply")(other, self);
           })
      .def("__truediv__",
           [](py::object self, py::object other) {
               py::object inv;
               try {
                   inv = py::float_(1.0) / other;
               } catch (py::error_already_set&) {
                   throw py::value_error("Tensor can only be divided by invertible scalars.");
               }
               return py::module_::import("cyten.tensors._tensors")
                 .attr("scalar_multiply")(inv, self);
           })
      .def("__neg__",
           [](py::object self) {
               return py::module_::import("cyten.tensors._tensors")
                 .attr("scalar_multiply")(-1, self);
           })
      .def("__pos__", [](Tensor::Ptr self) { return self; })
      .def("__matmul__", [](py::object self, py::object other) {
          return py::module_::import("cyten.tensors._tensors").attr("compose")(self, other);
      });

    cls.def("__repr__", &Tensor::__repr__).def("__str__", &Tensor::__str__);

    cls.def(
         "_as_codomain_leg",
         [](Tensor const& self, py::object idx) { return self._as_codomain_leg(as_leg_ref(idx)); },
         py::arg("idx"),
         R"pydoc(Return the leg, as if it was moved to the codomain.)pydoc")
      .def(
        "_as_domain_leg",
        [](Tensor const& self, py::object idx) { return self._as_domain_leg(as_leg_ref(idx)); },
        py::arg("idx"),
        R"pydoc(Return the leg, as if it was moved to the domain.)pydoc")
      .def("dbg", &Tensor::dbg)
      .def("_get_item",
           &Tensor::_get_item,
           py::arg("idx"),
           R"pydoc(
Implementation of :meth:`__getitem__`.

Can assume we have one non-negative integer index per leg.
)pydoc")
      .def(
        "_parse_leg_idx",
        [](Tensor const& self, py::object which_leg) {
            return self._parse_leg_idx(as_leg_ref(which_leg));
        },
        py::arg("which_leg"),
        R"pydoc(
Parse a leg index or a leg label.

Parameters
----------
idx: int | str
    An index referring to one of the :attr:`legs` *or* a label.

Returns
-------
in_domain: bool
    If the leg is in the domain.
co_domain_idx: int
    The index of the leg in the (co-)domain
legs_idx: int
    The index of the leg in :attr:`legs`. Same as input ``idx``, except
    it is guaranteed to be in ``range(num_legs)``.
)pydoc")
      .def("_repr_header_lines",
           &Tensor::_repr_header_lines,
           py::arg("indent"),
           py::arg("use_symm_str") = false);

    cls.def(
         "get_leg",
         [](Tensor const& self, py::object which_leg) -> py::object {
             if (!(py::isinstance<py::int_>(which_leg) || py::isinstance<py::str>(which_leg))) {
                 std::vector<std::variant<int64, std::string>> refs;
                 for (auto item : which_leg) {
                     refs.push_back(as_leg_ref(item));
                 }
                 py::list out;
                 for (auto const& leg : self.get_leg(refs)) {
                     out.append(leg);
                 }
                 return out;
             }
             return self.get_leg(as_leg_ref(which_leg));
         },
         py::arg("which_leg"),
         R"pydoc(Basically ``self.legs[which_leg]``, but allows labels and multiple indices.)pydoc")
      .def(
        "get_leg_co_domain",
        [](Tensor const& self, py::object which_leg) -> py::object {
            if (!(py::isinstance<py::int_>(which_leg) || py::isinstance<py::str>(which_leg))) {
                std::vector<std::variant<int64, std::string>> refs;
                for (auto item : which_leg) {
                    refs.push_back(as_leg_ref(item));
                }
                py::list out;
                for (auto const& leg : self.get_leg_co_domain(refs)) {
                    out.append(leg);
                }
                return out;
            }
            return self.get_leg_co_domain(as_leg_ref(which_leg));
        },
        py::arg("which_leg"),
        R"pydoc(
Get the specified leg from the domain or codomain.

This is the same as :meth:`get_leg` if the leg is in the codomain, and the respective
dual if the leg is in the domain.
)pydoc");

    cls.def(
         "set_labels",
         [](Tensor& self, py::object labels) -> Tensor& { return self.set_labels(labels); },
         py::arg("labels"),
         py::return_value_policy::reference,
         R"pydoc(Set the given labels, in-place. Return the modified instance.)pydoc")
      .def(
        "to_numpy",
        [](Tensor& self, py::object leg_order, py::object numpy_dtype, bool understood_braiding) {
            return self.to_numpy(optional_leg_order(leg_order), numpy_dtype, understood_braiding);
        },
        py::arg("leg_order") = py::none(),
        py::arg("numpy_dtype") = py::none(),
        py::arg("understood_braiding") = false,
        R"pydoc(Convert to a numpy array)pydoc");
}

} // namespace cyten
