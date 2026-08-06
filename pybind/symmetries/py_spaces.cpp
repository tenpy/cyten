#include "py_cyten_pybind11.h"

#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/symmetry.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <optional>
#include <vector>

namespace cyten {

namespace {

Symmetry::Ptr
symmetry_from_python(py::object symmetry_obj)
{
    if (py::isinstance<Symmetry>(symmetry_obj)) {
        return symmetry_obj.cast<Symmetry::Ptr>();
    }
    return symmetry_obj.attr("as_Symmetry")().cast<Symmetry::Ptr>();
}

std::optional<std::vector<int64>>
perm_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    py::array arr = py::array::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::type_error("basis permutation must be a 1D integer sequence or None");
    }
    auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

py::array_t<int64>
perm_to_numpy(std::vector<int64> const& perm)
{
    py::array_t<int64> arr(static_cast<py::ssize_t>(perm.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < perm.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = perm[i];
    }
    return arr;
}

py::object
dim_to_python(float64 dim)
{
    if (std::isfinite(dim) && std::floor(dim) == dim) {
        return py::int_(static_cast<long long>(dim));
    }
    return py::float_(dim);
}

} // namespace

void
bind_spaces(py::module_& m)
{
    py::class_<Leg, PyLeg, py::smart_holder> cls(m,
                                                 "Leg",
                                                 R"pydoc(
                                                 Common base class for a single leg of a tensor.

                                                 A single leg on a tensor can either be an :class:`ElementarySpace` or, e.g. as the result
                                                 of combining legs, a :class:`LegPipe`.

                                                 Attributes
                                                 ----------
                                                 symmetry : Symmetry
                                                     The symmetry associated with this leg.
                                                 dim : int or float
                                                     The (quantum-)dimension of this leg.
                                                     Is integer if ``symmetry.can_be_dropped``, otherwise may be float.
                                                 is_dual : bool
                                                     A boolean flag that changes when the :attr:`dual` is taken. May or may not have additional
                                                     meaning and implications, depending on the concrete subclass of :class:`Leg`.
                                                 )pydoc");

    cls.def(
      py::init(
        [](py::object symmetry_obj, py::object dim_obj, bool is_dual, py::object basis_perm) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            float64 dim = py::float_(dim_obj).cast<float64>();
            return std::make_shared<PyLeg>(
              std::move(symmetry), dim, is_dual, perm_from_python(basis_perm));
        }),
      py::arg("symmetry"),
      py::arg("dim"),
      py::arg("is_dual"),
      py::arg("basis_perm") = py::none());

    cls.def_readwrite("symmetry", &Leg::symmetry)
      .def_property(
        "dim",
        [](Leg const& self) { return dim_to_python(self.dim); },
        [](Leg& self, py::object dim_obj) { self.dim = py::float_(dim_obj).cast<float64>(); })
      .def_readwrite("is_dual", &Leg::is_dual);

    cls
      .def_property_readonly("dual",
                             &Leg::dual,
                             R"pydoc(
                             The dual leg, that is obtained when bending this leg.
                             )pydoc")
      .def_property_readonly("is_trivial", &Leg::is_trivial)
      .def_property(
        "basis_perm",
        [](Leg const& self) { return perm_to_numpy(self.basis_perm()); },
        [](Leg& self, py::object basis_perm) {
            self.set_basis_perm(perm_from_python(basis_perm));
        },
        R"pydoc(
        Permutation that translates between public and internal basis order.

        For the inverse permutation, see :attr:`inverse_basis_perm`.

        The tensor manipulations of ``cyten`` benefit from choosing a canonical order for the
        basis of vector spaces. This attribute translates between the "public" order of the basis,
        in which e.g. the inputs to :meth:`from_dense_block` are interpreted to this internal order,
        such that ``public_basis[basis_perm] == internal_basis``.
        The internal order is such that the basis vectors are grouped and sorted by sector.
        We can translate indices as ``public_idx == basis_perm[internal_idx]``.
        Only available if ``symmetry.can_be_dropped``, as otherwise there is no well-defined
        notion of a basis.

        ``_basis_perm`` is the internal version which may be ``None`` if the permutation is trivial.
        See also :meth:`apply_basis_perm`.
        )pydoc")
      .def_property(
        "inverse_basis_perm",
        [](Leg const& self) { return perm_to_numpy(self.inverse_basis_perm()); },
        [](Leg& self, py::object inverse_basis_perm) {
            self.set_inverse_basis_perm(perm_from_python(inverse_basis_perm));
        },
        R"pydoc(
        Inverse permutation of :attr:`basis_perm`.
        )pydoc")
      .def_property_readonly("flat_legs",
                             &Leg::flat_legs,
                             R"pydoc(
                             Flatten until there are no more pipes.

                             See Also
                             --------
                             flat_spaces : Keeps :class:`AbelianLegPipes` nested.
                             )pydoc")
      .def_property_readonly("flat_spaces",
                             &Leg::flat_spaces,
                             R"pydoc(
                             Flatten until we get spaces.

                             See Also
                             --------
                             flat_legs : Also flattens :class:`AbelianLegPipes`.
                             )pydoc")
      .def_property_readonly("num_flat_legs",
                             &Leg::num_flat_legs,
                             R"pydoc(
                             The number of :attr:`flat_legs`.
                             )pydoc")
      .def_property_readonly("ascii_arrow",
                             &Leg::ascii_arrow,
                             R"pydoc(
                             A single character arrow, for use in tensor diagrams

                             Indicates (a) if the leg is a pipe and (b) for ElementarySpaces, the duality
                             )pydoc");

    cls
      .def("test_sanity",
           &Leg::test_sanity,
           R"pydoc(
           Perform sanity checks.
           )pydoc")
      .def("as_Space",
           &Leg::as_Space,
           R"pydoc(
           Convert to (an appropriate subclass of) :class:`Space`.
           )pydoc")
      .def("as_ElementarySpace",
           &Leg::as_ElementarySpace,
           py::arg("is_dual") = false,
           R"pydoc(
           Convert to an isomorphic :class:`ElementarySpace`
           )pydoc")
      .def("_flat_leg_permutation",
           &Leg::_flat_leg_permutation,
           py::arg("offset") = 0,
           R"pydoc(
           Leg permutation such that combining legs would be in C style.
           )pydoc")
      .def("__eq__", &Leg::operator==, py::arg("other"))
      .def("apply_basis_perm",
           &Leg::apply_basis_perm,
           py::arg("arr"),
           py::arg("axis") = 0,
           py::arg("inverse") = false,
           py::arg("pre_compose") = false,
           R"pydoc(
           Apply the basis_perm, i.e. form ``arr[self.basis_perm]``.

           This is the preferred method of accessing the permutation, since we may skip applying
           trivial permutations.

           Parameters
           ----------
           arr : numpy array
               The data to act on.
           axis : int
               Which axis of ``arr`` to act on. We use ``numpy.take(arr, perm, axis)``.
           inverse : bool
               If we should apply the inverse permutation :attr:`inverse_basis_perm` instead.
           pre_compose : bool
               If we should pre-compose instead, i.e. form ``basis_perm[arr]``.
               Note that in that case, `axis` is ignored.
           )pydoc");
}

} // namespace cyten
