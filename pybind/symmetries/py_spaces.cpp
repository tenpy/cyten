#include "py_cyten_pybind11.h"

#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/symmetry.h>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <optional>
#include <string>
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

std::optional<std::vector<int64>>
multiplicities_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return perm_from_python(obj);
}

std::optional<std::string>
sector_order_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<std::string>();
}

py::array_t<float64>
f64_vector_to_numpy(std::vector<float64> const& v)
{
    py::array_t<float64> arr(static_cast<py::ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = v[i];
    }
    return arr;
}

py::object
slices_to_numpy(std::optional<std::vector<std::array<int64, 2>>> const& slices)
{
    if (!slices) {
        return py::none();
    }
    py::array_t<int64> arr({ static_cast<py::ssize_t>(slices->size()), py::ssize_t{ 2 } });
    auto buf = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < slices->size(); ++i) {
        buf(static_cast<py::ssize_t>(i), 0) = (*slices)[i][0];
        buf(static_cast<py::ssize_t>(i), 1) = (*slices)[i][1];
    }
    return arr;
}

std::optional<std::vector<int64>>
drop_which_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::str>(obj)) {
        auto s = obj.cast<std::string>();
        if (s == "all") {
            return std::nullopt;
        }
        throw py::value_error("which must be 'all', an int, or a list of ints");
    }
    if (py::isinstance<py::int_>(obj)) {
        return std::vector<int64>{ obj.cast<int64>() };
    }
    return obj.cast<std::vector<int64>>();
}

SectorArray
sector_array_from_python(py::handle obj, Symmetry const& symmetry)
{
    if (py::isinstance<SectorArray>(obj)) {
        return obj.cast<SectorArray>();
    }
    if (py::isinstance<Sector>(obj)) {
        return SectorArray::from_sector(obj.cast<Sector>());
    }
    auto arr = sector_array_from_numpy(obj);
    if (arr.sector_ind_len() == 0 && arr.size() == 0) {
        return SectorArray::empty(symmetry.sector_ind_len);
    }
    return arr;
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

    py::class_<Space, PySpace, py::smart_holder> space(m,
                                                       "Space",
                                                       R"pydoc(
                                                       Base class for symmetry spaces, see :class:`ElementarySpace` for the standard case.

                                                       A symmetry space is e.g. a vector space with a representation of a symmetry group.

                                                       Each symmetry space is equivalent to a direct sum of sectors, that
                                                       is :math:`V \cong \bigoplus_a \bigoplus_{\mu=1}{N_a} a`.
                                                       This is e.g. because the representation of the symmetry group is equivalent to a direct sum of
                                                       irreducible representations. From a different perspective, the vector space decomposes into
                                                       different charge sectors of the conserved charge. The unique sectors :math:`a` that appear in
                                                       the decomposition at least once, e.g. with `N_a > 0`, are stored in :attr:`sector_decomposition`
                                                       in a canonical order, while their multiplicities :math:`N_a` are stored in :attr:`multiplicities`.

                                                       Attributes
                                                       ----------
                                                       symmetry: Symmetry
                                                           The symmetry associated with this space.
                                                       sector_decomposition : 2D numpy array of int
                                                           The unique sectors that appear in the sector decomposition. A 2D array of integers with
                                                           axes [s, q] where s goes over different sectors and q over the (one or more) numbers needed
                                                           to label a sector. The sectors (to be precise, the rows ``sector_decomposition[i, :]``) are
                                                           unique. We use :attr:`multiplicities` to  account for duplicates.
                                                       sector_order : 'sorted' | 'dual_sorted' | None
                                                           Indicates if (and how) the :attr:`sector_decomposition` is sorted.
                                                           If ``'sorted'``, indicates that they are sorted by sector, i.e. such that
                                                           ``np.lexsort(sector_decomposition.T) == np.arange(num_sectors)``.
                                                           If ``'dual_sorted'``, indicated that the duals are sorted, i.e. such that
                                                           ``np.lexsort(dual_sectors(sector_decomposition).T) == np.arange(num_sectors)``.
                                                           If ``None``, no particular order is guaranteed.
                                                       multiplicities : 1D numpy array of int | None
                                                           How often each of the sectors in :attr:`sector_decomposition` appears. A 1D array of positive
                                                           integers with axis [s]. ``sector_decomposition[i, :]`` appears ``multiplicities[i]`` times.
                                                           ``None`` is equivalent to a sequence of ``1`` of appropriate length.
                                                       num_sectors : int
                                                           The number of sectors in the :attr:`sector_decomposition`.
                                                           This is the number of *unique* sectors, regardless of their multiplicity, and different
                                                           from the total number of sectors ``sum(multiplicities)``.
                                                       sector_dims : 1D array of int | None
                                                           If ``symmetry.can_be_dropped``, the integer dimension of each sector of the
                                                           :attr:`sector_decomposition`. Otherwise, not defined and set to ``None``.
                                                       sector_qdims : 1D array of float
                                                           The (quantum) dimension of each of the sectors. Unlike :attr:`sector_dims` this is always
                                                           defined, but may not always be integer.
                                                       dim : int | float
                                                           The total dimension. Is integer if ``symmetry.can_be_dropped``, otherwise may be float.
                                                       slices : 2D numpy array of int | None
                                                           For every sector ``sector_decomposition[n]``, the start ``slices[n, 0]`` and stop
                                                           ``slices[n, 1]`` of indices (in the *internal* basis order) that belong to this sector.
                                                           Conversely, ``basis_perm[slices[n, 0]:slices[n, 1]]`` are the elements of the public
                                                           basis that live in ``sector_decomposition[n]``. Only available if ``symmetry.can_be_dropped``.
                                                       )pydoc");

    space.def(py::init([](py::object symmetry_obj,
                          py::object sector_decomposition,
                          py::object multiplicities,
                          py::object sector_order) {
                  auto symmetry = symmetry_from_python(symmetry_obj);
                  return std::make_shared<PySpace>(
                    symmetry,
                    sector_array_from_python(sector_decomposition, *symmetry),
                    multiplicities_from_python(multiplicities),
                    sector_order_from_python(sector_order));
              }),
              py::arg("symmetry"),
              py::arg("sector_decomposition"),
              py::arg("multiplicities") = py::none(),
              py::arg("sector_order") = py::none());

    space.def_readwrite("symmetry", &Space::symmetry)
      .def_readwrite("sector_decomposition", &Space::sector_decomposition)
      .def_readwrite("sector_order", &Space::sector_order)
      .def_property(
        "multiplicities",
        [](Space const& self) { return perm_to_numpy(self.multiplicities); },
        [](Space& self, py::object obj) {
            auto m = multiplicities_from_python(obj);
            if (!m) {
                self.multiplicities.assign(static_cast<std::size_t>(self.num_sectors), 1);
            } else {
                self.multiplicities = std::move(*m);
            }
        })
      .def_readonly("num_sectors", &Space::num_sectors)
      .def_property_readonly("sector_dims",
                             [](Space const& self) -> py::object {
                                 if (!self.sector_dims) {
                                     return py::none();
                                 }
                                 return perm_to_numpy(*self.sector_dims);
                             })
      .def_property_readonly(
        "sector_qdims", [](Space const& self) { return f64_vector_to_numpy(self.sector_qdims); })
      .def_property_readonly("slices",
                             [](Space const& self) { return slices_to_numpy(self.slices); })
      .def_property_readonly("dim", [](Space const& self) { return dim_to_python(self.dim); });

    space
      .def_property_readonly("dual",
                             &Space::dual,
                             R"pydoc(
                             The dual space of the same type.

                             A dual space necessarily has a :attr:`sector_decomposition` which consists of the
                             :meth:`Symmetry.dual_sectors` of the original (though not necessarily in order).

                             Strictly speaking, this only guarantees to give one possible choice for a dual space and
                             might differ from *the* dual space by an irrelevant isomorphism.
                             )pydoc")
      .def_property_readonly("is_trivial",
                             &Space::is_trivial,
                             R"pydoc(
                             If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.

                             A trivial space is one-dimensional and transforms trivially under a symmetry group.
                             In category speak, it is (isomorphic to) the monoidal unit.
                             )pydoc");

    space
      .def("test_sanity",
           &Space::test_sanity,
           R"pydoc(
           Perform sanity checks.
           )pydoc")
      .def("__eq__", &Space::operator==, py::arg("other"))
      .def("is_isomorphic_to",
           &Space::is_isomorphic_to,
           py::arg("other"),
           R"pydoc(
           If the two spaces are isomorphic, i.e. have the same :attr:`sector_decomposition`.
           )pydoc")
      .def("is_subspace_of",
           &Space::is_subspace_of,
           py::arg("other"),
           R"pydoc(
           Whether self is (isomorphic to) a subspace of other.

           Per convention, self is never a subspace of other, if the :attr:`symmetry` are different.

           See Also
           --------
           ElementarySpace.from_largest_common_subspace
           )pydoc")
      .def("as_ElementarySpace",
           &Space::as_ElementarySpace,
           py::arg("is_dual") = false,
           R"pydoc(
           Convert to an isomorphic :class:`ElementarySpace`.
           )pydoc")
      .def(
        "change_symmetry",
        [](Space& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorMapFn map = [sector_map](SectorArray const& sectors) {
                return sector_map(sectors).cast<SectorArray>();
            };
            return self.change_symmetry(std::move(symmetry), std::move(map), injective);
        },
        py::arg("symmetry"),
        py::arg("sector_map"),
        py::arg("injective") = false,
        R"pydoc(
        Change the symmetry by specifying how the sectors change.

        .. note ::
            This interface assumes that a single sector of the old symmetry is mapped to a single
            sector of the new symmetry, i.e. that the functor that we realize here preserves
            simple objects. This does e.g. not cover the case of relaxing SU(2) to its U(1)
            subgroup.

        Parameters
        ----------
        symmetry : :class:`~cyten.groups.Symmetry`
            The symmetry of the new space
        sector_map : function (SectorArray,) -> (SectorArray,)
            A map of sectors (2D int arrays), such that ``new_sectors = sector_map(old_sectors)``.
            The map is assumed to cooperate with duality, i.e. we assume without checking that
            ``symmetry.dual_sectors(sector_map(old_sectors))`` is the same as
            ``sector_map(old_symmetry.dual_sectors(old_sectors))``.
        injective: bool
            If ``True``, the `sector_map` is assumed to be injective, i.e. produce a list of
            unique outputs, if the inputs are unique.

        Returns
        -------
        A space with the new symmetry. The order of the basis is preserved, but every
        basis element lives in a new sector, according to `sector_map`.
        )pydoc")
      .def(
        "drop_symmetry",
        [](Space& self, py::object which) {
            return self.drop_symmetry(drop_which_from_python(which));
        },
        py::arg("which") = "all",
        R"pydoc(
        Drop some or all symmetries.

        Parameters
        ----------
        which : 'all' | (list of) int
            If ``'all'`` (default) the entire symmetry is dropped and the result has ``no_symmetry``.
            An integer or list of integers indicates to drop the :attr:`~cyten.Symmetry.factors` with
            those indices.
        )pydoc")
      .def("as_Space", &Space::as_Space)
      .def(
        "sector_decomposition_where",
        [](Space const& self, Sector sector) -> py::object {
            auto idx = self.sector_decomposition_where(sector);
            if (!idx) {
                return py::none();
            }
            return py::int_(*idx);
        },
        py::arg("sector"),
        R"pydoc(
        Find the index of a given sector in the :attr:`sector_decomposition`.

        Returns
        -------
        idx : int | None
            If the `sector` is found the :attr:`sector_decomposition`, its index there such
            that ``sector_decomposition[idx] == sector``. Otherwise ``None``.
        )pydoc")
      .def("sector_multiplicity",
           &Space::sector_multiplicity,
           py::arg("sector"),
           R"pydoc(
           The multiplicity of a given sector in the :attr:`sector_decomposition`.
           )pydoc");

    py::class_<LegPipe, Leg, PyLegPipe, py::smart_holder> pipe(m,
                                                               "LegPipe",
                                                               R"pydoc(
                                                               A group of legs, i.e. resulting from :func:`~cyten.tensors.combine_legs`.

                                                               Note that the abelian backend defines a custom subclass.

                                                               The :attr:`dual` of a pipe is given by another :class:`LegPipe`, which consists of the
                                                               dual of each of the :attr:`legs`, *in reverse order*. We also flip the :attr:`is_dual`
                                                               attribute to keep track of that (but the attribute has no further meaning).

                                                               Attributes
                                                               ----------
                                                               legs
                                                                   The legs that were grouped, and that this pipe can be split into.
                                                               combine_cstyle : bool
                                                                   The leg pipe defines an order in which multi-indices (one per leg) are combined into
                                                                   a single index. This can either be C-style (where the index for the last leg is varied the
                                                                   fastest) or F-style (where the first index is varied the fastest). For compatibility with
                                                                   the default behavior of ``np.reshape``, we favor C-style. However, if the `legs` were in
                                                                   the domain (at the top) of a tensor before combining, the conventional leg order implies
                                                                   a reversal of their order in ``Tensor.legs``. Thus, pipes in the domain should have F-style
                                                                   combine. Consistent with this expectation, the style is flipped on taking the :attr:`dual`

                                                               See Also
                                                               --------
                                                               TensorProduct
                                                               )pydoc");

    pipe.def(py::init([](py::sequence legs_obj, bool is_dual, bool combine_cstyle) {
                 std::vector<Leg::Ptr> legs;
                 legs.reserve(static_cast<std::size_t>(legs_obj.size()));
                 for (py::handle item : legs_obj) {
                     legs.push_back(item.cast<Leg::Ptr>());
                 }
                 return std::make_shared<PyLegPipe>(std::move(legs), is_dual, combine_cstyle);
             }),
             py::arg("legs"),
             py::arg("is_dual") = false,
             py::arg("combine_cstyle") = true);

    pipe.def_readwrite("legs", &LegPipe::legs)
      .def_readonly("num_legs", &LegPipe::num_legs)
      .def_readwrite("combine_cstyle", &LegPipe::combine_cstyle);

    pipe.def("test_sanity", &LegPipe::test_sanity)
      .def("__eq__",
           [](LegPipe const& self, py::object other) -> py::object {
               if (!py::isinstance<LegPipe>(other)) {
                   return py::reinterpret_borrow<py::object>(Py_NotImplemented);
               }
               return py::cast(
                 self.operator==(static_cast<Leg const&>(other.cast<LegPipe const&>())));
           })
      .def("__getitem__", &LegPipe::operator[], py::arg("idx"))
      .def("__len__", [](LegPipe const& self) { return self.num_legs; })
      .def(
        "__iter__",
        [](LegPipe& self) { return py::make_iterator(self.legs.begin(), self.legs.end()); },
        py::keep_alive<0, 1>())
      .def("__repr__", [](LegPipe const& self) { return self.repr(); })
      .def("repr", &LegPipe::repr, py::arg("show_symmetry") = true, py::arg("one_line") = false);
}

} // namespace cyten
