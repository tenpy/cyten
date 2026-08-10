#include "py_cyten_pybind11.h"

#include "backends/casters.hpp"
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
    // empty list / sequence → empty SectorArray with correct sector_ind_len
    // (np.asarray([]) is 1D and rejected by sector_array_from_numpy)
    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj) &&
        !py::isinstance<py::array>(obj)) {
        auto seq = py::reinterpret_borrow<py::sequence>(obj);
        if (seq.size() == 0) {
            return SectorArray::empty(symmetry.sector_ind_len);
        }
    }
    auto arr = sector_array_from_numpy(obj);
    if (arr.sector_ind_len() == 0 && arr.size() == 0) {
        return SectorArray::empty(symmetry.sector_ind_len);
    }
    return arr;
}

Symmetry::Ptr
optional_symmetry_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return nullptr;
    }
    return symmetry_from_python(py::reinterpret_borrow<py::object>(obj));
}

SectorMapFn
sector_map_from_python(py::function sector_map)
{
    return
      [sector_map](SectorArray const& sectors) { return sector_map(sectors).cast<SectorArray>(); };
}

Sector
sector_from_python(py::handle obj)
{
    if (py::isinstance<Sector>(obj)) {
        return obj.cast<Sector>();
    }
    return sector_from_numpy(obj);
}

py::slice
index_slice_to_python(IndexSlice const& slc)
{
    return py::slice(py::int_(slc.start), py::int_(slc.stop), py::none());
}

std::vector<py::object>
objects_from_python(py::handle obj)
{
    std::vector<py::object> out;
    for (py::handle item : obj) {
        out.push_back(py::reinterpret_borrow<py::object>(item));
    }
    return out;
}

py::list
objects_to_python(std::vector<py::object> const& objects)
{
    py::list out;
    for (auto const& obj : objects) {
        out.append(obj);
    }
    return out;
}

void bind_elementary_space(py::module_& m);

void bind_tensor_product(py::module_& m);

void bind_abelian_leg_pipe(py::module_& m);

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
        "_basis_perm",
        [](Leg const& self) -> py::object {
            if (!self.has_custom_basis_perm()) {
                return py::none();
            }
            return perm_to_numpy(self.basis_perm());
        },
        [](Leg& self, py::object basis_perm) {
            self.set_basis_perm(perm_from_python(basis_perm));
        })
      .def_property(
        "inverse_basis_perm",
        [](Leg const& self) { return perm_to_numpy(self.inverse_basis_perm()); },
        [](Leg& self, py::object inverse_basis_perm) {
            self.set_inverse_basis_perm(perm_from_python(inverse_basis_perm));
        },
        R"pydoc(
        Inverse permutation of :attr:`basis_perm`.
        )pydoc")
      .def_property(
        "_inverse_basis_perm",
        [](Leg const& self) -> py::object {
            if (!self.has_custom_basis_perm()) {
                return py::none();
            }
            return perm_to_numpy(self.inverse_basis_perm());
        },
        [](Leg& self, py::object inverse_basis_perm) {
            self.set_inverse_basis_perm(perm_from_python(inverse_basis_perm));
        })
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
      .def("as_Space", &Space::as_Space,
      R"pydoc(
      Convert to (an appropriate subclass of) :class:`Space`.
      )pydoc")
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

    pipe.def("test_sanity", &LegPipe::test_sanity,
    R"pydoc(
    Perform sanity checks.
    )pydoc")
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

    bind_elementary_space(m);
    bind_tensor_product(m);
    bind_abelian_leg_pipe(m);
}

namespace {

void
bind_elementary_space(py::module_& m)
{
    py::class_<ElementarySpace, Space, Leg, PyElementarySpace, py::smart_holder> cls(
      m,
      "ElementarySpace",
      R"pydoc(
      A :class:`Space` that is defined as (the dual of) a direct sum of sectors.

      While every :class:`Space` is isomorphic to a direct sum of sectors, an :class:`ElementarySpace`
      is by definition *equal* to such a direct sum, or to the dual of such a sum. We distinguish
      "ket" spaces :math:`V_k := a_1 \oplus a_2 \oplus \dots \plus a_N` with ``is_dual=False`` and
      "bra" spaces :math:`V_b := [b_1 \oplus b_2 \oplus \dots \plus b_N]^*` with ``is_dual=True``.
      The listed sectors, :math:`\{a_n\}` for the ket space :math:`V_k` and the :math:`\{b_n\}`
      for the bra space, are the :attr:`defining_sectors` of the space. For a ket space, they coincide
      with the :attr:`sector_decomposition`, while for a bra space they are mutually dual, since
      we have :math:`V_b \cong \bar{b}_1 \oplus \bar{b}_2 \oplus \dots \plus \bar{b}_N`.

      We impose a canonical order of sectors, such that the :attr:`defining_sectors` are sorted.
      This in turn means that the :attr:`sector_order` is ``'sorted'`` for ket spaces and
      ``'dual_sorted'`` for bra spaces.

      If the symmetry :attr:`Symmetry.can_be_dropped`, there is a notion of a basis for the
      spaces. We demand the basis to be compatible with the symmetry, i.e. each basis vector
      needs to lie in one of the sectors of the symmetry. The *internal* basis order that results
      from demanding that the sectors are contiguous and sorted may, however, not be the desired
      basis order, e.g. for matrix representations.

      Parameters
      ----------
      symmetry, sectors, multiplicities, is_dual, basis_perm
          Like attributes of the same name, except nested sequences are allowed in place of arrays.

      Attributes
      ----------
      is_dual: bool
          If this is a ket space (``False``) or a bra space (``True``).
      defining_sectors: 2D array of int
          The defining sectors, see class docstring of :class:`ElementarySpace`.
          Is ``np.lexsort( .T)``-ed.
          The :attr:`sector_decomposition` is equal for ket spaces (``is_dual=False``) or given by
          the respective :meth:`~cyten.symmetries.Symmetry.dual_sectors` for bra spaces.
      )pydoc");

    cls.def(py::init([](py::object symmetry_obj,
                        py::object defining_sectors,
                        py::object multiplicities,
                        bool is_dual,
                        py::object basis_perm) {
                auto symmetry = symmetry_from_python(symmetry_obj);
                return std::make_shared<PyElementarySpace>(
                  symmetry,
                  sector_array_from_python(defining_sectors, *symmetry),
                  multiplicities_from_python(multiplicities),
                  is_dual,
                  perm_from_python(basis_perm));
            }),
            py::arg("symmetry"),
            py::arg("defining_sectors"),
            py::arg("multiplicities") = py::none(),
            py::arg("is_dual") = false,
            py::arg("basis_perm") = py::none());

    // symmetry and dim exist on both the Space and the Leg base; keep them in sync.
    cls
      .def_property(
        "symmetry",
        [](ElementarySpace const& self) { return self.Space::symmetry; },
        [](ElementarySpace& self, py::object symmetry_obj) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            self.Space::symmetry = symmetry;
            self.Leg::symmetry = std::move(symmetry);
        })
      .def_property(
        "dim",
        [](ElementarySpace const& self) { return dim_to_python(self.Space::dim); },
        [](ElementarySpace& self, py::object dim_obj) {
            auto const dim = py::float_(dim_obj).cast<float64>();
            self.Space::dim = dim;
            self.Leg::dim = dim;
        })
      .def_readwrite("defining_sectors", &ElementarySpace::defining_sectors)
      .def_property_readonly(
        "sectors_of_basis",
        [](ElementarySpace const& self) { return sector_array_to_numpy(self.sectors_of_basis()); },
        R"pydoc(
        The sector (from the :attr:`sector_decomposition`) of each basis vector.
        )pydoc")
      .def_property_readonly("dual",
                             &ElementarySpace::dual_es,
                             R"pydoc(
                             The dual space, i.e. the same sectors with opposite :attr:`is_dual`.
                             )pydoc");

    cls.def_static(
      "from_basis",
      [](py::object symmetry_obj, py::object sectors_of_basis) {
          auto symmetry = symmetry_from_python(symmetry_obj);
          return ElementarySpace::from_basis(
            symmetry, sector_array_from_python(sectors_of_basis, *symmetry));
      },
      py::arg("symmetry"),
      py::arg("sectors_of_basis"),
      R"pydoc(
      Create an ElementarySpace by specifying the sector of every basis element.

      This requires that the symmetry :attr:`~cyten.symmetries.Symmetry.can_be_dropped`, such
      that there is a useful notion of a basis.

      .. note ::
          Unlike :meth:`from_defining_sectors`, this method expects the same sector to be listed
          multiple times, if the sector is multi-dimensional.

      .. note ::
          This classmethod always creates ket-spaces with ``is_dual=False``.
          Use :attr:`dual` or :meth:`as_bra_space` to create bra spaces.

      Parameters
      ----------
      symmetry: Symmetry
          The symmetry associated with this space.
      sectors_of_basis : iterable of iterable of int
          Specifies the basis. ``sectors_of_basis[n]`` is the sector of the ``n``-th basis element.
          In particular, for a ``d`` dimensional sector, we expect an integer multiple of ``d``
          occurrences. They need not be contiguous though.

      See Also
      --------
      :attr:`sectors_of_basis`
          Reproduces the `sectors_of_basis` parameter.
      from_defining_sectors
          Similar to the constructor, but with fewer requirements.
      )pydoc");

    cls.def_static(
      "from_independent_symmetries",
      [](py::sequence independent_descriptions) {
          std::vector<ElementarySpace::Ptr> descriptions;
          descriptions.reserve(static_cast<std::size_t>(independent_descriptions.size()));
          for (py::handle item : independent_descriptions) {
              descriptions.push_back(item.cast<ElementarySpace::Ptr>());
          }
          return ElementarySpace::from_independent_symmetries(descriptions);
      },
      py::arg("independent_descriptions"),
      R"pydoc(
      Create an ElementarySpace with multiple independent symmetries.

      Parameters
      ----------
      independent_descriptions : list of :class:`ElementarySpace`
          Each entry describes the resulting :class:`ElementarySpace` in terms of *one* of
          the independent symmetries. Spaces with a :class:`NoSymmetry` are ignored.
      )pydoc");

    cls.def_static(
      "from_largest_common_subspace",
      [](py::args spaces_obj, bool is_dual) {
          std::vector<Space::Ptr> spaces;
          spaces.reserve(static_cast<std::size_t>(spaces_obj.size()));
          for (py::handle item : spaces_obj) {
              spaces.push_back(item.cast<Space::Ptr>());
          }
          return ElementarySpace::from_largest_common_subspace(spaces, is_dual);
      },
      py::arg("is_dual") = false,
      R"pydoc(
      The largest common subspace of a list of spaces.

      The largest :class:`ElementarySpace` that :meth:`is_subspace_of` all of the `spaces`.
      I.e. the :attr:`sector_decomposition` is given by the "sector-wise minimum" of all
      multiplicities of the `spaces`.

      See Also
      --------
      is_subspace_of
      )pydoc");

    cls.def_static(
      "from_null_space",
      [](py::object symmetry_obj, bool is_dual) {
          return ElementarySpace::from_null_space(symmetry_from_python(symmetry_obj), is_dual);
      },
      py::arg("symmetry"),
      py::arg("is_dual") = false,
      R"pydoc(
      The zero-dimensional space, i.e. the span of the empty set.
      )pydoc");

    cls.def_static(
      "from_defining_sectors",
      [](py::object symmetry_obj,
         py::object defining_sectors,
         py::object multiplicities,
         bool is_dual,
         py::object basis_perm,
         bool unique_sectors,
         bool return_sorting_perm) -> py::object {
          auto symmetry = symmetry_from_python(symmetry_obj);
          std::vector<std::size_t> sort;
          auto res = ElementarySpace::from_defining_sectors(
            symmetry,
            sector_array_from_python(defining_sectors, *symmetry),
            multiplicities_from_python(multiplicities),
            is_dual,
            perm_from_python(basis_perm),
            unique_sectors,
            return_sorting_perm ? &sort : nullptr);
          if (!return_sorting_perm) {
              return py::cast(res);
          }
          std::vector<int64> const sort_perm(sort.begin(), sort.end());
          return py::make_tuple(py::cast(res), perm_to_numpy(sort_perm));
      },
      py::arg("symmetry"),
      py::arg("defining_sectors"),
      py::arg("multiplicities") = py::none(),
      py::arg("is_dual") = false,
      py::arg("basis_perm") = py::none(),
      py::arg("unique_sectors") = false,
      py::arg("return_sorting_perm") = false,
      R"pydoc(
      Similar to the constructor, but with fewer requirements.

      .. note ::
          Unlike :meth:`from_basis`, this method expects a multi-dimensional sector to be listed
          only once to mean its entire multiplet of basis states.

      Parameters
      ----------
      symmetry: Symmetry
          The symmetry associated with this space.
      defining_sectors: 2D array_like of int
          Like the :attr:`defining_sectors` attribute, but can be in any order and may contain
          duplicates (see `unique_sectors`).
      multiplicities: 1D array_like of int, optional
          How often each of the `defining_sectors` appears. A 1D array of positive integers with
          axis [s]. ``defining_sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
          If not given, a multiplicity ``1`` is assumed for all `defining_sectors`.
      is_dual: bool
          If the result is a bra- or a ket space, like the attribute :attr:`is_dual`.
          Note that this changes the meaning of the `defining_sectors`.
      basis_perm: ndarray, optional
          The permutation from the desired public basis to the basis described by
          `defining_sectors` and `multiplicities`.
      unique_sectors: bool
          If ``True``, the `sectors` are assumed to be duplicate-free.
      return_sorting_perm: bool
          If ``True``, the permutation ``np.lexsort(sectors.T)`` is returned too.

      Returns
      -------
      space: ElementarySpace
          The new space
      sector_sort: 1D array, optional
          Only ``if return_sorting_perm``. The permutation that sorts the `defining_sectors`.
      )pydoc");

    cls.def_static(
      "from_sector_decomposition",
      [](py::object symmetry_obj,
         py::object sector_decomposition,
         py::object multiplicities,
         bool is_dual,
         py::object basis_perm,
         bool unique_sectors) {
          auto symmetry = symmetry_from_python(symmetry_obj);
          return ElementarySpace::from_sector_decomposition(
            symmetry,
            sector_array_from_python(sector_decomposition, *symmetry),
            multiplicities_from_python(multiplicities),
            is_dual,
            perm_from_python(basis_perm),
            unique_sectors);
      },
      py::arg("symmetry"),
      py::arg("sector_decomposition"),
      py::arg("multiplicities") = py::none(),
      py::arg("is_dual") = false,
      py::arg("basis_perm") = py::none(),
      py::arg("unique_sectors") = false,
      R"pydoc(
      Create a :class:`ElementarySpace` that has a given :attr:`sector_decomposition`.

      Parameters
      ----------
      symmetry: Symmetry
          The symmetry associated with this space.
      sector_decomposition: 2D array_like of int
          Like the :attr:`sector_decomposition` attribute, but can be in any order and may contain
          duplicates (see `unique_sectors`).
      multiplicities: 1D array_like of int, optional
          How often each of the `sector_decomposition` appears. A 1D array of positive integers
          with axis [s]. ``sector_decomposition[i_s, :]`` appears ``multiplicities[i_s]`` times.
          If not given, a multiplicity ``1`` is assumed for all `sector_decomposition`.
      is_dual: bool
          If the result is a bra- or a ket space, like the attribute :attr:`is_dual`.
      basis_perm: ndarray, optional
          The permutation from the desired public basis to the basis described by
          `sector_decomposition` and `multiplicities`.
      unique_sectors: bool
          If ``True``, the `sectors` are assumed to be duplicate-free.

      See Also
      --------
      from_defining_sectors
      )pydoc");

    cls.def_static(
      "from_trivial_sector",
      [](int64 dim, py::object symmetry_obj, bool is_dual, py::object basis_perm) {
          return ElementarySpace::from_trivial_sector(dim,
                                                      optional_symmetry_from_python(symmetry_obj),
                                                      is_dual,
                                                      perm_from_python(basis_perm));
      },
      py::arg("dim") = 1,
      py::arg("symmetry") = py::none(),
      py::arg("is_dual") = false,
      py::arg("basis_perm") = py::none(),
      R"pydoc(
      Create an ElementarySpace that lives in the trivial sector (i.e. it is symmetric).

      Parameters
      ----------
      dim : int
          The dimension of the space.
      symmetry : :class:`~cyten.Symmetry`
          The symmetry of the space. Defaults to ``no_symmetry``.
      is_dual : bool
          If the space should be bra or a ket space.
      )pydoc");

    cls
      .def("test_sanity",
           &ElementarySpace::test_sanity,
           R"pydoc(
           Perform sanity checks.
           )pydoc")
      .def("__repr__", [](ElementarySpace const& self) { return self.repr(); })
      .def("repr",
           &ElementarySpace::repr,
           py::arg("show_symmetry") = true,
           py::arg("one_line") = false)
      .def("__eq__",
           [](ElementarySpace const& self, py::object other) -> py::object {
               if (!py::isinstance<ElementarySpace>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(self.equals_es(other.cast<ElementarySpace const&>()));
           })
      .def("as_Space", &ElementarySpace::as_Space,
      R"pydoc(
      Convert to (an appropriate subclass of) :class:`Space`.
      )pydoc")
      .def("as_ElementarySpace", &ElementarySpace::as_ElementarySpace, py::arg("is_dual") = false,
      R"pydoc(
      Convert to an isomorphic :class:`ElementarySpace`.
      )pydoc")
      .def("as_ket_space",
           &ElementarySpace::as_ket_space,
           R"pydoc(
           The ket space (``is_dual=False``) isomorphic or equal to self.
           )pydoc")
      .def("as_bra_space",
           &ElementarySpace::as_bra_space,
           R"pydoc(
           The bra space (``is_dual=True``) isomorphic or equal to self.
           )pydoc")
      .def(
        "change_symmetry",
        [](ElementarySpace& self,
           py::object symmetry_obj,
           py::function sector_map,
           bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
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
        "direct_sum",
        [](ElementarySpace const& self, py::args others_obj) {
            std::vector<ElementarySpace::Ptr> others;
            others.reserve(static_cast<std::size_t>(others_obj.size()));
            for (py::handle item : others_obj) {
                others.push_back(item.cast<ElementarySpace::Ptr>());
            }
            return self.direct_sum(others);
        },
        R"pydoc(
        Form the direct sum (i.e. stacking).

        The basis of the new space results from concatenating the individual bases.

        Spaces must have the same symmetry and is_dual.
        The result is a space with the same symmetry and is_dual, whose sectors are those
        that appear in any of the spaces and multiplicities are the sum of the multiplicities
        in each of the spaces.
        )pydoc")
      .def(
        "drop_symmetry",
        [](ElementarySpace& self, py::object which) {
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
      .def(
        "parse_index",
        [](ElementarySpace const& self, int64 idx) {
            auto const [sector_idx, multiplicity_idx] = self.parse_index(idx);
            return py::make_tuple(sector_idx, multiplicity_idx);
        },
        py::arg("idx"),
        R"pydoc(
        Utility function to translate an index.

        Parameters
        ----------
        idx : int
            An index of the leg, labelling an element of the public computational basis of self.

        Returns
        -------
        sector_idx : int
            The index of the corresponding sector, indicating that the `idx`-th basis element
            lives in ``self.sector_decomposition[sector_idx]``.
        multiplicity_idx : int
            The index "within the sector", in
            ``range(sector_dim * self.multiplicities[sector_index])``.
        )pydoc")
      .def("idx_to_sector", &ElementarySpace::idx_to_sector, py::arg("idx"))
      .def(
        "take_slice",
        [](ElementarySpace& self, py::object blockmask) {
            return self.take_slice(py::array::ensure(blockmask));
        },
        py::arg("blockmask"),
        R"pydoc(
        Take a "slice" of the leg, keeping only some of the basis states.

        Parameters
        ----------
        blockmask : 1D array-like of bool
            For every basis state of self, in the public basis order,
            if it should be kept (``True``) or discarded (``False``).
        )pydoc")
      .def("with_opposite_duality",
           &ElementarySpace::with_opposite_duality,
           R"pydoc(
           A space isomorphic to self with opposite ``is_dual`` attribute.
           )pydoc")
      .def("with_is_dual",
           &ElementarySpace::with_is_dual,
           py::arg("is_dual"),
           R"pydoc(
           A space isomorphic to self with given ``is_dual`` attribute.
           )pydoc")
      .def("save_hdf5",
           &ElementarySpace::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &ElementarySpace::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

void
bind_tensor_product(py::module_& m)
{
    py::class_<TensorProduct, Space, PyTensorProduct, py::smart_holder> cls(m,
                                                                            "TensorProduct",
                                                                            R"pydoc(
                                                                            Represents a tensor product of :class:`Spaces`\ s, e.g. the (co-)domain of a tensor.

                                                                            Attributes
                                                                            ----------
                                                                            factors : list[Space | LegPipe]
                                                                                The factors in the tensor product, e.g. some of the legs of a tensor.
                                                                            num_factors : int
                                                                                The number of :attr:`factors`.
                                                                            _sector_decomposition, _multiplicities
                                                                                If the sectors, multiplicities are already known, recomputation can be skipped.
                                                                                Warning: If given, they are not checked for correctness!

                                                                            See Also
                                                                            --------
                                                                            LegPipe
                                                                                A :class:`LegPipe` has the same mathematical idea as the :class:`TensorProduct`.
                                                                                There are two main differences:
                                                                                Firstly, for a :class:`TensorProduct`, we compute the :attr:`sector_decomposition`, which
                                                                                we do not do for a :class`LegPipe`. This is reflected in the fact that only
                                                                                :class:`TensorProduct`s are :class:`Space`s, while :class:`LegPipe`s are not.
                                                                                Secondly, we only keep track of duality with an explicit flag for :class:`Leg`s, to have
                                                                                arrows on our tensor legs. A :class:`TensorProduct` has no ``is_dual`` attribute.
                                                                            )pydoc");

    cls.def(py::init([](py::iterable factors_obj,
                        py::object symmetry_obj,
                        py::object sector_decomposition,
                        py::object multiplicities) {
                auto factors = objects_from_python(factors_obj);
                auto symmetry = optional_symmetry_from_python(symmetry_obj);
                std::optional<SectorArray> sectors;
                if (!sector_decomposition.is_none()) {
                    auto sym = symmetry;
                    if (!sym && !factors.empty()) {
                        sym = symmetry_from_python(factors.front().attr("symmetry"));
                    }
                    if (!sym) {
                        throw py::value_error("If spaces is empty, the symmetry arg is required.");
                    }
                    sectors = sector_array_from_python(sector_decomposition, *sym);
                }
                return std::make_shared<PyTensorProduct>(
                  std::move(factors),
                  std::move(symmetry),
                  std::move(sectors),
                  multiplicities_from_python(multiplicities));
            }),
            py::arg("factors"),
            py::arg("symmetry") = py::none(),
            py::arg("_sector_decomposition") = py::none(),
            py::arg("_multiplicities") = py::none());

    cls
      .def_property(
        "factors",
        [](TensorProduct const& self) { return objects_to_python(self.factors); },
        [](TensorProduct& self, py::iterable factors_obj) {
            self.factors = objects_from_python(factors_obj);
            self.num_factors = static_cast<int64>(self.factors.size());
        })
      .def_readonly("num_factors", &TensorProduct::num_factors)
      .def_property_readonly("dual", &TensorProduct::dual_space,
      R"pydoc(
      The dual space of the same type.
      
      A dual space necessarily has a :attr:`sector_decomposition` which consists of the
      :meth:`Symmetry.dual_sectors` of the original (though not necessarily in order).
      
      Strictly speaking, this only guarantees to give one possible choice for a dual space and
      might differ from *the* dual space by an irrelevant isomorphism.
      )pydoc")
      .def_property_readonly("has_pipes",
                             &TensorProduct::has_pipes,
                             R"pydoc(
                             Is any of the :attr:`factors` a pipe?
                             )pydoc")
      .def_property_readonly("flat_legs",
                             &TensorProduct::flat_legs,
                             R"pydoc(
                             Flatten until there are no more pipes.

                             See Also
                             --------
                             flat_spaces : Keeps :class:`AbelianLegPipes` nested.
                             )pydoc")
      .def_property_readonly("flat_spaces",
                             &TensorProduct::flat_spaces,
                             R"pydoc(
                             Flatten until we get spaces.

                             See Also
                             --------
                             flat_legs : Also flattens :class:`AbelianLegPipes`.
                             )pydoc")
      .def_property_readonly("num_flat_legs",
                             &TensorProduct::num_flat_legs,
                             R"pydoc(
                             The number of :attr:`flat_legs`.
                             )pydoc");

    cls.def_static(
      "from_partial_products",
      [](py::args factors_obj) {
          std::vector<TensorProduct::Ptr> factors;
          factors.reserve(static_cast<std::size_t>(factors_obj.size()));
          for (py::handle item : factors_obj) {
              factors.push_back(item.cast<TensorProduct::Ptr>());
          }
          return TensorProduct::from_partial_products(factors);
      },
      R"pydoc(
      Form the :class:`TensorProduct` of all :attr:`spaces` from partial products.

      The result has as :attr:`spaces` all those spaces that appear on the `factors`.
      I.e. we form :math:`V_1 \otimes V_2 \otimes W_1 \otimes W_2 \dots` from
      :math:`V_1 \otimes V_2` and :math:`W_1 \otimes W_2 \dots`.
      )pydoc");

    cls
      .def("test_sanity",
           &TensorProduct::test_sanity,
           R"pydoc(
           Perform sanity checks.
           )pydoc")
      .def(
        "block_size",
        [](TensorProduct const& self, py::object coupled) {
            if (py::isinstance<py::int_>(coupled)) {
                return self.block_size(coupled.cast<int64>());
            }
            return self.block_size(sector_from_python(coupled));
        },
        py::arg("coupled"),
        R"pydoc(
        The size of a block.

        Parameters
        ----------
        coupled : Sector or int
            Specify the coupled sector, either directly as a sector or as an integer, which
            is interpreted as an index, i.e. is equivalent to the sector
            ``self.sector_decomposition[coupled]``.
        )pydoc")
      .def(
        "change_symmetry",
        [](TensorProduct& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
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
        [](TensorProduct& self, py::object which) {
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
      .def("flat_legs_nesting",
           &TensorProduct::flat_legs_nesting,
           R"pydoc(
           The indices into :attr:`flat_legs`, that combine to each :attr:`factor`.
           )pydoc")
      .def("flat_leg_idcs",
           &TensorProduct::flat_leg_idcs,
           py::arg("i"),
           R"pydoc(
           All indices into the :meth:`flat_legs` that the leg ``factors[i]`` flattens to.
           )pydoc")
      .def(
        "forest_block_size",
        [](TensorProduct const& self, py::object uncoupled, py::object coupled) {
            return self.forest_block_size(sector_array_from_python(uncoupled, *self.symmetry),
                                          sector_from_python(coupled));
        },
        py::arg("uncoupled"),
        py::arg("coupled"),
        R"pydoc(
        The size of a forest-block
        )pydoc")
      .def(
        "forest_block_slice",
        [](TensorProduct const& self, py::object uncoupled, py::object coupled) {
            return index_slice_to_python(self.forest_block_slice(
              sector_array_from_python(uncoupled, *self.symmetry), sector_from_python(coupled)));
        },
        py::arg("uncoupled"),
        py::arg("coupled"),
        R"pydoc(
        The range of indices of a forest-block within its block, as a slice.
        )pydoc")
      .def("insert_multiply",
           &TensorProduct::insert_multiply,
           py::arg("other"),
           py::arg("pos"),
           R"pydoc(
           Insert a new space into the product at position `pos`.
           )pydoc")
      .def(
        "iter_tree_blocks",
        [](TensorProduct const& self, py::object coupled) {
            py::list out;
            for (auto const& item :
                 self.iter_tree_blocks(sector_array_from_python(coupled, *self.symmetry))) {
                out.append(py::make_tuple(py::cast(item.tree),
                                          index_slice_to_python(item.slice),
                                          perm_to_numpy(item.multiplicities),
                                          item.coupled_idx));
            }
            return py::iter(out);
        },
        py::arg("coupled"),
        R"pydoc(
        Iterate over tree blocks. Helper function for :class:`FusionTreeBackend`.

        See :ref:`fusion_tree_backend__blocks` for definitions of blocks and tree blocks.

        Yields
        ------
        tree : FusionTree
            A fusion tree whose uncoupled sectors are consistent with `self` and whose
            coupled sector is ``coupled[i]``
        slc : slice
            The slice of the tree-block associated with `tree` in its block.
        mults : 1D array of int
            The multiplicities of the uncoupled sectors of `tree` within their ``self.factor``.
        i : int
            The index of the current coupled sector in `coupled`

        See Also
        --------
        iter_forest_blocks
        iter_uncoupled
        )pydoc")
      .def(
        "iter_forest_blocks",
        [](TensorProduct const& self, py::object coupled) {
            py::list out;
            for (auto const& item :
                 self.iter_forest_blocks(sector_array_from_python(coupled, *self.symmetry))) {
                out.append(py::make_tuple(
                  py::cast(item.uncoupled), index_slice_to_python(item.slice), item.coupled_idx));
            }
            return py::iter(out);
        },
        py::arg("coupled"),
        R"pydoc(
        Iterate over forest blocks. Helper function for :class:`FusionTreeBackend`.

        See :ref:`fusion_tree_backend__blocks` for definitions of blocks and forest blocks.

        Yields
        ------
        uncoupled : tuple of Sector
            A tuple of uncoupled sectors that can fuse to a coupled sector ``coupled[i]``
        slc : slice
            The slice of the tree-block associated with `tree` in its block.
        i : int
            The index of the current coupled sector in `coupled`

        See Also
        --------
        iter_tree_blocks
        iter_uncoupled
        )pydoc")
      .def(
        "iter_uncoupled",
        [](TensorProduct const& self, bool yield_slices) {
            py::list out;
            for (auto const& item : self.iter_uncoupled(yield_slices)) {
                auto uncoupled = py::cast(item.uncoupled);
                auto mults = perm_to_numpy(item.multiplicities);
                if (!yield_slices) {
                    out.append(py::make_tuple(std::move(uncoupled), std::move(mults)));
                    continue;
                }
                py::list slices;
                for (auto const& slc : *item.slices) {
                    slices.append(index_slice_to_python(slc));
                }
                out.append(
                  py::make_tuple(std::move(uncoupled), std::move(mults), std::move(slices)));
            }
            return py::iter(out);
        },
        py::arg("yield_slices") = false,
        R"pydoc(
        Iterate over all combinations of sectors from the :attr:`flat_legs`.

        Yields
        ------
        uncoupled : 2D array of int
            A combination of uncoupled sectors, where
            ``uncoupled[i] == self.flat_legs[i].sector_decomposition[some_idx]``.
        multiplicities : 1D array of int
            The corresponding multiplicities
            ``multiplicities[i] == self.flat_legs[i].multiplicities[some_idx]``.
        slices : list of slice, optional
            Only if ``yield_slices``, the corresponding entry of :attr:`Space.slices`, as a slice.
            I.e. ``slices[i] == slice(*self.flat_legs[i].slices[some_idx])``.

        Notes
        -----
        For a TensorProduct of zero spaces, i.e. with ``num_factors == 0``,
        we *do* yield once, where the yielded arrays are empty (e.g. ``len(uncoupled) == 0``).
        )pydoc")
      .def("left_multiply",
           &TensorProduct::left_multiply,
           py::arg("other"),
           R"pydoc(
           Add a new factor at the left / beginning of the spaces
           )pydoc")
      .def("permuted",
           &TensorProduct::permuted,
           py::arg("perm"),
           R"pydoc(
           A product of the same :attr:`factors` in a different order.
           )pydoc")
      .def("right_multiply",
           &TensorProduct::right_multiply,
           py::arg("other"),
           R"pydoc(
           Add a new factor at the right / end of the spaces
           )pydoc")
      .def(
        "tree_block_size",
        [](TensorProduct const& self, py::object uncoupled) {
            return self.tree_block_size(sector_array_from_python(uncoupled, *self.symmetry));
        },
        py::arg("uncoupled"),
        R"pydoc(
        The size of a tree-block
        )pydoc")
      .def(
        "tree_block_slice",
        [](TensorProduct const& self, FusionTree const& tree) {
            return index_slice_to_python(self.tree_block_slice(tree));
        },
        py::arg("tree"),
        R"pydoc(
        The range of indices of a tree-block within its block, as a slice.
        )pydoc");

    cls
      .def("__eq__",
           [](TensorProduct const& self, py::object other) -> py::object {
               if (!py::isinstance<TensorProduct>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(
                 self.operator==(static_cast<Space const&>(other.cast<TensorProduct const&>())));
           })
      .def("__getitem__",
           [](TensorProduct const& self, py::object idx) -> py::object {
               if (py::isinstance<py::slice>(idx)) {
                   return objects_to_python(self.factors)[idx];
               }
               return self[idx.cast<int64>()];
           })
      .def("__len__", [](TensorProduct const& self) { return self.num_factors; })
      .def("__iter__",
           [](TensorProduct const& self) { return py::iter(objects_to_python(self.factors)); })
      .def("__repr__",
           &TensorProduct::repr,
           py::arg("show_symmetry") = true,
           py::arg("one_line") = false)
      .def(
        "repr", &TensorProduct::repr, py::arg("show_symmetry") = true, py::arg("one_line") = false)
      .def("save_hdf5",
           &TensorProduct::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &TensorProduct::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

void
bind_abelian_leg_pipe(py::module_& m)
{
    // Diamond MI: the Python MRO is AbelianLegPipe, LegPipe, ElementarySpace, Space, Leg.
    py::class_<AbelianLegPipe, LegPipe, ElementarySpace, PyAbelianLegPipe, py::smart_holder> cls(
      m,
      "AbelianLegPipe",
      R"pydoc(
      Special case of a :class:`LegPipe` for abelian group symmetries.

      This class essentially exists to allow specialized handling of combined legs in the
      :class:`AbelianBackend`. For this backend, we want to treat combined legs, i.e. pipes, exactly
      the same as regular legs. This is why this class also inherits from :class:`ElementarySpace`,
      which are the "uncombined" legs. Crucially, this allows the pipe to have
      :attr:`defining_sectors` for the :attr:`cyten.backends.abelian.AbelianBackendData.block_inds`
      to point to, to have a well-behaved :attr:`is_dual` attribute and to have a :attr:`basis_perm`,
      which can account for the basis permutation that is induced by going from sectors of the
      individual legs to a sorted list of coupled sectors on the pipe.

      Attributes
      ----------
      legs:
          The individual legs that form this pipe, and that the pipe can be split into.
          In particular, these are such that the pipe, as an :class:`ElementarySpace`, is isomorphic
          to their tensor product ``TensorProduct(legs)``, i.e. has the same
          :attr:`sector_decomposition`.
      sector_strides : 1D numpy array of int
          Strides for the shape ``[leg.num_sectors for leg in self.legs]``. Is either C-style or
          F-style, depending on `combine_cstyle`. This allows one-to-one mapping between
          multi-indices (one block_ind per space) to a single index.
          Used in :meth:`AbelianBackend.combine_legs`.
      fusion_outcomes_sort : 1D numpy array of int
          The permutation that sorts the list of fusion outcomes.
          To calculate the :attr:`sector_decomposition` of the pipe, we go through all combinations
          of sectors from the :attr:`legs` in F-style order, i.e. varying sectors from the first leg
          the fastest. For each combination of sectors, we perform their fusion, which yields a
          single sector in the abelian case assumed here. The resulting list of fused sectors is in
          general neither sorted nor unique. This permutation (stable) sorts the resulting list.
          We use F-style to match the sorting convention of :attr:`block_ind_map`.
      block_ind_map_slices : 1D numpy array of int
          Slices for embedding the unique fused sectors in the sorted list of all fusion outcomes.
          Shape is ``(K,)`` where ``K == pipe.num_sectors + 1``.
          Fusing all sectors from the :attr:`sector_decomposition` of all legs and sorting the
          outcomes gives a list which contains (in general) duplicates.
          The slice ``block_ind_map_slices[n]:block_ind_map_slices[n + 1]`` within this sorted list
          contains the same entry, namely ``pipe.sector_decomposition[n]``.
          Used in :math:`AbelianBackend.split_legs`.
      block_ind_map : BlockInds
          Map for the embedding of uncoupled to coupled indices, see notes of the Python class.
          Shape is ``(M, N)`` where ``M`` is the number of combinations of sectors,
          i.e. ``M == prod(leg.num_sectors for leg in legs)`` and ``N == 3 + len(legs)``.
      )pydoc");

    cls.def(py::init([](py::sequence legs_obj, bool is_dual, bool combine_cstyle) {
                std::vector<ElementarySpace::Ptr> legs;
                legs.reserve(static_cast<std::size_t>(legs_obj.size()));
                for (py::handle item : legs_obj) {
                    legs.push_back(item.cast<ElementarySpace::Ptr>());
                }
                return std::make_shared<PyAbelianLegPipe>(
                  std::move(legs), is_dual, combine_cstyle);
            }),
            py::arg("legs"),
            py::arg("is_dual") = false,
            py::arg("combine_cstyle") = true);

    cls
      .def_property_readonly(
        "sector_strides",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.sector_strides); })
      .def_property_readonly(
        "fusion_outcomes_sort",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.fusion_outcomes_sort); })
      .def_property_readonly(
        "block_ind_map_slices",
        [](AbelianLegPipe const& self) { return perm_to_numpy(self.block_ind_map_slices); })
      .def_property_readonly("block_ind_map",
                             [](AbelianLegPipe const& self) { return self.block_ind_map; });

    cls
      .def_property_readonly("dual",
                             &AbelianLegPipe::dual_pipe,
                             R"pydoc(
                             The dual pipe, i.e. the dual of each leg in reverse order.
                             )pydoc")
      .def_property_readonly("is_trivial", &AbelianLegPipe::is_trivial,
      R"pydoc(
      If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.
      
      A trivial space is one-dimensional and transforms trivially under a symmetry group.
      In category speak, it is (isomorphic to) the monoidal unit.
      )pydoc")
      .def_property_readonly("flat_spaces",
                             &AbelianLegPipe::flat_spaces,
                             R"pydoc(
                             ``[self]`` -- unlike a plain :class:`LegPipe`, an
                             :class:`AbelianLegPipe` is already a space.
                             )pydoc")
      .def_property_readonly("ascii_arrow", &AbelianLegPipe::ascii_arrow,
      R"pydoc(
      A single character arrow, for use in tensor diagrams
      
      Indicates (a) if the leg is a pipe and (b) for ElementarySpaces, the duality
      )pydoc");

    cls.def_static(
      "from_independent_symmetries",
      [](py::sequence independent_descriptions) {
          std::vector<AbelianLegPipe::Ptr> descriptions;
          descriptions.reserve(static_cast<std::size_t>(independent_descriptions.size()));
          for (py::handle item : independent_descriptions) {
              descriptions.push_back(item.cast<AbelianLegPipe::Ptr>());
          }
          return AbelianLegPipe::from_independent_symmetries(descriptions);
      },
      py::arg("independent_descriptions"),
      R"pydoc(
      Create an AbelianLegPipe with multiple independent symmetries.

      Parameters
      ----------
      independent_descriptions : list of :class:`AbelianLegPipe`
          Each entry describes the resulting pipe in terms of *one* of the independent symmetries.
      )pydoc");

    // The unsupported ElementarySpace factories. They are bound (and raise) such that they
    // shadow the inherited versions, which would silently return a plain ElementarySpace.
    cls
      .def_static("from_basis",
                  [](py::args, py::kwargs) -> py::object {
                      throw py::type_error("from_basis is not supported for AbelianLegPipe");
                  },
                  R"pydoc(
                  Create an ElementarySpace by specifying the sector of every basis element.
                  
                  This requires that the symmetry :attr:`~cyten.symmetries.Symmetry.can_be_dropped`, such
                  that there is a useful notion of a basis.
                  
                  .. note ::
                      Unlike :meth:`from_defining_sectors`, this method expects the same sector to be listed
                      multiple times, if the sector is multi-dimensional. The Hilbert Space of a spin-one-half
                      D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
                      or as ``ElementarySpace.from_defining_sectors(su2, [spin_half])``. In the former case
                      we need to list the same sector both for the spin up and spin down state.
                  
                  .. note ::
                      This classmethod always creates ket-spaces with ``is_dual=False``. This is to make
                      it unambiguous if `sectors_of_basis` refers to the :attr:`sector_decomposition` or the
                      :attr:`defining_sectors`, since they coincide for ket spaces.
                      Use :attr:`dual` or :meth:`as_bra_space` to create bra spaces.
                  
                  Parameters
                  ----------
                  symmetry: Symmetry
                      The symmetry associated with this space.
                  sectors_of_basis : iterable of iterable of int
                      Specifies the basis. ``sectors_of_basis[n]`` is the sector of the ``n``-th basis element.
                      In particular, for a ``d`` dimensional sector, we expect an integer multiple of ``d``
                      occurrences. They need not be contiguous though. They will be grouped by order of
                      appearance, such that they ``m``-th time a sector appears, that basis state is interpreted
                      as the ``(m % d)``-th state of the multiplet.
                  
                  See Also
                  --------
                  :attr:`sectors_of_basis`
                      Reproduces the `sectors_of_basis` parameter.
                  from_defining_sectors
                      Similar to the constructor, but with fewer requirements.
                  )pydoc")
      .def_static("from_null_space",
                  [](py::args, py::kwargs) -> py::object {
                      throw py::type_error("from_null_space is not supported for AbelianLegPipe");
                  },
                  R"pydoc(
                  The zero-dimensional space, i.e. the span of the empty set.
                  )pydoc")
      .def_static("from_defining_sectors",
                  [](py::args, py::kwargs) -> py::object {
                      throw py::type_error(
                        "from_defining_sectors is not supported for AbelianLegPipe");
                  },
                  R"pydoc(
                  Similar to the constructor, but with fewer requirements.
                  
                  .. note ::
                      Unlike :meth:`from_basis`, this method expects a multi-dimensional sector to be listed
                      only once to mean its entire multiplet of basis states. The Hilbert Space of a spin-1/2
                      D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
                      or as ``ElementarySpace.from_defining_sectors(su2, [spin_half])``. In the former case
                      we need to list the same sector both for the spin up and spin down state.
                  
                  Parameters
                  ----------
                  symmetry: Symmetry
                      The symmetry associated with this space.
                  defining_sectors: 2D array_like of int
                      Like the :attr:`defining_sectors` attribute, but can be in any order and may contain
                      duplicates (see `unique_sectors`).
                  multiplicities: 1D array_like of int, optional
                      How often each of the `defining_sectors` appears. A 1D array of positive integers with
                      axis [s]. ``defining_sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
                      If not given, a multiplicity ``1`` is assumed for all `defining_sectors`.
                  is_dual: bool
                      If the result is a bra- or a ket space, like the attribute :attr:`is_dual`.
                      Note that this changes the meaning of the `defining_sectors`.
                  basis_perm: ndarray, optional
                      The permutation from the desired public basis to the basis described by
                      `defining_sectors` and `multiplicities`.
                  unique_sectors: bool
                      If ``True``, the `sectors` are assumed to be duplicate-free.
                  return_sorting_perm: bool
                      If ``True``, the permutation ``np.lexsort(sectors.T)`` is returned too.
                  
                  Returns
                  -------
                  space: ElementarySpace
                      The new space
                  sector_sort: 1D array, optional
                      Only ``if return_sorting_perm``. The permutation that sorts the `defining_sectors`.
                  )pydoc")
      .def_static("from_trivial_sector", [](py::args, py::kwargs) -> py::object {
          throw py::type_error("from_trivial_sector is not supported for AbelianLegPipe");
      },
      R"pydoc(
      Create an ElementarySpace that lives in the trivial sector (i.e. it is symmetric).
      
      Parameters
      ----------
      dim : int
          The dimension of the space.
      symmetry : :class:`~cyten.Symmetry`
          The symmetry of the space.
      is_dual : bool
          If the space should be bra or a ket space.
      )pydoc");

    cls
      .def("test_sanity",
           &AbelianLegPipe::test_sanity,
           R"pydoc(
           Perform sanity checks.
           )pydoc")
      .def("as_Space", &AbelianLegPipe::as_Space,
      R"pydoc(
      Convert to (an appropriate subclass of) :class:`Space`.
      )pydoc")
      .def("as_ElementarySpace", &AbelianLegPipe::as_ElementarySpace, py::arg("is_dual") = false,
      R"pydoc(
      Convert to an isomorphic :class:`ElementarySpace`.
      )pydoc")
      .def(
        "change_symmetry",
        [](
          AbelianLegPipe& self, py::object symmetry_obj, py::function sector_map, bool injective) {
            return self.change_symmetry(
              symmetry_from_python(symmetry_obj), sector_map_from_python(sector_map), injective);
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
        [](AbelianLegPipe& self, py::object which) {
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
      .def(
        "set_basis_perm",
        [](AbelianLegPipe& self, py::args, py::kwargs) { self.set_basis_perm(std::nullopt); },
        R"pydoc(
        Not supported: an :class:`AbelianLegPipe` determines its own ``basis_perm``.
        )pydoc")
      .def(
        "take_slice",
        [](AbelianLegPipe& self, py::object blockmask) {
            return self.take_slice(py::array::ensure(blockmask));
        },
        py::arg("blockmask"),
        R"pydoc(
        Take a "slice" of the leg, keeping only some of the basis states.

        Loses the product (pipe) structure and results in a plain :class:`ElementarySpace`.
        )pydoc")
      .def("with_opposite_duality",
           &AbelianLegPipe::with_opposite_duality,
           R"pydoc(
           A pipe of the same legs with opposite ``is_dual`` attribute.
           )pydoc")
      .def(
        "_get_fusion_outcomes_perm",
        [](AbelianLegPipe const& self, py::object multiplicities) {
            auto mults = multiplicities_from_python(multiplicities);
            return perm_to_numpy(
              self.get_fusion_outcomes_perm(mults.value_or(self.multiplicities)));
        },
        py::arg("multiplicities"),
        R"pydoc(
        Get the permutation of basis elements that is introduced by the fusion.
        )pydoc")
      .def("__eq__",
           [](AbelianLegPipe const& self, py::object other) -> py::object {
               if (!py::isinstance<LegPipe>(other)) {
                   return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
               }
               return py::cast(
                 self.operator==(static_cast<Leg const&>(other.cast<LegPipe const&>())));
           })
      .def("__repr__", [](AbelianLegPipe const& self) { return self.repr(); })
      .def("repr",
           &AbelianLegPipe::repr,
           py::arg("show_symmetry") = true,
           py::arg("one_line") = false)
      .def("save_hdf5",
           &AbelianLegPipe::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &AbelianLegPipe::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));

    m.def("swap_gate",
          &swap_gate,
          py::arg("V"),
          py::arg("W"),
          R"pydoc(
          The swap gate (numpy representation of the braid).

              |   V   W
              |   │   │
              |   v   v
              |    ╲ ╱
              |     ╲          <-  overbraid == underbraid is assumed
              |    ╱ ╲
              |   v   v
              |   │   │
              |   W   V

          Returns
          -------
          A numpy representation of the above tensor with axes ``[W, V, W*, V*]``.

          See Also
          --------
          :meth:`cyten.Symmetry.swap_gate`
              The swap gate for single sectors.
          )pydoc");

    m.def("twist_gate",
          &twist_gate,
          py::arg("V"),
          R"pydoc(
          The topological twist on a whole space, as numpy representation.

          Returns
          -------
          A numpy representation of the above tensor with axes ``[V, V*]``.

          See Also
          --------
          :meth:`cyten.Symmetry.topological_twist`
              The twist on a single sector, given in the form of a prefactor for the identity map.
          )pydoc");

    m.def("_twist_gate_diag", &twist_gate_diag, py::arg("V"));

    m.def(
      "_flat_leg_permutation",
      [](py::sequence legs_obj) {
          std::vector<Leg::Ptr> legs;
          legs.reserve(static_cast<std::size_t>(legs_obj.size()));
          for (py::handle item : legs_obj) {
              legs.push_back(item.cast<Leg::Ptr>());
          }
          return flat_leg_permutation(legs);
      },
      py::arg("legs"),
      R"pydoc(
      Leg permutation such that combining / splitting legs would be in C style.

      Returns
      -------
      perm
          The permutation of the flat legs such that combining or splitting them in C style after
          applying this permutation corresponds to combining / splitting them with respect to their
          :attr:`combine_c_style` without applying this permuatation.
          This is useful when working with the flat legs of nested pipes that may have different
          :attr:`combine_c_style`, as done in the fusion tree backend.
      )pydoc");

    m.def(
      "_unique_sorted_sectors",
      [](SectorArray const& sectors, py::object multiplicities) {
          auto mults = multiplicities_from_python(multiplicities).value_or(std::vector<int64>{});
          auto [s, m, perm] = unique_sorted_sectors(sectors, mults);
          return py::make_tuple(
            s, perm_to_numpy(m), perm_to_numpy(std::vector<int64>(perm.begin(), perm.end())));
      },
      py::arg("unsorted_sectors"),
      py::arg("unsorted_multiplicities"),
      R"pydoc(
      Sort sectors and merge duplicates.
      )pydoc");

    m.def(
      "_sort_sectors",
      [](SectorArray const& sectors, py::object multiplicities) {
          auto mults = multiplicities_from_python(multiplicities).value_or(std::vector<int64>{});
          auto [s, m, perm] = sort_sectors_public(sectors, mults);
          return py::make_tuple(
            s, perm_to_numpy(m), perm_to_numpy(std::vector<int64>(perm.begin(), perm.end())));
      },
      py::arg("sectors"),
      py::arg("multiplicities"));

    m.def(
      "_parse_inputs_drop_symmetry",
      [](py::object which, py::object symmetry_obj) -> py::tuple {
          auto symmetry = symmetry_from_python(symmetry_obj);
          auto [factors, remaining] =
            parse_inputs_drop_symmetry_public(drop_which_from_python(which), std::move(symmetry));
          if (!factors) {
              return py::make_tuple(py::str("all"), remaining);
          }
          return py::make_tuple(perm_to_numpy(*factors), remaining);
      },
      py::arg("which"),
      py::arg("symmetry"),
      R"pydoc(
      Input parsing for :meth:`Space.drop_symmetry`.

      Returns
      -------
      which : 'all' | list of int
          Which symmetries to drop, as integers in ``range(symmetry.num_factors)``.
          ``'all'`` indicates to drop all.
      remaining_symmetry : Symmetry
          The symmetry that remains.
      )pydoc");
}

} // namespace

} // namespace cyten
