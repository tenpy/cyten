#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/sector.h>
#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/trees.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace cyten {

namespace {

std::vector<std::uint8_t>
are_dual_from_python(py::handle obj, std::size_t expected_len)
{
    if (obj.is_none()) {
        return std::vector<std::uint8_t>(expected_len, 0);
    }
    py::array arr = py::array::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::type_error("are_dual must be a 1D sequence of bool");
    }
    auto casted =
      py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    if (static_cast<std::size_t>(r.shape(0)) != expected_len) {
        throw py::value_error("are_dual length mismatch");
    }
    std::vector<std::uint8_t> out(expected_len);
    for (std::size_t i = 0; i < expected_len; ++i) {
        out[i] = r(static_cast<py::ssize_t>(i)) ? 1 : 0;
    }
    return out;
}

std::optional<std::vector<int64>>
multiplicities_from_python(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    py::array arr = py::array::ensure(obj);
    if (!arr || arr.ndim() != 1) {
        throw py::type_error("multiplicities must be a 1D integer sequence");
    }
    auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

SectorArray
sector_array_from_python(py::handle obj, Symmetry const& symmetry)
{
    if (obj.is_none()) {
        return symmetry.empty_sector_array;
    }
    if (py::isinstance<SectorArray>(obj)) {
        return obj.cast<SectorArray>();
    }
    if (py::isinstance<Sector>(obj)) {
        return SectorArray::from_sector(obj.cast<Sector>());
    }
    // empty list / sequence → empty SectorArray with correct sector_ind_len
    if (py::isinstance<py::sequence>(obj) && !py::isinstance<py::str>(obj)) {
        auto seq = obj.cast<py::sequence>();
        if (seq.size() == 0) {
            return symmetry.empty_sector_array;
        }
    }
    return sector_array_from_numpy(obj);
}

Symmetry::Ptr
symmetry_from_python(py::object symmetry_obj)
{
    if (py::isinstance<Symmetry>(symmetry_obj)) {
        return symmetry_obj.cast<Symmetry::Ptr>();
    }
    return symmetry_obj.attr("as_Symmetry")().cast<Symmetry::Ptr>();
}

BlockBackend*
block_backend_from_python(py::object backend)
{
    if (backend.is_none()) {
        return nullptr;
    }
    if (py::isinstance<BlockBackend>(backend)) {
        return backend.cast<BlockBackend*>();
    }
    return backend.attr("block_backend").cast<BlockBackend*>();
}

/// Match historical Python: real topological coeffs as float, not complex(x+0j).
/// Complex coeffs make is_real fusion-tree mappings multiply float blocks by
/// complex128 and trip ComplexWarning-as-error under pytest.
py::object
scalar_to_python(complex128 z)
{
    if (z.imag() == 0.0) {
        return py::cast(z.real());
    }
    return py::cast(z);
}

py::dict
linear_combination_to_python(FusionTreeLinearCombination const& lc)
{
    py::dict out;
    for (auto const& [tree, coeff] : lc) {
        out[py::cast(tree)] = scalar_to_python(coeff);
    }
    return out;
}

py::dict
pair_linear_combination_to_python(FusionTreePairLinearCombination const& lc)
{
    py::dict out;
    for (auto const& [pair, coeff] : lc) {
        out[py::cast(pair)] = scalar_to_python(coeff);
    }
    return out;
}

} // namespace

void
bind_trees(py::module_& m)
{
    py::class_<FusionTree> cls(m,
                               "FusionTree",
                               R"pydoc(
                               A fusion tree, which represents the map from uncoupled to coupled sectors.

                               Consider the following example tree::

                                   FusionTree(
                                       symmetry=symmetry,
                                       coupled=coupled,
                                       uncoupled=[a, b, c, d],
                                       are_dual=[False, True, True, False],
                                       inner_sectors=[x, y],
                                       multiplicities=[i, j, k],
                                   )

                               Graphically::

                                   |    a     b     c     d     <- isomorphic to pre_Z_uncoupled
                                   |    v     ^     ^     v        e.g. dual(b) iso to pre_Z_uncoupled[1]
                                   |    │     Z     Z     │
                                   |    v     v     v     v
                                   |    a     b     c     d     <- uncoupled
                                   |    ╰──i──╯     │     │
                                   |      x│        │     │
                                   |       ╰───j────╯     │
                                   |          y│          │
                                   |           ╰────k─────╯
                                   |                │
                                   |                coupled

                               Attributes
                               ----------
                               symmetry : Symmetry
                                   The symmetry.
                               uncoupled : SectorArray
                                   N uncoupled sectors. These are the sectors *below* any Z isos.
                               coupled : Sector
                                   The coupled sector at the bottom of the tree.
                               are_dual : 1D array of bool
                                   N flags: is there a Z isomorphism above the uncoupled sector.
                               inner_sectors : SectorArray
                                   N - 2 internal sectors, at the internal edges of the tree.
                               multiplicities : 1D array of int
                                   N - 1 multiplicity labels, at the fusion vertices of the tree.
                               )pydoc");

    cls.def(py::init([](py::object symmetry_obj,
                        py::object uncoupled,
                        Sector coupled,
                        py::object are_dual,
                        py::object inner_sectors,
                        py::object multiplicities) {
                auto symmetry = symmetry_from_python(symmetry_obj);
                SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
                SectorArray inner = sector_array_from_python(inner_sectors, *symmetry);
                auto dual = are_dual_from_python(are_dual, unc.size());
                return FusionTree(std::move(symmetry),
                                  std::move(unc),
                                  coupled,
                                  std::move(dual),
                                  std::move(inner),
                                  multiplicities_from_python(multiplicities));
            }),
            py::arg("symmetry"),
            py::arg("uncoupled"),
            py::arg("coupled"),
            py::arg("are_dual"),
            py::arg("inner_sectors"),
            py::arg("multiplicities") = py::none());

    cls.def_readwrite("symmetry", &FusionTree::symmetry)
      .def_readwrite("uncoupled", &FusionTree::uncoupled)
      .def_readwrite("coupled", &FusionTree::coupled)
      .def_readwrite("inner_sectors", &FusionTree::inner_sectors)
      .def_readonly("num_uncoupled", &FusionTree::num_uncoupled)
      .def_readonly("num_vertices", &FusionTree::num_vertices)
      .def_readonly("num_inner_edges", &FusionTree::num_inner_edges)
      .def_readonly("fusion_style", &FusionTree::fusion_style)
      .def_readonly("is_abelian", &FusionTree::is_abelian)
      .def_readonly("braiding_style", &FusionTree::braiding_style);

    // Writable NumPy views into C++ storage so in-place updates (e.g. fusion_trees)
    // remain visible on the FusionTree instance.
    cls
      .def_property(
        "are_dual",
        [](py::object self_obj) {
            auto& self = self_obj.cast<FusionTree&>();
            return py::array(py::dtype::of<bool>(),
                             { self.are_dual.size() },
                             { sizeof(std::uint8_t) },
                             self.are_dual.data(),
                             self_obj);
        },
        [](FusionTree& self, py::object value) {
            self.are_dual = are_dual_from_python(value, self.num_uncoupled);
        })
      .def_property(
        "multiplicities",
        [](py::object self_obj) {
            auto& self = self_obj.cast<FusionTree&>();
            return py::array(py::dtype::of<int64>(),
                             { self.multiplicities.size() },
                             { sizeof(int64) },
                             self.multiplicities.data(),
                             self_obj);
        },
        [](FusionTree& self, py::object value) {
            auto opt = multiplicities_from_python(value);
            if (!opt) {
                self.multiplicities.assign(self.num_vertices, 0);
            } else {
                if (opt->size() != self.num_vertices) {
                    throw py::value_error("multiplicities length mismatch");
                }
                self.multiplicities = std::move(*opt);
            }
        });

    cls.def_property_readonly("pre_Z_uncoupled",
                              &FusionTree::pre_Z_uncoupled,
                              R"pydoc(
                              The uncoupled sectors *above* any Z isomorphisms.
                              )pydoc");

    cls
      .def("test_sanity",
           &FusionTree::test_sanity,
           R"pydoc(
           Perform sanity checks.
           )pydoc")
      .def_static(
        "from_abelian_symmetry",
        [](py::object symmetry_obj, py::object uncoupled, py::object are_dual) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
            auto dual = are_dual_from_python(are_dual, unc.size());
            return FusionTree::from_abelian_symmetry(std::move(symmetry), unc, dual);
        },
        py::arg("symmetry"),
        py::arg("uncoupled"),
        py::arg("are_dual"),
        R"pydoc(
        Assume an abelian symmetry and build the unique tree with the given `uncoupled`.

        For an abelian symmetry, two sectors fuse to a single other sector, such that the entire
        tree is determined by the uncoupled sectors alone.
        )pydoc")
      .def_static(
        "from_empty",
        [](py::object symmetry_obj) {
            return FusionTree::from_empty(symmetry_from_python(symmetry_obj));
        },
        py::arg("symmetry"),
        R"pydoc(
        The empty tree with no uncoupled sectors.
        )pydoc")
      .def_static(
        "from_sector",
        [](py::object symmetry_obj, Sector sector, bool is_dual) {
            return FusionTree::from_sector(symmetry_from_python(symmetry_obj), sector, is_dual);
        },
        py::arg("symmetry"),
        py::arg("sector"),
        py::arg("is_dual"),
        R"pydoc(
        A tree with a single uncoupled sector and no nodes.
        )pydoc")
      .def("__hash__", &FusionTree::hash)
      .def("__eq__", &FusionTree::operator==, py::arg("other"))
      .def("ascii_diagram",
           &FusionTree::ascii_diagram,
           py::arg("dagger") = false,
           R"pydoc(
           Visual representation of the tree as ASCII art.
           )pydoc")
      .def_static(
        "_str_uncoupled_coupled",
        [](Symmetry const& symmetry,
           SectorArray const& uncoupled,
           Sector coupled,
           py::object are_dual) {
            auto dual = are_dual_from_python(are_dual, uncoupled.size());
            return FusionTree::str_uncoupled_coupled(symmetry, uncoupled, coupled, dual);
        },
        py::arg("symmetry"),
        py::arg("uncoupled"),
        py::arg("coupled"),
        py::arg("are_dual"),
        R"pydoc(
        Helper function for string representation.

        Generates a string that represents the uncoupled sectors before the Z isos,
        the uncoupled sectors after and the coupled sector.

        Is also used by ``fusion_trees.__str__``.
        )pydoc")
      .def_static(
        "bend_leg",
        [](FusionTree const& X, FusionTree const& Y, bool bend_downward, bool do_conj) {
            return pair_linear_combination_to_python(
              FusionTree::bend_leg(X, Y, bend_downward, do_conj));
        },
        py::arg("X"),
        py::arg("Y"),
        py::arg("bend_downward"),
        py::arg("do_conj") = false,
        R"pydoc(
        Bend a leg on a tree-pair, return the resulting linear combination of tree-pairs.

        Graphically::

            |    bend_downward=True                    bend_downward=False
            |
            |   │   │   │   ╭────╮                    │   │   │   │    │
            |   ┢━━━┷━━━┷━━━┷━┓  │                    ┢━━━┷━━━┷━━━┷━┓  │
            |   ┡━━━━━━━━━━━━━┛  │                    ┡━━━━━━━━━━━━━┛  │
            |   │                │                    │                │
            |   ┢━━━━━━━━━━━━━┓  │                    ┢━━━━━━━━━━━━━┓  │
            |   ┡━━━┯━━━┯━━━┯━┛  │                    ┡━━━┯━━━┯━━━┯━┛  │
            |   │   │   │   │    │                    │   │   │   ╰────╯

        Parameters
        ----------
        X, Y : FusionTree
            The original tree pair, such that we modify ``hconj(X) @ Y``.
            Note that `X` is a fusion tree that represents the splitting tree ``hconj(X)``.
        bend_downward : bool
            Whether the rightmost leg of `Y` is bent down (``bend_downward == True``) or the rightmost
            leg of ``hconj(X)`` is bent up (``bend_downward == False``).
        do_conj : bool
            If ``True``, return the conjugate of the coefficients instead.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            The bent tree pair is a linear combination ``bent = sum_i a_i hconj(Y_i) @ X_i`` of tree
            pairs (where ``Y_i`` is a fusion tree and thus ``hconj(Y_i)`` a splitting tree).
            The returned dictionary has entries ``linear_combination[Y_i, X_i] = a_i`` for the
            contributions to this linear combination (i.e. tree pairs for which the coefficient
            vanishes are omitted).
        )pydoc")
      .def(
        "braid",
        [](FusionTree const& self, int64 j, bool overbraid, float64 cutoff, bool do_conj) {
            return linear_combination_to_python(self.braid(j, overbraid, cutoff, do_conj));
        },
        py::arg("j"),
        py::arg("overbraid"),
        py::arg("cutoff") = 1e-16,
        py::arg("do_conj") = false,
        R"pydoc(
        Braid a leg on a fusion tree, return the resulting linear combination of trees.

        Graphically::

            |   overbraid:                  underbraid
            |
            |   │   │   │   │               │   │   │   │
            |   │    ╲ ╱    │               │    ╲ ╱    │
            |   │     ╱     │               │     ╲     │
            |   │    ╱ ╲    │               │    ╱ ╲    │
            |   │   j  j+1  │               │   j  j+1  │
            |   ┢━━━┷━━━┷━━━┷━┓             ┢━━━┷━━━┷━━━┷━┓
            |   ┡━━━━━━━━━━━━━┛             ┡━━━━━━━━━━━━━┛
            |   │                           │

        .. warning ::
            When braiding splitting trees (daggers of fusion trees), consider the notes below.

        Parameters
        ----------
        j : int
            The index for the braid. We braid ``uncoupled[j]`` with ``uncoupled[j + 1]``.
        overbraid : bool
            If we apply an overbraid or an underbraid (see graphic above).
        cutoff : float
            We skip contributions with a prefactor below this.
        do_conj : bool
            If ``True``, return the conjugate of the coefficients instead.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            The braided fusion tree is a linear combination ``braided_self = sum_i a_i X_i``.
            The returned dictionary has entries ``linear_combination[X_i] = a_i`` for the
            contributions to this linear combination (i.e. trees for which the coefficient vanishes
            may be omitted).
        )pydoc")
      .def("vertex_labels",
           &FusionTree::vertex_labels,
           py::arg("n"),
           R"pydoc(
           For the ``n``-th fusion vertex, get the respective sectors.

           Returns
           -------
           a, b, mu, c
               The sectors and multiplicity label around the ``n``-th vertex of the tree::

                   |   (n-1 higher vertices)      │
                   |                      │       │
                   |                      a       b
                   |                      ╰───µ───╯
                   |                          c
                   |                          │
                   |                          (possibly lower vertices)
           )pydoc")
      .def("modify_vertex_labels",
           &FusionTree::modify_vertex_labels,
           py::arg("n"),
           py::arg("a"),
           py::arg("b"),
           py::arg("mu"),
           py::arg("c"),
           py::arg("copy") = true,
           R"pydoc(
           Update the multiplicity and the three sectors around the ``n``-th vertex.

           Parameters
           ----------
           n : int
               The vertex.
           a, b, mu, c
               Three sectors and a multiplicity, like the returns of :meth:`vertex_labels`.
               ``None`` place-holders indicate to not update that value.
           copy : bool
               If ``True``, we return a modified copy. If ``False``, we modify in place and return
               the modified instance.
           )pydoc")
      .def("__str__", &FusionTree::str)
      .def("__repr__", &FusionTree::repr)
      .def(
        "to_dense_block",
        [](
          FusionTree const& self, py::object backend, py::object dtype, bool understood_braiding) {
            std::optional<Dtype> dt;
            if (!dtype.is_none()) {
                dt = dtype.cast<Dtype>();
            }
            return self.to_dense_block(
              block_backend_from_python(backend), dt, understood_braiding);
        },
        py::arg("backend") = py::none(),
        py::arg("dtype") = py::none(),
        py::arg("understood_braiding") = false,
        R"pydoc(
        Get the matrix elements of the map as a backend Block.

        Parameters
        ----------
        backend : TensorBackend, optional
            The backend for the resulting block. By default, we return a numpy array.
        dtye : Dtype, optional
            The dtype for the resulting block. By default, inferred from the symmetry
        understood_braiding : bool
            For symmetries with non-trivial (but symmetric) braiding, e.g. fermions, the resulting
            dense block does no longer capture the braiding statistics correctly. This means that
            :func:`permute_legs` is not consistently reproduced by e.g. ``numpy.transpose`` on
            the dense block representation. Permuting its legs would require e.g. explicit swap
            gates. When using the result, special care needs to be taken regarding the leg order.
            To avoid this pitfall, we raise an error by default. Set this flag to ``True`` to
            disable the error. It is then your responsibility to take care of leg orders and braids.
            See :mod:`cyten.testing.swap_gate_numpy` for manipulations on these dense blocks.

        Returns
        -------
        The matrix elements with axes ``[m_a1, m_a2, ..., m_aJ, m_c]``.
        )pydoc")
      .def("copy",
           &FusionTree::copy,
           py::arg("deep") = true,
           R"pydoc(
           Return a shallow (or deep) copy.
           )pydoc")
      .def("extended",
           &FusionTree::extended,
           py::arg("new_uncoupled"),
           py::arg("mu"),
           py::arg("new_coupled"),
           py::arg("is_dual"),
           R"pydoc(
           A new tree, from adding a new fusion node at the bottom, below the coupled sector.

           Graphically::

               |               │
               |              (Z)
               |               v
               |   (self)     new_uncoupled
               |       │       │
               |       ╰───µ───╯
               |           │
               |          new_coupled

           See Also
           --------
           insert
               Can insert nodes "above"
           split_topmost
               Split off the topmost node.
           )pydoc")
      .def("insert",
           &FusionTree::insert,
           py::arg("t2"),
           R"pydoc(
           Insert a tree `t2` above the first uncoupled sector.

           See Also
           --------
           insert_at
               Inserting at general position
           split
               Split into two separate fusion trees.
           )pydoc")
      .def(
        "insert_at",
        [](FusionTree const& self, int64 n, FusionTree const& t2, float64 eps) {
            return linear_combination_to_python(self.insert_at(n, t2, eps));
        },
        py::arg("n"),
        py::arg("t2"),
        py::arg("eps") = 1.0e-14,
        R"pydoc(
        Insert a tree `t2` above the `n`-th uncoupled sector.

        The result is (in general) not a canonical tree.
        We transform it to canonical form via a series of F moves.
        This yields the result as a linear combination of canonical trees.
        We return a dictionary, with those trees as keys and the prefactors as values.

        Parameters
        ----------
        n : int
            The position to insert at. `t2` is inserted above ``t1.uncoupled[n]``.
            We must have have ``self.are_dual[n] is False``, as we can not have a Z between trees.
        t2 : :class:`FusionTree`
            The fusion tree to insert
        eps : float
            F symbols whose absolute values are smaller than this number are treated as zero.

        Returns
        -------
        coefficients : dict
            Trees and coefficients that form the composite map as a linear combination.
            Abusing notation (``FusionTree`` instances can not actually be scaled or added),
            this means ``map = sum(c * t for t, c in coefficient.items())``.

        See Also
        --------
        insert
            The same insertion, but restricted to ``n=0``, and returns that tree directly, no dict.
        split
            Split into two separate fusion trees.
        )pydoc")
      .def(
        "outer",
        [](FusionTree const& self, FusionTree const& right_tree, float64 eps) {
            return linear_combination_to_python(self.outer(right_tree, eps));
        },
        py::arg("right_tree"),
        py::arg("eps") = 1.0e-14,
        R"pydoc(
        Outer product with another tree.

        Fuse with `right_tree` at the coupled sector (-> new coupled sectors are all sectors that
        are allowed fusion channels of the coupled sectors).

        Parameters
        ----------
        right_tree : FusionTree
            Tree to be combined with at the coupled sector from the right.
        eps : float
            F symbols whose absolute values are smaller than this number are treated as zero.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            Result expressed as linear combination of fusion trees in the canonical basis with the
            corresponding coefficients.

        See Also
        --------
        insert_at
            Similar insertion, but the tree is inserted above of an uncoupled sector rather than
            fused with the coupled sector.
        )pydoc")
      .def("split",
           &FusionTree::split,
           py::arg("n"),
           R"pydoc(
           Split into two separate fusion trees.

           Parameters
           ----------
           n : int
               Where to split. Must fulfill ``2 <= n < self.num_uncoupled``.

           Returns
           -------
           t1 : :class:`FusionTree`
               The part that fuses the ``uncoupled_sectors[:n]`` to ``inner_sectors[n - 2]``
           t2 : :class:`FusionTree`
               The part that fuses ``inner_sectors[n - 2]`` and ``uncoupled_sectors[n:]``
               to ``coupled``.

           See Also
           --------
           insert
           )pydoc")
      .def("split_bottom_vertex",
           &FusionTree::split_bottom_vertex,
           R"pydoc(
           Split off the bottom vertex.

           Graphically::

               |   a b x y z           a  b  x  y     z
               |   │ │ │ │ │           │  │  │  │     │
               |   (self_tree)    =    (rest_tree)    │
               |       │                    │         │
               |       c                    ╰────µ────╯
               |                                 │
               |                                 c

           where `rest_tree` might be empty if ``self.num_uncoupled == 1`` or consist of
           only a single sector with no fusion vertex if ``self.num_uncoupled == 2``.

           Returns
           -------
           rest_tree : FusionTree
               The remaining tree, with one fewer vertex.
           c : Sector
               The old coupled sector.
           mu : int
               The old bottom multiplicity label.
           z : Sector
               The old last uncoupled sector.

           See Also
           --------
           extended
           )pydoc")
      .def(
        "twist",
        [](FusionTree const& self, std::vector<int64> const& idcs, bool overtwist) {
            return linear_combination_to_python(self.twist(idcs, overtwist));
        },
        py::arg("idcs"),
        py::arg("overtwist"),
        R"pydoc(
        Twist some legs above a tree, return the resulting linear combination of trees.

        Parameters
        ----------
        idcs : list of int
            Which uncoupled legs to twist
        overtwist : bool
            The chirality of the twist. If the loop is to the right of the wires, an overtwist is
            such that the free end is on top. See notes below.

        Returns
        -------
        linear_combination : dict {FusionTree: complex}
            The composite object of tree and twist is a linear combination
            ``twisted_self = sum_i a_i X_i``. The returned dictionary has entries
            ``linear_combination[X_i] = a_i`` for the contributions to this linear combination
            (i.e. trees for which the coefficient vanishes may be omitted).

        Notes
        -----
        See the following graphical examples for braid chiralities::

            |   idcs = [-1]                    idcs = [-1]
            |   overtwist = True               overtwist = False
            |
            |   │   │   │   │                  │   │   │   │
            |   │   │   │   │   ╭─╮            │   │   │   │   ╭─╮
            |   │   │   │    ╲ ╱  │            │   │   │    ╲ ╱  │
            |   │   │   │     ╱   │            │   │   │     ╲   │
            |   │   │   │    ╱ ╲  │            │   │   │    ╱ ╲  │
            |   ┢━━━┷━━━┷━━━┷━┓ ╰─╯            ┢━━━┷━━━┷━━━┷━┓ ╰─╯
            |   ┡━━━━━━━━━━━━━┛                ┡━━━━━━━━━━━━━┛
            |   │                              │

        For multiple legs (``len(idcs) > 1``), we twist the together, e.g. here for
        ``idcs=[-2, -1]`` and ``overtwist=True``::

            |   │   │   │   │   ╭──────╮
            |   │   │    ╲   ╲ ╱       │
            |   │   │     ╲   ╱   ╭─╮  │
            |   │   │      ╲ ╱ ╲ ╱  │  │
            |   │   │       ╱   ╱   │  │
            |   │   │      ╱ ╲ ╱ ╲  │  │
            |   │   │     ╱   ╱   ╰─╯  │
            |   │   │    ╱   ╱ ╲       │
            |   ┢━━━┷━━━┷━━━┷━┓ ╰──────╯
            |   ┡━━━━━━━━━━━━━┛
            |   │
        )pydoc");

    py::class_<fusion_trees> ft(m,
                                "fusion_trees",
                                R"pydoc(
                                Iterable over all :class:`FusionTree`\ s with given uncoupled and coupled sectors.

                                This custom iterator has efficient implementations of ``len`` and :meth:`index`, which
                                avoid generating all intermediate trees.

                                TODO elaborate on canonical order of trees -> reference in module level docstring.
                                )pydoc");

    ft.def(
      py::init(
        [](py::object symmetry_obj, py::object uncoupled, Sector coupled, py::object are_dual) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
            std::optional<std::vector<std::uint8_t>> dual;
            if (!are_dual.is_none()) {
                dual = are_dual_from_python(are_dual, unc.size());
            }
            return fusion_trees(std::move(symmetry), std::move(unc), coupled, std::move(dual));
        }),
      py::arg("symmetry"),
      py::arg("uncoupled"),
      py::arg("coupled"),
      py::arg("are_dual") = py::none());

    ft.def_readwrite("symmetry", &fusion_trees::symmetry)
      .def_readwrite("uncoupled", &fusion_trees::uncoupled)
      .def_readwrite("coupled", &fusion_trees::coupled)
      .def_readonly("num_uncoupled", &fusion_trees::num_uncoupled)
      .def_property(
        "are_dual",
        [](py::object self_obj) {
            auto& self = self_obj.cast<fusion_trees&>();
            return py::array(py::dtype::of<bool>(),
                             { self.are_dual.size() },
                             { sizeof(std::uint8_t) },
                             self.are_dual.data(),
                             self_obj);
        },
        [](fusion_trees& self, py::object value) {
            self.are_dual = are_dual_from_python(value, self.num_uncoupled);
        });

    ft.def("__iter__",
           [](fusion_trees const& self) {
               py::list out;
               for (auto const& t : self.all_trees()) {
                   out.append(t);
               }
               return py::iter(out);
           })
      .def("__len__", &fusion_trees::size)
      .def("__str__", &fusion_trees::str)
      .def("__repr__", &fusion_trees::repr)
      .def("index",
           &fusion_trees::index,
           py::arg("tree"),
           R"pydoc(
           The index of a given tree in the iterator.
           )pydoc");
}

} // namespace cyten
