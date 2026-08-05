#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/sector_ops.h>
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
    auto casted = py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast>::ensure(arr);
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
        return sector_array_from_sector(obj.cast<Sector>());
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

} // namespace

void
bind_trees(py::module_& m)
{
    py::class_<FusionTree> cls(
      m,
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
                auto dual = are_dual_from_python(are_dual, unc.num_sectors);
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
    cls.def_property(
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

    cls.def_property_readonly("pre_Z_uncoupled", &FusionTree::pre_Z_uncoupled);

    cls.def("test_sanity", &FusionTree::test_sanity, "Perform sanity checks.")
      .def_static(
        "from_abelian_symmetry",
        [](py::object symmetry_obj, py::object uncoupled, py::object are_dual) {
            auto symmetry = symmetry_from_python(symmetry_obj);
            SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
            auto dual = are_dual_from_python(are_dual, unc.num_sectors);
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
        "The empty tree with no uncoupled sectors.")
      .def_static(
        "from_sector",
        [](py::object symmetry_obj, Sector sector, bool is_dual) {
            return FusionTree::from_sector(symmetry_from_python(symmetry_obj), sector, is_dual);
        },
        py::arg("symmetry"),
        py::arg("sector"),
        py::arg("is_dual"),
        "A tree with a single uncoupled sector and no nodes.")
      .def("__hash__", &FusionTree::hash)
      .def("__eq__", &FusionTree::operator==, py::arg("other"))
      .def("ascii_diagram",
           &FusionTree::ascii_diagram,
           py::arg("dagger") = false,
           "Visual representation of the tree as ASCII art.")
      .def_static(
        "_str_uncoupled_coupled",
        [](Symmetry const& symmetry,
           SectorArray const& uncoupled,
           Sector coupled,
           py::object are_dual) {
            auto dual = are_dual_from_python(are_dual, uncoupled.num_sectors);
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
      .def_static("bend_leg",
                  &FusionTree::bend_leg,
                  py::arg("X"),
                  py::arg("Y"),
                  py::arg("bend_downward"),
                  py::arg("do_conj") = false,
                  "Bend a leg on a tree-pair, return the resulting linear combination of tree-pairs.")
      .def("braid",
           &FusionTree::braid,
           py::arg("j"),
           py::arg("overbraid"),
           py::arg("cutoff") = 1e-16,
           py::arg("do_conj") = false,
           "Braid a leg on a fusion tree, return the resulting linear combination of trees.")
      .def("vertex_labels",
           &FusionTree::vertex_labels,
           py::arg("n"),
           "For the ``n``-th fusion vertex, get the respective sectors.")
      .def("modify_vertex_labels",
           &FusionTree::modify_vertex_labels,
           py::arg("n"),
           py::arg("a"),
           py::arg("b"),
           py::arg("mu"),
           py::arg("c"),
           py::arg("copy") = true,
           "Update the multiplicity and the three sectors around the ``n``-th vertex.")
      .def("__str__", &FusionTree::str)
      .def("__repr__", &FusionTree::repr)
      .def(
        "to_dense_block",
        [](FusionTree const& self,
           py::object backend,
           py::object dtype,
           bool understood_braiding) {
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
        "Get the matrix elements of the map as a backend Block.")
      .def("copy", &FusionTree::copy, py::arg("deep") = true, "Return a shallow (or deep) copy.")
      .def("extended",
           &FusionTree::extended,
           py::arg("new_uncoupled"),
           py::arg("mu"),
           py::arg("new_coupled"),
           py::arg("is_dual"),
           "A new tree, from adding a new fusion node at the bottom, below the coupled sector.")
      .def("insert",
           &FusionTree::insert,
           py::arg("t2"),
           "Insert a tree `t2` above the first uncoupled sector.")
      .def("insert_at",
           &FusionTree::insert_at,
           py::arg("n"),
           py::arg("t2"),
           py::arg("eps") = 1.0e-14,
           "Insert a tree `t2` above the `n`-th uncoupled sector.")
      .def("outer",
           &FusionTree::outer,
           py::arg("right_tree"),
           py::arg("eps") = 1.0e-14,
           "Outer product with another tree.")
      .def("split",
           &FusionTree::split,
           py::arg("n"),
           "Split into two separate fusion trees.")
      .def("split_bottom_vertex",
           &FusionTree::split_bottom_vertex,
           "Split off the bottom vertex.")
      .def("twist",
           &FusionTree::twist,
           py::arg("idcs"),
           py::arg("overtwist"),
           "Twist some legs above a tree, return the resulting linear combination of trees.");

    py::class_<fusion_trees> ft(
      m,
      "fusion_trees",
      R"pydoc(
      Iterable over all :class:`FusionTree`\ s with given uncoupled and coupled sectors.

      This custom iterator has efficient implementations of ``len`` and :meth:`index`, which
      avoid generating all intermediate trees.

      TODO elaborate on canonical order of trees -> reference in module level docstring.
      )pydoc");

    ft.def(py::init([](py::object symmetry_obj,
                       py::object uncoupled,
                       Sector coupled,
                       py::object are_dual) {
               auto symmetry = symmetry_from_python(symmetry_obj);
               SectorArray unc = sector_array_from_python(uncoupled, *symmetry);
               std::optional<std::vector<std::uint8_t>> dual;
               if (!are_dual.is_none()) {
                   dual = are_dual_from_python(are_dual, unc.num_sectors);
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
           "The index of a given tree in the iterator.");
}

} // namespace cyten
