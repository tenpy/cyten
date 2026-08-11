#include <cyten/tools/mappings.h>

#include "../py_cyten_pybind11.h"

#include <complex>
#include <utility>
#include <vector>

namespace cyten {

namespace {

template<typename Mapping>
void
bind_sparse_mapping(py::module_& m, char const* name, char const* doc)
{
    using KT = typename Mapping::Key;
    using Inner = typename Mapping::Inner;

    py::class_<Mapping> cls(m, name);
    cls.doc() = doc;

    cls.def(py::init<>())
      .def(py::init<typename Mapping::Outer>(), py::arg("data"))
      .def_readwrite("data", &Mapping::data)
      .def_static("from_identity",
                  &Mapping::from_identity,
                  py::arg("keys"),
                  R"pydoc(The identity mapping ``e_j -> e_j`` on the given keys)pydoc")
      .def("pre_compose",
           &Mapping::pre_compose,
           py::arg("other"),
           R"pydoc(
The composite ``res_{ik} = \sum_j other_{ij} self{jk}``, such that self acts first.

I.e. we pre-compose self with other, i.e. compose other with self, i.e.::

    pre_compose(self, other) : x ↦ other(self(x)) = (other ∘ self)(x)
)pydoc")
      .def(
        "nonzero_rows",
        &Mapping::nonzero_rows,
        R"pydoc(The idcs ``i`` for which there are entries ``self_{ij} = self[j][i]`` set.)pydoc")
      .def(
        "nonzero_cols",
        &Mapping::nonzero_cols,
        R"pydoc(The idcs ``j`` for which there are entries ``self_{ij} = self[j][i]`` set.)pydoc")
      .def("prune",
           &Mapping::prune,
           py::arg("tol"),
           R"pydoc(Remove small contributions with ``abs(coefficient) <= tol`` in-place.)pydoc")
      .def("__len__", &Mapping::size)
      .def(
        "__contains__",
        [](Mapping const& self, KT const& j) { return self.contains(j); },
        py::arg("j"))
      .def(
        "__getitem__",
        [](Mapping const& self, KT const& j) {
            auto it = self.data.find(j);
            if (it == self.data.end()) {
                throw py::key_error("SparseMapping key not found");
            }
            return it->second; // copy to Python dict
        },
        py::arg("j"))
      .def(
        "__setitem__",
        [](Mapping& self, KT const& j, Inner const& inner) { self.data[j] = inner; },
        py::arg("j"),
        py::arg("value"))
      .def("items",
           [](Mapping const& self) {
               py::list out;
               for (auto const& [j, inner] : self.data) {
                   out.append(py::make_tuple(j, inner));
               }
               return out;
           })
      .def("keys",
           [](Mapping const& self) {
               py::list out;
               for (auto const& [j, inner] : self.data) {
                   (void)inner;
                   out.append(j);
               }
               return out;
           })
      .def("values", [](Mapping const& self) {
          py::list out;
          for (auto const& [j, inner] : self.data) {
              (void)j;
              out.append(inner);
          }
          return out;
      });
}

template<typename IdMapping>
void
bind_identity_mapping(py::module_& m, char const* name)
{
    using KT = typename IdMapping::Key;

    py::class_<IdMapping> cls(m, name);
    cls.doc() =
      R"pydoc(An identity mapping with same call structure as :class:`SparseMapping`)pydoc";

    cls.def(py::init<>())
      .def(py::init<std::vector<KT> const&>(), py::arg("keys"))
      .def_readwrite("keys", &IdMapping::keys)
      .def("pre_compose",
           &IdMapping::pre_compose,
           py::arg("other"),
           R"pydoc(
The composite ``res_{ik} = \sum_j other_{ij} self{jk}``, such that self acts first.

I.e. we pre-compose self with other, i.e. compose other with self, i.e.::

    pre_compose(self, other) : x ↦ other(self(x)) = (other ∘ self)(x)
)pydoc")
      .def("nonzero_rows", &IdMapping::nonzero_rows)
      .def("nonzero_cols", &IdMapping::nonzero_cols)
      .def("prune",
           &IdMapping::prune,
           py::arg("tol"),
           R"pydoc(Remove small entries, in-place (no-op for identity).)pydoc");
}

} // namespace

void
bind_mappings(py::module_& m)
{
    bind_sparse_mapping<SparseMappingFusionTree>(m,
                                                 "SparseMappingFusionTree",
                                                 R"pydoc(
A sparse matrix, where the labels of basis states are a structured type, not just int.

Used in :class:`cyten.backends.fusion_tree_backend.TreePairMapping` and related objects.

To represent the mapping ``e_j -> \sum_i A_{ij} e_i``, we store ``self[j][i] = A_{ij}``.
I.e. a single entry ``self[j][i] = a`` represents the contribution ``e_j -> a e_i``.

Concrete instantiation with :class:`FusionTree` keys and ``complex128`` coefficients.
)pydoc");

    bind_sparse_mapping<SparseMappingFusionTreePair>(m,
                                                     "SparseMappingFusionTreePair",
                                                     R"pydoc(
SparseMapping with ``(FusionTree, FusionTree)`` keys (tree pairs) and ``complex128`` coefficients.
)pydoc");

    bind_identity_mapping<IdentityMappingFusionTree>(m, "IdentityMappingFusionTree");
    bind_identity_mapping<IdentityMappingFusionTreePair>(m, "IdentityMappingFusionTreePair");
}

} // namespace cyten
