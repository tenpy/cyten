#include <cyten/tools/mappings.h>

#include "../py_cyten_pybind11.h"
#include "../doc_plus.h"
#include "docstrings/tools/mappings.h"

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
                  doc_cpp_ref(R"pydoc(from_identity)pydoc", "cyten::Mapping::from_identity()"))
      .def("pre_compose",
           &Mapping::pre_compose,
           py::arg("other"),
           doc_cpp_ref(R"pydoc(pre_compose)pydoc", "cyten::Mapping::pre_compose()"))
      .def(
        "nonzero_rows",
        &Mapping::nonzero_rows,
        doc_cpp_ref(R"pydoc(nonzero_rows)pydoc", "cyten::Mapping::nonzero_rows()"))
      .def(
        "nonzero_cols",
        &Mapping::nonzero_cols,
        doc_cpp_ref(R"pydoc(nonzero_cols)pydoc", "cyten::Mapping::nonzero_cols()"))
      .def("prune",
           &Mapping::prune,
           py::arg("tol"),
           doc_cpp_ref(R"pydoc(prune)pydoc", "cyten::Mapping::prune()"))
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
    cls.doc() = doc_cpp_ref(R"pydoc(IdMapping)pydoc", "cyten::IdMapping");

    cls.def(py::init<>())
      .def(py::init<std::vector<KT> const&>(), py::arg("keys"))
      .def_readwrite("keys", &IdMapping::keys)
      .def("pre_compose",
           &IdMapping::pre_compose,
           py::arg("other"),
           doc_cpp_ref(R"pydoc(pre_compose)pydoc", "cyten::IdMapping::pre_compose()"))
      .def("nonzero_rows", &IdMapping::nonzero_rows)
      .def("nonzero_cols", &IdMapping::nonzero_cols)
      .def("prune",
           &IdMapping::prune,
           py::arg("tol"),
           doc_cpp_ref(R"pydoc(prune)pydoc", "cyten::IdMapping::prune()"));
}

} // namespace

void
bind_mappings(py::module_& m)
{
    bind_sparse_mapping<SparseMappingFusionTree>(m,
                                                 "SparseMappingFusionTree",
                                                 doc_cpp_ref(R"pydoc(prune)pydoc", "cyten::IdMapping::prune()"));

    bind_sparse_mapping<SparseMappingFusionTreePair>(m,
                                                     "SparseMappingFusionTreePair",
                                                     doc_cpp_ref(R"pydoc(prune)pydoc", "cyten::IdMapping::prune()"));

    bind_identity_mapping<IdentityMappingFusionTree>(m, "IdentityMappingFusionTree");
    bind_identity_mapping<IdentityMappingFusionTreePair>(m, "IdentityMappingFusionTreePair");
}

} // namespace cyten
