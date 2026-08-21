#include "../doc_plus.h"
#include "docstrings/symmetries/symmetry_factor.h"
#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/symmetry.h>
#include <cyten/symmetries/symmetry_factor.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_symmetry_factor(py::module_& m)
{
    py::class_<SymmetryFactor, BaseSymmetry, PySymmetryFactor, py::smart_holder> cls(
      m, "SymmetryFactor", DOC(cyten, SymmetryFactor));

    cls.def(py::init<FusionStyle,
                     BraidingStyle,
                     Sector,
                     std::string,
                     float64,
                     bool,
                     std::optional<std::string>,
                     bool>(),
            py::arg("fusion_style"),
            py::arg("braiding_style"),
            py::arg("trivial_sector"),
            py::arg("group_name"),
            py::arg("num_sectors"),
            py::arg("has_complex_topological_data"),
            py::arg("descriptive_name") = py::none(),
            py::arg("trivial_shift") = true);

    cls.def_readwrite("group_name", &SymmetryFactor::group_name)
      .def_readwrite("descriptive_name", &SymmetryFactor::descriptive_name)
      .def_readwrite("fusion_tensor_dtype", &SymmetryFactor::fusion_tensor_dtype);

    cls
      .def(
        "is_valid_sector",
        [](SymmetryFactor const& self, py::object a) {
            if (py::isinstance<Sector>(a)) {
                return self.is_valid_sector(a.cast<Sector>());
            }
            if (!py::isinstance<py::array>(a)) {
                return false;
            }
            try {
                return self.is_valid_sector(sector_from_numpy(a));
            } catch (...) {
                return false;
            }
        },
        py::arg("a"),
        doc_cpp_ref(R"pydoc(is_valid_sector)pydoc", "cyten::BaseSymmetry::is_valid_sector()"))
      .def(
        "are_valid_sectors",
        [](SymmetryFactor const& self, py::object sectors) {
            if (py::isinstance<SectorArray>(sectors)) {
                return self.are_valid_sectors(sectors.cast<SectorArray>());
            }
            if (!py::isinstance<py::array>(sectors)) {
                return false;
            }
            try {
                return self.are_valid_sectors(sector_array_from_numpy(sectors));
            } catch (...) {
                return false;
            }
        },
        py::arg("sectors"))
      .def("fusion_outcomes",
           &SymmetryFactor::fusion_outcomes,
           py::arg("a"),
           py::arg("b"),
           doc_cpp_ref(R"pydoc(fusion_outcomes)pydoc", "cyten::BaseSymmetry::fusion_outcomes()"))
      .def("__repr__", &SymmetryFactor::repr)
      .def(
        "is_equivalent_to",
        [](SymmetryFactor& self, py::object other) {
            if (py::isinstance<Symmetry>(other)) {
                return other.attr("is_equivalent_to")(py::cast(self)).cast<bool>();
            }
            return self._is_equivalent_factor(other.cast<SymmetryFactor const&>());
        },
        py::arg("other"))
      .def("_is_equivalent_factor",
           &SymmetryFactor::_is_equivalent_factor,
           py::arg("other"),
           DOC(cyten, SymmetryFactor, _is_equivalent_factor))
      .def(
        "as_Symmetry",
        [](py::object self) {
            // Use the Python-held shared_ptr; shared_from_this fails with smart_holder
            // trampolines.
            auto ptr = self.cast<SymmetryFactor::Ptr>();
            return py::cast(std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ ptr }));
        },
        DOC(cyten, SymmetryFactor, as_Symmetry))
      .def("__str__", &SymmetryFactor::str)
      .def("__mul__",
           [](py::object self, py::object other) -> py::object {
               auto self_ptr = self.cast<SymmetryFactor::Ptr>();
               if (py::isinstance<SymmetryFactor>(other)) {
                   return py::cast(std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{
                     self_ptr, other.cast<SymmetryFactor::Ptr>() }));
               }
               if (py::isinstance<Symmetry>(other)) {
                   auto const& sym = other.cast<Symmetry const&>();
                   std::vector<SymmetryFactor::Ptr> factors;
                   factors.reserve(1 + sym.factors.size());
                   factors.push_back(self_ptr);
                   factors.insert(factors.end(), sym.factors.begin(), sym.factors.end());
                   return py::cast(std::make_shared<Symmetry>(std::move(factors)));
               }
               return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
           })
      .def("__eq__",
           [](SymmetryFactor const& self, py::object other) {
               if (!py::isinstance<SymmetryFactor>(other)) {
                   return false;
               }
               return self.equals(other.cast<SymmetryFactor const&>());
           })
      .def(
        "save_hdf5",
        [](SymmetryFactor const& self,
           py::object saver,
           py::object h5gr,
           std::string const& subpath) {
            self.save_hdf5(saver, h5gr, subpath); // virtual dispatch
        },
        py::arg("hdf5_saver"),
        py::arg("h5gr"),
        py::arg("subpath"));
}

} // namespace cyten
