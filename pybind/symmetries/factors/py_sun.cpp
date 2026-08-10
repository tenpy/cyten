#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/sun.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_sun(py::module_& m)
{
    py::class_<SUN, Group, py::smart_holder> cls(m,
                                                 "SUN",
                                                 R"pydoc(
                                                 SU(N) group symmetry

                                                 The sectors are arrays of length N which correspond to first rows of normalized Gelfand-Tsetlin
                                                 patterns (see https://arxiv.org/pdf/1009.0437 ).
                                                 E.g. for SU(3) the 8 dimensional irreducible representation is labeled by [2,1,0]

                                                 Clebsch Gordan coefficients and F/R symbols need to be calculated within the
                                                 clebsch_gordan_coefficients package and exported as hdf5 file.

                                                 CGfile: hdf5 file containing the clebsch gordan coefficients
                                                 Ffile: hdf5 file containing the F symbols
                                                 Rfile: hdf5 file containing the R Symbols
                                                 )pydoc");

    cls.def(py::init<int, py::object, py::object, py::object, std::optional<std::string>>(),
            py::arg("N"),
            py::arg("CGfile"),
            py::arg("Ffile"),
            py::arg("Rfile"),
            py::arg("descriptive_name") = py::none());

    cls.def_readonly("N", &SUN::N)
      .def_readwrite("CGfile", &SUN::CGfile)
      .def_readwrite("Ffile", &SUN::Ffile)
      .def_readwrite("Rfile", &SUN::Rfile)
      .def_static(
        "from_hdf5", &SUN::from_hdf5, py::arg("hdf5_loader"), py::arg("h5gr"), py::arg("subpath"));

    cls.def("hweight_from_CG_hdf5", &SUN::hweight_from_CG_hdf5)
      .def("hweight_from_F_hdf5", &SUN::hweight_from_F_hdf5)
      .def("hweight_from_R_hdf5", &SUN::hweight_from_R_hdf5)
      .def("S_index_irrep_weight", &SUN::S_index_irrep_weight, py::arg("a"))
      .def("highest_irrep_in_decomp", &SUN::highest_irrep_in_decomp, py::arg("a"), py::arg("b"))
      .def("dims_of_irreps", &SUN::dims_of_irreps, py::arg("a"), py::arg("b"))
      .def(
        "outer_multiplicity_from_CG", &SUN::outer_multiplicity_from_CG, py::arg("a"), py::arg("b"))
      .def("clebschgordan",
           &SUN::clebschgordan,
           py::arg("a"),
           py::arg("q_a"),
           py::arg("b"),
           py::arg("q_b"),
           py::arg("c"),
           py::arg("q_c"),
           py::arg("mu"))
      .def("_f_symbol_from_CG",
           &SUN::_f_symbol_from_CG,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("_r_symbol_from_CG", &SUN::_r_symbol_from_CG, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("has_data_in_group", &SUN::has_data_in_group, py::arg("group"))
      .def("sanity_check_hdf5", &SUN::sanity_check_hdf5, py::arg("file"));
}

} // namespace cyten
