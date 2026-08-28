#include "../../doc_plus.h"
#include "docstrings/symmetries/factors/sun.h"
#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"

#include <cyten/symmetries/factors/sun.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_sun(py::module_& m)
{
    py::class_<SUN, Group, py::smart_holder> cls(m, "SUN", DOC(cyten, SUN));

    cls.def(py::init([](int N,
                        int64 hweight,
                        std::optional<int64> cg_hweight,
                        std::optional<int64> f_hweight,
                        std::optional<int64> r_hweight,
                        std::optional<std::string> path,
                        std::optional<std::string> filename_base,
                        std::optional<std::string> descriptive_name) {
                return SUN::from_config(N,
                                        hweight,
                                        cg_hweight,
                                        f_hweight,
                                        r_hweight,
                                        std::move(path),
                                        std::move(filename_base),
                                        std::move(descriptive_name));
            }),
            py::arg("N"),
            py::arg("hweight"),
            py::kw_only(),
            py::arg("cg_hweight") = py::none(),
            py::arg("f_hweight") = py::none(),
            py::arg("r_hweight") = py::none(),
            py::arg("path") = py::none(),
            py::arg("filename_base") = py::none(),
            py::arg("descriptive_name") = py::none());

    cls.def(py::init<int, py::object, py::object, py::object, std::optional<std::string>>(),
            py::arg("N"),
            py::arg("CGfile"),
            py::arg("Ffile"),
            py::arg("Rfile"),
            py::arg("descriptive_name") = py::none());

    m.def("su_n_data_filename",
          &su_n_data_filename,
          py::arg("N"),
          py::arg("kind"),
          py::arg("hweight"),
          py::arg("filename_base") = py::none(),
          DOC(cyten, su_n_data_filename));

    m.def("su_n_data_file_path",
          &su_n_data_file_path,
          py::arg("N"),
          py::arg("kind"),
          py::arg("hweight"),
          py::arg("path") = py::none(),
          py::arg("filename_base") = py::none(),
          DOC(cyten, su_n_data_file_path));

    cls.def_readonly("N", &SUN::N)
      .def_readwrite("CGfile", &SUN::CGfile)
      .def_readwrite("Ffile", &SUN::Ffile)
      .def_readwrite("Rfile", &SUN::Rfile)
      .def_static(
        "from_hdf5", &SUN::from_hdf5, py::arg("hdf5_loader"), py::arg("h5gr"), py::arg("subpath"));

    cls.def("hweight_from_CG_hdf5", &SUN::hweight_from_CG_hdf5)
      .def("hweight_from_F_hdf5", &SUN::hweight_from_F_hdf5)
      .def("hweight_from_R_hdf5", &SUN::hweight_from_R_hdf5)
      .def("S_index_irrep_weight",
           &SUN::S_index_irrep_weight,
           py::arg("a"),
           DOC(cyten, SUN, S_index_irrep_weight))
      .def("highest_irrep_in_decomp",
           &SUN::highest_irrep_in_decomp,
           py::arg("a"),
           py::arg("b"),
           DOC(cyten, SUN, highest_irrep_in_decomp))
      .def("dims_of_irreps",
           &SUN::dims_of_irreps,
           py::arg("a"),
           py::arg("b"),
           DOC(cyten, SUN, dims_of_irreps))
      .def("outer_multiplicity_from_CG",
           &SUN::outer_multiplicity_from_CG,
           py::arg("a"),
           py::arg("b"),
           DOC(cyten, SUN, outer_multiplicity_from_CG))
      .def("clebschgordan",
           &SUN::clebschgordan,
           py::arg("a"),
           py::arg("q_a"),
           py::arg("b"),
           py::arg("q_b"),
           py::arg("c"),
           py::arg("q_c"),
           py::arg("mu"),
           DOC(cyten, SUN, clebschgordan))
      .def("_f_symbol_from_CG",
           &SUN::_f_symbol_from_CG,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           DOC(cyten, SUN, _f_symbol_from_CG))
      .def("_r_symbol_from_CG",
           &SUN::_r_symbol_from_CG,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           DOC(cyten, SUN, _r_symbol_from_CG))
      .def("has_data_in_group", &SUN::has_data_in_group, py::arg("group"))
      .def("sanity_check_hdf5",
           &SUN::sanity_check_hdf5,
           py::arg("file"),
           DOC(cyten, SUN, sanity_check_hdf5));
}

} // namespace cyten
