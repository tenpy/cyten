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

                                                 Clebsch-Gordan coefficients and F/R symbols need to be calculated with the
                                                 clebsch_gordan_coefficients package and exported as HDF5 files.

                                                 There are two ways to construct an SUN:

                                                 1. From the standard data files, resolved from the config::

                                                        SUN(N, hweight, *, cg_hweight=None, f_hweight=None, r_hweight=None,
                                                            path=None, filename_base=None, descriptive_name=None)

                                                    The three files are looked up as
                                                    ``{su_n_data_path}/{su_n_data_filename_base}_N{N}_{CG|F|R}_hweight{H}.hdf5``,
                                                    with the two braced names taken from the cyten config (see :mod:`cyten.config`).
                                                    ``hweight`` sets all three highest weights; ``cg_hweight`` / ``f_hweight`` /
                                                    ``r_hweight`` override them individually. The CG highest weight must be >= the
                                                    F and R highest weights (they are usually all equal).
                                                    ``path`` and ``filename_base`` override the config options for this call only.
                                                    Use :func:`su_n_data_file_path` to see where cyten will look.

                                                 2. From open ``h5py.File`` handles::

                                                        SUN(N, CGfile, Ffile, Rfile, descriptive_name=None)

                                                    CGfile: hdf5 file containing the clebsch gordan coefficients
                                                    Ffile: hdf5 file containing the F symbols
                                                    Rfile: hdf5 file containing the R Symbols
                                                 )pydoc");

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
          R"pydoc(
          Standard file name for SU(N) symmetry data.

          ``'{base}_N{N}_{kind}_hweight{hweight}.hdf5'`` with ``kind`` in ``{'CG', 'F', 'R'}``
          (case-insensitive) and ``base`` defaulting to the ``su_n_data_filename_base`` config option.
          )pydoc");

    m.def("su_n_data_file_path",
          &su_n_data_file_path,
          py::arg("N"),
          py::arg("kind"),
          py::arg("hweight"),
          py::arg("path") = py::none(),
          py::arg("filename_base") = py::none(),
          R"pydoc(
          Full path where cyten looks for an SU(N) data file.

          ``path`` defaults to the ``su_n_data_path`` config option. Useful to check where cyten
          expects your generated data files to live.
          )pydoc");

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
           R"pydoc(
           To every SU(N) irrep, labeled by the first row of a GT pattern, we can assign an integer S.
           )pydoc")
      .def("highest_irrep_in_decomp",
           &SUN::highest_irrep_in_decomp,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Returns the highest irrep which appears in the decomposition of a x b.
           )pydoc")
      .def("dims_of_irreps",
           &SUN::dims_of_irreps,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Returns a dictionary with irreps as keys and their dimension as values.

           The irreps are the ones appearing in the decomposition of a x b
           Does not contain multiplicities!
           )pydoc")
      .def("outer_multiplicity_from_CG",
           &SUN::outer_multiplicity_from_CG,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Returns a dictionary with the outer multiplicities for the irreps in the decomposition of a x b.
           )pydoc")
      .def("clebschgordan",
           &SUN::clebschgordan,
           py::arg("a"),
           py::arg("q_a"),
           py::arg("b"),
           py::arg("q_b"),
           py::arg("c"),
           py::arg("q_c"),
           py::arg("mu"),
           R"pydoc(
           Evaluate a single Clebsch-Gordan coefficient.

           Parameters
           ----------
           a, b, c
               Sector for the fusion :math:`a \otimes b \mapsto c`.
           q_a, q_b, q_c:
               Indices of the Gelfand Tsetlin pattern
           mu:
               multiplicity index 1 <= mu

           Returns
           -------
           The CG coefficient for the given input
           )pydoc")
      .def("_f_symbol_from_CG",
           &SUN::_f_symbol_from_CG,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           R"pydoc(
           Returns the F symbol for the specified input irreps calculated from CG coefficients.

           a,b,c,d,e,f are irrep labels, i.e. first rows of GT patterns
           output is the conjugated F symbol [F^{abc}_{def}]^*_{mu,nu,kappa, lambda}
           where a x b = mu c, c x d =nu e, b x d= kappa f and a x f =lambda e

           Parameters
           ----------
           a, b, c, d, e, f:   Sector
               Irreps specifying the CG coefficient.
           )pydoc")
      .def("_r_symbol_from_CG",
           &SUN::_r_symbol_from_CG,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           Returns the R symbol for the specified input irreps calculated from CG coefficients.

           Parameters
           ----------
           a, b, c:   Sector
               Irreps specifying the R symbol.
           )pydoc")
      .def("has_data_in_group", &SUN::has_data_in_group, py::arg("group"))
      .def("sanity_check_hdf5",
           &SUN::sanity_check_hdf5,
           py::arg("file"),
           R"pydoc(
           Sanity check for Hdf5 files containing CG-coefficients, F-symbols or R-symbols.

           This method takes a Hdf5 file and checks if it has the required structure and if
           the necessary data has been saved to it. This excludes the possibility of using incompletely generated files,
           but cannot guarantee completeness of the file and correctness of the data in the file.
           In particular, consistency of the data in the file should be checked by the cyten tests for SU(N) symmetry.
           )pydoc");
}

} // namespace cyten
