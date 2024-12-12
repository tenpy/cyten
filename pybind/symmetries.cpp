#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <exception>

#include "cyten/symmetries.h"


using namespace std;
using namespace cyten;
namespace py = pybind11;
using namespace pybind11::literals; // provides "arg"_a literals
                                

void bind_symmetries(py::module_ &m){
    py::options options;
    options.disable_enum_members_docstring();

    py::register_exception<SymmetryError>(m, "SymmetryError");

    py::enum_<FusionStyle>(m, "FusionStyle", py::arithmetic(), R"pbdoc(
        Describes properties of fusion, i.e. of the tensor product.

        =================  =============================================================================
        Value              Meaning
        =================  =============================================================================
        single             Fusing sectors results in a single sector ``a ⊗ b = c``, e.g. abelian groups.
        -----------------  -----------------------------------------------------------------------------
        multiple_unique    Every sector appears at most once in pairwise fusion, ``N_symbol in [0, 1]``.
        -----------------  -----------------------------------------------------------------------------
        general            No assumptions, ``N_symbol in [0, 1, 2, 3, ...]``.
        =================  =============================================================================
        )pbdoc")
        .value("single", FusionStyle::single)
        .value("multiple_unique", FusionStyle::multiple_unique)
        .value("general", FusionStyle::general)
        .export_values();

    py::enum_<BraidingStyle>(m, "BraidingStyle", py::arithmetic(), R"pbdoc(
        Describes properties of braiding.

        =============  ===========================================
        Value
        =============  ===========================================
        bosonic        Symmetric braiding with trivial twist
        -------------  -------------------------------------------
        fermionic      Symmetric braiding with non-trivial twist
        -------------  -------------------------------------------
        anyonic        General, non-symmetric braiding
        -------------  -------------------------------------------
        no_braiding    Braiding is not defined
        =============  ===========================================
        )pbdoc")
        .value("bosonic", BraidingStyle::bosonic)
        .value("fermionic", BraidingStyle::fermionic)
        .value("anyonic", BraidingStyle::anyonic) 
        .value("no_braiding", BraidingStyle::no_braiding)
        .export_values();
    
    
    // template<typename Symmetry_subclass>
    // void def_symmetry(py::class_<Symmetry_subclass> sym) {
    //     sym.def_read("fusion_style", &Symmetry::fusion_style)
    //        .def_read("braiding_style", &Symmetry::braiding_style)
    //        .def_read("can_be_dropped", &Symmetry::can_be_dropped);
    // };

    // py::class_<Symmetry> symmetry(m, "Symmetry");
    
    // symmetry.def(py::init<FusionStyle, BraidingStyle, bool>())
}
