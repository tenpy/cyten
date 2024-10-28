#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "symmetries.h"

using namespace std;
using namespace cyten;
namespace py = pybind11;
using namespace pybind11::literals; // provides "arg"_a literals
                                

void bind_symmetries(py::module_ &m){

    py::enum_<FusionStyle>(m, "FusionStyle", py::arithmetic())
        .value("single", FusionStyle::single)
        .value("multiple_unique", FusionStyle::multiple_unique)
        .value("general", FusionStyle::general); 

    py::enum_<BraidingStyle>(m, "BraidingStyle", py::arithmetic())
        .value("bosonic", BraidingStyle::bosonic)
        .value("fermionic", BraidingStyle::fermionic)
        .value("anyonic", BraidingStyle::anyonic) 
        .value("no_braiding", BraidingStyle::no_braiding);

    // py::class_<Symmetry> symmetry(m, "Symmetry");
    
    // symmetry.def(py::init<FusionStyle, BraidingStyle, bool>())
    //     .def_read("fusion_style", &Symmetry::fusion_style)
    //     .def_read("braiding_style", &Symmetry::braiding_style)
    //     .def_read("can_be_dropped", &Symmetry::can_be_dropped);
}