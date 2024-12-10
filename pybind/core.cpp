// Implements main pybind11 python module cyten._core with all python bindings.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include "check.h"

using namespace std;
using namespace cyten;
namespace py = pybind11;
using namespace pybind11::literals; // provides "arg"_a literals

void bind_symmetries(py::module_ &m);


PYBIND11_MODULE(_core, m) {
    m.doc() = "Cyten python bindings using pybind11"; // optional module docstring
    
    m.def("add", &cyten::add, "A function that adds two numbers");
    
    bind_symmetries(m);

}
