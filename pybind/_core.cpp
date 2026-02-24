#include "cyten_pybind11.h"

using namespace cyten;
namespace py = pybind11;
using namespace pybind11::literals; // provides "arg"_a literals

#include <check.h> // TODO: remove check

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Cyten python bindings using pybind11"; // optional module docstring

    bind_config(m);
    bind_tools(m);
    bind_version(m);

    m.def("add", &cyten::add, "A function that adds two numbers");
}
