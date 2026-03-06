#include "py_cyten_pybind11.h"

using namespace cyten;

#include <check.h> // TODO: remove check

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Cyten python bindings using pybind11"; // optional module docstring

    bind_version(m);
    bind_tools(m);
    bind_config(m);
    bind_block_backend(m);

    m.def("add", &cyten::add, "A function that adds two numbers");
}
