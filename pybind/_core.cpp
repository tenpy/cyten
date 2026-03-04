#include "cyten_pybind11.h"

using namespace cyten;

#include <check.h> // TODO: remove check

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Cyten python bindings using pybind11"; // optional module docstring

    bind_block_backend(m);
    bind_block_backend_numpy(m);
    bind_config(m);
    bind_dtypes(m);
    bind_tools(m);
    bind_version(m);

    m.def("add", &cyten::add, "A function that adds two numbers");
}
