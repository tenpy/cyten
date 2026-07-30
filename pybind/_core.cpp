#include "py_cyten_pybind11.h"

using namespace cyten;

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Cyten python bindings using pybind11"; // optional module docstring

    bind_version(m);
    bind_tools(m);
    bind_config(m);
    bind_block_backend(m);

    bind_check(m); // TODO: remove check
}
