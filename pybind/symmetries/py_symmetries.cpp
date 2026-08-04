#include "py_cyten_pybind11.h"

namespace cyten {

void bind_symmetries_exceptions(py::module_& m);
void bind_symmetries_styles(py::module_& m);
void bind_base_symmetry(py::module_& m);
void bind_symmetry_factor(py::module_& m);
void bind_group(py::module_& m);
void bind_abelian_group(py::module_& m);
void bind_no_symmetry(py::module_& m);
void bind_u1(py::module_& m);
void bind_zn(py::module_& m);
void bind_symmetry(py::module_& m);

void
bind_symmetries(py::module_& m)
{
    bind_symmetries_exceptions(m);
    bind_symmetries_styles(m);
    bind_base_symmetry(m);
    bind_symmetry_factor(m);
    bind_group(m);
    bind_abelian_group(m);
    bind_no_symmetry(m);
    bind_u1(m);
    bind_zn(m);
    bind_symmetry(m);
}

} // namespace cyten
