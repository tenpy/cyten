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
void bind_su2(py::module_& m);
void bind_sun(py::module_& m);
void bind_fermion_number(py::module_& m);
void bind_fermion_parity(py::module_& m);
void bind_zn_anyon_category(py::module_& m);
void bind_zn_anyon_category2(py::module_& m);
void bind_quantum_double_zn_anyon_category(py::module_& m);
void bind_toric_code_category(py::module_& m);
void bind_fibonacci_anyon_category(py::module_& m);
void bind_ising_anyon_category(py::module_& m);
void bind_su2_k_anyon_category(py::module_& m);
void bind_su3_3_anyon_category(py::module_& m);
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
    bind_su2(m);
    bind_sun(m);
    bind_fermion_number(m);
    bind_fermion_parity(m);
    bind_zn_anyon_category(m);
    bind_zn_anyon_category2(m);
    bind_quantum_double_zn_anyon_category(m);
    bind_toric_code_category(m);
    bind_fibonacci_anyon_category(m);
    bind_ising_anyon_category(m);
    bind_su2_k_anyon_category(m);
    bind_su3_3_anyon_category(m);
    bind_symmetry(m);
}

} // namespace cyten
