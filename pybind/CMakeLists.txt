# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

target_sources(_core
  PUBLIC
  _core.cpp
  block_backend/py_block_backend.cpp
  py_config.cpp
  py_tools.cpp
  tools/py_cost_polynomials.cpp
  py_version.cpp
  py_check.cpp
  symmetries/py_symmetries.cpp
  symmetries/py_exceptions.cpp
  symmetries/py_styles.cpp
  symmetries/py_sector.cpp
  symmetries/py_base_symmetry.cpp
  symmetries/py_symmetry_factor.cpp
  symmetries/py_group.cpp
  symmetries/py_abelian_group.cpp
  symmetries/py_no_symmetry.cpp
  symmetries/py_u1.cpp
  symmetries/py_zn.cpp
  symmetries/py_su2.cpp
  symmetries/py_sun.cpp
  symmetries/py_fermion_number.cpp
  symmetries/py_fermion_parity.cpp
  symmetries/py_zn_anyon_category.cpp
  symmetries/py_zn_anyon_category2.cpp
  symmetries/py_quantum_double_zn_anyon_category.cpp
  symmetries/py_toric_code_category.cpp
  symmetries/py_fibonacci_anyon_category.cpp
  symmetries/py_ising_anyon_category.cpp
  symmetries/py_su2_k_anyon_category.cpp
  symmetries/py_su3_3_anyon_category.cpp
  symmetries/py_symmetry.cpp
  symmetries/py_trees.cpp
  symmetries/py_spaces.cpp
)
