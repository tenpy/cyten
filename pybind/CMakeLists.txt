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
  symmetries/py_base_symmetry.cpp
  symmetries/py_symmetry_factor.cpp
  symmetries/py_group.cpp
  symmetries/py_abelian_group.cpp
  symmetries/py_no_symmetry.cpp
  symmetries/py_u1.cpp
  symmetries/py_zn.cpp
  symmetries/py_su2.cpp
  symmetries/py_sun.cpp
  symmetries/py_symmetry.cpp
)
