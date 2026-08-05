# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

target_sources(cyten
  PRIVATE
  block_backend/array_api.cpp
  block_backend/block_backend.cpp
  block_backend/dtypes.cpp
  block_backend/numpy.cpp
  block_backend/torch.cpp
  check.cpp
  config.cpp
  symmetries/base_symmetry.cpp
  symmetries/symmetry_factor.cpp
  symmetries/symmetry.cpp
  symmetries/sector_numpy.cpp
  symmetries/sector_ops.cpp
  symmetries/group.cpp
  symmetries/abelian_group.cpp
  symmetries/no_symmetry.cpp
  symmetries/u1.cpp
  symmetries/zn.cpp
  symmetries/su2.cpp
  symmetries/sun.cpp
  symmetries/fermion_number.cpp
  symmetries/fermion_parity.cpp
  symmetries/zn_anyon_category.cpp
  symmetries/zn_anyon_category2.cpp
  symmetries/quantum_double_zn_anyon_category.cpp
  symmetries/toric_code_category.cpp
  symmetries/fibonacci_anyon_category.cpp
  symmetries/ising_anyon_category.cpp
  symmetries/su2_k_anyon_category.cpp
  symmetries/su3_3_anyon_category.cpp
  tools.cpp
  tools/cost_polynomials.cpp
  version.cpp
)


# #####################################################################
# ## build version.h
# #####################################################################
add_custom_target(gen-version-h
  COMMAND "${CMAKE_COMMAND}"
    "-D" "SRC_DIR=${PROJECT_SOURCE_DIR}/src"
    "-P" "${PROJECT_SOURCE_DIR}/src/version-h-gen.cmake"
  COMMENT "Generating git version file"
)

add_dependencies(cyten gen-version-h)
