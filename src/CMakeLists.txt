# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

target_sources(cyten
  PRIVATE
  backends/tensor_backend.cpp
  backends/no_symmetry.cpp
  backends/abelian.cpp
  backends/fusion_tree_backend.cpp
  backends/fusion_tree_mapping.cpp
  backends/fusion_tree_permute.cpp
  backends/backend_factory.cpp
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
  symmetries/fusion_symbol.cpp
  symmetries/sector_array.cpp
  symmetries/group.cpp
  symmetries/abelian_group.cpp
  symmetries/factors/no_symmetry.cpp
  symmetries/factors/u1.cpp
  symmetries/factors/zn.cpp
  symmetries/factors/su2.cpp
  symmetries/factors/sun.cpp
  symmetries/factors/fermion_number.cpp
  symmetries/factors/fermion_parity.cpp
  symmetries/factors/zn_anyon_category.cpp
  symmetries/factors/zn_anyon_category2.cpp
  symmetries/factors/quantum_double_zn_anyon_category.cpp
  symmetries/factors/toric_code_category.cpp
  symmetries/factors/fibonacci_anyon_category.cpp
  symmetries/factors/ising_anyon_category.cpp
  symmetries/factors/su2_k_anyon_category.cpp
  symmetries/factors/su3_3_anyon_category.cpp
  symmetries/trees.cpp
  symmetries/spaces.cpp
  tools.cpp
  tools/cost_polynomials.cpp
  tools/mappings.cpp
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
