# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

target_sources(cyten
  PRIVATE
  check.cpp
  config.cpp
  tools.cpp
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
