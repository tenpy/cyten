message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS " Project cyten, A Cross-section of Python & C++, Tensor network library ")
message(STATUS "------------------------------------------------------------------------")
message(STATUS "")

# #####################################################################
# ## CMAKE and CXX VERSION
# #####################################################################
cmake_minimum_required(VERSION 3.27)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message(STATUS " Generator: ${CMAKE_GENERATOR}")
message(STATUS " Build Target: ${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS " Installation Prefix: ${CMAKE_INSTALL_PREFIX}")

# #####################################################################
# ## Version
# #####################################################################

# Scikit-build-core sets version
if(NOT SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION 0.0.1)
endif()

# #####################################################################
# ## COMMAND LINE OPTIONS
# #####################################################################
option(BUILD_DOC "Build API documentation" OFF)
option(LINTING "Use clang-tidy for linting" OFF)
option(BUILD_TESTING "Build ctest tests" ON)


# #####################################################################
# ## PROJECT
# #####################################################################
project(
    cyten
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

include(CMakePrintHelpers)
cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(CMAKE_CXX_STANDARD)

# #####################################################################
# ## find python
# #####################################################################

set(Python_FIND_VIRTUALENV ONLY)
set(Python_FIND_STRATEGY LOCATION)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed)

find_package(pybind11 CONFIG REQUIRED)
# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
cmake_print_variables(Python_VERSION)
cmake_print_variables(Python_INCLUDE_DIRS)
cmake_print_variables(Python_NumPy_VERSION)
cmake_print_variables(Python_NumPy_INCLUDE_DIRS)


# #####################################################################
# ## find pytorch (same shared libs as `import torch`)
# #####################################################################
#
# Discover Torch via the active interpreter's installed package so cyten._core
# links the same libtorch / libc10 that Python's torch._C loads. That makes
# `import cyten` + `import torch` in one process safe (one shared libtorch).
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE CYTEN_TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE CYTEN_TORCH_CMAKE_PREFIX_RESULT
)
if(NOT CYTEN_TORCH_CMAKE_PREFIX_RESULT EQUAL 0 OR CYTEN_TORCH_CMAKE_PREFIX STREQUAL "")
  message(FATAL_ERROR
    "Could not obtain torch.utils.cmake_prefix_path from ${Python_EXECUTABLE}. "
    "Install the matching Python torch package in this environment.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${CYTEN_TORCH_CMAKE_PREFIX}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
cmake_print_variables(Torch_DIR)
cmake_print_variables(TORCH_INSTALL_PREFIX)
cmake_print_variables(TORCH_LIBRARIES)

# #####################################################################
# ## C++ LIBRARY
# #####################################################################

add_library(cyten STATIC)
set_property(TARGET cyten PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET cyten PROPERTY VISIBILITY_INLINES_HIDDEN ON)

target_include_directories(cyten
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
add_subdirectory(src)
target_compile_options(cyten PUBLIC -fPIC)

target_link_libraries(cyten PUBLIC pybind11::pybind11 "${TORCH_LIBRARIES}")

# #####################################################################
# ## PYTHON BINDINGS
# #####################################################################

pybind11_add_module(_core pybind/_core.cpp)

target_include_directories(_core
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind
  PUBLIC "${PROJECT_SOURCE_DIR}/include")
add_subdirectory(pybind)  # adds source files to _core

target_link_libraries(_core PUBLIC cyten)

# Resolve libtorch/c10 from the same place as Python torch._C.
# Installed layout: site-packages/cyten/_core*.so
#   $ORIGIN/../../..  -> env/lib (conda/micromamba shared torch libs)
#   $ORIGIN/../torch/lib -> site-packages/torch/lib (pip wheel layout)
# Do not use INSTALL_RPATH_USE_LINK_PATH: Torch/Caffe2 link dirs may inject
# unrelated paths (e.g. Intel MKL stubs) that we do not want on the module.
get_filename_component(_cyten_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib" ABSOLUTE)
set(_cyten_core_rpath "$ORIGIN/../../..;$ORIGIN/../torch/lib;${_cyten_torch_lib_dir}")
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  list(APPEND _cyten_core_rpath "$ENV{CONDA_PREFIX}/lib")
endif()
# list(APPEND) on a ;-string splits; rebuild a single RPATH string.
string(JOIN ":" _cyten_core_rpath_str ${_cyten_core_rpath})
set_target_properties(_core PROPERTIES
  BUILD_RPATH "${_cyten_core_rpath_str}"
  INSTALL_RPATH "${_cyten_core_rpath_str}"
  INSTALL_RPATH_USE_LINK_PATH FALSE
)


# The install directory is the output (wheel) directory
install(TARGETS _core LIBRARY DESTINATION cyten)


# #####################################################################
# ## DOCUMENTATION
# #####################################################################
if(BUILD_DOC)
  find_package(Doxygen REQUIRED)

  if(DOXYGEN_FOUND)
    configure_file(cyten_Doxyfile.in cytenDoxyfile)
    add_custom_target(doc ALL
      ${DOXYGEN_EXECUTABLE} ${PROJECT_SOURCE_DIR}/docs/Doxyfile
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/docs
      COMMENT "Generating API documentation with Doxygen in docs/build_docs/" VERBATIM
    )
  endif(DOXYGEN_FOUND)

  message(STATUS " Build Documentation: YES")
  message(STATUS "  - Doxygen: ${DOXYGEN_EXECUTABLE}")
else()
  message(STATUS " Build Documentation: NO")
endif()

if(DOXYGEN_FOUND)
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/docs/
    DESTINATION docs
    COMPONENT documentation
  )
endif()


# #####################################################################
# ## clang-tidy for linting
# #####################################################################
if (LINTING)
  find_program(CLANG_TIDY_EXE "clang-tidy")
  if(CLANG_TIDY_EXE)
    message(STATUS "clang-tidy found: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY ${CLANG_TIDY_EXE})
  else()
    message(WARNING "clang-tidy not found!")
  endif()
endif()


# #####################################################################
# ## CTest
# #####################################################################

include(CTest)
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests/ctest)
endif()
