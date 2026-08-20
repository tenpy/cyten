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
# Checked-in pybind/docstrings/<path>.h mirrors include/cyten/<path>.h.
# Regeneration is optional and only enabled when pybind11_mkdoc + clang/libclang
# are available (see environment.yml). Without them, builds use checked-in files.
option(CYTEN_GENERATE_DOCSTRINGS
  "Regenerate pybind/docstrings/*.h with pybind11_mkdoc when tools are available"
  ON)


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

# Torch's Caffe2/mkl.cmake sets caffe2::mkl INTERFACE_LINK_DIRECTORIES to
# ${MKL_ROOT}/lib/intel64 (etc.). When MKL_ROOT is unset those become the
# bogus absolute paths /lib/intel64, and executables that link cyten (e.g.
# CommonCppTests) fail to resolve libtorch_cpu's MKL/sleef/omp deps. Conda
# ships those libs in $CONDA_PREFIX/lib.
if(NOT MKL_ROOT AND DEFINED ENV{CONDA_PREFIX})
  set(MKL_ROOT "$ENV{CONDA_PREFIX}")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Belt-and-suspenders: overwrite any empty-MKL_ROOT stub dirs Torch still set.
if(TARGET caffe2::mkl AND MKL_ROOT)
  set_property(TARGET caffe2::mkl PROPERTY INTERFACE_LINK_DIRECTORIES
    "${MKL_ROOT}/lib")
  # Executables that link Torch (e.g. CommonCppTests) must resolve libtorch_cpu's
  # DT_NEEDED deps (MKL, sleef, omp) at link time; -L alone is not enough for
  # that — pass -rpath-link. Shared modules like _core do not need this.
  target_link_options(caffe2::mkl INTERFACE "LINKER:-rpath-link=${MKL_ROOT}/lib")
endif()

cmake_print_variables(Torch_DIR)
cmake_print_variables(TORCH_INSTALL_PREFIX)
cmake_print_variables(TORCH_LIBRARIES)

# #####################################################################
# ## C++ LIBRARY
# #####################################################################

add_library(cyten STATIC)
set_property(TARGET cyten PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET cyten PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET cyten PROPERTY VISIBILITY_INLINES_HIDDEN ON)
# CXX_VISIBILITY_PRESET is ignored for STATIC libraries; pybind11 still needs
# -fvisibility=hidden on every TU that includes pybind types (see FAQ:
# "'SomeClass' declared with greater visibility than the type of its field").
# pybind11_add_module(_core) already adds the flag; cyten must get it explicitly.
target_compile_options(cyten PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
)

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

# #####################################################################
# ## pybind11_mkdoc (optional: one docstring header per include/cyten/*.h)
# #####################################################################
# Layout: include/cyten/<rel>  →  pybind/docstrings/<rel>
# Each pybind TU includes only the docstring headers it needs, so regenerating
# one include does not force recompiling unrelated py_*.cpp files.
#
# Pilot list — extend as more bindings switch to DOC(...):
set(CYTEN_MKDOC_HEADERS
  tensors/ops_algebra.h
)

set(CYTEN_DOCSTRINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind/docstrings")
set(CYTEN_CAN_GENERATE_DOCSTRINGS FALSE)

if(CYTEN_GENERATE_DOCSTRINGS)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11_mkdoc"
    RESULT_VARIABLE _cyten_mkdoc_import
    OUTPUT_QUIET ERROR_QUIET
  )
  execute_process(
    COMMAND clang -print-resource-dir
    OUTPUT_VARIABLE CYTEN_CLANG_RESOURCE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _cyten_clang_res
  )
  set(_cyten_libclang "")
  if(DEFINED ENV{LIBCLANG_PATH} AND EXISTS "$ENV{LIBCLANG_PATH}")
    set(_cyten_libclang "$ENV{LIBCLANG_PATH}")
  elseif(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib/libclang.so")
    set(_cyten_libclang "$ENV{CONDA_PREFIX}/lib/libclang.so")
  endif()

  if(_cyten_mkdoc_import EQUAL 0
     AND _cyten_clang_res EQUAL 0
     AND NOT CYTEN_CLANG_RESOURCE_DIR STREQUAL ""
     AND NOT _cyten_libclang STREQUAL "")
    set(CYTEN_CAN_GENERATE_DOCSTRINGS TRUE)
  else()
    message(STATUS
      "CYTEN_GENERATE_DOCSTRINGS: tools missing (pybind11_mkdoc / clang / "
      "libclang); using checked-in ${CYTEN_DOCSTRINGS_DIR}/")
  endif()
endif()

set(_cyten_mkdoc_stamps "")
if(CYTEN_CAN_GENERATE_DOCSTRINGS)
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
    OUTPUT_VARIABLE CYTEN_PYBIND11_INCLUDE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
  )
  set(CYTEN_MKDOC_FIX "${CMAKE_CURRENT_SOURCE_DIR}/scripts/fix_mkdoc_numpy_sections.py")
endif()

foreach(_cyten_hdr_rel IN LISTS CYTEN_MKDOC_HEADERS)
  set(_cyten_doc_out "${CYTEN_DOCSTRINGS_DIR}/${_cyten_hdr_rel}")
  if(NOT EXISTS "${_cyten_doc_out}" AND NOT CYTEN_CAN_GENERATE_DOCSTRINGS)
    message(FATAL_ERROR
      "Missing ${_cyten_doc_out}. Either commit/restore that file or install "
      "pybind11-mkdoc + clangdev + libclang and configure with "
      "CYTEN_GENERATE_DOCSTRINGS=ON.")
  endif()

  if(CYTEN_CAN_GENERATE_DOCSTRINGS)
    set(_cyten_src_hdr "${CMAKE_CURRENT_SOURCE_DIR}/include/cyten/${_cyten_hdr_rel}")
    set(_cyten_safe_rel "${_cyten_hdr_rel}")
    string(REPLACE "/" "_" _cyten_safe_rel "${_cyten_safe_rel}")
    string(REPLACE "." "_" _cyten_safe_rel "${_cyten_safe_rel}")
    set(_cyten_raw "${CMAKE_CURRENT_BINARY_DIR}/docstrings_${_cyten_safe_rel}.raw.h")
    set(_cyten_gen "${CMAKE_CURRENT_BINARY_DIR}/docstrings_${_cyten_safe_rel}.generated.h")
    set(_cyten_stamp "${CMAKE_CURRENT_BINARY_DIR}/docstrings_${_cyten_safe_rel}.stamp")
    get_filename_component(_cyten_doc_outdir "${_cyten_doc_out}" DIRECTORY)

    # Stamp in the build tree so `ninja clean` does not delete checked-in files.
    add_custom_command(
      OUTPUT ${_cyten_stamp}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_cyten_doc_outdir}
      COMMAND ${CMAKE_COMMAND} -E env "LIBCLANG_PATH=${_cyten_libclang}"
              ${Python_EXECUTABLE} -m pybind11_mkdoc
              -o ${_cyten_raw}
              -std=c++23
              -I${CMAKE_CURRENT_SOURCE_DIR}/include
              -I${CYTEN_PYBIND11_INCLUDE}
              -I${Python_INCLUDE_DIRS}
              -I${CYTEN_CLANG_RESOURCE_DIR}/include
              ${_cyten_src_hdr}
      COMMAND ${Python_EXECUTABLE} ${CYTEN_MKDOC_FIX} ${_cyten_raw}
              -o ${_cyten_gen}
              --source-rel ${_cyten_hdr_rel}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_cyten_gen} ${_cyten_doc_out}
      COMMAND ${CMAKE_COMMAND} -E touch ${_cyten_stamp}
      DEPENDS ${_cyten_src_hdr} ${CYTEN_MKDOC_FIX}
      COMMENT "Regenerating pybind/docstrings/${_cyten_hdr_rel}"
      VERBATIM
    )
    list(APPEND _cyten_mkdoc_stamps ${_cyten_stamp})
  endif()
endforeach()

if(CYTEN_CAN_GENERATE_DOCSTRINGS)
  add_custom_target(cyten_mkdoc_docstrings DEPENDS ${_cyten_mkdoc_stamps})
  add_dependencies(_core cyten_mkdoc_docstrings)
  message(STATUS
    "CYTEN_GENERATE_DOCSTRINGS: ON (${CYTEN_MKDOC_HEADERS})")
else()
  add_custom_target(cyten_mkdoc_docstrings
    COMMENT "cyten_mkdoc_docstrings skipped (using checked-in pybind/docstrings/)"
  )
  message(STATUS "CYTEN_GENERATE_DOCSTRINGS: OFF (using checked-in pybind/docstrings/)")
endif()


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
