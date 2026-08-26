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

# Prefer the conda-forge compiler (headers + libstdc++) that conda Python
# actually loads. A newer system g++ (e.g. Fedora GCC 16) emits symbols
# such as GLIBCXX_3.4.36 that conda's libstdc++ (GCC 15) does not provide;
# `import cyten` then fails because Python has already loaded the older
# libstdc++. Install with: conda install -c conda-forge cxx-compiler
if(NOT CMAKE_CXX_COMPILER AND DEFINED ENV{CONDA_PREFIX})
  foreach(_cyten_cxx
      "$ENV{CONDA_PREFIX}/bin/c++"
      "$ENV{CONDA_PREFIX}/bin/g++")
    if(EXISTS "${_cyten_cxx}")
      set(CMAKE_CXX_COMPILER "${_cyten_cxx}")
      break()
    endif()
  endforeach()
  if(NOT CMAKE_CXX_COMPILER)
    file(GLOB _cyten_conda_gxx "$ENV{CONDA_PREFIX}/bin/*-conda-*-g++")
    if(_cyten_conda_gxx)
      list(GET _cyten_conda_gxx 0 CMAKE_CXX_COMPILER)
    endif()
  endif()
endif()

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

# Reject a compiler whose libstdc++ is newer than the one conda Python loads.
function(cyten_glibcxx_max so_file out_var)
  set(_max "")
  if(EXISTS "${so_file}")
    execute_process(
      COMMAND strings -a "${so_file}"
      COMMAND grep "^GLIBCXX_[0-9]"
      COMMAND sort -V
      COMMAND tail -1
      OUTPUT_VARIABLE _sym
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET)
    string(REGEX REPLACE "^GLIBCXX_" "" _max "${_sym}")
  endif()
  set(${out_var} "${_max}" PARENT_SCOPE)
endfunction()

set(_cyten_conda_stdcxx "")
if(DEFINED ENV{CONDA_PREFIX}
   AND EXISTS "$ENV{CONDA_PREFIX}/lib/libstdc++.so.6")
  set(_cyten_conda_stdcxx "$ENV{CONDA_PREFIX}/lib/libstdc++.so.6")
endif()
if(_cyten_conda_stdcxx)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so
    OUTPUT_VARIABLE _cyten_cxx_stdcxx
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(_cyten_cxx_stdcxx "${_cyten_cxx_stdcxx}" REALPATH)
  get_filename_component(_cyten_conda_stdcxx_real
    "${_cyten_conda_stdcxx}" REALPATH)
  cyten_glibcxx_max("${_cyten_cxx_stdcxx}" _cyten_cxx_glibcxx)
  cyten_glibcxx_max("${_cyten_conda_stdcxx_real}" _cyten_conda_glibcxx)
  if(_cyten_cxx_glibcxx AND _cyten_conda_glibcxx
     AND _cyten_cxx_glibcxx VERSION_GREATER _cyten_conda_glibcxx)
    message(FATAL_ERROR
      "C++ compiler ${CMAKE_CXX_COMPILER} uses libstdc++ ${_cyten_cxx_stdcxx} "
      "(GLIBCXX_${_cyten_cxx_glibcxx}), but conda Python loads "
      "${_cyten_conda_stdcxx_real} (GLIBCXX_${_cyten_conda_glibcxx}). "
      "Install the matching compiler and reconfigure from a clean build dir:\n"
      "  conda install -c conda-forge cxx-compiler\n"
      "  conda activate $CONDA_PREFIX\n"
      "  rm -rf <build-dir>  # or delete CMakeCache.txt\n"
      "The env compiler is selected automatically via CXX / PATH.")
  endif()
endif()


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
# PUBLIC so dependents (examples, ctests, user programs) get the same flag;
# otherwise SpinSite/Coupling headers warn about py::array / py::object fields.
# pybind11_add_module(_core) already adds the flag for the extension module.
target_compile_options(cyten PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-fvisibility=hidden>
)
target_compile_options(cyten PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
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
# ## Docstrings (Doxygen XML → pybind/docstrings/<rel>, always at build)
# #####################################################################
# Layout: include/cyten/<rel>  →  pybind/docstrings/<rel> (gitignored; IDE-visible)
# Each pybind TU includes only the docstring headers it needs.
# Requires doxygen (see environment.yml).
#
# Pilot list — extend as more bindings switch to DOC(...):
set(CYTEN_MKDOC_HEADERS
  tensors/ops_algebra.h
  tensors/constructors.h
  tensors/ops_elementwise.h
  tensors/ops_legs.h
  tensors/decompositions.h
  tensors/helpers.h
  tensors/labels.h
  tensors/vector_like.h
  tensors/direct_sum.h
  tensors/tensor.h
  tensors/symmetric_tensor.h
  tensors/diagonal_tensor.h
  tensors/mask.h
  tensors/charged_tensor.h
  tensors/sparse.h
  tensors/planar.h
  tensors/krylov_based.h
  symmetries/factors/fermion_number.h
  symmetries/factors/fermion_parity.h
  symmetries/factors/fibonacci_anyon_category.h
  symmetries/factors/ising_anyon_category.h
  symmetries/factors/no_symmetry.h
  symmetries/factors/quantum_double_zn_anyon_category.h
  symmetries/factors/su2.h
  symmetries/factors/su2_k_anyon_category.h
  symmetries/factors/su3_3_anyon_category.h
  symmetries/factors/sun.h
  symmetries/factors/toric_code_category.h
  symmetries/factors/u1.h
  symmetries/factors/zn.h
  symmetries/factors/zn_anyon_category.h
  symmetries/factors/zn_anyon_category2.h
  symmetries/abelian_group.h
  symmetries/base_symmetry.h
  symmetries/exceptions.h
  symmetries/group.h
  symmetries/sector.h
  symmetries/spaces.h
  symmetries/styles.h
  symmetries/symmetry.h
  symmetries/symmetry_factor.h
  symmetries/trees.h
  backends/abelian.h
  backends/backend_factory.h
  backends/block_inds.h
  backends/fusion_tree_backend.h
  backends/fusion_tree_mapping.h
  backends/no_symmetry.h
  backends/tensor_backend.h
  block_backend/array_api.h
  block_backend/block_backend.h
  block_backend/dtypes.h
  block_backend/numpy.h
  block_backend/torch.h
  tools/cost_polynomials.h
  tools/mappings.h
  models/couplings.h
  models/degrees_of_freedom.h
  models/sites.h
  tools.h
)

find_program(CYTEN_DOXYGEN_EXECUTABLE NAMES doxygen REQUIRED)
set(CYTEN_DOCSTRINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind/docstrings")
set(CYTEN_DOCSTRING_SCRIPT
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/doxygen_xml_to_docstrings.py")
set(CYTEN_DOCSTRINGS_XML_ROOT
  "${CMAKE_CURRENT_BINARY_DIR}/docstrings_doxygen")

set(_cyten_doc_outputs "")
foreach(_cyten_hdr_rel IN LISTS CYTEN_MKDOC_HEADERS)
  set(_cyten_src_hdr "${CMAKE_CURRENT_SOURCE_DIR}/include/cyten/${_cyten_hdr_rel}")
  set(_cyten_doc_out "${CYTEN_DOCSTRINGS_DIR}/${_cyten_hdr_rel}")
  set(_cyten_safe_rel "${_cyten_hdr_rel}")
  string(REPLACE "/" "_" _cyten_safe_rel "${_cyten_safe_rel}")
  string(REPLACE "." "_" _cyten_safe_rel "${_cyten_safe_rel}")
  set(_cyten_xml_out "${CYTEN_DOCSTRINGS_XML_ROOT}/${_cyten_safe_rel}")
  set(_cyten_doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_docstrings_${_cyten_safe_rel}")
  get_filename_component(_cyten_doc_outdir "${_cyten_doc_out}" DIRECTORY)

  file(WRITE "${_cyten_doxyfile}"
"PROJECT_NAME           = cyten
OUTPUT_DIRECTORY       = ${_cyten_xml_out}
INPUT                  = ${_cyten_src_hdr}
RECURSIVE              = NO
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
GENERATE_XML           = YES
EXTRACT_ALL            = YES
QUIET                  = YES
WARNINGS               = NO
MARKDOWN_SUPPORT       = NO
AUTOLINK_SUPPORT       = NO
")

  # Write into the source tree under pybind/docstrings/ so IDEs can open DOC()
  # headers next to the bindings. Files are gitignored; always regenerated.
  add_custom_command(
    OUTPUT ${_cyten_doc_out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_cyten_doc_outdir}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${_cyten_xml_out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_cyten_xml_out}
    COMMAND ${CYTEN_DOXYGEN_EXECUTABLE} ${_cyten_doxyfile}
    COMMAND ${Python_EXECUTABLE} ${CYTEN_DOCSTRING_SCRIPT}
            --xml-dir ${_cyten_xml_out}/xml
            --header-rel ${_cyten_hdr_rel}
            --source-root ${CMAKE_CURRENT_SOURCE_DIR}
            -o ${_cyten_doc_out}
    DEPENDS ${_cyten_src_hdr} ${CYTEN_DOCSTRING_SCRIPT} ${_cyten_doxyfile}
    COMMENT "Generating pybind/docstrings/${_cyten_hdr_rel}"
    VERBATIM
  )
  list(APPEND _cyten_doc_outputs ${_cyten_doc_out})
endforeach()

add_custom_target(cyten_generate_docstrings DEPENDS ${_cyten_doc_outputs})
add_custom_target(cyten_mkdoc_docstrings DEPENDS cyten_generate_docstrings)
add_dependencies(_core cyten_generate_docstrings)
message(STATUS
  "Docstrings: generating ${CYTEN_MKDOC_HEADERS} under pybind/docstrings/ "
  "(doxygen=${CYTEN_DOXYGEN_EXECUTABLE})")


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
