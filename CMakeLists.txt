message(STATUS "")
message(STATUS "------------------------------------------------------------------------")
message(STATUS " Project lasyten, A Cross-section of Python & C++, Tensor network library ")
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
# `import lasyten` then fails because Python has already loaded the older
# libstdc++. Install with: conda install -c conda-forge cxx-compiler
if(NOT CMAKE_CXX_COMPILER AND DEFINED ENV{CONDA_PREFIX})
  foreach(_lasyten_cxx
      "$ENV{CONDA_PREFIX}/bin/c++"
      "$ENV{CONDA_PREFIX}/bin/g++")
    if(EXISTS "${_lasyten_cxx}")
      set(CMAKE_CXX_COMPILER "${_lasyten_cxx}")
      break()
    endif()
  endforeach()
  if(NOT CMAKE_CXX_COMPILER)
    file(GLOB _lasyten_conda_gxx "$ENV{CONDA_PREFIX}/bin/*-conda-*-g++")
    if(_lasyten_conda_gxx)
      list(GET _lasyten_conda_gxx 0 CMAKE_CXX_COMPILER)
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
option(LASYTEN_WITH_HDF5 "Link hdf5_io_core for native HDF5 I/O" ON)
# When ON (default): build Numpy/Torch/Array-API backends, models, _SU2, and
# the pybind11 ``_core`` extension. NumpyBlockBackend requires a live Python
# interpreter (extension or pybind11::embed) — it is not a pure-C++ backend.
# When OFF: those Python-owned TUs are excluded from liblasyten; ``_core`` is
# not built. Public headers still use ``py::`` extensively (via lasyten.h), so
# pybind11 remains a compile/link requirement until those APIs are decoupled.
option(LASYTEN_WITH_PYTHON
  "Include Python/pybind11-dependent liblasyten sources and build _core" ON)


# #####################################################################
# ## PROJECT
# #####################################################################
project(
    lasyten
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
)

include(CMakePrintHelpers)
cmake_print_variables(PROJECT_SOURCE_DIR)
cmake_print_variables(CMAKE_CXX_STANDARD)

# #####################################################################
# ## find python / pybind11
# #####################################################################
# Required even with LASYTEN_WITH_PYTHON=OFF: include/lasyten/lasyten.h and many
# public headers declare py:: types. A Python-free configure is blocked until
# those headers are split (see message below when OFF).

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

if(NOT LASYTEN_WITH_PYTHON)
  message(WARNING
    "LASYTEN_WITH_PYTHON=OFF: excluding block_backend/{numpy,torch,array_api}.cpp, "
    "symmetries/factors/su2.cpp, and models/*.cpp from liblasyten; skipping _core.\n"
    "pybind11 is STILL required because public headers use py:: types. Remaining "
    "blockers for a Python-free liblasyten (non-exhaustive):\n"
    "  include/lasyten/lasyten.h  (includes pybind11; namespace py = pybind11)\n"
    "  include/lasyten/config.h\n"
    "  include/lasyten/tools.h, tools/warn.h, tools/hdf5_export.h\n"
    "  include/lasyten/block_backend/{block_backend,dtypes,numpy,torch,array_api}.h\n"
    "  include/lasyten/backends/{tensor_backend,abelian,fusion_tree_*,no_symmetry,\n"
    "                         backend_factory,block_inds_numpy}.h\n"
    "  include/lasyten/tensors/{tensor,diagonal_tensor,mask,symmetric_tensor,\n"
    "                        sparse,krylov_based}.h\n"
    "  include/lasyten/symmetries/{spaces,sector_numpy,fusion_symbol}.h\n"
    "  include/lasyten/models/{degrees_of_freedom,sites,couplings}.h\n"
    "Also: src TUs outside the gated list still call Numpy/Torch APIs "
    "(e.g. backends/backend_factory.cpp, tools/hdf5_export.cpp, "
    "symmetries/trees.cpp, block_backend/block_backend.cpp) — linking "
    "executables that pull those symbols will fail until further decoupling.")
endif()

# Reject a compiler whose libstdc++ is newer than the one conda Python loads.
function(lasyten_glibcxx_max so_file out_var)
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

set(_lasyten_conda_stdcxx "")
if(DEFINED ENV{CONDA_PREFIX}
   AND EXISTS "$ENV{CONDA_PREFIX}/lib/libstdc++.so.6")
  set(_lasyten_conda_stdcxx "$ENV{CONDA_PREFIX}/lib/libstdc++.so.6")
endif()
if(_lasyten_conda_stdcxx)
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libstdc++.so
    OUTPUT_VARIABLE _lasyten_cxx_stdcxx
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(_lasyten_cxx_stdcxx "${_lasyten_cxx_stdcxx}" REALPATH)
  get_filename_component(_lasyten_conda_stdcxx_real
    "${_lasyten_conda_stdcxx}" REALPATH)
  lasyten_glibcxx_max("${_lasyten_cxx_stdcxx}" _lasyten_cxx_glibcxx)
  lasyten_glibcxx_max("${_lasyten_conda_stdcxx_real}" _lasyten_conda_glibcxx)
  if(_lasyten_cxx_glibcxx AND _lasyten_conda_glibcxx
     AND _lasyten_cxx_glibcxx VERSION_GREATER _lasyten_conda_glibcxx)
    message(FATAL_ERROR
      "C++ compiler ${CMAKE_CXX_COMPILER} uses libstdc++ ${_lasyten_cxx_stdcxx} "
      "(GLIBCXX_${_lasyten_cxx_glibcxx}), but conda Python loads "
      "${_lasyten_conda_stdcxx_real} (GLIBCXX_${_lasyten_conda_glibcxx}). "
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
# Discover Torch via the active interpreter's installed package so lasyten._core
# links the same libtorch / libc10 that Python's torch._C loads. That makes
# `import lasyten` + `import torch` in one process safe (one shared libtorch).
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE LASYTEN_TORCH_CMAKE_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE LASYTEN_TORCH_CMAKE_PREFIX_RESULT
)
if(NOT LASYTEN_TORCH_CMAKE_PREFIX_RESULT EQUAL 0 OR LASYTEN_TORCH_CMAKE_PREFIX STREQUAL "")
  message(FATAL_ERROR
    "Could not obtain torch.utils.cmake_prefix_path from ${Python_EXECUTABLE}. "
    "Install the matching Python torch package in this environment.")
endif()
list(PREPEND CMAKE_PREFIX_PATH "${LASYTEN_TORCH_CMAKE_PREFIX}")

# Torch's Caffe2/mkl.cmake sets caffe2::mkl INTERFACE_LINK_DIRECTORIES to
# ${MKL_ROOT}/lib/intel64 (etc.). When MKL_ROOT is unset those become the
# bogus absolute paths /lib/intel64, and executables that link lasyten (e.g.
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

add_library(lasyten STATIC)
set_property(TARGET lasyten PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET lasyten PROPERTY CXX_VISIBILITY_PRESET hidden)
set_property(TARGET lasyten PROPERTY VISIBILITY_INLINES_HIDDEN ON)
# CXX_VISIBILITY_PRESET is ignored for STATIC libraries; pybind11 still needs
# -fvisibility=hidden on every TU that includes pybind types (see FAQ:
# "'SomeClass' declared with greater visibility than the type of its field").
# PUBLIC so dependents (examples, ctests, user programs) get the same flag;
# otherwise SpinSite/Coupling headers warn about py::array / py::object fields.
# pybind11_add_module(_core) already adds the flag for the extension module.
target_compile_options(lasyten PRIVATE
  $<$<COMPILE_LANGUAGE:C>:-fvisibility=hidden>
)
target_compile_options(lasyten PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility=hidden>
  $<$<COMPILE_LANGUAGE:CXX>:-fvisibility-inlines-hidden>
)

target_include_directories(lasyten
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
add_subdirectory(src)
target_compile_options(lasyten PUBLIC -fPIC)

# pybind11::headers are required while public APIs still use py:: types
# (include/lasyten/lasyten.h). LASYTEN_WITH_PYTHON gates which *sources* ship the
# numpy/torch/array_api backends; it does not remove this link dependency yet.
# NumpyBlockBackend specifically needs a Python interpreter at runtime.
target_link_libraries(lasyten PUBLIC pybind11::pybind11 "${TORCH_LIBRARIES}")

# #####################################################################
# ## hdf5_io_core (optional native HDF5)
# #####################################################################
if(LASYTEN_WITH_HDF5)
  set(_lasyten_hdf5_io_default "${CMAKE_CURRENT_SOURCE_DIR}/external/hdf5_io")
  # Prefer a sibling checkout that already has the Python-free core (Phase B).
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tenpy_other/hdf5_io/include/hdf5_io/core_io.h")
    set(_lasyten_hdf5_io_default "${CMAKE_CURRENT_SOURCE_DIR}/../tenpy_other/hdf5_io")
  elseif(NOT EXISTS "${_lasyten_hdf5_io_default}/include/hdf5_io/core_io.h")
    message(FATAL_ERROR
      "LASYTEN_WITH_HDF5=ON but hdf5_io core not found. "
      "Init submodule external/hdf5_io or set LASYTEN_HDF5_IO_DIR to a Phase-B hdf5_io tree.")
  endif()
  set(LASYTEN_HDF5_IO_DIR "${_lasyten_hdf5_io_default}" CACHE PATH
    "Source tree of hdf5_io (must provide hdf5_io_core)")
  set(HDF5_IO_BUILD_PYTHON OFF CACHE BOOL "Build hdf5_io Python module when embedded" FORCE)
  add_subdirectory("${LASYTEN_HDF5_IO_DIR}" "${CMAKE_BINARY_DIR}/_deps/hdf5_io")
  target_link_libraries(lasyten PUBLIC hdf5_io_core)
  target_compile_definitions(lasyten PUBLIC LASYTEN_WITH_HDF5=1)
  message(STATUS "LASYTEN_WITH_HDF5: using ${LASYTEN_HDF5_IO_DIR}")
endif()

# #####################################################################
# ## PYTHON BINDINGS
# #####################################################################
# hdf5_py_bridge.{h,cpp} live under pybind/tools/ and are compiled only into
# ``_core`` (not liblasyten).

if(LASYTEN_WITH_PYTHON)
  pybind11_add_module(_core pybind/_core.cpp)

  target_include_directories(_core
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pybind
    PUBLIC "${PROJECT_SOURCE_DIR}/include")
  add_subdirectory(pybind)  # adds source files to _core

  target_link_libraries(_core PUBLIC lasyten)
else()
  message(STATUS "LASYTEN_WITH_PYTHON=OFF: skipping pybind11 module _core")
endif()

# #####################################################################
# ## Docstrings (Doxygen XML → pybind/docstrings/<rel>, always at build)
# #####################################################################
# Layout: include/lasyten/<rel>  →  pybind/docstrings/<rel> (gitignored; IDE-visible)
# Each pybind TU includes only the docstring headers it needs.
# Requires doxygen (see environment.yml).
#
# Pilot list — extend as more bindings switch to DOC(...):
set(LASYTEN_MKDOC_HEADERS
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
  tensors/hidden_leg_tensor.h
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

if(LASYTEN_WITH_PYTHON)
find_program(LASYTEN_DOXYGEN_EXECUTABLE NAMES doxygen REQUIRED)
set(LASYTEN_DOCSTRINGS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind/docstrings")
set(LASYTEN_DOCSTRING_SCRIPT
  "${CMAKE_CURRENT_SOURCE_DIR}/scripts/doxygen_xml_to_docstrings.py")
set(LASYTEN_DOCSTRINGS_XML_ROOT
  "${CMAKE_CURRENT_BINARY_DIR}/docstrings_doxygen")

set(_lasyten_doc_outputs "")
foreach(_lasyten_hdr_rel IN LISTS LASYTEN_MKDOC_HEADERS)
  set(_lasyten_src_hdr "${CMAKE_CURRENT_SOURCE_DIR}/include/lasyten/${_lasyten_hdr_rel}")
  set(_lasyten_doc_out "${LASYTEN_DOCSTRINGS_DIR}/${_lasyten_hdr_rel}")
  set(_lasyten_safe_rel "${_lasyten_hdr_rel}")
  string(REPLACE "/" "_" _lasyten_safe_rel "${_lasyten_safe_rel}")
  string(REPLACE "." "_" _lasyten_safe_rel "${_lasyten_safe_rel}")
  set(_lasyten_xml_out "${LASYTEN_DOCSTRINGS_XML_ROOT}/${_lasyten_safe_rel}")
  set(_lasyten_doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile_docstrings_${_lasyten_safe_rel}")
  get_filename_component(_lasyten_doc_outdir "${_lasyten_doc_out}" DIRECTORY)

  file(WRITE "${_lasyten_doxyfile}"
"PROJECT_NAME           = lasyten
OUTPUT_DIRECTORY       = ${_lasyten_xml_out}
INPUT                  = ${_lasyten_src_hdr}
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
    OUTPUT ${_lasyten_doc_out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_lasyten_doc_outdir}
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${_lasyten_xml_out}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${_lasyten_xml_out}
    COMMAND ${LASYTEN_DOXYGEN_EXECUTABLE} ${_lasyten_doxyfile}
    COMMAND ${Python_EXECUTABLE} ${LASYTEN_DOCSTRING_SCRIPT}
            --xml-dir ${_lasyten_xml_out}/xml
            --header-rel ${_lasyten_hdr_rel}
            --source-root ${CMAKE_CURRENT_SOURCE_DIR}
            -o ${_lasyten_doc_out}
    DEPENDS ${_lasyten_src_hdr} ${LASYTEN_DOCSTRING_SCRIPT} ${_lasyten_doxyfile}
    COMMENT "Generating pybind/docstrings/${_lasyten_hdr_rel}"
    VERBATIM
  )
  list(APPEND _lasyten_doc_outputs ${_lasyten_doc_out})
endforeach()

add_custom_target(lasyten_generate_docstrings DEPENDS ${_lasyten_doc_outputs})
add_custom_target(lasyten_mkdoc_docstrings DEPENDS lasyten_generate_docstrings)
add_dependencies(_core lasyten_generate_docstrings)
message(STATUS
  "Docstrings: generating ${LASYTEN_MKDOC_HEADERS} under pybind/docstrings/ "
  "(doxygen=${LASYTEN_DOXYGEN_EXECUTABLE})")


# #####################################################################
# ## .pyi stub (pybind11-stubgen → lasyten/_core.pyi, for IDE type hints)
# #####################################################################
# Written into the source tree next to _core.so (gitignored; always regenerated)
# so IDEs/LSPs get parameter hints for the compiled extension. Optional: if
# pybind11-stubgen is unavailable or fails, warn and skip rather than fail the build.
find_program(LASYTEN_PYBIND11_STUBGEN_EXECUTABLE NAMES pybind11-stubgen)
set(LASYTEN_CORE_PYI "${CMAKE_CURRENT_SOURCE_DIR}/lasyten/_core.pyi")

if(LASYTEN_PYBIND11_STUBGEN_EXECUTABLE)
  add_custom_command(
    OUTPUT ${LASYTEN_CORE_PYI}
    COMMAND sh -c "PYTHONPATH=$<TARGET_FILE_DIR:_core> ${LASYTEN_PYBIND11_STUBGEN_EXECUTABLE} _core -o ${CMAKE_CURRENT_SOURCE_DIR}/lasyten \
        || echo 'WARNING: pybind11-stubgen failed; lasyten/_core.pyi not (re)generated'"
    DEPENDS _core
    COMMENT "Generating lasyten/_core.pyi via pybind11-stubgen"
    VERBATIM
  )
  add_custom_target(lasyten_generate_pyi_stub ALL DEPENDS ${LASYTEN_CORE_PYI})
  add_dependencies(lasyten_generate_pyi_stub _core)
else()
  message(WARNING
    "pybind11-stubgen not found; skipping lasyten/_core.pyi generation. "
    "Install it (pip install pybind11-stubgen, or the 'lasyten[stubs]' extra) for IDE type hints.")
endif()


# Resolve libtorch/c10 from the same place as Python torch._C.
# Installed layout: site-packages/lasyten/_core*.so
#   $ORIGIN/../../..  -> env/lib (conda/micromamba shared torch libs)
#   $ORIGIN/../torch/lib -> site-packages/torch/lib (pip wheel layout)
# Do not use INSTALL_RPATH_USE_LINK_PATH: Torch/Caffe2 link dirs may inject
# unrelated paths (e.g. Intel MKL stubs) that we do not want on the module.
get_filename_component(_lasyten_torch_lib_dir "${TORCH_INSTALL_PREFIX}/lib" ABSOLUTE)
set(_lasyten_core_rpath "$ORIGIN/../../..;$ORIGIN/../torch/lib;${_lasyten_torch_lib_dir}")
if(DEFINED ENV{CONDA_PREFIX} AND NOT "$ENV{CONDA_PREFIX}" STREQUAL "")
  list(APPEND _lasyten_core_rpath "$ENV{CONDA_PREFIX}/lib")
endif()
# list(APPEND) on a ;-string splits; rebuild a single RPATH string.
string(JOIN ":" _lasyten_core_rpath_str ${_lasyten_core_rpath})
set_target_properties(_core PROPERTIES
  BUILD_RPATH "${_lasyten_core_rpath_str}"
  INSTALL_RPATH "${_lasyten_core_rpath_str}"
  INSTALL_RPATH_USE_LINK_PATH FALSE
)


# The install directory is the output (wheel) directory
install(TARGETS _core LIBRARY DESTINATION lasyten)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/lasyten/py.typed" DESTINATION lasyten)
install(FILES ${LASYTEN_CORE_PYI} DESTINATION lasyten OPTIONAL)

endif() # LASYTEN_WITH_PYTHON


# #####################################################################
# ## DOCUMENTATION
# #####################################################################
if(BUILD_DOC)
  find_package(Doxygen REQUIRED)

  if(DOXYGEN_FOUND)
    configure_file(lasyten_Doxyfile.in lasytenDoxyfile)
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
