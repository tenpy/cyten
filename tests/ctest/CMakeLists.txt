# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

enable_testing()
# create the testing file and list of tests
set (TestsToRun
  test_sector.cpp
  test_sector_casters.cpp
  test_block_inds.cpp
)

create_test_sourcelist(Tests CommonCppTests.cpp ${TestsToRun})


# add the executable
add_executable(CommonCppTests ${Tests})
set_source_files_properties(CommonCppTests.cpp ${TestsToRun} PROPERTIES
  SKIP_LINTING ON
)

target_include_directories(CommonCppTests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest
  PRIVATE ${PROJECT_SOURCE_DIR}/pybind
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(CommonCppTests PUBLIC cyten pybind11::embed)

# Add all the ADD_TEST for each test
foreach (test ${TestsToRun})
  get_filename_component (TName ${test} NAME_WE)
  add_test (NAME ${TName} COMMAND CommonCppTests ${TName} )
endforeach ()

# User-guide example from docs/intro/examples/ (see docs/intro/first_steps.rst).
set(example_heisenberg_src
  ${PROJECT_SOURCE_DIR}/docs/intro/examples/heisenberg_two_site.cpp)
add_executable(example_heisenberg_two_site ${example_heisenberg_src})
set_source_files_properties(${example_heisenberg_src} PROPERTIES SKIP_LINTING ON)
target_include_directories(example_heisenberg_two_site
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(example_heisenberg_two_site PUBLIC cyten pybind11::embed)
add_test(NAME example_heisenberg_two_site COMMAND example_heisenberg_two_site)
