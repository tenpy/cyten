# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

enable_testing()
# create the testing file and list of tests
set (TestsToRun
  test_symmetries.cpp
)

create_test_sourcelist(Tests CommonCppTests.cpp ${TestsToRun})


# add the executable
add_executable (CommonCppTests ${Tests})
set_source_files_properties(CommonCppTests.cpp ${TestsToRun} PROPERTIES
    SKIP_LINTING ON
)

target_include_directories(CommonCppTests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/ctest
  PUBLIC ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(CommonCppTests PUBLIC cyten pybind11::embed)

# Add all the ADD_TEST for each test
foreach (test ${TestsToRun})
  get_filename_component (TName ${test} NAME_WE)
  add_test (NAME ${TName} COMMAND CommonCppTests ${TName} )
endforeach ()
