find_package(Git)

if (GIT_EXECUTABLE)
execute_process(
    COMMAND ${GIT_EXECUTABLE} log --pretty=format:%h -n 1
    WORKING_DIRECTORY ${SRC_DIR}
    OUTPUT_VARIABLE GIT_REV
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else()
  message(STATUS "Git not found!")
endif()

if("${GIT_REV}" STREQUAL "")
  message(STATUS "can't get version from git")
  set(GIT_REV "")
  set(GIT_DIFF "")
  set(GIT_TAG "unknown")
  set(GIT_BRANCH "")
else()
  execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --exact-match --tags
        OUTPUT_VARIABLE GIT_TAG
        WORKING_DIRECTORY ${SRC_DIR}
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
  execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
        OUTPUT_VARIABLE GIT_BRANCH
        WORKING_DIRECTORY ${SRC_DIR}
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
  execute_process(
        COMMAND bash -c "${GIT_EXECUTABLE} diff --quiet --exit-code || echo '-dirty'"
        OUTPUT_VARIABLE GIT_DIFF
        WORKING_DIRECTORY ${SRC_DIR}
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
  message(STATUS "version from git: ${GIT_TAG} ${GIT_REV}-${GIT_BRANCH}${GIT_DIFF}")
endif()


configure_file(
  "${SRC_DIR}/version.h.in"
  "${SRC_DIR}/version.h"
)
