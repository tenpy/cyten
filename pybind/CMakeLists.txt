# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

target_sources(_core
  PUBLIC
  _core.cpp
  block_backend/py_block_backend.cpp
  py_config.cpp
  py_tools.cpp
  py_version.cpp
)
