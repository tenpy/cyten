# #####################################################################
# ## ADD SUB-DIRECTORIES
# #####################################################################

target_sources(_core
  PUBLIC
  _core.cpp
  block_backend/block_backend.cpp
  block_backend/numpy.cpp
  block_backend/dtypes.cpp
  config.cpp
  tools.cpp
  version.cpp
)
