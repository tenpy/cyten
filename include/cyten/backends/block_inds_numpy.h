#pragma once

/// BlockInds ↔ NumPy helpers for libcyten (no type_casters required).

#include <cyten/backends/block_inds.h>
#include <cyten/cyten.h>

namespace cyten {

py::array_t<int64> block_inds_to_numpy(BlockInds const& src);
BlockInds block_inds_from_numpy(py::handle src);

} // namespace cyten
