#pragma once

/// BlockInds ↔ NumPy helpers for liblasyten (no type_casters required).

#include <lasyten/backends/block_inds.h>
#include <lasyten/lasyten.h>

namespace lasyten {

py::array_t<int64> block_inds_to_numpy(BlockInds const& src);
BlockInds block_inds_from_numpy(py::handle src);

} // namespace lasyten
