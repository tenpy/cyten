#pragma once

#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>
#include <memory>
#include <string>
#include <vector>

namespace cyten {

/// Abstract base class for dense blocks. Subclassed per backend (e.g. NumpyBlock).
/// Pass and return as BlockPtr=std::shared_ptr<Block>; use BlockCPtr=std::shared_ptr<const Block>
/// for inputs.
class Block
{
  public:
    virtual ~Block() = default;

    /// Shape of the block (one size per axis).
    virtual std::vector<cyten_int> shape() const = 0;

    /// Dtype of the block entries.
    virtual Dtype dtype() const = 0;

    /// Device string (e.g. "cpu", "cuda:0").
    virtual std::string device() const = 0;

    /// Element or sub-block access by indices. Returns a scalar (py::object) or a new Block for
    /// slices.
    virtual py::object operator[](std::vector<cyten_int> const& idcs) const = 0;
};

using BlockPtr = std::shared_ptr<Block>;
using BlockCPtr = std::shared_ptr<const Block>;

} // namespace cyten
