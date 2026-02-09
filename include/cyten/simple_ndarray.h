#pragma once

namespace cyten {

// @brief n-dimensional Array which is contiguous and C-style
template<typename DType>
class SimpleNDArray
{
  public:
    SimpleNDArray(std::vector<size_t> const& shape_);
    ~SimpleNDArray();
    DType& at(std::vector<size_t> const& inds);
    DType const& at(std::vector<size_t> const& inds) const;

  private:
    void* data;
    const size_t itemsize;
    size_t ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
};

} // namespace cyten

#include "./internal/simple_ndarray.hpp"
