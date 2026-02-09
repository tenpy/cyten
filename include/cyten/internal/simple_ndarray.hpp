#include "simple_ndarray.h"
#pragma once
// included by simple_ndarray.h

namespace cyten {

template<typename DType>
inline SimpleNDArray<DType>::SimpleNDArray(std::vector<size_t> const& shape_)
  : item_size(sizeof(DType))
  , ndim(shape_.size())
  , shape(shape_)
  , strides(shape_.size())
{
    size_t stride = 1;
    for (size_t i = ndim - 1; i >= 0; --i) {
        strides[i] = item_size * stride;
    }
    size_t data_size =
}

template<typename DType>
inline SimpleNDArray<DType>::~SimpleNDArray()
{
    delete[] data;
}
} // namespace cyten
