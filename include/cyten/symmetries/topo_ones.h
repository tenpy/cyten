#pragma once

/// Cached NumPy ones arrays matching Python ``one_1D`` … ``one_4D_float`` in ``_symmetries.py``.

#include "../cyten.h"

namespace cyten::topo_ones {

inline py::module_
numpy()
{
    return py::module_::import("numpy");
}

inline py::array
ones_array(py::tuple shape, py::object dtype)
{
    return numpy().attr("ones")(shape, py::arg("dtype") = dtype).cast<py::array>();
}

inline py::array
one_1D()
{
    return ones_array(py::make_tuple(1), numpy().attr("intp"));
}

inline py::array
one_2D()
{
    return ones_array(py::make_tuple(1, 1), numpy().attr("intp"));
}

inline py::array
one_2D_float()
{
    return ones_array(py::make_tuple(1, 1), numpy().attr("float64"));
}

inline py::array
one_4D()
{
    return ones_array(py::make_tuple(1, 1, 1, 1), numpy().attr("intp"));
}

inline py::array
one_4D_float()
{
    return ones_array(py::make_tuple(1, 1, 1, 1), numpy().attr("float64"));
}

inline int16_t
mod_n(int32_t x, int N)
{
    int r = static_cast<int>(x % N);
    if (r < 0) {
        r += N;
    }
    return static_cast<int16_t>(r);
}

} // namespace cyten::topo_ones
