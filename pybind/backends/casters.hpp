#pragma once

/// Optional conversion helpers: accept NumPy arrays as BlockInds arguments.
///
/// ``py::class_<BlockInds>`` owns the Python type. This caster only extends *load*
/// so call sites can still pass ndarrays; *cast* always produces bound BlockInds.

#include <cyten/backends/block_inds.h>
#include <cyten/backends/block_inds_numpy.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pybind11::detail {

template<>
struct type_caster<cyten::BlockInds> : type_caster_base<cyten::BlockInds>
{
    using base = type_caster_base<cyten::BlockInds>;

    bool load(handle src, bool convert)
    {
        if (base::load(src, convert)) {
            return true;
        }
        if (!convert) {
            return false;
        }
        try {
            cyten::BlockInds tmp = cyten::block_inds_from_numpy(src);
            value = new cyten::BlockInds(std::move(tmp));
            return true;
        } catch (...) {
            return false;
        }
    }
};

} // namespace pybind11::detail
