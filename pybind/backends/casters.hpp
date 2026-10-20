#pragma once

/// Optional conversion helpers: accept NumPy arrays as BlockInds arguments.
///
/// ``py::class_<BlockInds>`` owns the Python type. This caster only extends *load*
/// so call sites can still pass ndarrays; *cast* always produces bound BlockInds.

#include <lasyten/backends/block_inds.h>
#include <lasyten/backends/block_inds_numpy.h>

#include <pybind11/cast.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pybind11::detail {

template<>
struct type_caster<lasyten::BlockInds> : type_caster_base<lasyten::BlockInds>
{
    using base = type_caster_base<lasyten::BlockInds>;

    bool load(handle src, bool convert)
    {
        if (base::load(src, convert)) {
            return true;
        }
        if (!convert) {
            return false;
        }
        try {
            lasyten::BlockInds tmp = lasyten::block_inds_from_numpy(src);
            value = new lasyten::BlockInds(std::move(tmp));
            return true;
        } catch (...) {
            return false;
        }
    }
};

} // namespace pybind11::detail
