#pragma once

/// Native HDF5 I/O helpers for cyten ``src/`` (Python-free headers).

#include <hdf5_io/core_io.h>
#include <hdf5_io/h5_ops.h>

#include <highfive/highfive.hpp>
#include <string>

namespace cyten::hdf5 {

using Saver = ::hdf5_io::core::Saver;
using Loader = ::hdf5_io::core::Loader;

inline std::string
ensure_slash(std::string path)
{
    if (path.empty() || path.back() != '/')
        path.push_back('/');
    return path;
}

/// Saver rooted at an instance group (children use short relative names).
inline Saver
local_saver(HighFive::Group& h5gr)
{
    return Saver(h5gr);
}

inline Loader
local_loader(HighFive::Group& h5gr, bool ignore_unknown = true)
{
    return Loader(h5gr, ignore_unknown);
}

} // namespace cyten::hdf5
