#include <cassert>
#include <iostream>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "symmetries/casters.hpp"

#include <cyten/symmetries/sector.h>
#include <cyten/symmetries/sector_numpy.h>

namespace py = pybind11;
using namespace cyten;

int
test_sector_casters(int /*argc*/, char** /*args*/)
{
    py::scoped_interpreter guard{};
    py::module_::import("numpy");
    // Register py::class_<Sector> / SectorArray bindings used by type_caster_base.
    py::module_::import("cyten._core");

    Sector s{ 1, -2, 3 };
    py::object obj = py::cast(s);
    assert(py::isinstance<Sector>(obj));
    auto arr = sector_to_numpy(s);
    assert(arr.ndim() == 1);
    assert(arr.shape(0) == 3);

    Sector s2 = obj.cast<Sector>();
    assert(s2 == s);

    // ndarray / sequence → Sector via sector_from_numpy (same path as load caster)
    Sector from_list = sector_from_numpy(py::list(py::make_tuple(7, 8)));
    assert((from_list == Sector{ 7, 8 }));

    SectorArray sa(2, 2);
    sa[0] = Sector{ 1, 2 };
    sa[1] = Sector{ 3, 4 };
    py::object sa_obj = py::cast(sa);
    assert(py::isinstance<SectorArray>(sa_obj));
    auto sa_arr = sector_array_to_numpy(sa);
    assert(sa_arr.ndim() == 2);
    assert(sa_arr.shape(0) == 2);
    assert(sa_arr.shape(1) == 2);

    SectorArray sa2 = sa_obj.cast<SectorArray>();
    assert(sa2 == sa);

    SectorArray from_np = sector_array_from_numpy(sa_arr);
    assert(from_np == sa);

    std::cout << "test_sector_casters passed." << std::endl;
    return 0;
}
