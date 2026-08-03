#include <cassert>
#include <iostream>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "symmetries/casters.hpp"

#include <cyten/symmetries/sector.h>

namespace py = pybind11;
using namespace cyten;

int
test_sector_casters(int /*argc*/, char** /*args*/)
{
    py::scoped_interpreter guard{};
    py::module_::import("numpy");

    Sector s{ 1, -2, 3 };
    py::object obj = py::cast(s);
    auto arr = py::array::ensure(obj);
    assert(arr);
    assert(arr.ndim() == 1);
    assert(arr.shape(0) == 3);

    Sector s2 = py::cast<Sector>(obj);
    assert(s2 == s);

    // Python list → Sector via convert
    Sector from_list = py::cast<Sector>(py::list(py::make_tuple(7, 8)));
    assert((from_list == Sector{ 7, 8 }));

    SectorArray sa(2, 2);
    sa.set(0, Sector{ 1, 2 });
    sa.set(1, Sector{ 3, 4 });
    py::object sa_obj = py::cast(sa);
    auto sa_arr = py::array::ensure(sa_obj);
    assert(sa_arr.ndim() == 2);
    assert(sa_arr.shape(0) == 2);
    assert(sa_arr.shape(1) == 2);

    SectorArray sa2 = py::cast<SectorArray>(sa_obj);
    assert(sa2 == sa);

    std::cout << "test_sector_casters passed." << std::endl;
    return 0;
}
