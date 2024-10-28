#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <exception>

#include "symmetries.h"


using namespace std;
using namespace cyten;
namespace py = pybind11;
using namespace pybind11::literals; // provides "arg"_a literals
                                

void bind_symmetries(py::module_ &m){

    py::enum_<FusionStyle>(m, "FusionStyle", py::arithmetic())
        .value("single", FusionStyle::single)
        .value("multiple_unique", FusionStyle::multiple_unique)
        .value("general", FusionStyle::general)
        .export_values();

    py::enum_<BraidingStyle>(m, "BraidingStyle", py::arithmetic())
        .value("bosonic", BraidingStyle::bosonic)
        .value("fermionic", BraidingStyle::fermionic)
        .value("anyonic", BraidingStyle::anyonic) 
        .value("no_braiding", BraidingStyle::no_braiding)
        .export_values();
    
    
    py::class_<SectorArray>(m, "SectorArray", py::buffer_protocol())
        .def(py::init<size_t>())
        .def(py::init([](py::array_t<charge> array){
            if (array.ndim() != 2)
                throw std::runtime_error("expect Array of size 2");
            size_t size = array.shape()[0];
            size_t sector_len = array.shape()[1];
            SectorArray * copy = new SectorArray(sector_len);
            Sector s(sector_len);
            for (size_t i = 0; i < size; ++i)
                for (size_t j = 0; j < sector_len; ++j)
                    s[j] = array.at(i,j);
                copy->push_back(s);
            return copy;
        }))
        .def_buffer([](SectorArray & array) {
            return py::buffer_info(
                (void*) array.raw_pointer(),  // raw pointer
                sizeof(charge), // element size
                py::format_descriptor<charge>::format(), // python format descriptor
                2, // dimension
                { array.size(), array.sector_len }, // buffer dims
                { sizeof(charge) * array.sector_len, sizeof(charge) }, // strides (in bytes)
                true // readonly
            );
        })
        .def_readonly("sector_len", &SectorArray::sector_len)
        .def_property_readonly("shape", [](SectorArray const & a){ 
            return std::make_tuple(a.size(), a.sector_len);
            })
        .def("append", [](SectorArray & a, Sector const & s){
            if (s.size() != a.sector_len)
                throw std::length_error("Incompatible size of Sector to be added");
            a.push_back(s);
        }, "append a sector to the array")
        ;

    // py::class_<Symmetry> symmetry(m, "Symmetry");
    
    
    // symmetry.def(py::init<FusionStyle, BraidingStyle, bool>())
    //     .def_read("fusion_style", &Symmetry::fusion_style)
    //     .def_read("braiding_style", &Symmetry::braiding_style)
    //     .def_read("can_be_dropped", &Symmetry::can_be_dropped);
}