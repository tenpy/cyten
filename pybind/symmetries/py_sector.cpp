#include "py_cyten_pybind11.h"

#include <cyten/symmetries/sector_hdf5.h>
#include <cyten/symmetries/sector_numpy.h>

#include <string>

namespace cyten {

void
bind_sector(py::module_& m)
{
    // Bound under the public names ``Sector`` / ``SectorArray`` for HDF5 type tags.
    // Must NOT be ``py::class_<Sector>`` — that conflicts with ndarray type_casters.
    py::class_<SectorHdf5>(m, "Sector", R"pydoc(
        HDF5-exportable sector. The rest of the API uses NumPy ndarrays for sectors;
        this type exists so :class:`~cyten.tools.hdf5_io.Hdf5Saver` can call
        :meth:`save_hdf5` / :meth:`from_hdf5`.
        )pydoc")
      .def(py::init([](py::object arr) { return SectorHdf5{ sector_from_numpy(arr) }; }),
           py::arg("values"),
           "Construct from a 1D integer array.")
      .def(
        "__array__",
        [](SectorHdf5 const& self, py::object dtype) {
            py::array a = sector_to_numpy(self.sector);
            if (dtype.is_none()) {
                return a;
            }
            return a.attr("astype")(dtype, py::arg("copy") = false).cast<py::array>();
        },
        py::arg("dtype") = py::none())
      .def_property_readonly(
        "shape", [](SectorHdf5 const& self) { return py::make_tuple(self.sector.len()); })
      .def("__len__",
           [](SectorHdf5 const& self) { return static_cast<std::size_t>(self.sector.len()); })
      .def("__repr__",
           [](SectorHdf5 const& self) {
               return py::str(sector_to_numpy(self.sector).attr("__repr__")());
           })
      .def(
        "save_hdf5",
        [](SectorHdf5 const& self,
           py::object hdf5_saver,
           py::object h5gr,
           std::string const& subpath) { self.save_hdf5(hdf5_saver, h5gr, subpath); },
        py::arg("hdf5_saver"),
        py::arg("h5gr"),
        py::arg("subpath"),
        R"pydoc(
        Export this sector into a HDF5 file.

        Saves charge values under ``subpath + 'values'`` as a NumPy integer array.
        )pydoc")
      .def_static(
        "from_hdf5",
        [](py::object hdf5_loader, py::object h5gr, std::string const& subpath) {
            auto obj = SectorHdf5::from_hdf5(hdf5_loader, h5gr, subpath);
            py::object py_obj = py::cast(obj);
            hdf5_loader.attr("memorize_load")(h5gr, py_obj);
            return py_obj;
        },
        py::arg("hdf5_loader"),
        py::arg("h5gr"),
        py::arg("subpath"),
        R"pydoc(
        Reconstruct a sector from HDF5 data saved with :meth:`save_hdf5`.
        )pydoc");

    py::class_<SectorArrayHdf5>(m, "SectorArray", R"pydoc(
        HDF5-exportable batch of sectors. Prefer NumPy ndarrays elsewhere.
        )pydoc")
      .def(
        py::init([](py::object arr) { return SectorArrayHdf5{ sector_array_from_numpy(arr) }; }),
        py::arg("values"),
        "Construct from a 2D integer array.")
      .def(
        "__array__",
        [](SectorArrayHdf5 const& self, py::object dtype) {
            py::array a = sector_array_to_numpy(self.sectors);
            if (dtype.is_none()) {
                return a;
            }
            return a.attr("astype")(dtype, py::arg("copy") = false).cast<py::array>();
        },
        py::arg("dtype") = py::none())
      .def_property_readonly("shape",
                             [](SectorArrayHdf5 const& self) {
                                 return py::make_tuple(self.sectors.num_sectors,
                                                       self.sectors.sector_ind_len);
                             })
      .def(
        "save_hdf5",
        [](SectorArrayHdf5 const& self,
           py::object hdf5_saver,
           py::object h5gr,
           std::string const& subpath) { self.save_hdf5(hdf5_saver, h5gr, subpath); },
        py::arg("hdf5_saver"),
        py::arg("h5gr"),
        py::arg("subpath"))
      .def_static(
        "from_hdf5",
        [](py::object hdf5_loader, py::object h5gr, std::string const& subpath) {
            auto obj = SectorArrayHdf5::from_hdf5(hdf5_loader, h5gr, subpath);
            py::object py_obj = py::cast(obj);
            hdf5_loader.attr("memorize_load")(h5gr, py_obj);
            return py_obj;
        },
        py::arg("hdf5_loader"),
        py::arg("h5gr"),
        py::arg("subpath"));
}

} // namespace cyten
