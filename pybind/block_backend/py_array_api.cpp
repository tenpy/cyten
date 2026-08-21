#include "../doc_plus.h"
#include "docstrings/block_backend/array_api.h"
// NOTE: this file is #included from py_block_backend.cpp

#include "py_trampolines.hpp"
#include <cyten/block_backend/array_api.h>

namespace cyten {

void
bind_block_backend_array_api(py::module_& m)
{
    py::class_<ArrayApiBlockBackend, BlockBackend, PyArrayApiBlockBackend, py::smart_holder>
      array_api_block_backend(m, "ArrayApiBlockBackend");
    array_api_block_backend.doc() = DOC(cyten, ArrayApiBlockBackend);
    array_api_block_backend.def(py::init<py::object, std::string>(),
                                py::arg("api_namespace"),
                                py::arg("default_device") = "cpu");
    array_api_block_backend.def_property_readonly(
      "api", &ArrayApiBlockBackend::api, "The Array API namespace this backend dispatches to.");
    array_api_block_backend.def_static(
      "from_hdf5",
      &ArrayApiBlockBackend::from_hdf5,
      py::arg("hdf5_loader"),
      py::arg("h5gr"),
      py::arg("subpath"),
      "Load an ArrayApiBlockBackend from an HDF5 file (not generally supported).");

    py::class_<ArrayApiBlockBackend::Block, BlockBackend::Block, py::smart_holder>(
      array_api_block_backend,
      "BlockCls",
      "Block that holds an Array-API array as a Python object.")
      .def("to_numpy", py::overload_cast<>(&ArrayApiBlockBackend::Block::to_numpy, py::const_))
      .def("to_numpy",
           py::overload_cast<Dtype>(&ArrayApiBlockBackend::Block::to_numpy, py::const_),
           py::arg("dtype"))
      .def("save_hdf5",
           &ArrayApiBlockBackend::Block::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"),
           "Save block to HDF5 via numpy conversion.")
      .def_static("from_hdf5",
                  &ArrayApiBlockBackend::Block::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"),
                  "Load block from HDF5 (requires Array API context).");
}

} // namespace cyten
