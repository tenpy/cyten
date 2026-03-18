// NOTE: this file is #included from py_block_backend.cpp

#include "py_trampolines.hpp"
#include <cyten/block_backend/numpy.h>

namespace cyten {

void
bind_block_backend_numpy(py::module_& m)
{

    py::class_<NumpyBlockBackend, BlockBackend, py::smart_holder> numpy_block_backend(
      m, "NumpyBlockBackend");
    numpy_block_backend.doc() = R"pydoc(
        A block backend using numpy.

        No constructor available, use from_factory instead.
        Not to be subclassed.
        )pydoc";
    numpy_block_backend.def_static(
      "from_factory",
      &NumpyBlockBackend::from_factory,
      py::arg("device") = "cpu",
      py::return_value_policy::reference,
      "Get the backend instance for the given device (nearly-singleton per device).");
    numpy_block_backend.def_static("load_hdf5",
                                   &NumpyBlockBackend::load_hdf5,
                                   py::arg("hdf5_loader"),
                                   py::arg("h5gr"),
                                   py::arg("subpath"),
                                   "Load a block from an HDF5 file.");

    py::class_<NumpyBlockBackend::Block, BlockBackend::Block, py::smart_holder>(
      numpy_block_backend, "BlockCls", "Block that holds a numpy array in a py::object.")
      .def(py::init<py::array>(), py::arg("arr"))
      .def("to_numpy",
           py::overload_cast<>(&NumpyBlockBackend::Block::to_numpy, py::const_),
           py::return_value_policy::reference_internal)
      .def("to_numpy",
           py::overload_cast<Dtype>(&NumpyBlockBackend::Block::to_numpy, py::const_),
           py::arg("dtype"),
           py::return_value_policy::reference_internal);
    // NOTE: don't immplment __array__ since we don't want to allow Blocks to cast to numpy arrays
    // except via NOTE: no __mul__, __add__ etc since we have Block*Scalar defined in
    // BlockBackend.cpp
}

} // namespace cyten
