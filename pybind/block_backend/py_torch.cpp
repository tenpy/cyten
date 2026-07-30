// NOTE: this file is #included from py_block_backend.cpp

#include <cyten/block_backend/torch.h>

namespace cyten {

void
bind_block_backend_torch(py::module_& m)
{
    py::class_<TorchBlockBackend, BlockBackend, py::smart_holder> torch_block_backend(
      m, "TorchBlockBackend");
    torch_block_backend.doc() = R"pydoc(
        A block-backend using PyTorch.

        No constructor available, use from_factory instead.
        Not to be subclassed.
        )pydoc";
    torch_block_backend.def_static(
      "from_factory",
      &TorchBlockBackend::from_factory,
      py::arg("device") = "cpu:0",
      py::return_value_policy::reference,
      "Get the backend instance for the given device (nearly-singleton per device).");
    torch_block_backend.def_static("from_hdf5",
                                   &TorchBlockBackend::from_hdf5,
                                   py::arg("hdf5_loader"),
                                   py::arg("h5gr"),
                                   py::arg("subpath"),
                                   "Load a TorchBlockBackend from an HDF5 file.");

    py::class_<TorchBlockBackend::Block, BlockBackend::Block, py::smart_holder>(
      torch_block_backend, "BlockCls", "Block that holds a torch::Tensor.")
      .def("to_numpy",
           py::overload_cast<>(&TorchBlockBackend::Block::to_numpy, py::const_),
           py::return_value_policy::reference_internal)
      .def("to_numpy",
           py::overload_cast<Dtype>(&TorchBlockBackend::Block::to_numpy, py::const_),
           py::arg("dtype"),
           py::return_value_policy::reference_internal)
      .def("save_hdf5",
           &TorchBlockBackend::Block::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"),
           "Save block (via numpy) to HDF5.")
      .def_static("from_hdf5",
                  &TorchBlockBackend::Block::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"),
                  "Load block from HDF5.");
}

} // namespace cyten
