#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"
#include "docstrings/backends/abelian.h"

#include "backends/casters.hpp"

#include <cyten/backends/abelian.h>
#include <cyten/backends/block_inds_numpy.h>
#include <cyten/block_backend/numpy.h>
#include <cyten/block_backend/torch.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/symmetric_tensor.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyten {

namespace {

std::shared_ptr<BlockBackend>
as_shared_block_backend(py::object obj)
{
    if (py::isinstance<NumpyBlockBackend>(obj)) {
        auto* p = obj.cast<NumpyBlockBackend*>();
        return NumpyBlockBackend::from_factory_shared(p->default_device);
    }
    if (py::isinstance<TorchBlockBackend>(obj)) {
        auto* p = obj.cast<TorchBlockBackend*>();
        return TorchBlockBackend::from_factory_shared(p->default_device);
    }
    auto* raw = obj.cast<BlockBackend*>();
    return std::shared_ptr<BlockBackend>(raw, [](BlockBackend*) {});
}

} // namespace

void
bind_abelian_backend_data(py::module_& m)
{
    py::class_<AbelianBackendData, TensorBackend::Data, py::smart_holder> cls(
      m, "AbelianBackendData");
    cls.doc() = DOC(cyten, AbelianBackendData);

    cls.def(py::init([](Dtype dtype,
                        std::string device,
                        std::vector<BlockBackend::BlockPtr> blocks,
                        BlockInds block_inds,
                        bool is_sorted) {
                return std::make_shared<AbelianBackendData>(std::move(dtype),
                                                            std::move(device),
                                                            std::move(blocks),
                                                            std::move(block_inds),
                                                            is_sorted);
            }),
            py::arg("dtype"),
            py::arg("device"),
            py::arg("blocks"),
            py::arg("block_inds"),
            py::arg("is_sorted") = false);

    cls.def_readwrite("dtype", &AbelianBackendData::dtype)
      .def_readwrite("device", &AbelianBackendData::device)
      .def_readwrite("blocks", &AbelianBackendData::blocks)
      .def_readwrite("block_inds", &AbelianBackendData::block_inds);

    cls
      .def("get_block_num",
           &AbelianBackendData::get_block_num,
           py::arg("block_inds"),
           DOC(cyten, AbelianBackendData, get_block_num))
      .def("get_block",
           &AbelianBackendData::get_block,
           py::arg("block_inds"),
           DOC(cyten, AbelianBackendData, get_block))
      .def("save_hdf5",
           &AbelianBackendData::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &AbelianBackendData::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

void
bind_abelian_backend(py::module_& m)
{
    // AbelianBackendData must already be registered (bind_abelian_backend_data).
    py::class_<AbelianBackend, TensorBackend, py::smart_holder> cls(m, "AbelianBackend");
    cls.doc() = DOC(cyten, AbelianBackend);

    cls.def(py::init([](py::object block_backend) {
                return std::make_shared<AbelianBackend>(as_shared_block_backend(block_backend));
            }),
            py::arg("block_backend"));

    cls.def_static("wrap", &AbelianBackend::wrap, py::arg("data"));
    cls.def_static("unwrap", &AbelianBackend::unwrap, py::arg("data"));

    cls.def("leg_pipe_map_incoming_block_inds",
            &AbelianBackend::leg_pipe_map_incoming_block_inds,
            py::arg("pipe"),
            py::arg("incoming_block_inds"),
            DOC(cyten, AbelianBackend, leg_pipe_map_incoming_block_inds));

    cls.def(
      "partial_trace",
      [](AbelianBackend& self,
         SymmetricTensorCPtr tensor,
         std::vector<std::pair<int64, int64>> pairs,
         std::vector<std::optional<int64>> levels) -> py::object {
          auto [data, codomain, domain] =
            self.partial_trace(tensor, std::move(pairs), std::move(levels));
          if (!codomain && !domain) {
              // Match Python: return (scalar, None, None) for a full trace.
              auto abd = AbelianBackend::unwrap(data);
              if (abd->blocks.empty()) {
                  return py::make_tuple(
                    self.block_backend->as_scalar(dtype::zero_scalar(abd->dtype), abd->dtype),
                    py::none(),
                    py::none());
              }
              return py::make_tuple(
                self.block_backend->item(abd->blocks[0]), py::none(), py::none());
          }
          return py::make_tuple(std::move(data), std::move(codomain), std::move(domain));
      },
      py::arg("tensor"),
      py::arg("pairs"),
      py::arg("levels") = py::none(),
      DOC(cyten, AbelianBackend, partial_trace));

    // Expose under both the C++ name and the Python private name.
    m.def(
      "valid_block_inds",
      &valid_block_inds,
      py::arg("codomain"),
      py::arg("domain"),
      doc_cpp_ref(R"pydoc(valid_block_inds)pydoc", "cyten::AbelianBackend::valid_block_inds()"));
    m.attr("_valid_block_inds") = m.attr("valid_block_inds");
}

} // namespace cyten
