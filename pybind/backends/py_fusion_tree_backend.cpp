#include "../py_cyten_pybind11.h"
#include "../doc_plus.h"
#include "docstrings/backends/fusion_tree_backend.h"

#include "backends/casters.hpp"

#include <cyten/backends/fusion_tree_backend.h>
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

/// Convert a Python BlockBackend (often a non-owning factory singleton) to shared_ptr.
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
bind_fusion_tree_data(py::module_& m)
{
    py::class_<FusionTreeData, TensorBackend::Data, py::smart_holder> cls(m, "FusionTreeData");
    cls.doc() = DOC(cyten, FusionTreeData);

    cls.def(py::init([](BlockInds block_inds,
                        std::vector<BlockBackend::BlockPtr> blocks,
                        Dtype dtype,
                        std::string device,
                        bool is_sorted) {
                return std::make_shared<FusionTreeData>(
                  std::move(block_inds), std::move(blocks), dtype, std::move(device), is_sorted);
            }),
            py::arg("block_inds"),
            py::arg("blocks"),
            py::arg("dtype"),
            py::arg("device"),
            py::arg("is_sorted") = false);

    cls.def_readwrite("block_inds", &FusionTreeData::block_inds)
      .def_readwrite("blocks", &FusionTreeData::blocks)
      .def_readwrite("dtype", &FusionTreeData::dtype)
      .def_readwrite("device", &FusionTreeData::device);

    cls
      .def("block_ind_from_coupled",
           &FusionTreeData::block_ind_from_coupled,
           py::arg("coupled"),
           py::arg("domain"),
           DOC(cyten, FusionTreeData, block_ind_from_coupled))
      .def("block_ind_from_domain_sector_ind",
           &FusionTreeData::block_ind_from_domain_sector_ind,
           py::arg("domain_sector_ind"),
           DOC(cyten, FusionTreeData, block_ind_from_domain_sector_ind))
      .def(
        "discard_zero_blocks",
        [](FusionTreeData& self, py::object backend, float64 eps) {
            self.discard_zero_blocks(as_shared_block_backend(backend), eps);
        },
        py::arg("backend"),
        py::arg("eps"),
        DOC(cyten, FusionTreeData, discard_zero_blocks))
      .def("save_hdf5",
           &FusionTreeData::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &FusionTreeData::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

void
bind_fusion_tree_backend(py::module_& m)
{
    py::class_<FusionTreeBackend, TensorBackend, py::smart_holder> cls(m, "FusionTreeBackend");
    cls.doc() = DOC(cyten, FusionTreeBackend);

    cls.def(py::init([](py::object block_backend) {
                return std::make_shared<FusionTreeBackend>(as_shared_block_backend(block_backend));
            }),
            py::arg("block_backend"));
    cls.def_static("wrap", &FusionTreeBackend::wrap, py::arg("data"));
    cls.def_static("unwrap", &FusionTreeBackend::unwrap, py::arg("data"));

    cls.def(
      "partial_trace",
      [](FusionTreeBackend& self,
         SymmetricTensorCPtr tensor,
         std::vector<std::pair<int64, int64>> pairs,
         std::vector<std::optional<int64>> levels) -> py::object {
          auto [data, codomain, domain] =
            self.partial_trace(tensor, std::move(pairs), std::move(levels));
          if (!codomain && !domain) {
              // Match Python: return (scalar, None, None) for a full trace.
              auto ftd = FusionTreeBackend::unwrap(data);
              if (ftd->blocks.empty()) {
                  return py::make_tuple(
                    self.block_backend->as_scalar(dtype::zero_scalar(ftd->dtype), ftd->dtype),
                    py::none(),
                    py::none());
              }
              return py::make_tuple(
                self.block_backend->item(ftd->blocks[0]), py::none(), py::none());
          }
          return py::make_tuple(std::move(data), std::move(codomain), std::move(domain));
      },
      py::arg("tensor"),
      py::arg("pairs"),
      py::arg("levels") = py::none(),
      DOC(cyten, FusionTreeBackend, partial_trace));

    cls.def("apply_instructions",
            &FusionTreeBackend::apply_instructions,
            py::arg("tensor"),
            py::arg("instructions"),
            py::arg("codomain_idcs"),
            py::arg("domain_idcs"),
            py::arg("new_codomain"),
            py::arg("new_domain"),
            py::arg("mixes_codomain_domain"),
            DOC(cyten, FusionTreeBackend, apply_instructions));
}

} // namespace cyten
