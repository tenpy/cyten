#include "../py_cyten_pybind11.h"

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
    cls.doc() = R"pydoc(
Data stored in a Tensor for :class:`FusionTreeBackend`.

Attributes
----------
block_inds : BlockInds
    Indices that specify the coupled sectors of the non-zero blocks.
    Shape ``(N, 2)``. ``block_inds[n] == [i, j]`` indicates that the coupled sector for
    ``blocks[n]`` is given by ``tensor.codomain.sector_decomposition[i] == coupled ==
    tensor.domain.sector_decomposition[j]``.
blocks : list of 2D Block
    The nonzero blocks, ``blocks[n]`` corresponding to ``coupled_sectors[n]``.
dtype : Dtype
    The dtype of the tensor (and of the `blocks`).
device : str
    The device on which the blocks are currently stored.
    We currently only support tensors which have all blocks on a single device.
    Should be the device returned by :func:`BlockBackend.as_device`.
is_sorted : bool
    If ``False`` (default), we permute `blocks` and `block_inds` according to
    ``np.lexsort(block_inds.T)``.
    If ``True``, we assume they are sorted *without* checking.
)pydoc";

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
           R"pydoc(
Return `ind` such that ``blocks[ind]`` is associated with the `coupled` sector.

This is such that ``domain.sector_decomposition[block_inds[res][1]] == coupled``.

Note: we use the domain (and not the codomain), since only the :attr:`block_inds[:, 1]`
are sorted.
)pydoc")
      .def("block_ind_from_domain_sector_ind",
           &FusionTreeData::block_ind_from_domain_sector_ind,
           py::arg("domain_sector_ind"),
           R"pydoc(
Return `ind` such that ``block_inds[ind, 1] == domain_sector_ind``

Note: we use the domain (and not the codomain), since only the :attr:`block_inds[:, 1]`
are sorted.
)pydoc")
      .def(
        "discard_zero_blocks",
        [](FusionTreeData& self, py::object backend, float64 eps) {
            self.discard_zero_blocks(as_shared_block_backend(backend), eps);
        },
        py::arg("backend"),
        py::arg("eps"),
        R"pydoc(
        Discard blocks whose norm is below the threshold `eps`
        )pydoc")
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
    cls.doc() = R"pydoc(
        A backend based on fusion trees.

        Notes
        -----
        Data is :class:`FusionTreeData` (coupled-sector ``block_inds`` + forest blocks).
        )pydoc";

    cls.def(py::init([](py::object block_backend, float64 eps) {
                return std::make_shared<FusionTreeBackend>(as_shared_block_backend(block_backend),
                                                           eps);
            }),
            py::arg("block_backend"),
            py::arg("eps") = 5.0e-14);

    cls.def_readwrite("eps", &FusionTreeBackend::eps);
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
      R"pydoc(
      Perform an arbitrary number of traces. Pairs are converted to leg idcs.

      Returns ``data, codomain, domain``.
      )pydoc");

    cls.def("apply_instructions",
            &FusionTreeBackend::apply_instructions,
            py::arg("tensor"),
            py::arg("instructions"),
            py::arg("codomain_idcs"),
            py::arg("domain_idcs"),
            py::arg("new_codomain"),
            py::arg("new_domain"),
            py::arg("mixes_codomain_domain"),
            R"pydoc(
Apply a sequence of braid/bend/twist instructions (used by :meth:`permute_legs`).
)pydoc");
}

} // namespace cyten
