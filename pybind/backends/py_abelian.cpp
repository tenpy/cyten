#include "../py_cyten_pybind11.h"

#include <cyten/backends/abelian.h>
#include <cyten/block_backend/numpy.h>
#include <cyten/block_backend/torch.h>

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
    cls.doc() = R"pydoc(
Data stored in a Tensor for :class:`AbelianBackend`.

The :attr:`block_inds` can be visualized as follows::

    |           ---- codomain ---->  <--- domain ----
    |
    |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |    b |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |    l |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |    o |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |    c |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |    k |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |    s |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |      |    x  x  x  x  x  x  x  x  x  x  x  x  x
    |      v

Attributes
----------
dtype : Dtype
    The dtype of the data
device : str
    The device on which the blocks are currently stored.
    We currently only support tensors which have all blocks on a single device.
    Should be the device returned by :func:`BlockBackend.as_device`.
blocks : list of block
    A list of blocks containing the actual entries of the tensor.
    Leg order is ``[*codomain, *reversed(domain()]``, like ``Tensor.legs``.
block_inds : 2D ndarray
    A 2D array of positive integers with shape (len(blocks), num_legs).
    The block `blocks[n]` belongs to the `block_inds[n, m]`-th sector of ``leg``,
    that is to ``leg.sector_decomposition[block_inds[n, m]]``, where::

        leg == (codomain.spaces[m] if m < len(codomain) else domain.spaces[-1 - m])
            == tensor.get_leg_co_domain(m)

    Thus, the columns of `block_inds` follow the same ordering convention as :attr:`Tensor.legs`.
    By convention, we store `blocks` and `block_inds` such that ``np.lexsort(block_inds.T)``
    is sorted.

Parameters
----------
dtype, device, blocks, block_inds
    like attributes above, but not necessarily sorted
is_sorted : bool
    If ``False`` (default), we permute `blocks` and `block_inds` according to
    ``np.lexsort(block_inds.T)``.
    If ``True``, we assume they are sorted *without* checking.
)pydoc";

    cls.def(py::init([](Dtype dtype,
                        std::string device,
                        std::vector<BlockBackend::BlockPtr> blocks,
                        py::array_t<int64> block_inds,
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
           R"pydoc(
Return the index ``n`` of the block which matches the block_inds.

I.e. such that ``all(self.block_inds[n, :] == block_inds)``.
Return None if no such ``n`` exists.
)pydoc")
      .def("get_block",
           &AbelianBackendData::get_block,
           py::arg("block_inds"),
           R"pydoc(
Get the block at given block indices.

Return the block in :attr:`blocks` matching the given block_inds,
i.e. `self.blocks[n]` such that `all(self.block_inds[n, :] == blocks_inds)`
or None if no such block exists
)pydoc")
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
    cls.doc() = R"pydoc(
Backend for Abelian group symmetries.

Notes
-----
The data stored for the various tensor classes defined in ``cyten.tensors`` is::

    - ``SymmetricTensor``:
        An ``AbelianBackendData`` instance whose blocks have as many axes as the tensor has legs.

    - ``DiagonalTensor`` :
        An ``AbelianBackendData`` instance whose blocks have only a single axis.
        This is the diagonal of the corresponding 2D block in a ``Tensor``.

    - ``Mask`` :
        An ``AbelianBackendData`` instance whose blocks have only a single axis and bool values.
)pydoc";

    cls.def(py::init([](py::object block_backend) {
                auto backend =
                  std::make_shared<AbelianBackend>(as_shared_block_backend(block_backend));
                backend->DataCls = py::type::of<AbelianBackendData>();
                return backend;
            }),
            py::arg("block_backend"));

    cls.def_static("wrap", &AbelianBackend::wrap, py::arg("data"));
    cls.def_static("unwrap", &AbelianBackend::unwrap, py::arg("data"));

    cls.def("leg_pipe_map_incoming_block_inds",
            &AbelianBackend::leg_pipe_map_incoming_block_inds,
            py::arg("pipe"),
            py::arg("incoming_block_inds"));

    cls.def(
      "partial_trace",
      [](AbelianBackend& self,
         py::object tensor,
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
      py::arg("levels") = py::none());

    // Override static from_hdf5 (base throws NotImplemented).
    cls.def_static("from_hdf5",
                   &AbelianBackend::from_hdf5,
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"));

    // Expose under both the C++ name and the Python private name.
    m.def("valid_block_inds",
          &valid_block_inds,
          py::arg("codomain"),
          py::arg("domain"),
          R"pydoc(
Charge-allowed block index combinations for ``codomain`` / ``domain``, lexsorted.
)pydoc");
    m.attr("_valid_block_inds") = m.attr("valid_block_inds");
}

} // namespace cyten
