#include "../py_cyten_pybind11.h"

#include <cyten/backends/abelian.h>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

void
bind_abelian_backend_data(py::module_& m)
{
    py::class_<AbelianBackendData, TensorBackend::Data, py::smart_holder> cls(m,
                                                                              "AbelianBackendData");
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

    cls.def("get_block_num",
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

} // namespace cyten
