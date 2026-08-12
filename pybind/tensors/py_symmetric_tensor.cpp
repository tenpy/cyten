#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/symmetric_tensor.h>

#include "py_trampolines.hpp"

#include "../py_cyten_pybind11.h"

#include <pybind11/stl.h>

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

namespace {

std::optional<std::vector<std::variant<int64, std::string>>>
optional_leg_order(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<std::variant<int64, std::string>> out;
    for (auto item : to_iterable(obj)) {
        if (py::isinstance<py::str>(item)) {
            out.emplace_back(item.cast<std::string>());
        } else {
            out.emplace_back(item.cast<int64>());
        }
    }
    return out;
}

} // namespace

void
bind_tensors_symmetric_tensor(py::module_& m)
{
    py::class_<SymmetricTensor, Tensor, PySymmetricTensor, py::smart_holder> cls(m, "SymmetricTensor");
    cls.doc() = R"pydoc(
A tensor that is symmetric, i.e. invariant under the symmetry.

.. note ::
    The constructor is not particularly user friendly.
    Consider using the various classmethods instead.

Parameters
----------
codomain : TensorProduct | list[Space]
    The codomain.
domain : TensorProduct | list[Space] | None
    The domain. ``None`` (the default) is equivalent to ``[]``, i.e. no legs in the domain.
backend : TensorBackend
    The backend of the tensor.
labels: list[list[str | None]] | list[str | None] | None
    Specify the labels for the legs.
    Can either give two lists, one for the codomain, one for the domain.
    Or a single flat list for all legs in the order of the :attr:`legs`,
    such that ``[codomain_labels, domain_labels]`` is equivalent
    to ``[*codomain_legs, *reversed(domain_legs)]``.
dtype : Dtype
    The dtype of tensor entries.

Attributes
----------
data:
    Backend-specific data structure that contains the numerical data, i.e. the free parameters
    of tensors with the given symmetry.
)pydoc";

    cls.def(py::init<TensorBackend::DataPtr, py::object, py::object, TensorBackend::Ptr, py::object>(),
            py::arg("data"),
            py::arg("codomain"),
            py::arg("domain") = py::none(),
            py::arg("backend") = nullptr,
            py::arg("labels") = py::none());

    cls.def_property(
      "data",
      [](SymmetricTensor& self) -> py::object {
          // Match Python NoSymmetryBackend: expose the raw Block, not BlockData wrapper.
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              return py::cast(NoSymmetryBackend::unwrap(self.data));
          }
          return py::cast(self.data);
      },
      [](SymmetricTensor& self, py::object obj) {
          if (std::dynamic_pointer_cast<NoSymmetryBackend>(self.backend)) {
              self.data = NoSymmetryBackend::wrap(obj.cast<BlockBackend::BlockPtr>());
          } else {
              self.data = obj.cast<TensorBackend::DataPtr>();
          }
      });

    cls.def("test_sanity", &SymmetricTensor::test_sanity, "Perform sanity checks.");
    cls.def("verify_dtype", &SymmetricTensor::verify_dtype);

    cls.def_static(
      "from_block_func",
      &SymmetricTensor::from_block_func,
      py::arg("func"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("func_kwargs") = py::none(),
      py::arg("shape_kw") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
Initialize a :class:`SymmetricTensor` by generating its blocks from a function.

Here "the blocks of a tensor" are the backend-specific blocks that contain the free
parameters of the tensor in the :attr:`data`. The concrete meaning of these blocks depends
on the backend.

Parameters
----------
func: callable
    A function with two possible signatures. If `shape_kw` is given, we expect::

        ``func(*, shape_kw: tuple[int, ...], **kwargs) -> BlockLike``

    Otherwise::

        ``func(shape: tuple[int, ...], **kwargs) -> BlockLike``

    Where ``shape`` is the shape of the block to be generate and `func_kwargs` are passed
    as ``kwargs``. The output is converted to backend-specific blocks
    via ``backend.as_block``. In particular, it may be modified in-place after that.
codomain, domain, backend, labels
    Arguments for constructor of :class:`SymmetricTensor`.
func_kwargs: dict, optional
    Additional keyword arguments to be passed to ``func``.
shape_kw: str
    If given, the shape is passed to `func` as a kwarg with this keyword.
dtype: Dtype, None
    If given, the resulting blocks from `func` are converted to this dtype.
device: str, optional
    If given, the resulting blocks are moved to that device.
    Per default, if `func` returns backend-specific blocks, their device is used and
    otherwise the default device of the backend.

See Also
--------
from_sector_block_func
    Allows the `func` to take the current coupled sectors as an argument.
)pydoc");

    cls.def_static(
      "from_dense_block",
      &SymmetricTensor::from_dense_block,
      py::arg("block"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      py::arg("tol") = 1e-6,
      py::arg("understood_braiding") = false,
      R"pydoc(
Convert a dense block of the backend to a Tensor.

Parameters
----------
block : Block-like
    The data to be converted to a Tensor as a backend-specific block or some data that
    can be converted using :meth:`BlockBackend.as_block`.
    This includes e.g. nested python iterables or numpy arrays.
    The order of axes should match the :attr:`Tensor.legs`, i.e. first the codomain legs,
    then the domain leg *in reverse order*.
    The block should be given in the "public" basis order of the `legs`, e.g.
    according to :attr:`ElementarySpace.sectors_of_basis`.
codomain, domain, backend, labels
    Arguments, like for constructor of :class:`SymmetricTensor`.
dtype: Dtype, optional
    If given, the block is converted to that dtype and the resulting tensor will have that
    dtype. By default, we detect the dtype from the block.
device: str, optional
    If given, the block is moved to that device. Per default, try to use the device of
    the `block`, if it is a backend-specific block, or fall back to the backends default
    device.
understood_braiding : bool
    For symmetries with non-trivial (but symmetric) braiding, e.g. fermions, the input
    dense block does not capture the braiding statistics correctly. This means e.g. that
    :func:`permute_legs` is not consistently reproduced by e.g. ``numpy.transpose`` on
    the dense block representation. This means that the input dense block needs to be
    constructed in the correct leg order. To avoid this pitfall, we raise an error by
    default. Set this flag to ``True`` to disable the error. It is then your responsibility
    to take care of leg orders and braids.
)pydoc");

    cls.def_static("from_dense_block_trivial_sector",
                   &SymmetricTensor::from_dense_block_trivial_sector,
                   py::arg("vector"),
                   py::arg("space"),
                   py::arg("backend") = nullptr,
                   py::arg("device") = py::none(),
                   py::arg("label") = py::none(),
                   "Inverse of to_dense_block_trivial_sector.");

    cls.def_static(
      "from_eye",
      &SymmetricTensor::from_eye,
      py::arg("co_domain"),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      R"pydoc(
The identity map as a SymmetricTensor.

Parameters
----------
co_domain
    The domain *and* codomain of the resulting tensor.
labels
    Can either specify the labels for all legs of the resulting tensor, like
    in the constructor of :class:`SymmetricTensor`.
    Alternatively, can give labels only for the codomain (one list), and the domain labels
    are constructed as their dual labels i.e. ``'p' <-> 'p*'``.
backend: TensorBackend, optional
    The backend of the tensor.
dtype: Dtype
    The dtype of the tensor.
device: str
    The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
    the block backend.
)pydoc");

    cls.def_static(
      "from_random_normal",
      &SymmetricTensor::from_random_normal,
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("mean") = py::none(),
      py::arg("sigma") = 1.0,
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      R"pydoc(
Generate a sample from the normal distribution.

The probability density is

.. math ::
    p(T) \propto \mathrm{exp}\left[
        \frac{1}{2 \sigma^2} \mathrm{Tr} (T - \mathtt{mean}) (T - \mathtt{mean})^\dagger
    \right]

.. note ::
    For a complex `dtype`, the samples are taken from the complex normal distribution,
    which corresponds to sampling the real and imaginary parts independently from (real)
    normal distributions with half the variance of the complex normal distribution.

Parameters
----------
codomain, domain, backend, labels
    Arguments, like for constructor of :class:`SymmetricTensor`.
    If `mean` is given, all of them are optional and the respective attributes of
    `mean` are used.
dtype: Dtype
    The dtype.
mean: SymmetricTensor, optional
    The mean of the distribution. ``None`` is equivalent to zero mean.
sigma: float
    The standard deviation of the distribution
)pydoc");

    cls.def_static(
      "from_random_uniform",
      &SymmetricTensor::from_random_uniform,
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      R"pydoc(
Generate a tensor with uniformly random block-entries.

The block entries, i.e. the free parameters of the tensor are drawn independently and
uniformly. If dtype is a real type, they are drawn from [-1, 1], if it is complex, real and
imaginary part are drawn independently from [-1, 1].

.. note ::
    This is not a well defined probability distribution on the space of symmetric tensors,
    since the meaning of the uniformly drawn numbers depends on both the choice of the
    basis and on the backend.

Parameters
----------
codomain, domain, backend, labels
    Arguments, like for constructor of :class:`SymmetricTensor`.
dtype: Dtype
    The dtype for the tensor.
)pydoc");

    cls.def_static(
      "from_sector_block_func",
      &SymmetricTensor::from_sector_block_func,
      py::arg("func"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("func_kwargs") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
Initialize a :class:`SymmetricTensor` by generating its blocks from a function.

Here "the blocks of a tensor" are the backend-specific blocks that contain the free
parameters of the tensor in the :attr:`data`. The concrete meaning of these blocks depends
on the backend.

Unlike :meth:`from_block_func`, this classmethod supports a `func` that takes the current
coupled sector as an argument. The tensor, as a map from its domain to its codomain is
block-diagonal in the coupled sectors, i.e. in the ``domain.sector_decomposition``.
Thus, the free parameters of a tensor are associated with one block of this structure,
and thus with a given coupled sector. A value of ``coupled`` indicates that the generated
block is (part of) the components that maps from ``coupled`` in the domain to ``coupled``
in the codomain.

Parameters
----------
func: callable
    A function with the following signature::

        ``func(shape: tuple[int, ...], coupled: Sector, **kwargs) -> BlockLike``

    Where ``shape`` is the shape of the block to be generated, ``coupled`` is the current
    coupled sector and `func_kwargs` are passed as ``kwargs``.
    The output is converted to backend-specific blocks via ``backend.block_backend.as_block``.
codomain, domain, backend, labels
    Arguments, like for constructor of :class:`SymmetricTensor`.
func_kwargs: dict, optional
    Additional keyword arguments to be passed to ``func``.
shape_kw: str
    If given, the shape is passed to `func` as a kwarg with this keyword.
dtype: Dtype, None
    If given, the resulting blocks from `func` are converted to this dtype.
device: str, optional
    If given, the resulting blocks are moved to that device.
    Per default, if `func` returns backend-specific blocks, their device is used and
    otherwise the default device of the backend.

See Also
--------
from_block_func
)pydoc");

    cls.def_static("from_sector_projection",
                   &SymmetricTensor::from_sector_projection,
                   py::arg("co_domain"),
                   py::arg("sector"),
                   py::arg("backend") = nullptr,
                   py::arg("labels") = py::none(),
                   py::arg("dtype") = py::none(),
                   py::arg("device") = py::none(),
                   "A tensor that projects onto a given coupled sector of it domain.");

    cls.def_static(
      "from_tree_pairs",
      &SymmetricTensor::from_tree_pairs,
      py::arg("trees"),
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("device") = py::none(),
      R"pydoc(
Create a tensor from a linear combination of fusion-tree splitting-tree pairs.

Parameters
----------
trees : {(FusionTree, FusionTree): (J+K)-D Block}
    Specifies the linear combination that defines the resulting tensor.
    Each entry of the dict, ``{(X, Y): coeffs}`` represents several contributions to the
    linear combination, one per entry of the block ``coeffs``.
    The contribution with prefactor ``coeffs[n1, ..., nJ, mK, ..., m1]`` (note the axis order!)
    consists of the following steps as a map from domain to codomain::

        1. Project each leg ``k`` of the domain to a single sector, where the sector is
           given by ``Y.uncoupled[k]`` and the degeneracy index by ``mk`` (an index to
           the array ``coeffs``).

        2. Apply the fusion tree ``Y``.

        3. Apply the splitting tree ``X``.

        4. Apply inclusions on each leg ``j`` of the codomain, where the sector is given by
           ``X.uncoupled[j]`` and the degeneracy index by ``nj`` (an index to the array
           ``coeffs``).
codomain, domain, backend, labels
    Arguments, like for constructor of :class:`SymmetricTensor`.
)pydoc");

    cls.def_static(
      "from_zero",
      &SymmetricTensor::from_zero,
      py::arg("codomain"),
      py::arg("domain") = py::none(),
      py::arg("backend") = nullptr,
      py::arg("labels") = py::none(),
      py::arg("dtype") = Dtype::Complex128,
      py::arg("device") = py::none(),
      R"pydoc(
A zero tensor.

Parameters
----------
codomain, domain, backend, labels:
    Arguments, like for constructor of :class:`SymmetricTensor`.
dtype: Dtype
    The dtype for the entries.
device: str
    The device of the tensor. If ``None``, use the :attr:`BlockBackend.default_device` of
    the block backend.
)pydoc");

    cls.def_static("_parse_default_dtype",
                   &SymmetricTensor::_parse_default_dtype,
                   py::arg("dtype"),
                   py::arg("symmetry"));

    cls.def("as_dtype", &SymmetricTensor::as_dtype, py::arg("dtype"));
    cls.def("as_SymmetricTensor",
            &SymmetricTensor::as_SymmetricTensor,
            py::arg("guarantee_copy") = false,
            py::arg("warning") = py::none());
    cls.def("copy",
            &SymmetricTensor::copy,
            py::arg("deep") = true,
            py::arg("device") = py::none(),
            py::arg("dtype") = py::none());
    cls.def("diagonal",
            &SymmetricTensor::diagonal,
            py::arg("check_offdiagonal") = false,
            R"pydoc(
The diagonal part as a :class:`DiagonalTensor`.

Parameters
----------
check_offdiagonal: bool
    If we should check that the off-diagonal parts vanish.
)pydoc");
    cls.def("_get_item", &SymmetricTensor::_get_item, py::arg("idx"));
    cls.def("move_to_device", &SymmetricTensor::move_to_device, py::arg("device"));
    cls.def("to_backend",
            &SymmetricTensor::to_backend,
            py::arg("backend"),
            py::arg("dtype") = py::none(),
            py::arg("device") = py::none());
    cls.def(
      "to_dense_block",
      [](SymmetricTensor& self, py::object leg_order, std::optional<Dtype> dtype, bool understood_braiding) {
          return self.to_dense_block(optional_leg_order(leg_order), dtype, understood_braiding);
      },
      py::arg("leg_order") = py::none(),
      py::arg("dtype") = py::none(),
      py::arg("understood_braiding") = false);
    cls.def("to_dense_block_trivial_sector",
            &SymmetricTensor::to_dense_block_trivial_sector,
            R"pydoc(
Assumes self is a single-leg tensor and returns its components in the trivial sector.

See Also
--------
from_dense_block_trivial_sector
)pydoc");
    cls.def("save_hdf5",
            &SymmetricTensor::save_hdf5,
            py::arg("hdf5_saver"),
            py::arg("h5gr"),
            py::arg("subpath"),
            "Export SymmetricTensor to hdf5 such that it can be re-imported with from_hdf5");
    cls.def_static("from_hdf5",
                   &SymmetricTensor::from_hdf5,
                   py::arg("hdf5_loader"),
                   py::arg("h5gr"),
                   py::arg("subpath"),
                   "Import SymmetricTensor from hdf5");
}

} // namespace cyten
