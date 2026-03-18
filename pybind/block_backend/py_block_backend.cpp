#include "../py_cyten_pybind11.h"
#include "py_dtypes.cpp"
#include "py_numpy.cpp"
#include "py_trampolines.hpp"

#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/numpy.h>
#include <pybind11/detail/common.h>

namespace cyten {

void
bind_block_backend(py::module_& m)
{
    bind_block_backend_dtypes(m);

    py::class_<BlockBackend, PyBlockBackend, py::smart_holder> block_backend(m, "BlockBackend");
    block_backend.doc() = "Abstract base class that defines the operation on dense blocks.";

    py::class_<BlockBackend::Block, PyBlock, py::smart_holder>(
      block_backend, "BlockCls", "Abstract base for dense blocks.")
      .def_property_readonly(
        "shape",
        [&](const BlockBackend::Block& self) {
            return py::cast<py::tuple>(py::cast(self.shape()));
        },
        "The shape of the block.")
      .def_property_readonly("dtype", &BlockBackend::Block::dtype)
      .def_property_readonly("device", &BlockBackend::Block::device)
      .def("get_backend",
           &BlockBackend::Block::get_backend,
           py::return_value_policy::reference,
           "Return the backend for this block's device.")
      .def(
        "__add__",
        [](const BlockBackend::Block& self, const BlockCPtr& other) {
            return self.operator+(other);
        },
        py::arg("other"),
        "Elementwise addition with another block.")
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& s) { return self * s; },
        py::arg("other"),
        "Multiplication by a scalar.")
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, BlockBackend::Scalar s) { return self * s; },
        py::arg("other"),
        "Right multiplication by a scalar.")
      .def("__getitem__",
           py::overload_cast<py::object>(&BlockBackend::Block::get_item),
           py::arg("key"))
      // .def( /// python is not const-correct, so we can't provide a const access
      //   "__getitem__",
      //   py::overload_cast<py::object>(&BlockBackend::Block::get_item, py::const_),
      //   py::arg("key"))
      .def(
        "__setitem__",
        [](BlockBackend::Block& self, py::object key, py::object value) {
            self.set_item(key, value);
        },
        py::arg("key"),
        py::arg("value"))
      .def("to_numpy",
           py::overload_cast<Dtype>(&BlockBackend::Block::to_numpy, py::const_),
           py::arg("dtype"),
           "Convert to numpy array with the given Dtype.")
      .def("_item_as_complex128",
           &BlockBackend::Block::_item_as_complex128,
           "Return the element of a zero-dimensional block as a complex128.")
      .def("_item_as_int64",
           &BlockBackend::Block::_item_as_int64,
           "Return the element of a zero-dimensional block as a int64.");

    py::class_<BlockBackend::Scalar, py::smart_holder>(
      block_backend,
      "Scalar",
      "Scalar value with Dtype; use accessors to cast to float, complex, or bool.")
      .def(py::init<std::shared_ptr<BlockBackend::Block>>(),
           py::arg("block"),
           "Construct from a 0-d block (ndim == 0). Raises if block is null or ndim != 0.")
      .def_property_readonly("dtype", &BlockBackend::Scalar::dtype)
      .def("real", &BlockBackend::Scalar::real, "Real part (valid for any dtype).")
      .def("as_float64",
           &BlockBackend::Scalar::as_float64,
           "As float; raises if dtype is not Float32/Float64.")
      .def("as_complex128",
           &BlockBackend::Scalar::as_complex128,
           "As complex (real/bool have zero imaginary part).")
      .def("as_bool", &BlockBackend::Scalar::as_bool, "As bool; raises if dtype is not Bool.")
      .def("to_numpy",
           &BlockBackend::Scalar::to_numpy,
           "Return as numpy scalar (np.bool_, np.float64, etc.).")
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self + other;
        },
        py::arg("other"),
        "Addition with another scalar.")
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self - other;
        },
        py::arg("other"),
        "Subtraction with another scalar.")
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self * other;
        },
        py::arg("other"),
        "Multiplication with another scalar.")
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self / other;
        },
        py::arg("other"),
        "Division with another scalar.");

    block_backend // init and attributes
      .def(py::init<std::string>(), py::arg("device") = "cpu")
      .def_readonly("default_device", &BlockBackend::default_device);

    block_backend //  methods
      .def("__repr__",
           [](const BlockBackend& self) { return self.get_backend_name() + std::string("()"); })
      .def("__str__",
           [](const BlockBackend& self) { return self.get_backend_name() + std::string("()"); })
      .def("as_scalar",
           py::overload_cast<bool>(&BlockBackend::as_scalar),
           py::arg("value"),
           "Convert a bool to a scalar block.")
      .def("as_scalar",
           py::overload_cast<float64>(&BlockBackend::as_scalar),
           py::arg("value"),
           "Convert a float64 to a scalar block.")
      .def("as_scalar",
           py::overload_cast<complex64>(&BlockBackend::as_scalar),
           py::arg("value"),
           "Convert a complex64 to a scalar block.")
      .def("as_scalar",
           py::overload_cast<py::object, Dtype>(&BlockBackend::as_scalar),
           py::arg("value"),
           py::arg("dtype"),
           "Convert a Python object to a scalar block with the given Dtype.")
      .def("abs",
           &BlockBackend::abs,
           py::arg("a"),
           "The absolute value of a complex number, elementwise.")
      .def("apply_basis_perm",
           &BlockBackend::apply_basis_perm,
           py::arg("block"),
           py::arg("legs"),
           py::arg("inv") = false,
           "Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block")
      .def("apply_leg_permutations",
           &BlockBackend::apply_leg_permutations,
           py::arg("block"),
           py::arg("perms"),
           "Apply permutations to every axis of a dense block")
      .def("as_block",
           &BlockBackend::as_block,
           py::arg("a"),
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none(),
           R"pydoc(
           Convert objects to blocks.

           Should support blocks, numpy arrays, nested python containers. May support more.
           If `a` is already a block of correct dtype on the correct device, it may be returned
           un-modified.

           Returns
           -------
           block: Block
               The new block

           See Also
           --------
           block_copy
               Guarantees an independent copy.
           )pydoc")
      .def("as_device",
           &BlockBackend::as_device,
           py::arg("device"),
           R"pydoc(
           Convert input string to unambiguous device name.

           In particular, this should map any possible aliases to one unique name, e.g.
           for PyTorch, map ``'cuda'`` to ``'cuda:0'``.
           Also checks if that device is valid and available.
           )pydoc")
      .def("abs_argmax",
           &BlockBackend::abs_argmax,
           py::arg("block"),
           "Return the indices (one per axis) of the largest entry (by magnitude) of the block")
      .def("add_axis", &BlockBackend::add_axis, py::arg("a"), py::arg("pos"))
      .def("all",
           &BlockBackend::all,
           py::arg("a"),
           "Require a boolean block. If all of its entries are True")
      .def("allclose",
           &BlockBackend::allclose,
           py::arg("a"),
           py::arg("b"),
           py::arg("rtol") = 1e-05,
           py::arg("atol") = 1e-08)
      .def("angle",
           &BlockBackend::angle,
           py::arg("a"),
           "The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise.")
      .def("any",
           &BlockBackend::any,
           py::arg("a"),
           "Require a boolean block. If any of its entries are True")
      .def("apply_mask",
           &BlockBackend::apply_mask,
           py::arg("block"),
           py::arg("mask"),
           py::arg("ax"),
           "Apply a mask (1D boolean block) to a block, slicing/projecting that axis")
      .def("argsort",
           &BlockBackend::argsort,
           py::arg("block"),
           py::arg("sort") = py::none(),
           py::arg("axis") = 0,
           R"pydoc(
           Return the permutation that would sort a block along one axis.

           Parameters
           ----------
           block : Block
               The block to sort.
           sort : str
               Specify how the arguments should be sorted.

               ==================== =============================
               `sort`               order
               ==================== =============================
               ``'m>', 'LM'``       Largest magnitude first
               -------------------- -----------------------------
               ``'m<', 'SM'``       Smallest magnitude first
               -------------------- -----------------------------
               ``'>', 'LR', 'LA'``  Largest real part first
               -------------------- -----------------------------
               ``'<', 'SR', 'SA'``  Smallest real part first
               -------------------- -----------------------------
               ``'LI'``             Largest imaginary part first
               -------------------- -----------------------------
               ``'SI'``             Smallest imaginary part first
               ==================== =============================

           axis : int
               The axis along which to sort

           Returns
           -------
           1D block of int
               The indices that would sort the block
           )pydoc")
      .def("_argsort",
           &BlockBackend::_argsort,
           py::arg("block"),
           py::arg("axis"),
           "Like :meth:`block_argsort` but can assume real valued block, and sort ascending")
      .def("combine_legs",
           py::overload_cast<const BlockCPtr&,
                             const std::vector<std::vector<int64>>&,
                             const std::vector<bool>&>(&BlockBackend::combine_legs),
           py::arg("a"),
           py::arg("leg_idcs_combine"),
           py::arg("cstyles"),
           R"pydoc(
           Combine each group of legs in `leg_idcs_combine` into a single leg.

           The group of legs in each entry of `leg_idcs_combine` must be contiguous.
           The legs can be combined in C style (default) or F style; the style can
           be specified for each group of legs independently.
           )pydoc")
      .def("combine_legs",
           py::overload_cast<const BlockCPtr&, const std::vector<std::vector<int64>>&, bool>(
             &BlockBackend::combine_legs),
           py::arg("a"),
           py::arg("leg_idcs_combine"),
           py::arg("cstyles") = true)
      .def("conj", &BlockBackend::conj, py::arg("a"), "Complex conjugate of a block")
      .def("copy_block",
           &BlockBackend::copy_block,
           py::arg("a"),
           py::arg("device") = py::none(),
           R"pydoc(
           Create a new, independent block with the same data

           Parameters
           ----------
           a
               The block to copy
           device
               The device for the new block. Per default, use the same device as the old block.

           See Also
           --------
           as_block
               Function to guarantee dtype and device, without forcing copies.
           )pydoc")
      .def(
        "cutoff_inverse",
        &BlockBackend::cutoff_inverse,
        py::arg("a"),
        py::arg("cutoff"),
        "The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.")
      .def("dagger",
           &BlockBackend::dagger,
           py::arg("a"),
           "Permute axes to reverse order and elementwise conj.")
      .def("get_dtype", &BlockBackend::get_dtype, py::arg("a"))
      .def("eigh",
           &BlockBackend::eigh,
           py::arg("block"),
           py::arg("sort") = py::none(),
           R"pydoc(
           Eigenvalue decomposition of a 2D hermitian block.

           Return a 1D block of eigenvalues and a 2D block of eigenvectors

           Parameters
           ----------
           block : Block
               The block to decompose
           sort : {'m>', 'm<', '>', '<'}
               How the eigenvalues are sorted
           )pydoc")
      .def("eigvalsh",
           &BlockBackend::eigvalsh,
           py::arg("block"),
           py::arg("sort") = py::none(),
           R"pydoc(
           Eigenvalues of a 2D hermitian block.

           Return a 1D block of eigenvalues

           Parameters
           ----------
           block : Block
               The block to decompose
           sort : {'m>', 'm<', '>', '<'}
               How the eigenvalues are sorted
           )pydoc")
      .def("enlarge_leg",
           &BlockBackend::enlarge_leg,
           py::arg("block"),
           py::arg("mask"),
           py::arg("axis"))
      .def("exp",
           &BlockBackend::exp,
           py::arg("a"),
           R"pydoc(
           The *elementwise* exponential.

           Not to be confused with :meth:`matrix_exp`, the *matrix* exponential.
           )pydoc")
      .def("block_from_diagonal",
           &BlockBackend::block_from_diagonal,
           py::arg("diag"),
           "Return a 2D square block that has the 1D ``diag`` on the diagonal")
      .def("block_from_mask",
           &BlockBackend::block_from_mask,
           py::arg("mask"),
           py::arg("dtype"),
           R"pydoc(
           Convert a mask to a full block.

           Return a (N, M) of numbers (float or complex dtype) from a 1D bool-valued block shape (M,)
           where N is the number of True entries. The result is the coefficient matrix of the projection map.
           )pydoc")
      .def("block_from_numpy",
           &BlockBackend::block_from_numpy,
           py::arg("a"),
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none())
      .def("get_device", &BlockBackend::get_device, py::arg("a"))
      .def("get_diagonal",
           &BlockBackend::get_diagonal,
           py::arg("a"),
           py::arg("tol"),
           "Get the diagonal of a 2D block as a 1D block")
      .def("imag",
           &BlockBackend::imag,
           py::arg("a"),
           "The imaginary part of a complex number, elementwise.")
      .def("inner",
           &BlockBackend::inner,
           py::arg("a"),
           py::arg("b"),
           py::arg("do_dagger"),
           R"pydoc(
           Dense block version of tensors.inner.

           If do dagger, ``sum(conj(a[i1, i2, ..., iN]) * b[i1, ..., iN])``
           otherwise, ``sum(a[i1, ..., iN] * b[iN, ..., i2, i1])``.
           )pydoc")
      .def("is_real",
           &BlockBackend::is_real,
           py::arg("a"),
           R"pydoc(
           If the block is comprised of real numbers.

           Complex numbers with small or zero imaginary part still cause a `False` return.
           )pydoc")
      .def("item",
           &BlockBackend::item,
           py::arg("a"),
           "Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as "
           "python float or complex")
      .def("kron",
           &BlockBackend::kron,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           The kronecker product.

           Parameters
           ----------
           a, b
               Two blocks with the same number of dimensions.

           Notes
           -----
           The elements are products of elements from `a` and `b`::
               kron(a, b)[k0, k1, ..., kN] = a[i0, i1, ..., iN] * b[j0, j1, ..., jN]

           where::
               kt = it * st + jt,  t = 0,...,N

           (Taken from numpy docs)
           )pydoc")
      .def("linear_combination",
           &BlockBackend::linear_combination,
           py::arg("a"),
           py::arg("v"),
           py::arg("b"),
           py::arg("w"))
      .def("log",
           &BlockBackend::log,
           py::arg("a"),
           R"pydoc(
           The *elementwise* natural logarithm.

           Not to be confused with :meth:`matrix_log`, the *matrix* logarithm.
           )pydoc")
      .def("max", &BlockBackend::max, py::arg("a"))
      .def("max_abs", &BlockBackend::max_abs, py::arg("a"))
      .def("min", &BlockBackend::min, py::arg("a"))
      .def("mul", &BlockBackend::mul, py::arg("a"), py::arg("b"))
      .def("norm",
           &BlockBackend::norm,
           py::arg("a"),
           py::arg("order") = 2,
           py::arg("axis") = py::none(),
           R"pydoc(
           The p-norm vector-norm of a block.

           Parameters
           ----------
           order : float
               The order :math:`p` of the norm.
               Unlike numpy, we always compute vector norms, never matrix norms.
               We only support p-norms :math:`\Vert x \Vert = \sqrt[p]{\sum_i \abs{x_i}^p}`.
           axis : int | None
               ``axis=None`` means "all axes", i.e. norm of the flattened block.
               An integer means to broadcast the norm over all other axes.
           )pydoc")
      .def("outer",
           &BlockBackend::outer,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Outer product of blocks.

           ``res[i1,...,iN,j1,...,jM] = a[i1,...,iN] * b[j1,...,jM]``
           )pydoc")
      .def("permute_axes", &BlockBackend::permute_axes, py::arg("a"), py::arg("permutation"))
      .def("permute_combined_matrix",
           &BlockBackend::permute_combined_matrix,
           py::arg("block"),
           py::arg("dims1"),
           py::arg("idcs1"),
           py::arg("dims2"),
           py::arg("idcs2"),
           R"pydoc(
           For a matrix `a` with two combined multi-indices, permute the sub-indices.

           Parameters
           ----------
           a : 2D Block
               A matrix with combined axes ``[(m1.m2...mJ), (n1.n2...nK)]``.
           dims1 : list or 1D array of int
               The dimensions of the subindices ``[m1, m2, ..., mJ]``.
           idcs1 : list or 1D array of int
               Which of the axes ``[m1, m2, ..., mJ, n1, n2, ..., nK]`` should be in the first
               multi-index of the result.
           dims2 : list or 1D array of int
               The dimensions of the subindices ``[n1, n2, ..., nK]``.
           idcs2 : list or 1D array of int
               Which of the axes ``[m1, m2, ..., mJ, n1, n2, ..., nK]`` should be in the second
               multi-index of the result.

           Returns
           -------
           2D block
               A matrix with the same entries as `a`, but rearranged to the new axis order,
               e.g. ``[M, N]``, where ``M == combined([m1, m2, ..., mJ, n1, n2, ..., nK][idcs1])``
               and ``N == combined([m1, m2, ..., mJ, n1, n2, ..., nK][idcs2])``.

           See Also
           --------
           permute_combined_idx
           )pydoc")
      .def("permute_combined_idx",
           &BlockBackend::permute_combined_idx,
           py::arg("block"),
           py::arg("axis"),
           py::arg("dims"),
           py::arg("idcs"),
           R"pydoc(
           For a matrix `a` with a single combined multi-index, permute sub-indices.

           Parameters
           ----------
           a : 2D Block
               A matrix with axes ``[M, N]``, where either ``M = (m1.m2...mJ)`` or ``N = (n1.n2...nK)``
               is a multi-index *but not both*.
           axis : int
               Which of the two axes has the multi-indices
           dims : list or 1D array of int
               The dimensions of the sub-indices, e.g. ``[m1, m2, ..., mJ]``.
           idcs : list of 1D array of int
               The order of the sub-indices in the results, such that the result has
               axes ``[[m1, m2, ..., mJ][i] for i in idcs]``.

           Returns
           -------
           2D Block
               A matrix with the same entries as `a`, but rearranged to the new axis order,
               i.e. ``[M_new, N_new]`` where e.g. ``M_new = combined([m1, m2, ..., mJ][idcs])``.

           See Also
           --------
           permute_combined_matrix
           )pydoc")
      .def("random_normal",
           &BlockBackend::random_normal,
           py::arg("dims"),
           py::arg("dtype"),
           py::arg("sigma"),
           py::arg("device") = py::none())
      .def("random_uniform",
           &BlockBackend::random_uniform,
           py::arg("dims"),
           py::arg("dtype"),
           py::arg("device") = py::none())
      .def("real",
           &BlockBackend::real,
           py::arg("a"),
           "The real part of a complex number, elementwise.")
      .def("real_if_close",
           &BlockBackend::real_if_close,
           py::arg("a"),
           py::arg("tol"),
           R"pydoc(
           If a block is close to its real part, return the real part.

           Otherwise the original block. Elementwise.
           )pydoc")
      .def("tile",
           &BlockBackend::tile,
           py::arg("a"),
           py::arg("repeats"),
           "Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.")
      .def("_block_repr_lines",
           &BlockBackend::_block_repr_lines,
           py::arg("a"),
           py::arg("indent"),
           py::arg("max_width"),
           py::arg("max_lines"))
      .def("reshape", &BlockBackend::reshape, py::arg("a"), py::arg("shape"))
      .def("scale_axis",
           &BlockBackend::scale_axis,
           py::arg("block"),
           py::arg("factors"),
           py::arg("axis"),
           R"pydoc(
           Multiply block with the factors (a 1D block), along a given axis.

           E.g. if block is 4D and ``axis==2`` with numpy-like broadcasting, this is would be
           ``block * factors[None, None, :, None]``.
           )pydoc")
      .def(
        "get_shape",
        [](BlockBackend& self, const BlockCPtr& a) {
            return py::cast<py::tuple>(py::cast(self.get_shape(a)));
        },
        py::arg("a"))
      .def("split_legs",
           py::overload_cast<const BlockCPtr&,
                             const std::vector<int64>&,
                             const std::vector<std::vector<int64>>&,
                             const std::vector<bool>&>(&BlockBackend::split_legs),
           py::arg("a"),
           py::arg("idcs"),
           py::arg("dims"),
           py::arg("cstyles"),
           R"pydoc(
           Split legs into groups of legs with specified dimensions.

           The splitting of a leg can be in C style (default) or F style. In the
           latter case, the specified dimensions of the resulting group of legs
           *are reversed*. The style can be specified for each group of legs
           independently.
           )pydoc")
      .def("split_legs",
           py::overload_cast<const BlockCPtr&,
                             const std::vector<int64>&,
                             const std::vector<std::vector<int64>>&,
                             bool>(&BlockBackend::split_legs),
           py::arg("a"),
           py::arg("idcs"),
           py::arg("dims"),
           py::arg("cstyles") = true)
      .def("sqrt", &BlockBackend::sqrt, py::arg("a"), "The elementwise square root")
      .def("squeeze_axes", &BlockBackend::squeeze_axes, py::arg("a"), py::arg("idcs"))
      .def("stable_log",
           &BlockBackend::stable_log,
           py::arg("block"),
           py::arg("cutoff"),
           "Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.")
      .def("sum", &BlockBackend::sum, py::arg("a"), py::arg("ax"), "The sum over a single axis.")
      .def("sum_all",
           &BlockBackend::sum_all,
           py::arg("a"),
           R"pydoc(
           The sum of all entries of the block.

           If the block contains boolean values, this should return the number of ``True`` entries.
           )pydoc")
      .def("tdot",
           &BlockBackend::tdot,
           py::arg("a"),
           py::arg("b"),
           py::arg("idcs_a"),
           py::arg("idcs_b"))
      .def("tensor_outer",
           &BlockBackend::tensor_outer,
           py::arg("a"),
           py::arg("b"),
           py::arg("K"),
           R"pydoc(
           Version of ``tensors.outer`` on blocks.

           Note the different leg order to usual outer products::

               res[i1,...,iK,j1,...,jM,i{K+1},...,iN] == a[i1,...,iN] * b[j1,...,jM]

           intended to be used with ``K == a_num_codomain_legs``.
           )pydoc")
      .def("to_dtype", &BlockBackend::to_dtype, py::arg("a"), py::arg("dtype"))
      .def("to_numpy", &BlockBackend::to_numpy, py::arg("a"), py::arg("numpy_dtype") = py::none())
      .def("trace_full", &BlockBackend::trace_full, py::arg("a"))
      .def("trace_partial",
           &BlockBackend::trace_partial,
           py::arg("a"),
           py::arg("idcs1"),
           py::arg("idcs2"),
           py::arg("remaining_idcs"))
      .def("eye_block",
           &BlockBackend::eye_block,
           py::arg("legs"),
           py::arg("dtype"),
           py::arg("device") = py::none(),
           R"pydoc(
           The identity matrix, reshaped to a block.

           Note the unusual leg order ``[m1,...,mJ,mJ*,...,m1*]``,
           which is chosen to match :meth:`eye_data`.

           Note also that the ``legs`` only specify the dimensions of the first half,
           namely ``m1,...,mJ``.
           )pydoc")
      .def("eye_matrix",
           &BlockBackend::eye_matrix,
           py::arg("dim"),
           py::arg("dtype"),
           py::arg("device") = py::none(),
           "The ``dim x dim`` identity matrix")
      .def("get_block_element", &BlockBackend::get_block_element, py::arg("a"), py::arg("idcs"))
      .def("get_block_mask_element",
           &BlockBackend::get_block_mask_element,
           py::arg("a"),
           py::arg("large_leg_idx"),
           py::arg("small_leg_idx"),
           py::arg("sum_block") = 0,
           R"pydoc(
           Get an element of a mask.

           Mask elements are `True` if the entry `a[large_leg_idx]` is the `small_leg_idx`-th `True`
           in the block.

           Parameters
           ----------
           a
               The mask block
           large_leg_idx, small_leg_idx
               The block indices
           sum_block
               Number of `True` entries in the block, i.e., ``sum_block == self.sum_all(a)``. Agrees
               with the sector multiplicity of the small leg.
               (Only important if the sector dimension is larger than 1.)
           )pydoc")
      .def("matrix_dot",
           &BlockBackend::matrix_dot,
           py::arg("a"),
           py::arg("b"),
           "As in numpy.dot, both a and b might be matrix or vector.")
      .def("matrix_exp", &BlockBackend::matrix_exp, py::arg("matrix"))
      .def("matrix_log", &BlockBackend::matrix_log, py::arg("matrix"))
      .def("matrix_lq", &BlockBackend::matrix_lq, py::arg("a"), py::arg("full"))
      .def("matrix_qr",
           &BlockBackend::matrix_qr,
           py::arg("a"),
           py::arg("full"),
           "QR decomposition of a 2D block")
      .def("matrix_svd",
           &BlockBackend::matrix_svd,
           py::arg("a"),
           py::arg("algorithm"),
           "Perform a SVD decomposition of a matrix.")
      .def("possible_svd_algorithms",
           &BlockBackend::possible_svd_algorithms,
           "Possible algorithms for :meth:`matrix_svd`.")
      .def("ones_block",
           &BlockBackend::ones_block,
           py::arg("shape"),
           py::arg("dtype"),
           py::arg("device") = py::none())
      .def("synchronize",
           &BlockBackend::synchronize,
           "Wait for asynchronous processes (if any) to finish")
      .def("test_block_sanity",
           &BlockBackend::test_block_sanity,
           py::arg("block"),
           py::arg("expect_shape") = py::none(),
           py::arg("expect_dtype") = py::none(),
           py::arg("expect_device") = py::none())
      .def("zeros",
           &BlockBackend::zeros,
           py::arg("shape"),
           py::arg("dtype"),
           py::arg("device") = py::none())
      .def("save_hdf5",
           &BlockBackend::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &BlockBackend::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath")); // completed block_backend methods

    bind_block_backend_numpy(m);
}

} // namespace cyten
