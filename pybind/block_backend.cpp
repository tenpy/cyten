#include "cyten_pybind11.h"
#include <cyten/block.h>
#include <cyten/block_backend.h>
#include <cyten/numpy_block_backend.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace cyten {

void
bind_block_backend(py::module_& m)
{
    py::class_<Scalar, py::smart_holder>(
      m, "Scalar", "Scalar value with Dtype; use accessors to cast to float, complex, or bool.")
      .def(py::init<Dtype, cyten_complex>(),
           py::arg("dtype"),
           py::arg("value"),
           "Construct from Dtype and numeric value (int/float/complex; stored as complex "
           "internally).")
      .def_property_readonly("dtype", &Scalar::dtype)
      .def("real", &Scalar::real, "Real part (valid for any dtype).")
      .def(
        "cyten_double", &Scalar::cyten_double, "As float; raises if dtype is not Float32/Float64.")
      .def("as_complex", &Scalar::as_complex, "As complex (real/bool have zero imaginary part).")
      .def("as_bool", &Scalar::as_bool, "As bool; raises if dtype is not Bool.")
      .def("to_numpy", &Scalar::to_numpy, "Return as numpy scalar (np.bool_, np.float64, etc.).");

    py::class_<Block, py::smart_holder>(m, "Block", "Abstract base for dense blocks.")
      .def("shape", &Block::shape)
      .def("dtype", &Block::dtype)
      .def("device", &Block::device);

    py::class_<NumpyBlock, Block, py::smart_holder>(m, "NumpyBlock")
      .def(py::init<py::object>(), py::arg("arr"))
      .def("array", &NumpyBlock::array, py::return_value_policy::reference_internal)
      .def("__getitem__",
           [](NumpyBlock const& self, py::object key) -> py::object {
               py::object result = self.array().attr("__getitem__")(key);
               py::object sh = result.attr("shape");
               if (py::len(sh) == 0)
                   return result.attr("item")();
               return py::cast(std::make_shared<NumpyBlock>(result));
           })
      .def("__setitem__",
           [](NumpyBlock& self, py::object key, py::object value) {
               self.array().attr("__setitem__")(key, value);
           })
      .def(
        "__array__",
        [](NumpyBlock const& self, py::object dtype) {
            if (dtype.is_none())
                return self.array();
            return self.array().attr("astype")(dtype);
        },
        py::arg("dtype") = py::none())
      .def("__mul__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__mul__")(other));
           })
      .def("__rmul__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__rmul__")(other));
           })
      .def("__truediv__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__truediv__")(other));
           })
      .def("__add__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__add__")(other));
           })
      .def("__radd__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__radd__")(other));
           })
      .def("__sub__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__sub__")(other));
           })
      .def("__rsub__",
           [](NumpyBlock const& self, py::object other) {
               return std::make_shared<NumpyBlock>(self.array().attr("__rsub__")(other));
           })
      .def_property_readonly("shape",
                             [](NumpyBlock const& self) { return self.array().attr("shape"); })
      .def_property_readonly("dtype",
                             [](NumpyBlock const& self) { return self.array().attr("dtype"); });

    py::class_<BlockBackend, py::smart_holder>(
      m, "BlockBackend", "Abstract base class that defines the operation on dense blocks.")
      .def_readwrite("default_device", &BlockBackend::default_device)
      .def_readwrite("svd_algorithms", &BlockBackend::svd_algorithms)
      .def("__repr__", [](BlockBackend const& self) { return self.get_backend_name() + "()"; })
      .def("__str__", [](BlockBackend const& self) { return self.get_backend_name() + "()"; })
      .def("abs",
           &BlockBackend::abs,
           py::arg("a"),
           R"pydoc(The absolute value of a complex number, elementwise.)pydoc")
      .def("apply_leg_permutations",
           &BlockBackend::apply_leg_permutations,
           R"pydoc(Apply permutations to every axis of a dense block)pydoc")
      .def(
        "apply_basis_perm",
        &BlockBackend::apply_basis_perm,
        py::arg("block"),
        py::arg("legs"),
        py::arg("inv") = false,
        R"pydoc(Apply basis_perm of a ElementarySpace (or its inverse) on every axis of a dense block)pydoc")
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
        block : Block
            The new block

        See Also
        --------
        block_copy : Guarantees an independent copy.
        )pydoc")
      .def("as_device",
           &BlockBackend::as_device,
           py::arg("device") = py::none(),
           R"pydoc(
          Convert input string to unambiguous device name.

          In particular, this should map any possible aliases to one unique name, e.g.
          for PyTorch, map 'cuda' to 'cuda:0'. Also checks if that device is valid and available.
          )pydoc")
      .def(
        "abs_argmax",
        &BlockBackend::abs_argmax,
        R"pydoc(Return the indices (one per axis) of the largest entry (by magnitude) of the block)pydoc")
      .def("add_axis", &BlockBackend::add_axis)
      .def("block_all",
           &BlockBackend::block_all,
           R"pydoc(Require a boolean block. If all of its entries are True)pydoc")
      .def("allclose",
           &BlockBackend::allclose,
           py::arg("a"),
           py::arg("b"),
           py::arg("rtol") = 1e-5,
           py::arg("atol") = 1e-8)
      .def(
        "angle",
        &BlockBackend::angle,
        R"pydoc(The angle of a complex number such that a == exp(1.j * angle). Elementwise.)pydoc")
      .def("block_any",
           &BlockBackend::block_any,
           R"pydoc(Require a boolean block. If any of its entries are True)pydoc")
      .def("apply_mask",
           &BlockBackend::apply_mask,
           R"pydoc(Apply a mask (1D boolean block) to a block, slicing/projecting that axis)pydoc")
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
              sort                 order
              ==================== =============================
              'm>', 'LM'           Largest magnitude first
              -------------------- -----------------------------
              'm<', 'SM'            Smallest magnitude first
              -------------------- -----------------------------
              '>', 'LR', 'LA'       Largest real part first
              -------------------- -----------------------------
              '<', 'SR', 'SA'       Smallest real part first
              -------------------- -----------------------------
              'LI'                  Largest imaginary part first
              -------------------- -----------------------------
              'SI'                  Smallest imaginary part first
              ==================== =============================

          axis : int
              The axis along which to sort.

          Returns
          -------
          1D block of int
              The indices that would sort the block.
          )pydoc")
      .def("combine_legs",
           py::overload_cast<BlockCPtr const&, std::vector<std::vector<int>> const&, bool>(
             &BlockBackend::combine_legs),
           py::arg("a"),
           py::arg("leg_idcs_combine"),
           py::arg("cstyles") = true,
           R"pydoc(
          Combine each group of legs in leg_idcs_combine into a single leg.

          The group of legs in each entry of leg_idcs_combine must be contiguous.
          The legs can be combined in C style (default) or F style; the style can
          be specified for each group of legs independently.
          )pydoc")
      .def("conj", &BlockBackend::conj, R"pydoc(Complex conjugate of a block)pydoc")
      .def("copy_block",
           &BlockBackend::copy_block,
           py::arg("a"),
           py::arg("device") = py::none(),
           R"pydoc(
          Create a new, independent block with the same data.

          Parameters
          ----------
          a : Block
              The block to copy
          device : str, optional
              The device for the new block. Per default, use the same device as the old block.

          See Also
          --------
          as_block : Function to guarantee dtype and device, without forcing copies.
          )pydoc")
      .def("dagger",
           &BlockBackend::dagger,
           R"pydoc(Permute axes to reverse order and elementwise conj.)pydoc")
      .def("get_dtype", &BlockBackend::get_dtype)
      .def("eigh",
           &BlockBackend::eigh,
           py::arg("block"),
           py::arg("sort") = py::none(),
           R"pydoc(
          Eigenvalue decomposition of a 2D hermitian block.

          Return a 1D block of eigenvalues and a 2D block of eigenvectors.

          Parameters
          ----------
          block : Block
              The block to decompose
          sort : str, optional
              How the eigenvalues are sorted: 'm>', 'm<', '>', '<'
          )pydoc")
      .def("eigvalsh",
           &BlockBackend::eigvalsh,
           py::arg("block"),
           py::arg("sort") = py::none(),
           R"pydoc(
          Eigenvalues of a 2D hermitian block.

          Return a 1D block of eigenvalues.

          Parameters
          ----------
          block : Block
              The block to decompose
          sort : str, optional
              How the eigenvalues are sorted: 'm>', 'm<', '>', '<'
          )pydoc")
      .def(
        "exp",
        &BlockBackend::exp,
        R"pydoc(The *elementwise* exponential. Not to be confused with matrix_exp, the *matrix* exponential.)pydoc")
      .def("block_from_diagonal",
           &BlockBackend::block_from_diagonal,
           R"pydoc(Return a 2D square block that has the 1D diag on the diagonal)pydoc")
      .def("block_from_mask",
           &BlockBackend::block_from_mask,
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
      .def("get_device", &BlockBackend::get_device)
      .def("get_diagonal",
           &BlockBackend::get_diagonal,
           py::arg("a"),
           py::arg("tol") = py::none(),
           R"pydoc(Get the diagonal of a 2D block as a 1D block)pydoc")
      .def("imag",
           &BlockBackend::imag,
           R"pydoc(The imaginary part of a complex number, elementwise.)pydoc")
      .def(
        "inner",
        [](BlockBackend& self, BlockCPtr const& a, BlockCPtr const& b, bool do_dagger) {
            return Scalar(Dtype::Complex128, self.inner(a, b, do_dagger)).to_numpy();
        },
        py::arg("a"),
        py::arg("b"),
        py::arg("do_dagger") = false,
        R"pydoc(
        Dense block version of tensors.inner.

        If do_dagger, sum(conj(a[i1, i2, ..., iN]) * b[i1, ..., iN])
        otherwise, sum(a[i1, ..., iN] * b[iN, ..., i2, i1]).
        )pydoc")
      .def("is_real",
           &BlockBackend::is_real,
           R"pydoc(
          If the block is comprised of real numbers.

          Complex numbers with small or zero imaginary part still cause a False return.
          )pydoc")
      .def(
        "item",
        &BlockBackend::item,
        R"pydoc(Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as python float or complex)pydoc")
      .def("kron",
           &BlockBackend::kron,
           R"pydoc(
          The kronecker product.

          Parameters
          ----------
          a, b : Block
              Two blocks with the same number of dimensions.

          Notes
          -----
          The elements are products of elements from a and b:
          kron(a, b)[k0, k1, ..., kN] = a[i0, i1, ..., iN] * b[j0, j1, ..., jN]
          where kt = it * st + jt, t = 0,...,N. (Taken from numpy docs)
          )pydoc")
      .def(
        "log",
        &BlockBackend::log,
        R"pydoc(The *elementwise* natural logarithm. Not to be confused with matrix_log, the *matrix* logarithm.)pydoc")
      .def("max", &BlockBackend::max)
      .def("max_abs", &BlockBackend::max_abs)
      .def("min", &BlockBackend::min)
      .def("norm",
           &BlockBackend::norm,
           py::arg("a"),
           py::arg("order") = 2.0,
           py::arg("axis") = py::none(),
           R"pydoc(
           The p-norm vector-norm of a block.

           Parameters
           ----------
           order : float
               The order p of the norm. Unlike numpy, we always compute vector norms, never matrix norms.
               We only support p-norms: ||x|| = (sum_i |x_i|^p)^(1/p).
           axis : int or None
               axis=None means "all axes", i.e. norm of the flattened block.
               An integer means to broadcast the norm over all other axes.
           )pydoc")
      .def(
        "outer",
        &BlockBackend::outer,
        R"pydoc(Outer product of blocks. res[i1,...,iN,j1,...,jM] = a[i1,...,iN] * b[j1,...,jM])pydoc")
      .def("permute_axes", &BlockBackend::permute_axes)
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
           R"pydoc(The real part of a complex number, elementwise.)pydoc")
      .def("real_if_close",
           &BlockBackend::real_if_close,
           py::arg("a"),
           py::arg("tol"),
           R"pydoc(
          If a block is close to its real part, return the real part.

          Otherwise the original block. Elementwise.
          )pydoc")
      .def("reshape", &BlockBackend::reshape)
      .def("get_shape", &BlockBackend::get_shape)
      .def("sqrt", &BlockBackend::sqrt, R"pydoc(The elementwise square root)pydoc")
      .def("squeeze_axes", &BlockBackend::squeeze_axes)
      .def(
        "stable_log",
        &BlockBackend::stable_log,
        R"pydoc(Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.)pydoc")
      .def("sum",
           &BlockBackend::sum,
           py::arg("a"),
           py::arg("ax"),
           R"pydoc(The sum over a single axis.)pydoc")
      .def("sum_all",
           &BlockBackend::sum_all,
           R"pydoc(
           The sum of all entries of the block.

           If the block contains boolean values, this should return the number of True entries.
           )pydoc")
      .def("tdot", &BlockBackend::tdot)
      .def("to_dtype", &BlockBackend::to_dtype)
      .def("to_numpy", &BlockBackend::to_numpy, py::arg("a"), py::arg("numpy_dtype") = py::none())
      .def("trace_full", &BlockBackend::trace_full)
      .def("trace_partial", &BlockBackend::trace_partial)
      .def("eye_block",
           &BlockBackend::eye_block,
           py::arg("legs"),
           py::arg("dtype"),
           py::arg("device") = py::none(),
           R"pydoc(
           The identity matrix, reshaped to a block.

           Note the unusual leg order [m1,...,mJ,mJ*,...,m1*], which is chosen to match eye_data.
           Note also that the legs only specify the dimensions of the first half, namely m1,...,mJ.
           )pydoc")
      .def("eye_matrix",
           &BlockBackend::eye_matrix,
           py::arg("dim"),
           py::arg("dtype"),
           py::arg("device") = py::none(),
           R"pydoc(The dim x dim identity matrix)pydoc")
      .def("get_block_element", &BlockBackend::get_block_element)
      .def("get_block_mask_element",
           &BlockBackend::get_block_mask_element,
           py::arg("a"),
           py::arg("large_leg_idx"),
           py::arg("small_leg_idx"),
           py::arg("sum_block") = 0,
           R"pydoc(
           Get an element of a mask.

           Mask elements are True if the entry a[large_leg_idx] is the small_leg_idx-th True in the block.

           Parameters
           ----------
           a : Block
               The mask block
           large_leg_idx, small_leg_idx : int
               The block indices
           sum_block : int, optional
               Number of True entries in the block, i.e. sum_block == self.sum_all(a). Agrees
               with the sector multiplicity of the small leg. (Only important if the sector dimension is larger than 1.)
           )pydoc")
      .def("matrix_dot",
           &BlockBackend::matrix_dot,
           R"pydoc(As in numpy.dot, both a and b might be matrix or vector.)pydoc")
      .def("matrix_exp", &BlockBackend::matrix_exp)
      .def("matrix_log", &BlockBackend::matrix_log)
      .def("matrix_lq", &BlockBackend::matrix_lq)
      .def("matrix_qr", &BlockBackend::matrix_qr, R"pydoc(QR decomposition of a 2D block)pydoc")
      .def("matrix_svd",
           &BlockBackend::matrix_svd,
           py::arg("a"),
           py::arg("algorithm") = py::none(),
           R"pydoc(Internal version of matrix_svd, to be implemented by subclasses.)pydoc")
      .def("ones_block",
           &BlockBackend::ones_block,
           py::arg("shape"),
           py::arg("dtype"),
           py::arg("device") = py::none())
      .def("synchronize",
           &BlockBackend::synchronize,
           R"pydoc(Wait for asynchronous processes (if any) to finish)pydoc")
      .def(
        "test_block_sanity",
        [](BlockBackend& self,
           BlockCPtr const& block,
           py::object expect_shape,
           py::object expect_dtype,
           py::object expect_device) {
            std::optional<std::vector<cyten_int>> shape_opt;
            if (!expect_shape.is_none()) {
                std::vector<cyten_int> sh;
                for (py::handle h : expect_shape) {
                    sh.push_back(py::cast<cyten_int>(h));
                }
                shape_opt = std::move(sh);
            }
            std::optional<Dtype> dtype_opt;
            if (!expect_dtype.is_none())
                dtype_opt = py::cast<Dtype>(expect_dtype);
            std::optional<std::string> device_opt;
            if (!expect_device.is_none())
                device_opt = py::cast<std::string>(expect_device);
            self.test_block_sanity(block, shape_opt, dtype_opt, device_opt);
        },
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
                  &BlockBackend::load_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"))
      .def(
        "cutoff_inverse",
        &BlockBackend::cutoff_inverse,
        R"pydoc(The elementwise cutoff-inverse: 1 / a where abs(a) >= cutoff, otherwise 0.)pydoc")
      .def("enlarge_leg", &BlockBackend::enlarge_leg)
      .def("linear_combination", &BlockBackend::linear_combination)
      .def("mul", &BlockBackend::mul)
      .def("permute_combined_matrix",
           &BlockBackend::permute_combined_matrix,
           py::arg("block"),
           py::arg("dims1"),
           py::arg("idcs1"),
           py::arg("dims2"),
           py::arg("idcs2"),
           R"pydoc(
          For a matrix with two combined multi-indices, permute the sub-indices.

          Parameters
          ----------
          block : 2D Block
              A matrix with combined axes [(m1.m2...mJ), (n1.n2...nK)].
          dims1 : list or 1D array of int
              The dimensions of the subindices [m1, m2, ..., mJ].
          idcs1 : list or 1D array of int
              Which of the axes [m1, m2, ..., mJ, n1, n2, ..., nK] should be in the first
              multi-index of the result.
          dims2 : list or 1D array of int
              The dimensions of the subindices [n1, n2, ..., nK].
          idcs2 : list or 1D array of int
              Which of the axes [m1, m2, ..., mJ, n1, n2, ..., nK] should be in the second
              multi-index of the result.

          Returns
          -------
          2D block
              A matrix with the same entries as block, but rearranged to the new axis order,
              e.g. [M, N], where M == combined([m1,...,mJ,n1,...,nK][idcs1]) and
              N == combined([m1,...,mJ,n1,...,nK][idcs2]).

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
          block : 2D Block
              A matrix with axes [M, N], where either M = (m1.m2...mJ) or N = (n1.n2...nK)
              is a multi-index but not both.
          axis : int
              Which of the two axes has the multi-indices.
          dims : list or 1D array of int
              The dimensions of the sub-indices, e.g. [m1, m2, ..., mJ].
          idcs : list or 1D array of int
              The order of the sub-indices in the results, such that the result has
              axes [[m1, m2, ..., mJ][i] for i in idcs].

          Returns
          -------
          2D Block
              A matrix with the same entries as `a`, but rearranged to the new axis order,
              i.e. [M_new, N_new] where e.g. M_new = combined([m1, m2, ..., mJ][idcs]).

          See Also
          --------
          permute_combined_matrix
          )pydoc")
      .def("scale_axis",
           &BlockBackend::scale_axis,
           py::arg("block"),
           py::arg("factors"),
           py::arg("axis"),
           R"pydoc(
          Multiply block with the factors (a 1D block), along a given axis.

          E.g. if block is 4D and axis==2 with numpy-like broadcasting, this would be
          block * factors[None, None, :, None].
          )pydoc")
      .def("split_legs",
           py::overload_cast<BlockCPtr const&,
                             std::vector<int> const&,
                             std::vector<std::vector<cyten_int>> const&,
                             bool>(&BlockBackend::split_legs),
           py::arg("a"),
           py::arg("idcs"),
           py::arg("dims"),
           py::arg("cstyles") = true,
           R"pydoc(
           Split legs into groups of legs with specified dimensions.

           The splitting of a leg can be in C style (default) or F style. In the latter case,
           the specified dimensions of the resulting group of legs are reversed. The style can
           be specified for each group of legs independently.
           )pydoc")
      .def("tensor_outer",
           &BlockBackend::tensor_outer,
           R"pydoc(
           Version of tensors.outer on blocks.

           Note the different leg order: res[i1,...,iK,j1,...,jM,i{K+1},...,iN] == a[i1,...,iN] * b[j1,...,jM]
           intended to be used with K == a_num_codomain_legs.
           )pydoc")
      .def(
        "tile",
        &BlockBackend::tile,
        R"pydoc(Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.)pydoc")
      .def("_block_repr_lines", &BlockBackend::_block_repr_lines);

    py::class_<NumpyBlockBackend, BlockBackend, py::smart_holder>(m, "NumpyBlockBackend")
      .def(py::init<>());
}

} // namespace cyten
