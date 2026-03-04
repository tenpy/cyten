#include <cyten/dtypes.h>
#include <cyten/numpy_block_backend.h>

#include <pybind11/numpy.h>
#include <stdexcept>

namespace cyten {

namespace {
py::module_
numpy_module()
{
    return py::module_::import("numpy");
}

py::object
np_attr(char const* name)
{
    return numpy_module().attr(name);
}
} // namespace

// -----------------------------------------------------------------------------
// NumpyBlock
// -----------------------------------------------------------------------------

NumpyBlock::NumpyBlock(py::object arr)
  : arr_(std::move(arr))
{
}

std::vector<cyten_int>
NumpyBlock::shape() const
{
    py::tuple t = py::cast<py::tuple>(arr_.attr("shape"));
    std::vector<cyten_int> s;
    s.reserve(t.size());
    for (auto const& item : t)
        s.push_back(py::cast<cyten_int>(item));
    return s;
}

Dtype
NumpyBlock::dtype() const
{
    return dtype::from_numpy_dtype(arr_.attr("dtype"));
}

std::string
NumpyBlock::device() const
{
    return "cpu";
}

py::object
NumpyBlock::operator[](std::vector<cyten_int> const& idcs) const
{
    py::tuple key = py::cast(idcs);
    py::object result = arr_.attr("__getitem__")(key);
    py::object sh = result.attr("shape");
    if (py::len(sh) == 0)
        return result.attr("item")();
    return py::cast(std::make_shared<NumpyBlock>(result));
}

// -----------------------------------------------------------------------------
// NumpyBlockBackend helpers
// -----------------------------------------------------------------------------

NumpyBlock const*
NumpyBlockBackend::ptr(BlockCPtr const& b)
{
    auto* p = dynamic_cast<NumpyBlock const*>(b.get());
    if (!p)
        throw std::invalid_argument("block is not a NumpyBlock");
    return p;
}

py::object
NumpyBlockBackend::obj(BlockCPtr const& b)
{
    return ptr(b)->array();
}

BlockPtr
NumpyBlockBackend::wrap(py::object arr)
{
    return std::make_shared<NumpyBlock>(std::move(arr));
}

// -----------------------------------------------------------------------------
// NumpyBlockBackend
// -----------------------------------------------------------------------------

NumpyBlockBackend::NumpyBlockBackend()
  : BlockBackend("cpu")
{
    svd_algorithms = { "gesdd", "gesvd", "robust", "robust_silent" };
}

std::string
NumpyBlockBackend::get_backend_name() const
{
    return "NumpyBlockBackend";
}

bool
NumpyBlockBackend::is_correct_block_type(BlockCPtr const& block) const
{
    return dynamic_cast<NumpyBlock const*>(block.get()) != nullptr;
}

BlockPtr
NumpyBlockBackend::apply_leg_permutations(BlockCPtr const& block,
                                          std::vector<py::array_t<cyten_int>> const& perms)
{
    py::object a = obj(block);
    py::module_ np = numpy_module();
    py::list ix_parts;
    for (auto const& p : perms)
        ix_parts.append(p);
    py::object ix = np.attr("ix_")(*ix_parts);
    return wrap(a[ix]);
}

BlockPtr
NumpyBlockBackend::as_block(py::object a,
                            std::optional<Dtype> dtype_opt,
                            std::optional<std::string> device)
{
    (void)as_device(device);
    py::module_ np = numpy_module();
    py::object dt = dtype_opt ? dtype::to_numpy_dtype(*dtype_opt) : py::none();
    py::object arr = np.attr("asarray")(a, dt);
    // integer -> float64 like Python
    if (py::hasattr(arr, "dtype")) {
        py::object d = arr.attr("dtype");
        if (np.attr("issubdtype")(d, np.attr("integer")).cast<bool>())
            arr = arr.attr("astype")(np.attr("float64"), py::arg("copy") = false);
    }
    return wrap(arr);
}

std::string
NumpyBlockBackend::as_device(std::optional<std::string> device)
{
    if (!device)
        return default_device;
    if (*device != default_device)
        throw std::invalid_argument("NumpyBlockBackend does not support device " + *device);
    return *device;
}

std::vector<cyten_int>
NumpyBlockBackend::abs_argmax(BlockCPtr const& block)
{
    py::object a = obj(block);
    py::module_ np = numpy_module();
    py::object idx =
      np.attr("unravel_index")(np.attr("argmax")(np.attr("abs")(a)), a.attr("shape"));
    py::tuple t = py::cast<py::tuple>(idx);
    std::vector<cyten_int> out;
    for (auto const& item : t)
        out.push_back(py::cast<cyten_int>(item));
    return out;
}

BlockPtr
NumpyBlockBackend::abs(BlockCPtr const& a)
{
    return wrap(np_attr("abs")(obj(a)));
}

BlockPtr
NumpyBlockBackend::add_axis(BlockCPtr const& a, int pos)
{
    return wrap(np_attr("expand_dims")(obj(a), pos));
}

bool
NumpyBlockBackend::block_all(BlockCPtr const& a)
{
    return np_attr("all")(obj(a)).cast<bool>();
}

bool
NumpyBlockBackend::allclose(BlockCPtr const& a, BlockCPtr const& b, double rtol, double atol)
{
    return np_attr("allclose")(obj(a), obj(b), py::arg("rtol") = rtol, py::arg("atol") = atol)
      .cast<bool>();
}

BlockPtr
NumpyBlockBackend::angle(BlockCPtr const& a)
{
    return wrap(np_attr("angle")(obj(a)));
}

bool
NumpyBlockBackend::block_any(BlockCPtr const& a)
{
    return np_attr("any")(obj(a)).cast<bool>();
}

BlockPtr
NumpyBlockBackend::apply_mask(BlockCPtr const& block, BlockCPtr const& mask, int ax)
{
    return wrap(np_attr("compress")(obj(mask), obj(block), ax));
}

BlockPtr
NumpyBlockBackend::_argsort(BlockCPtr const& block, int axis)
{
    return wrap(np_attr("argsort")(obj(block), py::arg("axis") = axis));
}

BlockPtr
NumpyBlockBackend::conj(BlockCPtr const& a)
{
    return wrap(np_attr("conj")(obj(a)));
}

BlockPtr
NumpyBlockBackend::copy_block(BlockCPtr const& a, std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(np_attr("copy")(obj(a)));
}

BlockPtr
NumpyBlockBackend::cutoff_inverse(BlockCPtr const& a, double cutoff)
{
    py::object arr = obj(a);
    py::module_ np = numpy_module();
    py::object denom = np.attr("where")(
      np.attr("less")(np.attr("abs")(arr), py::float_(cutoff)), np.attr("inf"), arr);
    return wrap(py::float_(1.0) / denom);
}

Dtype
NumpyBlockBackend::get_dtype(BlockCPtr const& a)
{
    return dtype::from_numpy_dtype(obj(a).attr("dtype"));
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::eigh(BlockCPtr const& block, std::optional<std::string> sort)
{
    py::object a = obj(block);
    py::tuple res = py::cast<py::tuple>(np_attr("linalg").attr("eigh")(a));
    py::object w = res[0];
    py::object v = res[1];
    if (sort) {
        BlockPtr perm = argsort(wrap(w), sort, 0);
        py::object perm_arr = obj(perm);
        w = np_attr("take")(w, perm_arr);
        v = np_attr("take")(v, perm_arr, py::arg("axis") = 1);
    }
    return { wrap(w), wrap(v) };
}

BlockPtr
NumpyBlockBackend::eigvalsh(BlockCPtr const& block, std::optional<std::string> sort)
{
    py::object w = np_attr("linalg").attr("eigvalsh")(obj(block));
    if (sort) {
        BlockPtr perm = argsort(wrap(w), sort, 0);
        w = np_attr("take")(w, obj(perm));
    }
    return wrap(w);
}

BlockPtr
NumpyBlockBackend::enlarge_leg(BlockCPtr const& block, BlockCPtr const& mask, int axis)
{
    py::object a = obj(block);
    py::object m = obj(mask);
    py::module_ np = numpy_module();
    py::list shape_list = py::cast<py::list>(a.attr("shape"));
    shape_list[axis] = py::len(m);
    py::object shape = py::tuple(shape_list);
    py::object res = np.attr("zeros")(shape, py::arg("dtype") = a.attr("dtype"));
    py::list idcs;
    for (int i = 0; i < py::len(shape_list); ++i)
        idcs.append(i == axis ? m : py::object(py::slice(py::none(), py::none(), py::none())));
    res[py::tuple(idcs)] = np_attr("copy")(a);
    return wrap(res);
}

BlockPtr
NumpyBlockBackend::exp(BlockCPtr const& a)
{
    return wrap(np_attr("exp")(obj(a)));
}

BlockPtr
NumpyBlockBackend::block_from_diagonal(BlockCPtr const& diag)
{
    return wrap(np_attr("diag")(obj(diag)));
}

BlockPtr
NumpyBlockBackend::block_from_mask(BlockCPtr const& mask, Dtype dtype)
{
    py::object m = obj(mask);
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    int M = py::cast<int>(m.attr("shape").attr("__getitem__")(0));
    int N = py::cast<int>(np.attr("sum")(m));
    py::object res = np.attr("zeros")(py::make_tuple(N, M), py::arg("dtype") = dt);
    res[py::make_tuple(np.attr("arange")(N), m)] = 1;
    return wrap(res);
}

BlockPtr
NumpyBlockBackend::block_from_numpy(py::array const& a,
                                    std::optional<Dtype> dtype_opt,
                                    std::optional<std::string> device)
{
    (void)as_device(device);
    if (!dtype_opt)
        return wrap(py::object(a));
    return wrap(np_attr("asarray")(a, dtype::to_numpy_dtype(*dtype_opt)));
}

std::string
NumpyBlockBackend::get_device(BlockCPtr const& /*a*/)
{
    return default_device;
}

BlockPtr
NumpyBlockBackend::get_diagonal(BlockCPtr const& a, std::optional<double> tol)
{
    py::object arr = obj(a);
    py::object res = np_attr("diagonal")(arr);
    if (tol) {
        py::object diag_mat = np_attr("diag")(res);
        if (!np_attr("allclose")(arr, diag_mat, py::arg("atol") = *tol).cast<bool>())
            throw std::invalid_argument("Not a diagonal block.");
    }
    return wrap(res);
}

bool
NumpyBlockBackend::get_block_mask_element(BlockCPtr const& a,
                                          cyten_int large_leg_idx,
                                          cyten_int small_leg_idx,
                                          cyten_int sum_block)
{
    py::object arr = obj(a);
    cyten_int dim0 = get_shape(a).at(0);
    cyten_int offset = (large_leg_idx / dim0) * sum_block;
    large_leg_idx %= dim0;
    if (!py::cast<bool>(arr.attr("__getitem__")(large_leg_idx)))
        return false;
    py::object prefix = arr.attr("__getitem__")(py::slice(0, large_leg_idx, 1));
    cyten_int running = py::cast<cyten_int>(np_attr("sum")(prefix));
    return small_leg_idx == offset + running;
}

BlockPtr
NumpyBlockBackend::imag(BlockCPtr const& a)
{
    return wrap(np_attr("imag")(obj(a)));
}

py::object
NumpyBlockBackend::item(BlockCPtr const& a)
{
    return obj(a).attr("item")();
}

BlockPtr
NumpyBlockBackend::kron(BlockCPtr const& a, BlockCPtr const& b)
{
    return wrap(np_attr("kron")(obj(a), obj(b)));
}

BlockPtr
NumpyBlockBackend::linear_combination(Scalar a_coef,
                                      BlockCPtr const& v,
                                      Scalar b_coef,
                                      BlockCPtr const& w)
{
    return wrap(a_coef.to_numpy() * obj(v) + b_coef.to_numpy() * obj(w));
}

BlockPtr
NumpyBlockBackend::log(BlockCPtr const& a)
{
    return wrap(np_attr("log")(obj(a)));
}

double
NumpyBlockBackend::max(BlockCPtr const& a)
{
    return py::cast<double>(np_attr("max")(obj(a)).attr("item")());
}

double
NumpyBlockBackend::max_abs(BlockCPtr const& a)
{
    return py::cast<double>(np_attr("max")(np_attr("abs")(obj(a))).attr("item")());
}

double
NumpyBlockBackend::min(BlockCPtr const& a)
{
    return py::cast<double>(np_attr("min")(obj(a)).attr("item")());
}

BlockPtr
NumpyBlockBackend::mul(py::object a, BlockCPtr const& b)
{
    return wrap(a * obj(b));
}

double
NumpyBlockBackend::norm(BlockCPtr const& a, double order, std::optional<int> axis)
{
    py::object arr = obj(a);
    if (!axis) {
        return py::cast<double>(np_attr("linalg")
                                  .attr("norm")(arr.attr("ravel")(), py::arg("ord") = order)
                                  .attr("item")());
    }
    return py::cast<double>(
      np_attr("linalg").attr("norm")(arr, py::arg("ord") = order, py::arg("axis") = *axis));
}

BlockPtr
NumpyBlockBackend::outer(BlockCPtr const& a, BlockCPtr const& b)
{
    return wrap(np_attr("tensordot")(obj(a), obj(b), py::make_tuple(py::tuple(), py::tuple())));
}

BlockPtr
NumpyBlockBackend::permute_axes(BlockCPtr const& a, std::vector<int> const& permutation)
{
    return wrap(np_attr("transpose")(obj(a), py::cast(permutation)));
}

BlockPtr
NumpyBlockBackend::random_normal(std::vector<cyten_int> const& dims,
                                 Dtype dtype,
                                 double sigma,
                                 std::optional<std::string> device)
{
    (void)as_device(device);
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    if (!dtype::is_real(dtype))
        sigma /= std::sqrt(2.0);
    py::object res = np.attr("random").attr("normal")(
      py::arg("loc") = 0, py::arg("scale") = sigma, py::arg("size") = dims);
    if (!dtype::is_real(dtype))
        res = res + py::cast(std::complex<double>(0, 1)) *
                      np.attr("random").attr("normal")(
                        py::arg("loc") = 0, py::arg("scale") = sigma, py::arg("size") = dims);
    return wrap(np.attr("asarray")(res, py::arg("dtype") = dt));
}

BlockPtr
NumpyBlockBackend::random_uniform(std::vector<cyten_int> const& dims,
                                  Dtype dtype,
                                  std::optional<std::string> device)
{
    (void)as_device(device);
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    py::object res = np.attr("random").attr("uniform")(
      py::arg("low") = -1, py::arg("high") = 1, py::arg("size") = dims);
    if (!dtype::is_real(dtype))
        res = res + py::cast(std::complex<double>(0, 1)) *
                      np.attr("random").attr("uniform")(
                        py::arg("low") = -1, py::arg("high") = 1, py::arg("size") = dims);
    return wrap(np.attr("asarray")(res, py::arg("dtype") = dt));
}

BlockPtr
NumpyBlockBackend::real(BlockCPtr const& a)
{
    return wrap(np_attr("real")(obj(a)));
}

BlockPtr
NumpyBlockBackend::real_if_close(BlockCPtr const& a, double tol)
{
    return wrap(np_attr("real_if_close")(obj(a), py::arg("tol") = tol));
}

BlockPtr
NumpyBlockBackend::scale_axis(BlockCPtr const& block, BlockCPtr const& factors, int axis)
{
    py::object a = obj(block);
    py::object f = obj(factors);
    py::module_ np = numpy_module();
    int ndim = static_cast<int>(get_shape(block).size());
    py::list idx;
    for (int i = 0; i < ndim; ++i)
        idx.append(i == axis ? py::object(py::slice(py::none(), py::none(), py::none()))
                             : py::object(py::none()));
    return wrap(a * f.attr("__getitem__")(py::tuple(idx)));
}

BlockPtr
NumpyBlockBackend::tile(BlockCPtr const& a, int repeats)
{
    return wrap(np_attr("tile")(obj(a), repeats));
}

std::vector<std::string>
NumpyBlockBackend::_block_repr_lines(BlockCPtr const& a,
                                     std::string const& indent,
                                     int max_width,
                                     int max_lines)
{
    py::module_ np = numpy_module();
    py::object arr = obj(a);
    np.attr("printoptions")(py::arg("linewidth") = max_width - static_cast<int>(indent.size()));
    py::str s = py::str(arr);
    py::list lines = s.attr("split")("\n");
    std::vector<std::string> out;
    int n = py::len(lines);
    int first = (max_lines - 1) / 2;
    int last = max_lines - 1 - first;
    for (int i = 0; i < std::min(first, n); ++i)
        out.push_back(indent + py::cast<std::string>(lines[i]));
    if (n > max_lines) {
        out.push_back(indent + "...");
        for (int i = std::max(n - last, first); i < n; ++i)
            out.push_back(indent + py::cast<std::string>(lines[i]));
    }
    return out;
}

BlockPtr
NumpyBlockBackend::reshape(BlockCPtr const& a, std::vector<cyten_int> const& shape)
{
    return wrap(np_attr("reshape")(obj(a), py::cast(shape)));
}

std::vector<cyten_int>
NumpyBlockBackend::get_shape(BlockCPtr const& a)
{
    return ptr(a)->shape();
}

BlockPtr
NumpyBlockBackend::sqrt(BlockCPtr const& a)
{
    return wrap(np_attr("sqrt")(obj(a)));
}

BlockPtr
NumpyBlockBackend::squeeze_axes(BlockCPtr const& a, std::vector<int> const& idcs)
{
    return wrap(np_attr("squeeze")(obj(a), py::cast(idcs)));
}

BlockPtr
NumpyBlockBackend::stable_log(BlockCPtr const& block, double cutoff)
{
    py::object arr = obj(block);
    py::module_ np = numpy_module();
    return wrap(
      np.attr("where")(np.attr("greater")(arr, py::float_(cutoff)), np_attr("log")(arr), 0.0));
}

BlockPtr
NumpyBlockBackend::sum(BlockCPtr const& a, int ax)
{
    return wrap(np_attr("sum")(obj(a), py::arg("axis") = ax));
}

std::complex<cyten_float>
NumpyBlockBackend::sum_all(BlockCPtr const& a)
{
    return py::cast<std::complex<cyten_float>>(np_attr("sum")(obj(a)).attr("item")());
}

BlockPtr
NumpyBlockBackend::multiply_blocks(BlockCPtr const& a, BlockCPtr const& b)
{
    return wrap(obj(a) * obj(b));
}

BlockPtr
NumpyBlockBackend::tdot(BlockCPtr const& a,
                        BlockCPtr const& b,
                        std::vector<int> const& idcs_a,
                        std::vector<int> const& idcs_b)
{
    return wrap(
      np_attr("tensordot")(obj(a), obj(b), py::make_tuple(py::cast(idcs_a), py::cast(idcs_b))));
}

BlockPtr
NumpyBlockBackend::to_dtype(BlockCPtr const& a, Dtype dtype)
{
    return wrap(np_attr("asarray")(obj(a), dtype::to_numpy_dtype(dtype)));
}

py::object
NumpyBlockBackend::to_numpy(BlockCPtr const& a, std::optional<py::object> numpy_dtype)
{
    py::object arr = obj(a);
    if (numpy_dtype)
        return np_attr("asarray")(arr, *numpy_dtype);
    return np_attr("asarray")(arr);
}

std::complex<cyten_float>
NumpyBlockBackend::trace_full(BlockCPtr const& a)
{
    py::object arr = obj(a);
    std::vector<cyten_int> sh = get_shape(a);
    size_t num_trace = sh.size() / 2;
    cyten_int trace_dim = 1;
    for (size_t i = 0; i < num_trace; ++i)
        trace_dim *= sh[i];
    std::vector<int> perm(sh.size());
    for (size_t i = 0; i < num_trace; ++i)
        perm[i] = static_cast<int>(i);
    for (size_t i = 0; i < num_trace; ++i)
        perm[num_trace + i] = static_cast<int>(2 * num_trace - 1 - i);
    arr = np_attr("transpose")(arr, py::cast(perm));
    arr = np_attr("reshape")(arr, py::make_tuple(trace_dim, trace_dim));
    return py::cast<std::complex<cyten_float>>(
      np_attr("trace")(arr, py::arg("axis1") = 0, py::arg("axis2") = 1).attr("item")());
}

BlockPtr
NumpyBlockBackend::trace_partial(BlockCPtr const& a,
                                 std::vector<int> const& idcs1,
                                 std::vector<int> const& idcs2,
                                 std::vector<int> const& remaining_idcs)
{
    py::object arr = obj(a);
    std::vector<int> perm = remaining_idcs;
    perm.insert(perm.end(), idcs1.begin(), idcs1.end());
    perm.insert(perm.end(), idcs2.begin(), idcs2.end());
    arr = np_attr("transpose")(arr, py::cast(perm));
    std::vector<cyten_int> sh = get_shape(a);
    cyten_int trace_dim = 1;
    for (int i : idcs1)
        trace_dim *= sh[i];
    py::tuple shape = arr.attr("shape");
    py::list new_shape;
    for (size_t i = 0; i < remaining_idcs.size(); ++i)
        new_shape.append(shape[i]);
    new_shape.append(trace_dim);
    new_shape.append(trace_dim);
    arr = np_attr("reshape")(arr, py::tuple(new_shape));
    return wrap(np_attr("trace")(arr, py::arg("axis1") = -2, py::arg("axis2") = -1));
}

BlockPtr
NumpyBlockBackend::eye_matrix(int dim, Dtype dtype, std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(np_attr("eye")(dim, py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

py::object
NumpyBlockBackend::get_block_element(BlockCPtr const& a, std::vector<cyten_int> const& idcs)
{
    return obj(a).attr("__getitem__")(py::cast(idcs)).attr("item")();
}

BlockPtr
NumpyBlockBackend::matrix_dot(BlockCPtr const& a, BlockCPtr const& b)
{
    return wrap(np_attr("dot")(obj(a), obj(b)));
}

BlockPtr
NumpyBlockBackend::matrix_exp(BlockCPtr const& matrix)
{
    return wrap(py::module_::import("scipy.linalg").attr("expm")(obj(matrix)));
}

BlockPtr
NumpyBlockBackend::matrix_log(BlockCPtr const& matrix)
{
    return wrap(py::module_::import("scipy.linalg").attr("logm")(obj(matrix)));
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_qr(BlockCPtr const& a, bool full)
{
    py::tuple qr = py::module_::import("scipy.linalg")
                     .attr("qr")(obj(a), py::arg("mode") = (full ? "full" : "economic"));
    return { wrap(qr[0]), wrap(qr[1]) };
}

std::tuple<BlockPtr, BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_svd(BlockCPtr const& a, std::optional<std::string> algorithm)
{
    std::string algo = algorithm ? *algorithm : "gesdd";
    if (algo == "gesdd") {
        py::tuple uvt = py::module_::import("scipy.linalg")
                          .attr("svd")(obj(a), py::arg("full_matrices") = false);
        return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
    }
    if (algo == "gesvd") {
        py::tuple uvt = py::module_::import("scipy.linalg")
                          .attr("svd")(obj(a),
                                       py::arg("full_matrices") = false,
                                       py::arg("lapack_driver") = "gesvd");
        return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
    }
    if (algo == "robust" || algo == "robust_silent") {
        try {
            py::tuple uvt = py::module_::import("scipy.linalg")
                              .attr("svd")(obj(a), py::arg("full_matrices") = false);
            return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
        } catch (py::error_already_set const&) {
            if (algo != "robust_silent")
                throw;
        }
        py::tuple uvt = py::module_::import("scipy.linalg")
                          .attr("svd")(obj(a),
                                       py::arg("full_matrices") = false,
                                       py::arg("lapack_driver") = "gesvd");
        return { wrap(uvt[0]), wrap(uvt[1]), wrap(uvt[2]) };
    }
    throw std::invalid_argument("SVD algorithm not supported: " + algo);
}

BlockPtr
NumpyBlockBackend::ones_block(std::vector<cyten_int> const& shape,
                              Dtype dtype,
                              std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(np_attr("ones")(py::cast(shape), py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

BlockPtr
NumpyBlockBackend::zeros(std::vector<cyten_int> const& shape,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(
      np_attr("zeros")(py::cast(shape), py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

std::shared_ptr<NumpyBlockBackend>
NumpyBlockBackend::load_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto obj = std::make_shared<NumpyBlockBackend>();
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    // std::vector<std::string> svd_algs = hdf5_loader.attr("load")(subpath +
    // std::string("svd_algorithms")); std::string default_dev = hdf5_loader.attr("load")(subpath +
    // std::string("default_device"));
    // TODO: could check svd_algorithms and default_dev for correctness.
    return obj;
}

} // namespace cyten
