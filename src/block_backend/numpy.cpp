#include <cyten/block_backend/dtypes.h>
#include <cyten/block_backend/numpy.h>

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

NumpyBlockBackend::Block::Block(py::array arr)
  : arr_(std::move(arr))
{
}

std::vector<int64>
NumpyBlockBackend::Block::shape() const
{
    py::tuple t = py::cast<py::tuple>(arr_.attr("shape"));
    std::vector<int64> s;
    s.reserve(t.size());
    for (auto const& item : t)
        s.push_back(py::cast<int64>(item));
    return s;
}

Dtype
NumpyBlockBackend::Block::dtype() const
{
    return dtype::from_numpy_dtype(arr_.attr("dtype"));
}

std::string
NumpyBlockBackend::Block::device() const
{
    return "cpu";
}

// -----------------------------------------------------------------------------
// NumpyBlockBackend helpers
// -----------------------------------------------------------------------------

NumpyBlockBackend::Block const*
NumpyBlockBackend::ptr(const BlockCPtr& b)
{
    auto* p = dynamic_cast<NumpyBlockBackend::Block const*>(b.get());
    if (!p)
        throw std::invalid_argument("block is not a NumpyBlock");
    return p;
}

py::object
NumpyBlockBackend::obj(const BlockCPtr& b)
{
    return ptr(b)->to_numpy();
}

BlockPtr
NumpyBlockBackend::wrap(py::object arr)
{
    return std::make_shared<NumpyBlockBackend::Block>(std::move(arr));
}

// -----------------------------------------------------------------------------
// NumpyBlockBackend
// -----------------------------------------------------------------------------

NumpyBlockBackend::NumpyBlockBackend()
  : BlockBackend("cpu")
{
}

std::string
NumpyBlockBackend::get_backend_name() const
{
    return "NumpyBlockBackend";
}

bool
NumpyBlockBackend::is_correct_block_type(const BlockCPtr& block) const
{
    return dynamic_cast<NumpyBlockBackend::Block const*>(block.get()) != nullptr;
}

BlockPtr
NumpyBlockBackend::apply_leg_permutations(const BlockCPtr& block,
                                          const std::vector<py::array_t<int64>>& perms)
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

std::vector<int64>
NumpyBlockBackend::abs_argmax(const BlockCPtr& block)
{
    py::object a = obj(block);
    py::module_ np = numpy_module();
    py::object idx =
      np.attr("unravel_index")(np.attr("argmax")(np.attr("abs")(a)), a.attr("shape"));
    py::tuple t = py::cast<py::tuple>(idx);
    std::vector<int64> out;
    for (auto const& item : t)
        out.push_back(py::cast<int64>(item));
    return out;
}

BlockPtr
NumpyBlockBackend::abs(const BlockCPtr& a)
{
    return wrap(np_attr("abs")(obj(a)));
}

BlockPtr
NumpyBlockBackend::add_axis(const BlockCPtr& a, int64 pos)
{
    return wrap(np_attr("expand_dims")(obj(a), pos));
}

bool
NumpyBlockBackend::block_all(const BlockCPtr& a)
{
    return np_attr("all")(obj(a)).cast<bool>();
}

bool
NumpyBlockBackend::allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol)
{
    return np_attr("allclose")(obj(a), obj(b), py::arg("rtol") = rtol, py::arg("atol") = atol)
      .cast<bool>();
}

BlockPtr
NumpyBlockBackend::angle(const BlockCPtr& a)
{
    return wrap(np_attr("angle")(obj(a)));
}

bool
NumpyBlockBackend::block_any(const BlockCPtr& a)
{
    return np_attr("any")(obj(a)).cast<bool>();
}

BlockPtr
NumpyBlockBackend::apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax)
{
    return wrap(np_attr("compress")(obj(mask), obj(block), ax));
}

BlockPtr
NumpyBlockBackend::_argsort(const BlockCPtr& block, int64 axis)
{
    return wrap(np_attr("argsort")(obj(block), py::arg("axis") = axis));
}

BlockPtr
NumpyBlockBackend::conj(const BlockCPtr& a)
{
    return wrap(np_attr("conj")(obj(a)));
}

BlockPtr
NumpyBlockBackend::copy_block(const BlockCPtr& a, std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(np_attr("copy")(obj(a)));
}

BlockPtr
NumpyBlockBackend::cutoff_inverse(const BlockCPtr& a, float64 cutoff)
{
    py::object arr = obj(a);
    py::module_ np = numpy_module();
    py::object denom = np.attr("where")(
      np.attr("less")(np.attr("abs")(arr), py::float_(cutoff)), np.attr("inf"), arr);
    return wrap(py::float_(1.0) / denom);
}

Dtype
NumpyBlockBackend::get_dtype(const BlockCPtr& a)
{
    return dtype::from_numpy_dtype(obj(a).attr("dtype"));
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::eigh(const BlockCPtr& block, std::optional<std::string> sort)
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
NumpyBlockBackend::eigvalsh(const BlockCPtr& block, std::optional<std::string> sort)
{
    py::object w = np_attr("linalg").attr("eigvalsh")(obj(block));
    if (sort) {
        BlockPtr perm = argsort(wrap(w), sort, 0);
        w = np_attr("take")(w, obj(perm));
    }
    return wrap(w);
}

BlockPtr
NumpyBlockBackend::enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis)
{
    py::object a = obj(block);
    py::object m = obj(mask);
    py::module_ np = numpy_module();
    py::list shape_list = py::cast<py::list>(a.attr("shape"));
    shape_list[axis] = py::len(m);
    py::object shape = py::tuple(shape_list);
    py::object res = np.attr("zeros")(shape, py::arg("dtype") = a.attr("dtype"));
    py::list idcs;
    for (int64 i = 0; i < py::len(shape_list); ++i)
        idcs.append(i == axis ? m : py::object(py::slice(py::none(), py::none(), py::none())));
    res[py::tuple(idcs)] = np_attr("copy")(a);
    return wrap(res);
}

BlockPtr
NumpyBlockBackend::exp(const BlockCPtr& a)
{
    return wrap(np_attr("exp")(obj(a)));
}

BlockPtr
NumpyBlockBackend::block_from_diagonal(const BlockCPtr& diag)
{
    return wrap(np_attr("diag")(obj(diag)));
}

BlockPtr
NumpyBlockBackend::block_from_mask(const BlockCPtr& mask, Dtype dtype)
{
    py::object m = obj(mask);
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    int64 M = py::cast<int64>(m.attr("shape").attr("__getitem__")(0));
    int64 N = py::cast<int64>(np.attr("sum")(m));
    py::object res = np.attr("zeros")(py::make_tuple(N, M), py::arg("dtype") = dt);
    res[py::make_tuple(np.attr("arange")(N), m)] = 1;
    return wrap(res);
}

BlockPtr
NumpyBlockBackend::block_from_numpy(const py::array& a,
                                    std::optional<Dtype> dtype_opt,
                                    std::optional<std::string> device)
{
    (void)as_device(device);
    if (!dtype_opt)
        return wrap(py::object(a));
    return wrap(np_attr("asarray")(a, dtype::to_numpy_dtype(*dtype_opt)));
}

std::string
NumpyBlockBackend::get_device(const BlockCPtr& /*a*/)
{
    return default_device;
}

BlockPtr
NumpyBlockBackend::get_diagonal(const BlockCPtr& a, std::optional<float64> tol)
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
NumpyBlockBackend::get_block_mask_element(const BlockCPtr& a,
                                          int64 large_leg_idx,
                                          int64 small_leg_idx,
                                          int64 sum_block)
{
    py::object arr = obj(a);
    int64 dim0 = get_shape(a).at(0);
    int64 offset = (large_leg_idx / dim0) * sum_block;
    large_leg_idx %= dim0;
    if (!py::cast<bool>(arr.attr("__getitem__")(large_leg_idx)))
        return false;
    py::object prefix = arr.attr("__getitem__")(py::slice(0, large_leg_idx, 1));
    int64 running = py::cast<int64>(np_attr("sum")(prefix));
    return small_leg_idx == offset + running;
}

BlockPtr
NumpyBlockBackend::imag(const BlockCPtr& a)
{
    return wrap(np_attr("imag")(obj(a)));
}

py::object
NumpyBlockBackend::item(const BlockCPtr& a)
{
    return obj(a).attr("item")();
}

BlockPtr
NumpyBlockBackend::kron(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(np_attr("kron")(obj(a), obj(b)));
}

BlockPtr
NumpyBlockBackend::linear_combination(Scalar a_coef,
                                      const BlockCPtr& v,
                                      Scalar b_coef,
                                      const BlockCPtr& w)
{
    return wrap(a_coef.to_numpy() * obj(v) + b_coef.to_numpy() * obj(w));
}

BlockPtr
NumpyBlockBackend::log(const BlockCPtr& a)
{
    return wrap(np_attr("log")(obj(a)));
}

float64
NumpyBlockBackend::max(const BlockCPtr& a)
{
    return py::cast<float64>(np_attr("max")(obj(a)).attr("item")());
}

float64
NumpyBlockBackend::max_abs(const BlockCPtr& a)
{
    return py::cast<float64>(np_attr("max")(np_attr("abs")(obj(a))).attr("item")());
}

float64
NumpyBlockBackend::min(const BlockCPtr& a)
{
    return py::cast<float64>(np_attr("min")(obj(a)).attr("item")());
}

BlockPtr
NumpyBlockBackend::mul(py::object a, const BlockCPtr& b)
{
    return wrap(a * obj(b));
}

float64
NumpyBlockBackend::norm(const BlockCPtr& a, float64 order, std::optional<int64> axis)
{
    py::object arr = obj(a);
    if (!axis) {
        return py::cast<float64>(np_attr("linalg")
                                   .attr("norm")(arr.attr("ravel")(), py::arg("ord") = order)
                                   .attr("item")());
    }
    return py::cast<float64>(
      np_attr("linalg").attr("norm")(arr, py::arg("ord") = order, py::arg("axis") = *axis));
}

BlockPtr
NumpyBlockBackend::outer(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(np_attr("tensordot")(obj(a), obj(b), py::make_tuple(py::tuple(), py::tuple())));
}

BlockPtr
NumpyBlockBackend::permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation)
{
    return wrap(np_attr("transpose")(obj(a), py::cast(permutation)));
}

BlockPtr
NumpyBlockBackend::random_normal(const std::vector<int64>& dims,
                                 Dtype dtype,
                                 float64 sigma,
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
        res = res + py::cast(complex128(0, 1)) *
                      np.attr("random").attr("normal")(
                        py::arg("loc") = 0, py::arg("scale") = sigma, py::arg("size") = dims);
    return wrap(np.attr("asarray")(res, py::arg("dtype") = dt));
}

BlockPtr
NumpyBlockBackend::random_uniform(const std::vector<int64>& dims,
                                  Dtype dtype,
                                  std::optional<std::string> device)
{
    (void)as_device(device);
    py::module_ np = numpy_module();
    py::object dt = dtype::to_numpy_dtype(dtype);
    py::object res = np.attr("random").attr("uniform")(
      py::arg("low") = -1, py::arg("high") = 1, py::arg("size") = dims);
    if (!dtype::is_real(dtype))
        res = res + py::cast(complex128(0, 1)) *
                      np.attr("random").attr("uniform")(
                        py::arg("low") = -1, py::arg("high") = 1, py::arg("size") = dims);
    return wrap(np.attr("asarray")(res, py::arg("dtype") = dt));
}

BlockPtr
NumpyBlockBackend::real(const BlockCPtr& a)
{
    return wrap(np_attr("real")(obj(a)));
}

BlockPtr
NumpyBlockBackend::real_if_close(const BlockCPtr& a, float64 tol)
{
    return wrap(np_attr("real_if_close")(obj(a), py::arg("tol") = tol));
}

BlockPtr
NumpyBlockBackend::scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis)
{
    py::object a = obj(block);
    py::object f = obj(factors);
    py::module_ np = numpy_module();
    int64 ndim = static_cast<int64>(get_shape(block).size());
    py::list idx;
    for (int64 i = 0; i < ndim; ++i)
        idx.append(i == axis ? py::object(py::slice(py::none(), py::none(), py::none()))
                             : py::object(py::none()));
    return wrap(a * f.attr("__getitem__")(py::tuple(idx)));
}

BlockPtr
NumpyBlockBackend::tile(const BlockCPtr& a, int64 repeats)
{
    return wrap(np_attr("tile")(obj(a), repeats));
}

std::vector<std::string>
NumpyBlockBackend::_block_repr_lines(const BlockCPtr& a,
                                     const std::string& indent,
                                     int64 max_width,
                                     int64 max_lines)
{
    py::module_ np = numpy_module();
    py::object arr = obj(a);
    np.attr("printoptions")(py::arg("linewidth") = max_width - static_cast<int64>(indent.size()));
    py::str s = py::str(arr);
    py::list lines = s.attr("split")("\n");
    std::vector<std::string> out;
    int64 n = py::len(lines);
    int64 first = (max_lines - 1) / 2;
    int64 last = max_lines - 1 - first;
    for (int64 i = 0; i < std::min(first, n); ++i)
        out.push_back(indent + py::cast<std::string>(lines[i]));
    if (n > max_lines) {
        out.push_back(indent + "...");
        for (int64 i = std::max(n - last, first); i < n; ++i)
            out.push_back(indent + py::cast<std::string>(lines[i]));
    }
    return out;
}

BlockPtr
NumpyBlockBackend::reshape(const BlockCPtr& a, const std::vector<int64>& shape)
{
    return wrap(np_attr("reshape")(obj(a), py::cast(shape)));
}

std::vector<int64>
NumpyBlockBackend::get_shape(const BlockCPtr& a)
{
    return ptr(a)->shape();
}

BlockPtr
NumpyBlockBackend::sqrt(const BlockCPtr& a)
{
    return wrap(np_attr("sqrt")(obj(a)));
}

BlockPtr
NumpyBlockBackend::squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    return wrap(np_attr("squeeze")(obj(a), py::cast(idcs)));
}

BlockPtr
NumpyBlockBackend::stable_log(const BlockCPtr& block, float64 cutoff)
{
    py::object arr = obj(block);
    py::module_ np = numpy_module();
    return wrap(
      np.attr("where")(np.attr("greater")(arr, py::float_(cutoff)), np_attr("log")(arr), 0.0));
}

BlockPtr
NumpyBlockBackend::sum(const BlockCPtr& a, int64 ax)
{
    return wrap(np_attr("sum")(obj(a), py::arg("axis") = ax));
}

complex128
NumpyBlockBackend::sum_all(const BlockCPtr& a)
{
    return py::cast<complex128>(np_attr("sum")(obj(a)).attr("item")());
}

BlockPtr
NumpyBlockBackend::multiply_blocks(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(obj(a) * obj(b));
}

BlockPtr
NumpyBlockBackend::tdot(const BlockCPtr& a,
                        const BlockCPtr& b,
                        const std::vector<int64>& idcs_a,
                        const std::vector<int64>& idcs_b)
{
    return wrap(
      np_attr("tensordot")(obj(a), obj(b), py::make_tuple(py::cast(idcs_a), py::cast(idcs_b))));
}

BlockPtr
NumpyBlockBackend::to_dtype(const BlockCPtr& a, Dtype dtype)
{
    return wrap(np_attr("asarray")(obj(a), dtype::to_numpy_dtype(dtype)));
}

py::object
NumpyBlockBackend::to_numpy(const BlockCPtr& a, std::optional<py::object> numpy_dtype)
{
    py::object arr = obj(a);
    if (numpy_dtype)
        return np_attr("asarray")(arr, *numpy_dtype);
    return np_attr("asarray")(arr);
}

std::complex<float64>
NumpyBlockBackend::trace_full(const BlockCPtr& a)
{
    py::object arr = obj(a);
    std::vector<int64> sh = get_shape(a);
    size_t num_trace = sh.size() / 2;
    int64 trace_dim = 1;
    for (size_t i = 0; i < num_trace; ++i)
        trace_dim *= sh[i];
    std::vector<int64> perm(sh.size());
    for (size_t i = 0; i < num_trace; ++i)
        perm[i] = static_cast<int>(i);
    for (size_t i = 0; i < num_trace; ++i)
        perm[num_trace + i] = static_cast<int>(2 * num_trace - 1 - i);
    arr = np_attr("transpose")(arr, py::cast(perm));
    arr = np_attr("reshape")(arr, py::make_tuple(trace_dim, trace_dim));
    return py::cast<complex128>(
      np_attr("trace")(arr, py::arg("axis1") = 0, py::arg("axis2") = 1).attr("item")());
}

BlockPtr
NumpyBlockBackend::trace_partial(const BlockCPtr& a,
                                 const std::vector<int64>& idcs1,
                                 const std::vector<int64>& idcs2,
                                 const std::vector<int64>& remaining_idcs)
{
    py::object arr = obj(a);
    std::vector<int64> perm = remaining_idcs;
    perm.insert(perm.end(), idcs1.begin(), idcs1.end());
    perm.insert(perm.end(), idcs2.begin(), idcs2.end());
    arr = np_attr("transpose")(arr, py::cast(perm));
    std::vector<int64> sh = get_shape(a);
    int64 trace_dim = 1;
    for (int64 i : idcs1)
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
NumpyBlockBackend::eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(np_attr("eye")(dim, py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

py::object
NumpyBlockBackend::get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    return obj(a).attr("__getitem__")(py::cast(idcs)).attr("item")();
}

BlockPtr
NumpyBlockBackend::matrix_dot(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(np_attr("dot")(obj(a), obj(b)));
}

BlockPtr
NumpyBlockBackend::matrix_exp(const BlockCPtr& matrix)
{
    return wrap(py::module_::import("scipy.linalg").attr("expm")(obj(matrix)));
}

BlockPtr
NumpyBlockBackend::matrix_log(const BlockCPtr& matrix)
{
    return wrap(py::module_::import("scipy.linalg").attr("logm")(obj(matrix)));
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_qr(const BlockCPtr& a, bool full)
{
    py::tuple qr = py::module_::import("scipy.linalg")
                     .attr("qr")(obj(a), py::arg("mode") = (full ? "full" : "economic"));
    return { wrap(qr[0]), wrap(qr[1]) };
}

std::tuple<BlockPtr, BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_svd(const BlockCPtr& a, std::optional<std::string> algorithm)
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
        } catch (const py::error_already_set&) {
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

const std::vector<std::string>&
NumpyBlockBackend::possible_svd_algorithms() const
{
    static const std::vector<std::string> algorithms = {
        "gesdd", "gesvd", "robust", "robust_silent"
    };
    return algorithms;
}

BlockPtr
NumpyBlockBackend::ones_block(const std::vector<int64>& shape,
                              Dtype dtype,
                              std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(np_attr("ones")(py::cast(shape), py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

BlockPtr
NumpyBlockBackend::zeros(const std::vector<int64>& shape,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    (void)as_device(device);
    return wrap(
      np_attr("zeros")(py::cast(shape), py::arg("dtype") = dtype::to_numpy_dtype(dtype)));
}

std::shared_ptr<NumpyBlockBackend>
NumpyBlockBackend::load_hdf5(py::object hdf5_loader, py::object h5gr, const std::string& subpath)
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
