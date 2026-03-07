#include <cyten/block_backend/numpy.h>

#include <map>
#include <mutex>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace cyten {

using std::complex_literals::operator""i;

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
    std::vector<int64> s;
    s.reserve(static_cast<size_t>(arr_.ndim()));
    for (py::ssize_t i = 0; i < arr_.ndim(); ++i)
        s.push_back(static_cast<int64>(arr_.shape(i)));
    return s;
}

Dtype
NumpyBlockBackend::Block::dtype() const
{
    return dtype::from_numpy_dtype(arr_.dtype());
}

const std::string&
NumpyBlockBackend::Block::device() const
{
    static std::string device = "cpu";
    return device;
}

py::array
NumpyBlockBackend::Block::to_numpy(Dtype dtype) const
{
    return py::array(arr_.attr("astype")(dtype::to_numpy_dtype(dtype)));
}

BlockBackend*
NumpyBlockBackend::Block::get_backend() const
{
    return NumpyBlockBackend::from_factory(device());
}

BlockCPtr
NumpyBlockBackend::Block::get_item(py::object key) const
{
    if (key.is_none())
        return shared_from_this();
    py::object key_copy = _item_key_cast_Blocks_to_numpy(key);
    py::array result = arr_.attr("__getitem__")(key_copy);
    return std::make_shared<const NumpyBlockBackend::Block>(std::move(result));
}

py::object
NumpyBlockBackend::Block::_item_key_cast_Blocks_to_numpy(py::object key) const
{
    if (py::isinstance<NumpyBlockBackend::Block>(key)) {
        NumpyBlockBackend::Block* block = key.cast<NumpyBlockBackend::Block*>();
        return block->to_numpy();
    } else if (py::isinstance<py::tuple>(key)) {
        // cast any Block objects in the key to numpy arrays and make a new tuple
        py::tuple key_tuple = py::reinterpret_borrow<py::tuple>(key);
        py::tuple new_tuple = py::tuple(key_tuple.size());
        for (py::ssize_t i = 0; i < key_tuple.size(); ++i) {
            if (py::isinstance<NumpyBlockBackend::Block>(key_tuple[i])) {
                NumpyBlockBackend::Block* block = key_tuple[i].cast<NumpyBlockBackend::Block*>();
                new_tuple[i] = block->to_numpy();
            } else {
                new_tuple[i] = key_tuple[i];
            }
        }
        return std::move(new_tuple);
    }
    return key;
}

BlockPtr
NumpyBlockBackend::Block::get_item(py::object key)
{
    if (key.is_none())
        return shared_from_this();
    py::object key_copy = _item_key_cast_Blocks_to_numpy(key);
    py::array result = arr_.attr("__getitem__")(key);
    return std::make_shared<NumpyBlockBackend::Block>(std::move(result));
}

void
NumpyBlockBackend::Block::set_item(py::object key, py::object value)
{
    py::object key_copy = _item_key_cast_Blocks_to_numpy(key);
    if (py::isinstance<NumpyBlockBackend::Block>(value)) {
        NumpyBlockBackend::Block* block = value.cast<NumpyBlockBackend::Block*>();
        value = block->to_numpy();
    }
    arr_.attr("__setitem__")(key, value);
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

bool
NumpyBlockBackend::is_correct_block_type(const BlockCPtr& block) const
{
    return dynamic_cast<NumpyBlockBackend::Block const*>(block.get()) != nullptr;
}


// -----------------------------------------------------------------------------
// NumpyBlockBackend
// -----------------------------------------------------------------------------

NumpyBlockBackend*
NumpyBlockBackend::from_factory(const std::string& device)
{
    return from_factory_shared(device).get();
}

std::shared_ptr<NumpyBlockBackend>
NumpyBlockBackend::from_factory_shared(const std::string& device)
{
    if (device != "cpu")
        throw std::invalid_argument(
          "NumpyBlockBackend::from_factory only supports device \"cpu\", got: " + device);
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<NumpyBlockBackend>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(device);
    if (it == cache.end()) {
        it =
          cache.emplace(device, std::shared_ptr<NumpyBlockBackend>(new NumpyBlockBackend())).first;
    }
    return it->second;
}

NumpyBlockBackend::NumpyBlockBackend()
  : BlockBackend("cpu")
{
}

std::string
NumpyBlockBackend::get_backend_name() const
{
    return "NumpyBlockBackend";
}


// CHECKME: the following was generated by ../pybind11_codegen/pybind11_codegen.py
// gen_cpp_definition --py-name NumpyBlockBackend --header-file include/cyten/block_backend/numpy.h
// --src-file src/block_backend/numpy_FROM_SCRIPT.cpp

BlockPtr
NumpyBlockBackend::abs(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.abs(a)
     */
    return np.abs(a);
}

BlockPtr
NumpyBlockBackend::as_block(py::object a,
                            std::optional<Dtype> dtype,
                            std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * block = np.asarray(a, dtype=self.backend_dtype_map[dtype])
     * if np.issubdtype(block.dtype, np.integer):
     *             block = block.astype(np.float64, copy=False)
     * return block
     */
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    auto /* CHECKME: type? */ block =
      np.asarray(a, backend_dtype_map[dtype]) /* CHECKME: call with keywords: ..., dtype */;
    if (np.issubdtype(block.dtype, np.integer)) {
        block = block.astype(np.float64, false) /* CHECKME: call with keywords: ..., copy */;
    }
    return block;
}

std::string
NumpyBlockBackend::as_device(std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * if device is None:
     *             return self.default_device
     * if device != self.default_device:
     *             msg = f'{self.__class__.__name__} does not support device {device}.'
     *             raise ValueError(msg)
     * return device
     */
    if (device == py::none()) {
        return default_device;
    }
    if (device != default_device) {
        auto /* CHECKME: type? */ msg =
          std::format("{} does not support device {}.", __class__.__name__, device);
        throw std::invalid_argument(msg);
    }
    return device;
}

BlockPtr
NumpyBlockBackend::add_axis(const BlockCPtr& a, int64 pos)
{
    /* CHECKME: converted from following python code:
     * return np.expand_dims(a, pos)
     */
    return np.expand_dims(a, pos);
}

std::vector<int64>
NumpyBlockBackend::abs_argmax(const BlockCPtr& block)
{
    /* CHECKME: converted from following python code:
     * return np.unravel_index(np.argmax(np.abs(block)), block.shape)
     */
    return np.unravel_index(np.argmax(np.abs(block)), block.shape);
}

bool
NumpyBlockBackend::block_all(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.all(a)
     */
    return np.all(a);
}

bool
NumpyBlockBackend::allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol)
{
    /* CHECKME: converted from following python code:
     * return np.allclose(a, b, rtol=rtol, atol=atol)
     */
    return np.allclose(a, b, rtol, atol) /* CHECKME: call with keywords: ..., rtol, atol */;
}

BlockPtr
NumpyBlockBackend::angle(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.angle(a)
     */
    return np.angle(a);
}

bool
NumpyBlockBackend::block_any(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.any(a)
     */
    return np.any(a);
}

BlockPtr
NumpyBlockBackend::apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax)
{
    /* CHECKME: converted from following python code:
     * return np.compress(mask, block, ax)
     */
    return np.compress(mask, block, ax);
}

BlockPtr
NumpyBlockBackend::_argsort(const BlockCPtr& block, int64 axis)
{
    /* CHECKME: converted from following python code:
     * return np.argsort(block, axis=axis)
     */
    return np.argsort(block, axis) /* CHECKME: call with keywords: ..., axis */;
}

BlockPtr
NumpyBlockBackend::conj(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.conj(a)
     */
    return np.conj(a);
}

BlockPtr
NumpyBlockBackend::copy_block(const BlockCPtr& a, std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.copy(a)
     */
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    return np.copy(a);
}

/// The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.
BlockPtr
NumpyBlockBackend::cutoff_inverse(const BlockCPtr& a, float64 cutoff)
{
    /* CHECKME: converted from following python code:
     * return 1 / np.where(np.abs(a) < cutoff, np.inf, a)
     */
    return 1 / np.where(np.abs(a) < cutoff, np.inf, a);
}

Dtype
NumpyBlockBackend::get_dtype(BlockPtr a)
{
    /* CHECKME: converted from following python code:
     * return self.cyten_dtype_map[a.dtype]
     */
    return cyten_dtype_map[a.dtype];
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::eigh(const BlockCPtr& block, std::optional<std::string> sort)
{
    /* CHECKME: converted from following python code:
     * w, v = np.linalg.eigh(block)
     * if sort is not None:
     *             perm = self.argsort(w, sort)
     *             w = np.take(w, perm)
     *             v = np.take(v, perm, axis=1)
     * return w, v
     */
    std::make_tuple(w, v) = np.linalg.eigh(block);
    if (sort != py::none()) {
        auto /* CHECKME: type? */ perm = argsort(w, sort);
        auto /* CHECKME: type? */ w = np.take(w, perm);
        auto /* CHECKME: type? */ v =
          np.take(v, perm, 1) /* CHECKME: call with keywords: ..., axis */;
    }
    return std::make_tuple(w, v);
}

BlockPtr
NumpyBlockBackend::eigvalsh(const BlockCPtr& block, std::optional<std::string> sort)
{
    /* CHECKME: converted from following python code:
     * w = np.linalg.eigvalsh(block)
     * if sort is not None:
     *             perm = self.argsort(w, sort)
     *             w = np.take(w, perm)
     * return w
     */
    auto /* CHECKME: type? */ w = np.linalg.eigvalsh(block);
    if (sort != py::none()) {
        auto /* CHECKME: type? */ perm = argsort(w, sort);
        w = np.take(w, perm);
    }
    return w;
}

BlockPtr
NumpyBlockBackend::enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis)
{
    /* CHECKME: converted from following python code:
     * # OPTIMIZE is there a numpy builtin function that does this? or at least part of this?
     * shape = list(block.shape)
     * shape[axis] = len(mask)
     * res = np.zeros(shape, dtype=block.dtype)
     * idcs = [slice(None, None, None)] * len(shape)
     * idcs[axis] = mask
     * res[tuple(idcs)] = block.copy()
     * # OPTIMIZE is the copy needed
     * return res
     */
    // OPTIMIZE is there a numpy builtin function that does this? or at least part of this?
    auto /* CHECKME: type? */ shape = list(block.shape);
    shape[axis] = len(mask);
    auto /* CHECKME: type? */ res =
      np.zeros(shape, block.dtype) /* CHECKME: call with keywords: ..., dtype */;
    auto /* CHECKME: type? */ idcs = { slice(py::none(), py::none(), py::none()) } * len(shape);
    idcs[axis] = mask;
    res[tuple(idcs)] = block.copy();
    /* OPTIMIZE is the copy needed */
    return res;
}

BlockPtr
NumpyBlockBackend::exp(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.exp(a)
     */
    return np.exp(a);
}

BlockPtr
NumpyBlockBackend::block_from_diagonal(const BlockCPtr& diag)
{
    /* CHECKME: converted from following python code:
     * return np.diag(diag)
     */
    return np.diag(diag);
}

BlockPtr
NumpyBlockBackend::block_from_mask(const BlockCPtr& mask, Dtype dtype)
{
    /* CHECKME: converted from following python code:
     * (M,) = mask.shape
     * N = np.sum(mask)
     * res = np.zeros((N, M), dtype=self.backend_dtype_map[dtype])
     * res[np.arange(N), mask] = 1
     * return res
     */
    std::make_tuple(M) = mask.shape;
    auto /* CHECKME: type? */ N = np.sum(mask);
    auto /* CHECKME: type? */ res =
      np.zeros(std::make_tuple(N, M),
               backend_dtype_map[dtype]) /* CHECKME: call with keywords: ..., dtype */;
    /* Multidimensional slice using std::gslice
     * NOTE: Requires res_strides to be defined as std::valarray<size_t>
     *       containing the memory strides for each dimension.
     * For a row-major 2D array with shape (n0, n1, ..., n1):
     *   res_strides = {n1, 1};
     * Alternative: Consider using xtensor, Eigen, or C++23 mdspan
     */
    res[std::gslice(np.arange(N) * res_strides[0] + mask * res_strides[1],
                    std::valarray<std::size_t>{ 1, 1 },
                    std::valarray<std::size_t>{ res_strides[0], res_strides[1] })] = 1;
    return res;
}

BlockPtr
NumpyBlockBackend::block_from_numpy(const py::array& a,
                                    std::optional<Dtype> dtype,
                                    std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * if dtype is None:
     *             return a
     * return np.asarray(a, self.backend_dtype_map[dtype])
     */
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    if (dtype == py::none()) {
        return a;
    }
    return np.asarray(a, backend_dtype_map[dtype]);
}

std::string
NumpyBlockBackend::get_device(BlockPtr a)
{
    /* CHECKME: converted from following python code:
     * return self.default_device
     */
    return default_device;
}

BlockPtr
NumpyBlockBackend::get_diagonal(const BlockCPtr& a, std::optional<float64> tol)
{
    /* CHECKME: converted from following python code:
     * res = np.diagonal(a)
     * if tol is not None:
     *             if not np.allclose(a, np.diag(res), atol=tol):
     *                 raise ValueError('Not a diagonal block.')
     * return res
     */
    auto /* CHECKME: type? */ res = np.diagonal(a);
    if (tol != py::none()) {
        if (!(np.allclose(a, np.diag(res), tol) /* CHECKME: call with keywords: ..., atol */)) {
            throw std::invalid_argument("Not a diagonal block.");
        }
    }
    return res;
}

BlockPtr
NumpyBlockBackend::imag(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.imag(a)
     */
    return np.imag(a);
}

float64
NumpyBlockBackend::inner(BlockPtr a, BlockPtr b, bool do_dagger)
{
    /* CHECKME: converted from following python code:
     * # OPTIMIZE use np.sum(a * b) instead?
     * if do_dagger:
     *             return np.tensordot(np.conj(a), b, a.ndim).item()
     * return np.tensordot(a, b, [list(range(a.ndim)), list(reversed(range(a.ndim)))]).item()
     */
    // OPTIMIZE use np.sum(a * b) instead?
    if (do_dagger) {
        return np.tensordot(np.conj(a), b, a.ndim).item();
    }
    return np.tensordot(a, b, { list(range(a.ndim)), list(reversed(range(a.ndim))) }).item();
}

py::object
NumpyBlockBackend::item(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return a.item()
     */
    return a.item();
}

BlockPtr
NumpyBlockBackend::kron(const BlockCPtr& a, const BlockCPtr& b)
{
    /* CHECKME: converted from following python code:
     * return np.kron(a, b)
     */
    return np.kron(a, b);
}

BlockPtr
NumpyBlockBackend::log(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.log(a)
     */
    return np.log(a);
}

float64
NumpyBlockBackend::max(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.max(a).item()
     */
    return np.max(a).item();
}

float64
NumpyBlockBackend::max_abs(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.max(np.abs(a)).item()
     */
    return np.max(np.abs(a)).item();
}

float64
NumpyBlockBackend::min(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.min(a).item()
     */
    return np.min(a).item();
}

float64
NumpyBlockBackend::norm(const BlockCPtr& a, float64 order, std::optional<int64> axis)
{
    /* CHECKME: converted from following python code:
     * if axis is None:
     *             return np.linalg.norm(a.ravel(), ord=order).item()
     * return np.linalg.norm(a, ord=order, axis=axis)
     */
    if (axis == py::none()) {
        return np.linalg.norm(a.ravel(), order) /* CHECKME: call with keywords: ..., ord */.item();
    }
    return np.linalg.norm(a, order, axis) /* CHECKME: call with keywords: ..., ord, axis */;
}

BlockPtr
NumpyBlockBackend::outer(const BlockCPtr& a, const BlockCPtr& b)
{
    /* CHECKME: converted from following python code:
     * return np.tensordot(a, b, ((), ()))
     */
    return np.tensordot(a, b, std::make_tuple(std::tuple<>(), std::tuple<>()));
}

BlockPtr
NumpyBlockBackend::permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation)
{
    /* CHECKME: converted from following python code:
     * return np.transpose(a, permutation)
     */
    return np.transpose(a, permutation);
}

BlockPtr
NumpyBlockBackend::random_normal(const std::vector<int64>& dims,
                                 Dtype dtype,
                                 float64 sigma,
                                 std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * # if sigma is standard deviation for complex numbers, need to divide by sqrt(2)
     * # to get standard deviation in real and imag parts
     * if not dtype.is_real:
     *             sigma /= np.sqrt(2)
     * _ = self.as_device(device)
     * # for input check only
     * res = np.random.normal(loc=0, scale=sigma, size=dims)
     * if not dtype.is_real:
     *             res = res + 1.0j * np.random.normal(loc=0, scale=sigma, size=dims)
     * return res
     */
    // if sigma is standard deviation for complex numbers, need to divide by sqrt(2)
    // to get standard deviation in real and imag parts
    if (!dtype.is_real) {
        sigma /= np.sqrt(2);
    }
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    auto /* CHECKME: type? */ res =
      np.random.normal(0, sigma, dims) /* CHECKME: call with keywords: ..., loc, scale, size */;
    if (!dtype.is_real) {
        res = res + 1.0i  *np.random.normal(
                      0, sigma, dims) /* CHECKME: call with keywords: ..., loc, scale, size */;
    }
    return res;
}

BlockPtr
NumpyBlockBackend::random_uniform(const std::vector<int64>& dims,
                                  Dtype dtype,
                                  std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * res = np.random.uniform(-1, 1, size=dims)
     * if not dtype.is_real:
     *             res = res + 1.0j * np.random.uniform(-1, 1, size=dims)
     * return res
     */
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    auto /* CHECKME: type? */ res =
      np.random.uniform(-1, 1, dims) /* CHECKME: call with keywords: ..., size */;
    if (!dtype.is_real) {
        res = res + 1.0i *np.random.uniform(
                      -1, 1, dims) /* CHECKME: call with keywords: ..., size */;
    }
    return res;
}

BlockPtr
NumpyBlockBackend::real(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.real(a)
     */
    return np.real(a);
}

BlockPtr
NumpyBlockBackend::real_if_close(const BlockCPtr& a, float64 tol)
{
    /* CHECKME: converted from following python code:
     * return np.real_if_close(a, tol=tol)
     */
    return np.real_if_close(a, tol) /* CHECKME: call with keywords: ..., tol */;
}

BlockPtr
NumpyBlockBackend::tile(const BlockCPtr& a, int64 repeats)
{
    /* CHECKME: converted from following python code:
     * return np.tile(a, repeats)
     */
    return np.tile(a, repeats);
}

std::vector<std::string>
NumpyBlockBackend::_block_repr_lines(const BlockCPtr& a,
                                     const std::string& indent,
                                     int64 max_width,
                                     int64 max_lines)
{
    /* CHECKME: converted from following python code:
     * with np.printoptions(linewidth=max_width - len(indent)):
     *             lines = [f'{indent}{line}' for line in str(a).split('\n')]
     * if len(lines) > max_lines:
     *             first = (max_lines - 1) // 2
     *             last = max_lines - 1 - first
     *             lines = lines[:first] + [f'{indent}...'] + lines[-last:]
     * return lines
     */
    /* FIXME: With: with np.printoptions(linewidth=max_width - len(indent)):
            lines = [f'{indent}{line}' for line in str(a).split('\n')] */
    if (len(lines) > max_lines) {
        auto /* CHECKME: type? */ first = max_lines - 1 / 2;
        auto /* CHECKME: type? */ last = max_lines - 1 - first;
        auto /* CHECKME: type? */ lines = lines[std::slice(0, first, 1)] +
                                          { std::format("{}...", indent) } +
                                          lines[std::slice(-last, (lines.size() - -last), 1)];
    }
    return lines;
}

BlockPtr
NumpyBlockBackend::reshape(const BlockCPtr& a, const std::vector<int64>& shape)
{
    /* CHECKME: converted from following python code:
     * return np.reshape(a, shape)
     */
    return np.reshape(a, shape);
}

tuple_int_
NumpyBlockBackend::get_shape(BlockPtr a)
{
    /* CHECKME: converted from following python code:
     * return np.shape(a)
     */
    return np.shape(a);
}

BlockPtr
NumpyBlockBackend::sqrt(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.sqrt(a)
     */
    return np.sqrt(a);
}

BlockPtr
NumpyBlockBackend::squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    /* CHECKME: converted from following python code:
     * return np.squeeze(a, tuple(idcs))
     */
    return np.squeeze(a, tuple(idcs));
}

BlockPtr
NumpyBlockBackend::stable_log(const BlockCPtr& block, float64 cutoff)
{
    /* CHECKME: converted from following python code:
     * return np.where(block > cutoff, np.log(block), 0.0)
     */
    return np.where(block > cutoff, np.log(block), 0.0);
}

BlockPtr
NumpyBlockBackend::sum(const BlockCPtr& a, int64 ax)
{
    /* CHECKME: converted from following python code:
     * return np.sum(a, axis=ax)
     */
    return np.sum(a, ax) /* CHECKME: call with keywords: ..., axis */;
}

complex128
NumpyBlockBackend::sum_all(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * return np.sum(a).item()
     */
    return np.sum(a).item();
}

BlockPtr
NumpyBlockBackend::tdot(const BlockCPtr& a,
                        const BlockCPtr& b,
                        const std::vector<int64>& idcs_a,
                        const std::vector<int64>& idcs_b)
{
    /* CHECKME: converted from following python code:
     * return np.tensordot(a, b, (idcs_a, idcs_b))
     */
    return np.tensordot(a, b, std::make_tuple(idcs_a, idcs_b));
}

BlockPtr
NumpyBlockBackend::to_dtype(const BlockCPtr& a, Dtype dtype)
{
    /* CHECKME: converted from following python code:
     * return np.asarray(a, dtype=self.backend_dtype_map[dtype])
     */
    return np.asarray(a, backend_dtype_map[dtype]) /* CHECKME: call with keywords: ..., dtype */;
}

complex128
NumpyBlockBackend::trace_full(const BlockCPtr& a)
{
    /* CHECKME: converted from following python code:
     * num_trace = a.ndim // 2
     * trace_dim = np.prod(a.shape[:num_trace])
     * perm = [*range(num_trace), *reversed(range(num_trace, 2 * num_trace))]
     * a = np.reshape(np.transpose(a, perm), (trace_dim, trace_dim))
     * return np.trace(a, axis1=0, axis2=1).item()
     */
    auto /* CHECKME: type? */ num_trace = a.ndim / 2;
    auto /* CHECKME: type? */ trace_dim = np.prod(a.shape[std::slice(0, num_trace, 1)]);
    auto /* CHECKME: type? */
      perm = { /* FIXME: Starred: *range(num_trace) */,
               /* FIXME: Starred: *reversed(range(num_trace, 2 * num_trace)) */ };
    a = np.reshape(np.transpose(a, perm), std::make_tuple(trace_dim, trace_dim));
    return np.trace(a, 0, 1) /* CHECKME: call with keywords: ..., axis1, axis2 */.item();
}

BlockPtr
NumpyBlockBackend::trace_partial(const BlockCPtr& a,
                                 const std::vector<int64>& idcs1,
                                 const std::vector<int64>& idcs2,
                                 const std::vector<int64>& remaining_idcs)
{
    /* CHECKME: converted from following python code:
     * a = np.transpose(a, remaining + idcs1 + idcs2)
     * trace_dim = np.prod(a.shape[len(remaining) : len(remaining) + len(idcs1)], dtype=int)
     * a = np.reshape(a, a.shape[: len(remaining)] + (trace_dim, trace_dim))
     * return np.trace(a, axis1=-2, axis2=-1)
     */
    a = np.transpose(a, remaining + idcs1 + idcs2);
    auto /* CHECKME: type? */ trace_dim = np.prod(
      a.shape[std::slice(len(remaining), (len(remaining) + len(idcs1) - len(remaining)), 1)],
      int) /* CHECKME: call with keywords: ..., dtype */;
    a = np.reshape(
      a, a.shape[std::slice(0, len(remaining), 1)] + std::make_tuple(trace_dim, trace_dim));
    return np.trace(a, -2, -1) /* CHECKME: call with keywords: ..., axis1, axis2 */;
}

BlockPtr
NumpyBlockBackend::eye_matrix(int64 dim, Dtype dtype, std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.eye(dim, dtype=self.backend_dtype_map[dtype])
     */
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    return np.eye(dim, backend_dtype_map[dtype]) /* CHECKME: call with keywords: ..., dtype */;
}

py::object
NumpyBlockBackend::get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    /* CHECKME: converted from following python code:
     * return a[tuple(idcs)].item()
     */
    return a[tuple(idcs)].item();
}

BlockPtr
NumpyBlockBackend::matrix_dot(const BlockCPtr& a, const BlockCPtr& b)
{
    /* CHECKME: converted from following python code:
     * return np.dot(a, b)
     */
    return np.dot(a, b);
}

BlockPtr
NumpyBlockBackend::matrix_exp(const BlockCPtr& matrix)
{
    /* CHECKME: converted from following python code:
     * return scipy.linalg.expm(matrix)
     */
    return scipy.linalg.expm(matrix);
}

BlockPtr
NumpyBlockBackend::matrix_log(const BlockCPtr& matrix)
{
    /* CHECKME: converted from following python code:
     * return scipy.linalg.logm(matrix)
     */
    return scipy.linalg.logm(matrix);
}

std::tuple<BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_qr(const BlockCPtr& a, bool full)
{
    /* CHECKME: converted from following python code:
     * return scipy.linalg.qr(a, mode='full' if full else 'economic')
     */
    return scipy.linalg.qr(
      a, full ? "full" : "economic") /* CHECKME: call with keywords: ..., mode */;
}

std::tuple<BlockPtr, BlockPtr, BlockPtr>
NumpyBlockBackend::matrix_svd(const BlockCPtr& a, std::optional<std::string> algorithm)
{
    /* CHECKME: converted from following python code:
     * if algorithm is None:
     *             algorithm = 'gesdd'
     * if algorithm == 'gesdd':
     *             return scipy.linalg.svd(a, full_matrices=False)
     *
     *         elif algorithm in ['robust', 'robust_silent']:
     *             silent = algorithm == 'robust_silent'
     *             try:
     *                 return scipy.linalg.svd(a, full_matrices=False)
     *             except np.linalg.LinAlgError:
     *                 if not silent:
     *                     raise NotImplementedError  # log warning
     *             return _svd_gesvd(a)
     *
     *         elif algorithm == 'gesvd':
     *             return _svd_gesvd(a)
     *
     *         else:
     *             raise ValueError(f'SVD algorithm not supported: {algorithm}')
     */
    if (algorithm == py::none()) {
        algorithm = "gesdd";
    }
    if (algorithm == "gesdd") {
        return scipy.linalg.svd(a, false) /* CHECKME: call with keywords: ..., full_matrices */;
    } else {
        if (algorithm /* FIXME: In */ { "robust", "robust_silent" }) {
            auto /* CHECKME: type? */ silent = algorithm == "robust_silent";
            /* FIXME: Try: try:
                return scipy.linalg.svd(a, full_matrices=False)
            except np.linalg.LinAlgError:
                if not silent:
                    raise NotImplementedError  # log warning */
            return _svd_gesvd(a);
        } else {
            if (algorithm == "gesvd") {
                return _svd_gesvd(a);
            } else {
                throw std::invalid_argument(
                  std::format("SVD algorithm not supported: {}", algorithm));
            }
        }
    }
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
    /* CHECKME: converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.ones(shape, dtype=self.backend_dtype_map[dtype])
     */
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    return np.ones(shape, backend_dtype_map[dtype]) /* CHECKME: call with keywords: ..., dtype */;
}

BlockPtr
NumpyBlockBackend::zeros(const std::vector<int64>& shape,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    /* CHECKME: converted from following python code:
     * _ = self.as_device(device)
     * # for input check only
     * return np.zeros(shape, dtype=self.backend_dtype_map[dtype])
     */
    auto /* CHECKME: type? */ _ = as_device(device);
    /* for input check only */
    return np.zeros(shape, backend_dtype_map[dtype]) /* CHECKME: call with keywords: ..., dtype */;
}

std::shared_ptr<NumpyBlockBackend>
NumpyBlockBackend::load_hdf5(py::object hdf5_loader, py::object h5gr, const std::string& subpath)
{
    auto obj = NumpyBlockBackend::from_factory_shared("cpu");
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

BlockPtr
NumpyBlockBackend::apply_leg_permutations(const BlockCPtr& block,
                                          const std::vector<py::array_t<int64>>& perms)
{
    // FIXME: implement
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
NumpyBlockBackend::mul(py::object a, const BlockCPtr& b)
{
    return wrap(a * obj(b));
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
NumpyBlockBackend::multiply_blocks(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(obj(a) * obj(b));
}


} // namespace cyten
