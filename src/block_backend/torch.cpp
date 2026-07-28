#include <cyten/block_backend/torch.h>
#include <cyten/tools.h>

#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <torch/torch.h>

namespace cyten {

using torch::indexing::None;
using torch::indexing::Slice;
using torch::indexing::TensorIndex;

namespace {

std::vector<int64_t>
to_int64_vec(const std::vector<int64>& v)
{
    return std::vector<int64_t>(v.begin(), v.end());
}

py::dtype
numpy_dtype_from_scalar_type(c10::ScalarType st)
{
    switch (st) {
        case torch::kBool:
            return py::dtype::of<bool>();
        case torch::kLong:
            return py::dtype::of<int64_t>();
        case torch::kFloat:
            return py::dtype::of<float>();
        case torch::kDouble:
            return py::dtype::of<double>();
        case torch::kComplexFloat:
            return py::dtype::of<std::complex<float>>();
        case torch::kComplexDouble:
            return py::dtype::of<std::complex<double>>();
        default:
            throw std::invalid_argument("unsupported torch dtype for numpy conversion");
    }
}

torch::Tensor
tensor_from_numpy_array(const py::array& a)
{
    // Force an owned C-contiguous copy so we never alias Python/numpy memory with ATen.
    py::module_ np = py::module_::import("numpy");
    py::array arr = py::reinterpret_borrow<py::array>(
      np.attr("array")(a, py::arg("copy") = true, py::arg("order") = "C"));
    if (!arr)
        throw std::invalid_argument("expected a numpy array");

    std::vector<int64_t> shape(static_cast<size_t>(arr.ndim()));
    for (py::ssize_t i = 0; i < arr.ndim(); ++i)
        shape[static_cast<size_t>(i)] = static_cast<int64_t>(arr.shape(i));

    Dtype dt = dtype::from_numpy_dtype(arr.dtype());
    auto opts = torch::TensorOptions().dtype(dtype::to_torch_dtype(dt)).device(torch::kCPU);
    torch::Tensor t = torch::empty(shape, opts);
    if (t.numel() > 0) {
        if (static_cast<size_t>(arr.nbytes()) != static_cast<size_t>(t.nbytes()))
            throw std::runtime_error("numpy/torch nbytes mismatch in tensor_from_numpy_array");
        std::memcpy(t.data_ptr(), arr.data(), static_cast<size_t>(t.nbytes()));
    }
    return t;
}

torch::Tensor
tensor_from_py_object(py::object a)
{
    if (py::isinstance<py::array>(a))
        return tensor_from_numpy_array(py::reinterpret_borrow<py::array>(a));
    py::module_ np = py::module_::import("numpy");
    return tensor_from_numpy_array(np.attr("asarray")(a).cast<py::array>());
}

} // namespace

// -----------------------------------------------------------------------------
// dtype ↔ torch
// -----------------------------------------------------------------------------

namespace dtype {

c10::ScalarType
to_torch_dtype(Dtype d)
{
    switch (d) {
        case Dtype::Bool:
            return torch::kBool;
        case Dtype::Int64:
            return torch::kLong;
        case Dtype::Float32:
            return torch::kFloat;
        case Dtype::Float64:
            return torch::kDouble;
        case Dtype::Complex64:
            return torch::kComplexFloat;
        case Dtype::Complex128:
            return torch::kComplexDouble;
        default:
            throw std::invalid_argument("unsupported Dtype for torch");
    }
}

Dtype
from_torch_dtype(c10::ScalarType torch_dtype)
{
    switch (torch_dtype) {
        case torch::kBool:
            return Dtype::Bool;
        case torch::kLong:
            return Dtype::Int64;
        case torch::kFloat:
            return Dtype::Float32;
        case torch::kDouble:
            return Dtype::Float64;
        case torch::kComplexFloat:
            return Dtype::Complex64;
        case torch::kComplexDouble:
            return Dtype::Complex128;
        default:
            throw std::invalid_argument("unsupported torch ScalarType for cyten Dtype");
    }
}

} // namespace dtype

// -----------------------------------------------------------------------------
// TorchBlock
// -----------------------------------------------------------------------------

TorchBlockBackend::Block::Block(torch::Tensor tensor)
  : tensor_(std::move(tensor))
  , device_()
{
    torch::Device d = tensor_.device();
    if (!d.has_index())
        d = torch::Device(d.type(), /*index=*/0);
    device_ = d.str();
    try {
        (void)dtype::from_torch_dtype(tensor_.scalar_type());
    } catch (std::invalid_argument const&) {
        throw std::invalid_argument("TorchBlockBackend::Block: unsupported torch dtype");
    }
}

std::vector<int64>
TorchBlockBackend::Block::shape() const
{
    std::vector<int64> s;
    s.reserve(static_cast<size_t>(tensor_.dim()));
    for (int64_t i = 0; i < tensor_.dim(); ++i)
        s.push_back(static_cast<int64>(tensor_.size(i)));
    return s;
}

Dtype
TorchBlockBackend::Block::dtype() const
{
    return dtype::from_torch_dtype(tensor_.scalar_type());
}

const std::string&
TorchBlockBackend::Block::device() const
{
    return device_;
}

py::array
TorchBlockBackend::Block::to_numpy() const
{
    torch::Tensor t = tensor_.detach().cpu().contiguous().resolve_conj().resolve_neg();
    std::vector<py::ssize_t> shape(static_cast<size_t>(t.dim()));
    for (int64_t i = 0; i < t.dim(); ++i)
        shape[static_cast<size_t>(i)] = static_cast<py::ssize_t>(t.size(i));
    py::array arr(numpy_dtype_from_scalar_type(t.scalar_type()), shape);
    if (t.numel() > 0)
        std::memcpy(arr.mutable_data(), t.data_ptr(), static_cast<size_t>(t.nbytes()));
    return arr;
}

py::array
TorchBlockBackend::Block::to_numpy(Dtype dt) const
{
    return TorchBlockBackend::Block(tensor_.to(dtype::to_torch_dtype(dt))).to_numpy();
}

BlockBackend*
TorchBlockBackend::Block::get_backend() const
{
    return TorchBlockBackend::from_factory(device());
}

namespace {

std::optional<TensorIndex>
try_py_key_to_tensor_index(py::handle key)
{
    if (py::isinstance<BlockBackend::Block>(key)) {
        return TensorIndex{ tensor_from_numpy_array(key.cast<BlockBackend::Block&>().to_numpy()) };
    }
    if (py::isinstance<py::array>(key)) {
        return TensorIndex{ tensor_from_numpy_array(py::reinterpret_borrow<py::array>(key)) };
    }
    if (py::isinstance<py::slice>(key)) {
        py::slice sl = py::reinterpret_borrow<py::slice>(key);
        py::object start = sl.attr("start");
        py::object stop = sl.attr("stop");
        py::object step = sl.attr("step");
        auto as_opt_int = [](py::object o) -> std::optional<int64_t> {
            if (o.is_none())
                return std::nullopt;
            return o.cast<int64_t>();
        };
        return TensorIndex{ Slice(as_opt_int(start), as_opt_int(stop), as_opt_int(step)) };
    }
    try {
        return TensorIndex{ key.cast<int64_t>() };
    } catch (py::cast_error const&) {
        return std::nullopt;
    }
}

std::optional<std::vector<TensorIndex>>
try_py_key_to_tensor_indices(py::object key)
{
    if (py::isinstance<py::tuple>(key)) {
        py::tuple t = py::reinterpret_borrow<py::tuple>(key);
        std::vector<TensorIndex> out;
        out.reserve(static_cast<size_t>(t.size()));
        for (auto item : t) {
            auto idx = try_py_key_to_tensor_index(item);
            if (!idx)
                return std::nullopt;
            out.push_back(*idx);
        }
        return out;
    }
    auto single = try_py_key_to_tensor_index(key);
    if (!single)
        return std::nullopt;
    return std::vector<TensorIndex>{ *single };
}

torch::Tensor
tensor_from_py_value(py::object value, torch::Device device)
{
    if (py::isinstance<TorchBlockBackend::Block>(value))
        return value.cast<TorchBlockBackend::Block&>().tensor().to(device);
    if (py::isinstance<BlockBackend::Block>(value))
        return tensor_from_numpy_array(value.cast<BlockBackend::Block&>().to_numpy()).to(device);
    if (py::isinstance<BlockBackend::Scalar>(value)) {
        auto block = value.cast<BlockBackend::Scalar&>()._block();
        if (auto const* tb = dynamic_cast<TorchBlockBackend::Block const*>(block.get()))
            return tb->tensor().to(device);
        return tensor_from_numpy_array(block->to_numpy()).to(device);
    }
    return tensor_from_py_object(value).to(device);
}

} // namespace

BlockCPtr
TorchBlockBackend::Block::get_item(py::object key) const
{
    if (auto idcs = try_py_key_to_tensor_indices(key))
        return std::make_shared<const TorchBlockBackend::Block>(tensor_.index(*idcs));
    // Escape hatch: index via numpy (arbitrary Python keys).
    py::array arr = to_numpy();
    if (py::isinstance<BlockBackend::Block>(key))
        key = key.cast<BlockBackend::Block&>().to_numpy();
    py::object result = arr.attr("__getitem__")(key);
    return std::make_shared<const TorchBlockBackend::Block>(
      tensor_from_py_object(result).to(tensor_.device()));
}

BlockPtr
TorchBlockBackend::Block::get_item(py::object key)
{
    if (auto idcs = try_py_key_to_tensor_indices(key))
        return wrap(tensor_.index(*idcs));
    py::array arr = to_numpy();
    if (py::isinstance<BlockBackend::Block>(key))
        key = key.cast<BlockBackend::Block&>().to_numpy();
    py::object result = arr.attr("__getitem__")(key);
    return wrap(tensor_from_py_object(result).to(tensor_.device()));
}

void
TorchBlockBackend::Block::set_item(py::object key, py::object value)
{
    torch::Tensor val = tensor_from_py_value(value, tensor_.device()).to(tensor_.options());
    if (auto idcs = try_py_key_to_tensor_indices(key)) {
        tensor_.index_put_(*idcs, val);
        return;
    }
    if (py::isinstance<BlockBackend::Block>(key))
        key = key.cast<BlockBackend::Block&>().to_numpy();
    py::array arr = to_numpy();
    py::object np_val = TorchBlockBackend::Block(val.detach().cpu()).to_numpy();
    arr.attr("__setitem__")(key, np_val);
    tensor_ = tensor_from_numpy_array(arr).to(tensor_.device());
    torch::Device d = tensor_.device();
    if (!d.has_index())
        d = torch::Device(d.type(), 0);
    device_ = d.str();
}

void
TorchBlockBackend::Block::set_item(const std::vector<int64>& key, const Scalar& value)
{
    std::vector<TensorIndex> idcs;
    idcs.reserve(key.size());
    for (int64 k : key)
        idcs.emplace_back(static_cast<int64_t>(k));
    torch::Tensor v = tens(value._block()).to(tensor_.options());
    tensor_.index_put_(idcs, v);
}

void
TorchBlockBackend::Block::set_item(int64 idx, const Scalar& value)
{
    if (tensor_.dim() != 1)
        throw std::invalid_argument("integer set_item requires 1D block");
    set_item(std::vector<int64>{ idx }, value);
}

complex128
TorchBlockBackend::Block::_item_as_complex128() const
{
    torch::Tensor t = tensor_.detach().cpu().reshape({});
    if (c10::isComplexType(t.scalar_type())) {
        auto z = t.item<c10::complex<double>>();
        return complex128{ z.real(), z.imag() };
    }
    if (t.scalar_type() == torch::kBool)
        return complex128{ t.item<bool>() ? 1.0 : 0.0, 0.0 };
    if (t.scalar_type() == torch::kLong)
        return complex128{ static_cast<float64>(t.item<int64_t>()), 0.0 };
    return complex128{ t.to(torch::kDouble).item<double>(), 0.0 };
}

int64
TorchBlockBackend::Block::_item_as_int64() const
{
    torch::Tensor t = tensor_.detach().cpu().reshape({});
    if (t.scalar_type() == torch::kLong)
        return static_cast<int64>(t.item<int64_t>());
    if (t.scalar_type() == torch::kBool)
        return t.item<bool>() ? 1 : 0;
    throw std::invalid_argument("block item is not an integer dtype");
}

namespace {

torch::Tensor
as_torch_tensor(const BlockBackend::Block& other, torch::Device device)
{
    if (auto const* tb = dynamic_cast<TorchBlockBackend::Block const*>(&other))
        return tb->tensor().to(device);
    return tensor_from_numpy_array(other.to_numpy()).to(device);
}

} // namespace

BlockPtr
TorchBlockBackend::Block::operator+(const BlockBackend::Block& other) const
{
    return wrap(tensor_ + as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator-(const BlockBackend::Block& other) const
{
    return wrap(tensor_ - as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator*(const BlockBackend::Block& other) const
{
    return wrap(tensor_ * as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator/(const BlockBackend::Block& other) const
{
    return wrap(tensor_ / as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator<(const BlockBackend::Block& other) const
{
    return wrap(tensor_ < as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator<=(const BlockBackend::Block& other) const
{
    return wrap(tensor_ <= as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator>(const BlockBackend::Block& other) const
{
    return wrap(tensor_ > as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator>=(const BlockBackend::Block& other) const
{
    return wrap(tensor_ >= as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator==(const BlockBackend::Block& other) const
{
    return wrap(tensor_ == as_torch_tensor(other, tensor_.device()));
}
BlockPtr
TorchBlockBackend::Block::operator!=(const BlockBackend::Block& other) const
{
    return wrap(tensor_ != as_torch_tensor(other, tensor_.device()));
}

BlockPtr
TorchBlockBackend::Block::pow(const BlockBackend::Scalar& exponent) const
{
    return wrap(torch::pow(tensor_, as_torch_tensor(*exponent._block(), tensor_.device())));
}
BlockPtr
TorchBlockBackend::Block::pow(const BlockBackend::Block& exponent) const
{
    return wrap(torch::pow(tensor_, as_torch_tensor(exponent, tensor_.device())));
}

void
TorchBlockBackend::Block::save_hdf5(py::object hdf5_saver,
                                    py::object /*h5gr*/,
                                    const std::string& subpath)
{
    hdf5_saver.attr("save")(to_numpy(), subpath + std::string("arr"));
}

std::shared_ptr<TorchBlockBackend::Block>
TorchBlockBackend::Block::from_hdf5(py::object hdf5_loader,
                                    py::object h5gr,
                                    const std::string& subpath)
{
    py::array arr = hdf5_loader.attr("load")(subpath + std::string("arr")).cast<py::array>();
    auto obj = std::make_shared<TorchBlockBackend::Block>(tensor_from_numpy_array(arr));
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

// -----------------------------------------------------------------------------
// helpers
// -----------------------------------------------------------------------------

TorchBlockBackend::Block const*
TorchBlockBackend::ptr(const BlockCPtr& b)
{
    auto* p = dynamic_cast<TorchBlockBackend::Block const*>(b.get());
    if (!p)
        throw std::invalid_argument("block is not a TorchBlock");
    return p;
}

const torch::Tensor&
TorchBlockBackend::tens(const BlockCPtr& b)
{
    return ptr(b)->tensor();
}

BlockPtr
TorchBlockBackend::wrap(torch::Tensor tensor)
{
    return std::make_shared<TorchBlockBackend::Block>(std::move(tensor));
}

bool
TorchBlockBackend::is_correct_block_type(const BlockCPtr& block) const
{
    return dynamic_cast<TorchBlockBackend::Block const*>(block.get()) != nullptr;
}

torch::Device
TorchBlockBackend::parse_device(const std::string& device) const
{
    torch::Device d(device);
    if (!d.has_index())
        d = torch::Device(d.type(), /*index=*/0);
    return d;
}

std::pair<torch::Tensor, torch::Tensor>
TorchBlockBackend::to_same_dtype(const torch::Tensor& a,
                                 const torch::Tensor& b,
                                 std::optional<c10::ScalarType> at_least) const
{
    c10::ScalarType st = torch::promote_types(a.scalar_type(), b.scalar_type());
    if (at_least)
        st = torch::promote_types(st, *at_least);
    torch::Tensor aa = a.scalar_type() == st ? a : a.to(st);
    torch::Tensor bb = b.scalar_type() == st ? b : b.to(st);
    return { aa, bb };
}

BlockBackend::Scalar
TorchBlockBackend::as_scalar(const torch::Tensor& value)
{
    torch::Tensor v = value;
    if (v.dim() != 0)
        v = v.reshape({});
    return Scalar(wrap(std::move(v)));
}

BlockBackend::Scalar
TorchBlockBackend::as_scalar(complex128 value, Dtype dt)
{
    auto opts =
      torch::TensorOptions().dtype(dtype::to_torch_dtype(dt)).device(parse_device(default_device));
    if (dtype::is_complex(dt)) {
        c10::complex<double> z{ value.real(), value.imag() };
        return as_scalar(torch::scalar_tensor(z, opts));
    }
    if (dt == Dtype::Bool)
        return as_scalar(torch::scalar_tensor(value.real() != 0.0, opts));
    if (dt == Dtype::Int64)
        return as_scalar(torch::scalar_tensor(static_cast<int64_t>(value.real()), opts));
    return as_scalar(torch::scalar_tensor(value.real(), opts));
}

BlockBackend::Scalar
TorchBlockBackend::as_scalar(py::object value, Dtype dt)
{
    if (py::isinstance<BlockBackend::Scalar>(value)) {
        auto scalar = py::cast<BlockBackend::Scalar>(value);
        if (scalar.dtype() == dt)
            return scalar;
        BlockPtr converted =
          to_dtype(std::const_pointer_cast<BlockBackend::Block>(scalar._block()), dt);
        return Scalar(std::move(converted));
    }
    torch::Tensor t = tensor_from_py_object(value).to(dtype::to_torch_dtype(dt));
    if (t.dim() != 0)
        t = t.reshape({});
    return as_scalar(t.to(parse_device(default_device)));
}

BlockBackend::Scalar
TorchBlockBackend::as_scalar(bool b)
{
    return as_scalar(torch::scalar_tensor(
      b, torch::TensorOptions().dtype(torch::kBool).device(parse_device(default_device))));
}
BlockBackend::Scalar
TorchBlockBackend::as_scalar(int64 x)
{
    return as_scalar(torch::scalar_tensor(
      static_cast<int64_t>(x),
      torch::TensorOptions().dtype(torch::kLong).device(parse_device(default_device))));
}
BlockBackend::Scalar
TorchBlockBackend::as_scalar(float32 x)
{
    return as_scalar(torch::scalar_tensor(
      x, torch::TensorOptions().dtype(torch::kFloat).device(parse_device(default_device))));
}
BlockBackend::Scalar
TorchBlockBackend::as_scalar(float64 x)
{
    return as_scalar(torch::scalar_tensor(
      x, torch::TensorOptions().dtype(torch::kDouble).device(parse_device(default_device))));
}
BlockBackend::Scalar
TorchBlockBackend::as_scalar(complex64 z)
{
    c10::complex<float> cz{ z.real(), z.imag() };
    return as_scalar(torch::scalar_tensor(
      cz,
      torch::TensorOptions().dtype(torch::kComplexFloat).device(parse_device(default_device))));
}
BlockBackend::Scalar
TorchBlockBackend::as_scalar(complex128 z)
{
    c10::complex<double> cz{ z.real(), z.imag() };
    return as_scalar(torch::scalar_tensor(
      cz,
      torch::TensorOptions().dtype(torch::kComplexDouble).device(parse_device(default_device))));
}

// -----------------------------------------------------------------------------
// factory
// -----------------------------------------------------------------------------

TorchBlockBackend*
TorchBlockBackend::from_factory(const std::string& device)
{
    return from_factory_shared(device).get();
}

std::shared_ptr<TorchBlockBackend>
TorchBlockBackend::from_factory_shared(const std::string& device)
{
    // Normalize via a temporary so cache keys are canonical ("cpu:0", ...).
    static std::mutex mutex;
    static std::map<std::string, std::shared_ptr<TorchBlockBackend>> cache;
    std::string key = device;
    {
        torch::Device d(device);
        if (!d.has_index())
            d = torch::Device(d.type(), 0);
        key = d.str();
    }
    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(key);
    if (it == cache.end()) {
        it =
          cache.emplace(key, std::shared_ptr<TorchBlockBackend>(new TorchBlockBackend(key))).first;
    }
    return it->second;
}

TorchBlockBackend::TorchBlockBackend(const std::string& default_device_in)
  : BlockBackend([&] {
      torch::Device d(default_device_in);
      if (!d.has_index())
          d = torch::Device(d.type(), 0);
      return d.str();
  }())
{
}

std::string
TorchBlockBackend::get_backend_name() const
{
    return "TorchBlockBackend";
}

std::shared_ptr<TorchBlockBackend>
TorchBlockBackend::from_hdf5(py::object hdf5_loader, py::object h5gr, const std::string& subpath)
{
    std::string device =
      hdf5_loader.attr("load")(subpath + std::string("default_device")).cast<std::string>();
    auto obj = TorchBlockBackend::from_factory_shared(device);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

// -----------------------------------------------------------------------------
// backend methods
// -----------------------------------------------------------------------------

BlockPtr
TorchBlockBackend::abs(const BlockCPtr& a)
{
    return wrap(torch::abs(tens(a)));
}

std::string
TorchBlockBackend::as_device(std::optional<std::string> device)
{
    if (!device)
        return default_device;
    return parse_device(*device).str();
}

BlockPtr
TorchBlockBackend::as_block(py::object a,
                            std::optional<Dtype> dtype_opt,
                            std::optional<std::string> device)
{
    std::string dev = as_device(device);
    if (py::isinstance<TorchBlockBackend::Block>(a)) {
        BlockPtr block = a.cast<BlockPtr>();
        if (dtype_opt)
            block = to_dtype(block, *dtype_opt);
        if (block->device() != dev)
            block = copy_block(block, dev);
        return block;
    }
    if (py::isinstance<BlockBackend::Block>(a)) {
        BlockPtr block = a.cast<BlockPtr>();
        return block_from_numpy(block->to_numpy(), dtype_opt, dev);
    }
    torch::Tensor t = tensor_from_py_object(a).to(parse_device(dev));
    if (dtype_opt) {
        t = t.to(dtype::to_torch_dtype(*dtype_opt));
    }
    // Match Python: force integer tensors to float (unless bool was requested).
    if ((!dtype_opt || *dtype_opt != Dtype::Bool) &&
        c10::isIntegralType(t.scalar_type(), /*includeBool=*/false)) {
        t = t.to(torch::kDouble);
    }
    return wrap(std::move(t));
}

std::vector<int64>
TorchBlockBackend::abs_argmax(const BlockCPtr& block)
{
    torch::Tensor flat_idx = torch::argmax(torch::abs(tens(block)));
    int64_t idx = flat_idx.item<int64_t>();
    auto sizes = tens(block).sizes();
    std::vector<int64> idcs;
    idcs.reserve(static_cast<size_t>(sizes.size()));
    for (auto it = sizes.rbegin(); it != sizes.rend(); ++it) {
        int64_t dim = *it;
        idcs.push_back(static_cast<int64>(idx % dim));
        idx /= dim;
    }
    std::reverse(idcs.begin(), idcs.end());
    return idcs;
}

BlockPtr
TorchBlockBackend::add_axis(const BlockCPtr& a, int64 pos)
{
    return wrap(tens(a).unsqueeze(static_cast<int64_t>(pos)));
}

bool
TorchBlockBackend::all(const BlockCPtr& a)
{
    return torch::all(tens(a)).item<bool>();
}

bool
TorchBlockBackend::allclose(const BlockCPtr& a, const BlockCPtr& b, float64 rtol, float64 atol)
{
    auto [aa, bb] = to_same_dtype(tens(a), tens(b));
    return torch::allclose(aa, bb, rtol, atol);
}

BlockPtr
TorchBlockBackend::angle(const BlockCPtr& a)
{
    return wrap(torch::angle(tens(a)));
}

bool
TorchBlockBackend::any(const BlockCPtr& a)
{
    return torch::any(tens(a)).item<bool>();
}

BlockPtr
TorchBlockBackend::apply_mask(const BlockCPtr& block, const BlockCPtr& mask, int64 ax)
{
    torch::Tensor t = tens(block);
    std::vector<TensorIndex> idx(static_cast<size_t>(t.dim()), Slice());
    idx[static_cast<size_t>(ax)] = tens(mask);
    return wrap(t.index(idx));
}

BlockPtr
TorchBlockBackend::_argsort(const BlockCPtr& block, int64 axis)
{
    return wrap(torch::argsort(tens(block), static_cast<int64_t>(axis)));
}

BlockPtr
TorchBlockBackend::conj(const BlockCPtr& a)
{
    return wrap(torch::conj_physical(tens(a)));
}

BlockPtr
TorchBlockBackend::copy_block(const BlockCPtr& a, std::optional<std::string> device)
{
    torch::Tensor res = tens(a).clone().detach();
    if (device)
        res = res.to(parse_device(as_device(device)));
    return wrap(std::move(res));
}

BlockPtr
TorchBlockBackend::cutoff_inverse(const BlockCPtr& a, float64 cutoff)
{
    torch::Tensor t = tens(a);
    torch::Tensor denom = torch::where(torch::abs(t) < cutoff, torch::full_like(t, INFINITY), t);
    return wrap(1.0 / denom);
}

std::tuple<BlockPtr, BlockPtr>
TorchBlockBackend::eigh(const BlockCPtr& block, std::optional<std::string> sort)
{
    auto pair = torch::linalg_eigh(tens(block));
    torch::Tensor w = std::get<0>(pair);
    torch::Tensor v = std::get<1>(pair);
    if (sort) {
        BlockPtr perm = argsort(wrap(w), sort, /*axis=*/0);
        torch::Tensor p = tens(perm);
        w = w.index({ p });
        v = v.index({ Slice(), p });
    }
    return { wrap(std::move(w)), wrap(std::move(v)) };
}

BlockPtr
TorchBlockBackend::eigvalsh(const BlockCPtr& block, std::optional<std::string> sort)
{
    torch::Tensor w = torch::linalg_eigvalsh(tens(block));
    if (sort) {
        BlockPtr perm = argsort(wrap(w), sort, /*axis=*/0);
        w = w.index({ tens(perm) });
    }
    return wrap(std::move(w));
}

BlockPtr
TorchBlockBackend::enlarge_leg(const BlockCPtr& block, const BlockCPtr& mask, int64 axis)
{
    torch::Tensor a = tens(block);
    torch::Tensor m = tens(mask);
    std::vector<int64_t> shape = a.sizes().vec();
    shape[static_cast<size_t>(axis)] = m.numel();
    torch::Tensor res = torch::zeros(shape, a.options());
    std::vector<TensorIndex> idcs(static_cast<size_t>(a.dim()), Slice());
    idcs[static_cast<size_t>(axis)] = m;
    res.index_put_(idcs, a.clone());
    return wrap(std::move(res));
}

BlockPtr
TorchBlockBackend::exp(const BlockCPtr& a)
{
    return wrap(torch::exp(tens(a)));
}

BlockPtr
TorchBlockBackend::block_from_diagonal(const BlockCPtr& diag)
{
    return wrap(torch::diag(tens(diag)));
}

BlockPtr
TorchBlockBackend::block_from_mask(const BlockCPtr& mask, Dtype dt)
{
    torch::Tensor m = tens(mask);
    int64_t M = m.size(0);
    int64_t N = m.sum().item<int64_t>();
    auto opts = torch::TensorOptions().dtype(dtype::to_torch_dtype(dt)).device(m.device());
    torch::Tensor res = torch::zeros({ N, M }, opts);
    torch::Tensor rows =
      torch::arange(N, torch::TensorOptions().dtype(torch::kLong).device(m.device()));
    res.index_put_({ rows, m }, 1);
    return wrap(std::move(res));
}

BlockPtr
TorchBlockBackend::block_from_numpy(const py::array& a,
                                    std::optional<Dtype> dtype_opt,
                                    std::optional<std::string> device)
{
    std::string dev = as_device(device);
    torch::Tensor t = tensor_from_numpy_array(a).to(parse_device(dev));
    if (dtype_opt)
        t = t.to(dtype::to_torch_dtype(*dtype_opt));
    return wrap(std::move(t));
}

BlockPtr
TorchBlockBackend::get_diagonal(const BlockCPtr& a, std::optional<float64> tol)
{
    torch::Tensor t = tens(a);
    torch::Tensor res = torch::diagonal(t);
    if (tol) {
        if (!torch::allclose(t, torch::diag(res), /*rtol=*/0.0, /*atol=*/*tol))
            throw std::invalid_argument("Not a diagonal block.");
    }
    return wrap(std::move(res));
}

BlockPtr
TorchBlockBackend::imag(const BlockCPtr& a)
{
    torch::Tensor t = tens(a);
    if (!c10::isComplexType(t.scalar_type()))
        return wrap(torch::zeros_like(t));
    return wrap(torch::imag(t));
}

BlockBackend::Scalar
TorchBlockBackend::inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger)
{
    auto [aa, bb] = to_same_dtype(tens(a), tens(b), torch::kHalf);
    torch::Tensor res;
    if (do_dagger) {
        std::vector<int64_t> all_dims(static_cast<size_t>(aa.dim()));
        std::iota(all_dims.begin(), all_dims.end(), 0);
        res = torch::tensordot(torch::conj_physical(aa), bb, all_dims, all_dims);
    } else {
        std::vector<int64_t> idcs_a(static_cast<size_t>(aa.dim()));
        std::vector<int64_t> idcs_b(static_cast<size_t>(aa.dim()));
        for (int64_t i = 0; i < aa.dim(); ++i) {
            idcs_a[static_cast<size_t>(i)] = i;
            idcs_b[static_cast<size_t>(i)] = aa.dim() - 1 - i;
        }
        res = torch::tensordot(aa, bb, idcs_a, idcs_b);
    }
    return item(wrap(std::move(res)));
}

BlockBackend::Scalar
TorchBlockBackend::item(const BlockCPtr& a)
{
    torch::Tensor t = tens(a);
    if (t.numel() != 1)
        throw std::invalid_argument("item() requires a single-element block");
    return as_scalar(t.reshape({}));
}

BlockPtr
TorchBlockBackend::kron(const BlockCPtr& a, const BlockCPtr& b)
{
    auto [aa, bb] = to_same_dtype(tens(a), tens(b));
    return wrap(torch::kron(aa, bb));
}

BlockPtr
TorchBlockBackend::linear_combination(const Scalar& a_coef,
                                      const BlockCPtr& v,
                                      const Scalar& b_coef,
                                      const BlockCPtr& w)
{
    torch::Tensor av = tens(a_coef._block()) * tens(v);
    torch::Tensor bw = tens(b_coef._block()) * tens(w);
    auto [aa, bb] = to_same_dtype(av, bw);
    return wrap(aa + bb);
}

BlockPtr
TorchBlockBackend::log(const BlockCPtr& a)
{
    return wrap(torch::log(tens(a)));
}

BlockBackend::Scalar
TorchBlockBackend::max(const BlockCPtr& a)
{
    return item(wrap(torch::max(tens(a))));
}

BlockBackend::Scalar
TorchBlockBackend::max_abs(const BlockCPtr& a)
{
    return item(wrap(torch::max(torch::abs(tens(a)))));
}

BlockBackend::Scalar
TorchBlockBackend::min(const BlockCPtr& a)
{
    return item(wrap(torch::min(tens(a))));
}

BlockPtr
TorchBlockBackend::mul(const Scalar& a, const BlockCPtr& b)
{
    return wrap(tens(a._block()) * tens(b));
}

BlockBackend::Scalar
TorchBlockBackend::norm(const BlockCPtr& a, float64 order, std::optional<int64> axis)
{
    torch::Tensor t = tens(a);
    torch::Tensor res;
    if (axis)
        res = torch::linalg_vector_norm(t, order, { static_cast<int64_t>(*axis) });
    else
        res = torch::linalg_vector_norm(t, order);
    return item(wrap(std::move(res)));
}

BlockPtr
TorchBlockBackend::outer(const BlockCPtr& a, const BlockCPtr& b)
{
    // torch::tensordot(..., dims=([], [])) uses addmm and does not support Bool.
    // Broadcasted multiply matches tensordot for all dtypes, including bool.
    auto [aa, bb] = to_same_dtype(tens(a), tens(b));
    std::vector<int64_t> a_view = aa.sizes().vec();
    a_view.insert(a_view.end(), bb.dim(), 1);
    std::vector<int64_t> b_view(aa.dim(), 1);
    auto b_sizes = bb.sizes().vec();
    b_view.insert(b_view.end(), b_sizes.begin(), b_sizes.end());
    return wrap(aa.reshape(a_view) * bb.reshape(b_view));
}

BlockPtr
TorchBlockBackend::permute_axes(const BlockCPtr& a, const std::vector<int64>& permutation)
{
    return wrap(tens(a).permute(to_int64_vec(permutation)).clone());
}

BlockPtr
TorchBlockBackend::random_uniform(const std::vector<int64>& dims,
                                  Dtype dt,
                                  std::optional<std::string> device)
{
    auto opts = torch::TensorOptions()
                  .dtype(dtype::to_torch_dtype(dt))
                  .device(parse_device(as_device(device)));
    torch::Tensor u = torch::rand(to_int64_vec(dims), opts);
    if (dtype::is_complex(dt)) {
        c10::complex<double> offset{ -1.0, -1.0 };
        return wrap(torch::scalar_tensor(offset, opts) + 2.0 * u);
    }
    return wrap(-1.0 + 2.0 * u);
}

BlockPtr
TorchBlockBackend::random_normal(const std::vector<int64>& dims,
                                 Dtype dt,
                                 float64 sigma,
                                 std::optional<std::string> device)
{
    auto device_obj = parse_device(as_device(device));
    auto opts = torch::TensorOptions().dtype(dtype::to_torch_dtype(dt)).device(device_obj);
    torch::Tensor mean = torch::zeros(to_int64_vec(dims), opts);
    // Keep std real (kDouble); complex std is rejected by torch::normal.
    torch::Tensor std_t =
      sigma *
      torch::ones_like(mean, torch::TensorOptions().dtype(torch::kDouble).device(device_obj));
    return wrap(at::normal(mean, std_t));
}

BlockPtr
TorchBlockBackend::real(const BlockCPtr& a)
{
    return wrap(torch::real(tens(a)));
}

BlockPtr
TorchBlockBackend::real_if_close(const BlockCPtr& a, float64 tol)
{
    torch::Tensor t = tens(a);
    // Match Python: eps hardcoded for float64; compare imag
    constexpr float64 eps = 2.2204460492503131e-16;
    if (c10::isComplexType(t.scalar_type()) &&
        torch::all(torch::abs(torch::imag(t)) < tol * eps).item<bool>()) {
        t = torch::real(t);
    }
    return wrap(std::move(t));
}

BlockPtr
TorchBlockBackend::scale_axis(const BlockCPtr& block, const BlockCPtr& factors, int64 axis)
{
    torch::Tensor t = tens(block);
    torch::Tensor f = tens(factors);
    std::vector<TensorIndex> fidx(static_cast<size_t>(t.dim()), None);
    fidx[static_cast<size_t>(axis)] = Slice();
    return wrap(t * f.index(fidx));
}

BlockPtr
TorchBlockBackend::tile(const BlockCPtr& a, int64 repeats)
{
    return wrap(tens(a).repeat({ static_cast<int64_t>(repeats) }));
}

std::vector<std::string>
TorchBlockBackend::_block_repr_lines(const BlockCPtr& a,
                                     const std::string& indent,
                                     int64 max_width,
                                     int64 max_lines)
{
    // Prefer numpy repr to avoid importing the Python torch package (which can conflict with
    // libtorch already linked into cyten._core).
    (void)max_width;
    py::array arr = ptr(a)->to_numpy();
    std::string rep = py::str(py::repr(arr));
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= rep.size()) {
        std::size_t end = rep.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(indent + rep.substr(start));
            break;
        }
        lines.push_back(indent + rep.substr(start, end - start));
        start = end + 1;
    }
    if (static_cast<int64>(lines.size()) > max_lines) {
        int64 first = (max_lines - 1) / 2;
        int64 last = max_lines - 1 - first;
        std::vector<std::string> trimmed(lines.begin(), lines.begin() + first);
        trimmed.push_back(indent + "...");
        trimmed.insert(trimmed.end(), lines.end() - last, lines.end());
        return trimmed;
    }
    return lines;
}

BlockPtr
TorchBlockBackend::reshape(const BlockCPtr& a, const std::vector<int64>& shape)
{
    return wrap(tens(a).reshape(to_int64_vec(shape)));
}

BlockPtr
TorchBlockBackend::sqrt(const BlockCPtr& a)
{
    return wrap(torch::sqrt(tens(a)));
}

BlockPtr
TorchBlockBackend::squeeze_axes(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    torch::Tensor t = tens(a);
    std::vector<TensorIndex> idx;
    idx.reserve(static_cast<size_t>(t.dim()));
    for (int64_t ax = 0; ax < t.dim(); ++ax) {
        bool squeeze = std::find(idcs.begin(), idcs.end(), static_cast<int64>(ax)) != idcs.end();
        idx.emplace_back(squeeze ? TensorIndex{ 0 } : TensorIndex{ Slice() });
    }
    return wrap(t.index(idx));
}

BlockPtr
TorchBlockBackend::stable_log(const BlockCPtr& block, float64 cutoff)
{
    torch::Tensor t = tens(block);
    return wrap(torch::where(t > cutoff, torch::log(t), torch::zeros_like(t)));
}

BlockPtr
TorchBlockBackend::sum(const BlockCPtr& a, int64 ax)
{
    return wrap(torch::sum(tens(a), { static_cast<int64_t>(ax) }));
}

BlockBackend::Scalar
TorchBlockBackend::sum_all(const BlockCPtr& a)
{
    return item(wrap(torch::sum(tens(a))));
}

BlockPtr
TorchBlockBackend::multiply_blocks(const BlockCPtr& a, const BlockCPtr& b)
{
    return wrap(tens(a) * tens(b));
}

BlockPtr
TorchBlockBackend::tdot(const BlockCPtr& a,
                        const BlockCPtr& b,
                        const std::vector<int64>& idcs_a,
                        const std::vector<int64>& idcs_b)
{
    auto [aa, bb] = to_same_dtype(tens(a), tens(b), torch::kHalf);
    return wrap(torch::tensordot(aa, bb, to_int64_vec(idcs_a), to_int64_vec(idcs_b)));
}

BlockPtr
TorchBlockBackend::to_dtype(const BlockCPtr& a, Dtype dt)
{
    return wrap(tens(a).to(dtype::to_torch_dtype(dt)));
}

BlockBackend::Scalar
TorchBlockBackend::trace_full(const BlockCPtr& a)
{
    torch::Tensor t = tens(a);
    int64_t ndim = t.dim();
    int64_t num_trace = ndim / 2;
    int64_t trace_dim = 1;
    for (int64_t i = 0; i < num_trace; ++i)
        trace_dim *= t.size(i);
    std::vector<int64_t> perm(static_cast<size_t>(ndim));
    for (int64_t i = 0; i < num_trace; ++i)
        perm[static_cast<size_t>(i)] = i;
    for (int64_t i = 0; i < num_trace; ++i)
        perm[static_cast<size_t>(num_trace + i)] = 2 * num_trace - 1 - i;
    t = t.permute(perm).reshape({ trace_dim, trace_dim });
    return item(wrap(t.diagonal(/*offset=*/0, /*dim1=*/0, /*dim2=*/1).sum(0)));
}

BlockPtr
TorchBlockBackend::trace_partial(const BlockCPtr& a,
                                 const std::vector<int64>& idcs1,
                                 const std::vector<int64>& idcs2,
                                 const std::vector<int64>& remaining_idcs)
{
    torch::Tensor t = tens(a);
    std::vector<int64_t> perm;
    perm.reserve(remaining_idcs.size() + idcs1.size() + idcs2.size());
    for (int64 i : remaining_idcs)
        perm.push_back(i);
    for (int64 i : idcs1)
        perm.push_back(i);
    for (int64 i : idcs2)
        perm.push_back(i);
    t = t.permute(perm);
    int64_t trace_dim = 1;
    auto sizes = t.sizes();
    for (size_t i = 0; i < idcs1.size(); ++i)
        trace_dim *= sizes[remaining_idcs.size() + i];
    std::vector<int64_t> new_shape;
    for (size_t i = 0; i < remaining_idcs.size(); ++i)
        new_shape.push_back(sizes[i]);
    new_shape.push_back(trace_dim);
    new_shape.push_back(trace_dim);
    t = t.reshape(new_shape);
    return wrap(t.diagonal(/*offset=*/0, /*dim1=*/-1, /*dim2=*/-2).sum(-1));
}

BlockPtr
TorchBlockBackend::eye_matrix(int64 dim, Dtype dt, std::optional<std::string> device)
{
    auto opts = torch::TensorOptions()
                  .dtype(dtype::to_torch_dtype(dt))
                  .device(parse_device(as_device(device)));
    return wrap(torch::eye(static_cast<int64_t>(dim), opts));
}

BlockBackend::Scalar
TorchBlockBackend::get_block_element(const BlockCPtr& a, const std::vector<int64>& idcs)
{
    torch::Tensor t = tens(a);
    std::vector<TensorIndex> key;
    key.reserve(idcs.size());
    for (int64 i : idcs)
        key.emplace_back(static_cast<int64_t>(i));
    return item(wrap(t.index(key)));
}

BlockPtr
TorchBlockBackend::matrix_dot(const BlockCPtr& a, const BlockCPtr& b)
{
    auto [aa, bb] = to_same_dtype(tens(a), tens(b));
    return wrap(torch::matmul(aa, bb));
}

BlockPtr
TorchBlockBackend::matrix_exp(const BlockCPtr& matrix)
{
    return wrap(torch::linalg_matrix_exp(tens(matrix)));
}

std::tuple<BlockPtr, BlockPtr>
TorchBlockBackend::matrix_qr(const BlockCPtr& a, bool full)
{
    auto pair = torch::linalg_qr(tens(a), full ? "complete" : "reduced");
    return { wrap(std::get<0>(pair)), wrap(std::get<1>(pair)) };
}

std::tuple<BlockPtr, BlockPtr, BlockPtr>
TorchBlockBackend::matrix_svd(const BlockCPtr& a, std::optional<std::string> algorithm)
{
    torch::Tensor t = tens(a);
    std::optional<std::string_view> driver;
    if (t.device().is_cuda()) {
        std::string algo = algorithm ? *algorithm : "gesvd";
        bool ok = false;
        for (auto const& s : possible_svd_algorithms()) {
            if (s == algo) {
                ok = true;
                break;
            }
        }
        if (!ok)
            throw std::invalid_argument("unsupported SVD algorithm: " + algo);
        driver = algo;
    } else {
        if (algorithm && *algorithm == "gesvd")
            algorithm.reset();
        if (algorithm)
            throw std::invalid_argument(
              "For torch, the algorithm keyword is only supported on CUDA hardware");
    }
    auto triple = torch::linalg_svd(t, /*full_matrices=*/false, driver);
    return { wrap(std::get<0>(triple)), wrap(std::get<1>(triple)), wrap(std::get<2>(triple)) };
}

const std::vector<std::string>&
TorchBlockBackend::possible_svd_algorithms() const
{
    static const std::vector<std::string> algorithms = { "gesvdj", "gesvda", "gesvd" };
    return algorithms;
}

BlockPtr
TorchBlockBackend::ones_block(const std::vector<int64>& shape,
                              Dtype dt,
                              std::optional<std::string> device)
{
    auto opts = torch::TensorOptions()
                  .dtype(dtype::to_torch_dtype(dt))
                  .device(parse_device(as_device(device)));
    return wrap(torch::ones(to_int64_vec(shape), opts));
}

BlockPtr
TorchBlockBackend::zeros(const std::vector<int64>& shape,
                         Dtype dt,
                         std::optional<std::string> device)
{
    auto opts = torch::TensorOptions()
                  .dtype(dtype::to_torch_dtype(dt))
                  .device(parse_device(as_device(device)));
    return wrap(torch::zeros(to_int64_vec(shape), opts));
}

BlockPtr
TorchBlockBackend::apply_leg_permutations(const BlockCPtr& block,
                                          const std::vector<py::array_t<int64>>& perms)
{
    // Reuse numpy ix_ via to_numpy for correctness; result back on block device.
    py::module_ np = py::module_::import("numpy");
    py::object a = ptr(block)->to_numpy();
    py::list ix_parts;
    for (auto const& p : perms)
        ix_parts.append(p);
    py::object ix = np.attr("ix_")(*ix_parts);
    py::array result = a[ix].cast<py::array>();
    return wrap(tensor_from_numpy_array(result).to(tens(block).device()));
}

} // namespace cyten
