#include <cyten/tensors/direct_sum.h>

#include <cyten/block_backend/dtypes.h>
#include <cyten/tensors/ops_algebra.h>

#include <format>
#include <stdexcept>
#include <utility>

namespace cyten {

namespace {

int64
normalize_index(int64 i, std::size_t n)
{
    auto const N = static_cast<int64>(n);
    if (i < 0) {
        i += N;
    }
    if (i < 0 || i >= N) {
        throw std::out_of_range(std::format("DirectSum index {} out of range for size {}", i, N));
    }
    return i;
}

} // namespace

DirectSum::DirectSum(std::vector<TensorPtr> components)
  : _components(std::move(components))
{
    if (_components.empty()) {
        throw std::invalid_argument("DirectSum requires at least one tensor");
    }
    for (auto const& t : _components) {
        if (!t) {
            throw std::invalid_argument("DirectSum components must be non-null tensors");
        }
    }
    _backend = _components[0]->backend;
    _device = _components[0]->device;
    std::vector<Dtype> dtypes;
    dtypes.reserve(_components.size());
    for (auto const& t : _components) {
        if (t->backend != _backend) {
            throw std::invalid_argument("DirectSum components must share a backend");
        }
        if (t->device != _device) {
            throw std::invalid_argument("DirectSum components must share a device");
        }
        dtypes.push_back(t->dtype);
    }
    _dtype = dtype::common(dtypes);
}

TensorPtr
DirectSum::at(int64 i) const
{
    return _components[static_cast<std::size_t>(normalize_index(i, _components.size()))];
}

DirectSum::Ptr
DirectSum::copy(bool deep) const
{
    std::vector<TensorPtr> copied;
    copied.reserve(_components.size());
    for (auto const& t : _components) {
        copied.push_back(t->copy(deep));
    }
    return std::make_shared<DirectSum>(std::move(copied));
}

VectorLike::Ptr
DirectSum::clone() const
{
    return copy();
}

Dtype
DirectSum::vector_dtype() const
{
    return _dtype;
}

std::string
DirectSum::vector_device() const
{
    return _device;
}

TensorBackend::Ptr
DirectSum::vector_backend() const
{
    return _backend;
}

DirectSumCPtr
DirectSum::as_direct_sum(VectorLike::CPtr other, char const* op) const
{
    auto const ds = std::dynamic_pointer_cast<DirectSum const>(other);
    if (!ds) {
        throw std::invalid_argument(std::format("{}: expected a DirectSum", op));
    }
    if (ds->_components.size() != _components.size()) {
        throw std::invalid_argument(std::format("{}: DirectSum size mismatch ({} vs {})",
                                                op,
                                                _components.size(),
                                                ds->_components.size()));
    }
    return ds;
}

BlockBackend::Scalar
DirectSum::vector_norm() const
{
    auto acc = norm(TensorCPtr(_components[0]));
    acc = acc * acc;
    for (std::size_t i = 1; i < _components.size(); ++i) {
        auto ni = norm(TensorCPtr(_components[i]));
        acc = acc + ni * ni;
    }
    return acc.sqrt();
}

BlockBackend::Scalar
DirectSum::vector_inner(VectorLike::CPtr other, bool do_dagger) const
{
    auto const ds = as_direct_sum(other, "inner");
    auto acc = inner(TensorCPtr(_components[0]), TensorCPtr(ds->_components[0]), do_dagger);
    for (std::size_t i = 1; i < _components.size(); ++i) {
        acc = acc + inner(TensorCPtr(_components[i]), TensorCPtr(ds->_components[i]), do_dagger);
    }
    return acc;
}

VectorLike::Ptr
DirectSum::scaled(BlockBackend::Scalar const& a) const
{
    std::vector<TensorPtr> out;
    out.reserve(_components.size());
    for (auto const& t : _components) {
        out.push_back(scalar_multiply(a, TensorCPtr(t)));
    }
    return std::make_shared<DirectSum>(std::move(out));
}

VectorLike::Ptr
DirectSum::axpy(BlockBackend::Scalar const& a, VectorLike::CPtr other) const
{
    auto const ds = as_direct_sum(other, "axpy");
    auto one = _backend->block_backend->as_scalar(1.0);
    std::vector<TensorPtr> out;
    out.reserve(_components.size());
    for (std::size_t i = 0; i < _components.size(); ++i) {
        out.push_back(
          linear_combination(a, TensorCPtr(_components[i]), one, TensorCPtr(ds->_components[i])));
    }
    return std::make_shared<DirectSum>(std::move(out));
}

bool
DirectSum::compatible_with(VectorLike::CPtr other) const
{
    auto const ds = std::dynamic_pointer_cast<DirectSum const>(other);
    if (!ds || ds->_components.size() != _components.size()) {
        return false;
    }
    for (std::size_t i = 0; i < _components.size(); ++i) {
        if (!_components[i]->compatible_with(ds->_components[i])) {
            return false;
        }
    }
    return true;
}

} // namespace cyten
