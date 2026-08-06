#include <cyten/symmetries/spaces.h>

#include <cyten/symmetries/exceptions.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace cyten {

namespace {

[[nodiscard]] std::size_t
dim_as_size(float64 dim)
{
    assert(dim >= 0.);
    assert(std::floor(dim) == dim);
    return static_cast<std::size_t>(dim);
}

[[nodiscard]] std::vector<int64>
arange(std::size_t n)
{
    std::vector<int64> out(n);
    std::iota(out.begin(), out.end(), int64{ 0 });
    return out;
}

[[nodiscard]] std::vector<int64>
inverse_permutation(std::vector<int64> const& perm)
{
    std::vector<int64> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        auto const idx = perm[i];
        assert(idx >= 0);
        assert(static_cast<std::size_t>(idx) < perm.size());
        inv[static_cast<std::size_t>(idx)] = static_cast<int64>(i);
    }
    return inv;
}

[[nodiscard]] py::array_t<int64>
vector_to_array(std::vector<int64> const& v)
{
    py::array_t<int64> arr(static_cast<py::ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (std::size_t i = 0; i < v.size(); ++i) {
        buf(static_cast<py::ssize_t>(i)) = v[i];
    }
    return arr;
}

} // namespace

Leg::Leg(Symmetry::Ptr symmetry_,
         float64 dim_,
         bool is_dual_,
         std::optional<std::vector<int64>> basis_perm)
  : symmetry(std::move(symmetry_))
  , dim(dim_)
  , is_dual(is_dual_)
{
    if (!basis_perm) {
        _basis_perm = std::nullopt;
        _inverse_basis_perm = std::nullopt;
    } else {
        if (!symmetry->can_be_dropped()) {
            throw SymmetryError(std::format("basis_perm is meaningless for {}.", symmetry->str()));
        }
        assert(basis_perm->size() == dim_as_size(dim));
        _basis_perm = std::move(basis_perm);
        _inverse_basis_perm = inverse_permutation(*_basis_perm);
    }
}

void
Leg::test_sanity() const
{
    if (!symmetry->can_be_dropped()) {
        assert(!_basis_perm);
    }
    if (!_basis_perm) {
        assert(!_inverse_basis_perm);
    } else {
        assert(_inverse_basis_perm);
        auto const n = dim_as_size(dim);
        assert(_basis_perm->size() == n);
        assert(_inverse_basis_perm->size() == n);
        // is a permutation
        assert(std::unordered_set<int64>(_basis_perm->begin(), _basis_perm->end()).size() == n);
        assert(std::unordered_set<int64>(_inverse_basis_perm->begin(), _inverse_basis_perm->end())
                 .size() == n);
        for (std::size_t i = 0; i < n; ++i) {
            assert((*_basis_perm)[static_cast<std::size_t>((*_inverse_basis_perm)[i])] ==
                   static_cast<int64>(i));
        }
    }
}

py::object
Leg::as_ElementarySpace(bool is_dual_)
{
    // can be overridden for performance
    return as_Space().attr("as_ElementarySpace")(py::arg("is_dual") = is_dual_);
}

std::vector<int64>
Leg::basis_perm() const
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("basis_perm is meaningless for {}.", symmetry->str()));
    }
    if (!_basis_perm) {
        return arange(dim_as_size(dim));
    }
    return *_basis_perm;
}

void
Leg::set_basis_perm(std::optional<std::vector<int64>> basis_perm)
{
    if (!basis_perm) {
        _basis_perm = std::nullopt;
        _inverse_basis_perm = std::nullopt;
        return;
    }
    assert(basis_perm->size() == dim_as_size(dim));
    _basis_perm = std::move(basis_perm);
    _inverse_basis_perm = inverse_permutation(*_basis_perm);
}

std::vector<int64>
Leg::inverse_basis_perm() const
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("basis_perm is meaningless for {}.", symmetry->str()));
    }
    if (!_inverse_basis_perm) {
        return arange(dim_as_size(dim));
    }
    return *_inverse_basis_perm;
}

void
Leg::set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm)
{
    if (!inverse_basis_perm) {
        _basis_perm = std::nullopt;
        _inverse_basis_perm = std::nullopt;
        return;
    }
    assert(inverse_basis_perm->size() == dim_as_size(dim));
    _inverse_basis_perm = std::move(inverse_basis_perm);
    _basis_perm = inverse_permutation(*_inverse_basis_perm);
}

std::vector<Leg::Ptr>
Leg::flat_legs()
{
    return { shared_from_this() };
}

std::vector<Leg::Ptr>
Leg::flat_spaces()
{
    return { shared_from_this() };
}

int64
Leg::num_flat_legs() const
{
    return 1;
}

std::vector<int64>
Leg::_flat_leg_permutation(int64 offset) const
{
    return { offset };
}

std::string
Leg::ascii_arrow() const
{
    // Subclasses (ElementarySpace / LegPipe) override. Pure Leg should not appear in diagrams.
    throw std::runtime_error("ascii_arrow not implemented for this Leg subclass");
}

py::array
Leg::apply_basis_perm(py::array arr, int64 axis, bool inverse, bool pre_compose) const
{
    // this implementation assumes _basis_perm. AbelianLegPipe overrides this method.
    auto const& perm = inverse ? _inverse_basis_perm : _basis_perm;
    if (!perm) {
        // perm is identity permutation
        return arr;
    }
    auto perm_arr = vector_to_array(*perm);
    if (pre_compose) {
        assert(axis == 0);
        return perm_arr[py::make_tuple(arr)];
    }
    auto np = py::module_::import("numpy");
    return np.attr("take")(arr, perm_arr, py::arg("axis") = axis);
}

} // namespace cyten
