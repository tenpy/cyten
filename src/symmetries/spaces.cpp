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

namespace {

[[nodiscard]] std::vector<int64>
py_array_to_i64(py::array arr)
{
    auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

[[nodiscard]] std::vector<float64>
py_array_to_f64(py::array arr)
{
    auto casted = py::array_t<float64, py::array::c_style | py::array::forcecast>::ensure(arr);
    auto r = casted.unchecked<1>();
    std::vector<float64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

[[nodiscard]] SectorArray
take_or_all(SectorArray const& arr, std::optional<std::vector<std::size_t>> const& perm)
{
    if (!perm) {
        return arr;
    }
    return arr.take(*perm);
}

[[nodiscard]] std::vector<int64>
gather_or_all(std::vector<int64> const& vals, std::optional<std::vector<std::size_t>> const& perm)
{
    if (!perm) {
        return vals;
    }
    std::vector<int64> out(perm->size());
    for (std::size_t i = 0; i < perm->size(); ++i) {
        out[i] = vals[(*perm)[i]];
    }
    return out;
}

[[nodiscard]] bool
is_identity_lexsort(std::vector<std::size_t> const& indices)
{
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != i) {
            return false;
        }
    }
    return true;
}

} // namespace

Space::Space(Symmetry::Ptr symmetry_,
             SectorArray sector_decomposition_,
             std::optional<std::vector<int64>> multiplicities_,
             std::optional<std::string> sector_order_)
  : symmetry(std::move(symmetry_))
  , sector_decomposition(std::move(sector_decomposition_))
  , sector_order(std::move(sector_order_))
{
    if (sector_decomposition.sector_ind_len() != symmetry->sector_ind_len) {
        throw std::invalid_argument(
          std::format("Wrong sectors.shape: Expected (*, {}), got ({}, {}).",
                      symmetry->sector_ind_len,
                      sector_decomposition.size(),
                      sector_decomposition.sector_ind_len()));
    }
    num_sectors = static_cast<int64>(sector_decomposition.size());
    auto const n = static_cast<std::size_t>(num_sectors);
    if (!multiplicities_) {
        multiplicities.assign(n, 1);
    } else {
        multiplicities = std::move(*multiplicities_);
        assert(multiplicities.size() == n);
    }
    if (symmetry->can_be_dropped()) {
        sector_dims = py_array_to_i64(symmetry->batch_sector_dim(sector_decomposition));
        sector_qdims.assign(sector_dims->begin(), sector_dims->end());
        std::vector<std::array<int64, 2>> sl(n);
        int64 running = 0;
        for (std::size_t i = 0; i < n; ++i) {
            sl[i][0] = running;
            running += multiplicities[i] * (*sector_dims)[i];
            sl[i][1] = running;
        }
        slices = std::move(sl);
        dim = static_cast<float64>(running);
    } else {
        sector_dims = std::nullopt;
        sector_qdims = py_array_to_f64(symmetry->batch_qdim(sector_decomposition));
        slices = std::nullopt;
        float64 total = 0.;
        for (std::size_t i = 0; i < n; ++i) {
            total += sector_qdims[i] * static_cast<float64>(multiplicities[i]);
        }
        dim = total;
    }
}

void
Space::test_sanity() const
{
    assert(dim >= 0.);
    // sectors
    if (static_cast<int64>(sector_decomposition.size()) != num_sectors ||
        sector_decomposition.sector_ind_len() != symmetry->sector_ind_len) {
        throw std::runtime_error("wrong sectors.shape");
    }
    assert(symmetry->are_valid_sectors(sector_decomposition));
    {
        std::vector<std::int64_t> ones(static_cast<std::size_t>(num_sectors), 1);
        auto const [unique, um, perm] = sector_decomposition.unique_sorted(ones);
        assert(static_cast<int64>(unique.size()) == num_sectors);
        (void)um;
        (void)perm;
    }
    if (sector_order == "sorted") {
        assert(is_identity_lexsort(sector_decomposition.lexsort_indices()));
    } else if (sector_order == "dual_sorted") {
        auto expect_sorted = symmetry->dual_sectors(sector_decomposition);
        assert(is_identity_lexsort(expect_sorted.lexsort_indices()));
    } else if (!sector_order) {
        // nothing to check
    } else {
        throw std::runtime_error(std::format("Invalid sector_order: {}", *sector_order));
    }
    // multiplicities
    assert(multiplicities.size() == static_cast<std::size_t>(num_sectors));
    assert(std::ranges::all_of(multiplicities, [](int64 m) { return m > 0; }));
    if (symmetry->can_be_dropped()) {
        assert(slices);
        assert(sector_dims);
        assert(slices->size() == static_cast<std::size_t>(num_sectors));
        auto expect_dims = py_array_to_i64(symmetry->batch_sector_dim(sector_decomposition));
        assert(*sector_dims == expect_dims);
        for (std::size_t i = 0; i < static_cast<std::size_t>(num_sectors); ++i) {
            assert((*slices)[i][1] - (*slices)[i][0] == (*sector_dims)[i] * multiplicities[i]);
        }
        // slices should be consecutive
        if (num_sectors > 0) {
            assert((*slices)[0][0] == 0);
            for (std::size_t i = 1; i < static_cast<std::size_t>(num_sectors); ++i) {
                assert((*slices)[i][0] == (*slices)[i - 1][1]);
            }
            assert((*slices)[static_cast<std::size_t>(num_sectors) - 1][1] ==
                   static_cast<int64>(dim));
        }
    }
}

bool
Space::is_trivial() const
{
    if (num_sectors > 1) {
        return false;
    }
    if (multiplicities[0] > 1) {
        return false;
    }
    return sector_decomposition[0] == symmetry->trivial_sector;
}

bool
Space::operator==(Space const& /*other*/) const
{
    throw py::type_error(
      "Space does not support \"==\" comparison. Use `is_isomorphic_to` instead.");
}

bool
Space::is_isomorphic_to(Space const& other) const
{
    if (!symmetry->equals(*other.symmetry)) {
        throw SymmetryError("Incompatible symmetries");
    }
    if (num_sectors != other.num_sectors) {
        return false;
    }

    // find perm1 and perm2 such that ``self.sector_decomposition[perm1]`` and
    // ``other.sector_decomposition[perm2]`` have the same sorting convention
    std::optional<std::vector<std::size_t>> perm1;
    std::optional<std::vector<std::size_t>> perm2;
    if (!sector_order) {
        if (other.sector_order == "sorted") {
            perm1 = sector_decomposition.lexsort_indices();
            perm2 = std::nullopt;
        } else if (other.sector_order == "dual_sorted") {
            perm1 = symmetry->dual_sectors(sector_decomposition).lexsort_indices();
            perm2 = std::nullopt;
        } else {
            perm1 = sector_decomposition.lexsort_indices();
            perm2 = other.sector_decomposition.lexsort_indices();
        }
    } else if (!other.sector_order) {
        if (sector_order == "sorted") {
            perm1 = std::nullopt;
            perm2 = other.sector_decomposition.lexsort_indices();
        } else if (sector_order == "dual_sorted") {
            perm1 = std::nullopt;
            perm2 = symmetry->dual_sectors(other.sector_decomposition).lexsort_indices();
        } else {
            throw std::runtime_error("unreachable sector_order case");
        }
    } else if (sector_order == other.sector_order) {
        perm1 = std::nullopt;
        perm2 = std::nullopt;
    } else if (sector_order == "sorted") {
        perm1 = std::nullopt;
        perm2 = other.sector_decomposition.lexsort_indices();
    } else if (other.sector_order == "sorted") {
        perm1 = sector_decomposition.lexsort_indices();
        perm2 = std::nullopt;
    } else {
        throw std::runtime_error("unreachable sector_order case");
    }

    if (gather_or_all(multiplicities, perm1) != gather_or_all(other.multiplicities, perm2)) {
        return false;
    }
    return take_or_all(sector_decomposition, perm1) ==
           take_or_all(other.sector_decomposition, perm2);
}

bool
Space::is_subspace_of(Space const& other) const
{
    if (!symmetry->is_equivalent_to(*other.symmetry)) {
        return false;
    }
    if (num_sectors == 0) {
        return true;
    }
    if (sector_order == "sorted" && other.sector_order == "sorted") {
        // sectors are sorted, so we can just iterate over both of them
        std::size_t n_self = 0;
        for (std::size_t i = 0; i < other.sector_decomposition.size(); ++i) {
            if (sector_decomposition[n_self] == other.sector_decomposition[i]) {
                if (multiplicities[n_self] > other.multiplicities[i]) {
                    return false;
                }
                ++n_self;
            }
            if (static_cast<int64>(n_self) == num_sectors) {
                // have checked all sectors of self
                return true;
            }
        }
        // reaching this line means self has sectors which other does not have
        return false;
    }

    // OPTIMIZE sort once instead of looking up each time
    int64 num_sectors_checked = 0;
    for (std::size_t i = 0; i < other.sector_decomposition.size(); ++i) {
        auto const m = sector_multiplicity(other.sector_decomposition[i]);
        if (m == 0) {
            continue;
        }
        if (m > other.multiplicities[i]) {
            return false;
        }
        ++num_sectors_checked;
    }
    if (num_sectors_checked < num_sectors) {
        // this means self has some sectors that other doesn't have
        return false;
    }
    return true;
}

py::object
Space::as_ElementarySpace(bool is_dual_)
{
    SectorArray defining_sectors;
    bool is_sorted = false;
    if (is_dual_) {
        defining_sectors = symmetry->dual_sectors(sector_decomposition);
        is_sorted = sector_order == "dual_sorted";
    } else {
        defining_sectors = sector_decomposition;
        is_sorted = sector_order == "sorted";
    }

    auto ElementarySpace = py::module_::import("cyten.symmetries.spaces").attr("ElementarySpace");
    if (is_sorted) {
        return ElementarySpace(py::arg("symmetry") = symmetry,
                               py::arg("defining_sectors") = defining_sectors,
                               py::arg("multiplicities") = multiplicities,
                               py::arg("is_dual") = is_dual_);
    }
    return ElementarySpace.attr("from_defining_sectors")(
      py::arg("symmetry") = symmetry,
      py::arg("defining_sectors") = defining_sectors,
      py::arg("multiplicities") = multiplicities,
      py::arg("is_dual") = is_dual_,
      py::arg("unique_sectors") = true);
}

Space::Ptr
Space::as_Space()
{
    return shared_from_this();
}

std::optional<int64>
Space::sector_decomposition_where(Sector sector) const
{
    // OPTIMIZE : if sector_order allows it, use that sectors are sorted to speed up the lookup
    auto idx = sector_decomposition.row_where(sector);
    if (!idx) {
        return std::nullopt;
    }
    return static_cast<int64>(*idx);
}

int64
Space::sector_multiplicity(Sector sector) const
{
    auto idx = sector_decomposition_where(sector);
    if (!idx) {
        return 0;
    }
    return multiplicities[static_cast<std::size_t>(*idx)];
}

} // namespace cyten
