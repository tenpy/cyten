#include <cyten/symmetries/spaces.h>

#include <cyten/config.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/symmetries/factors/no_symmetry.h>
#include <cyten/tools.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numeric>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <tuple>
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

/// ``repr`` of a bool, using the Python spelling.
[[nodiscard]] char const*
bool_repr(bool value)
{
    return value ? "True" : "False";
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
{
    init_leg(std::move(symmetry_), dim_, is_dual_, std::move(basis_perm));
}

void
Leg::init_leg(Symmetry::Ptr symmetry_,
              float64 dim_,
              bool is_dual_,
              std::optional<std::vector<int64>> basis_perm)
{
    symmetry = std::move(symmetry_);
    dim = dim_;
    is_dual = is_dual_;
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

Leg::Ptr
Leg::shared_leg()
{
    return std::dynamic_pointer_cast<Leg>(shared_from_this());
}

std::vector<Leg::Ptr>
Leg::flat_legs()
{
    return { shared_leg() };
}

std::vector<Leg::Ptr>
Leg::flat_spaces()
{
    return { shared_leg() };
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

py::object
Leg::apply_basis_perm(py::object arr, int64 axis, bool inverse, bool pre_compose) const
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

    ElementarySpace::Ptr es;
    if (is_sorted) {
        es = std::make_shared<ElementarySpace>(
          symmetry, defining_sectors, multiplicities, is_dual_, std::nullopt);
    } else {
        es = ElementarySpace::from_defining_sectors(
          symmetry, defining_sectors, multiplicities, is_dual_, std::nullopt, true);
    }
    return py::cast(es);
}

Space::Ptr
Space::shared_space()
{
    return std::dynamic_pointer_cast<Space>(shared_from_this());
}

Space::Ptr
Space::as_Space()
{
    return shared_space();
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

namespace {

[[nodiscard]] std::optional<std::vector<int64>>
combined_basis_perm(std::vector<Leg::Ptr> const& legs, bool combine_cstyle)
{
    bool any_custom = false;
    for (auto const& leg : legs) {
        if (leg->has_custom_basis_perm()) {
            any_custom = true;
            break;
        }
    }
    if (!any_custom) {
        return std::nullopt;
    }
    auto misc = py::module_::import("cyten.tools.misc");
    py::list perms;
    for (auto const& leg : legs) {
        perms.append(vector_to_array(leg->basis_perm()));
    }
    py::array combined =
      misc.attr("combine_permutations")(perms, py::arg("cstyle") = combine_cstyle);
    auto casted = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(combined);
    auto r = casted.unchecked<1>();
    std::vector<int64> out(static_cast<std::size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) {
        out[static_cast<std::size_t>(i)] = r(i);
    }
    return out;
}

} // namespace

// note: Leg is a virtual base and can therefore not be initialized here, see Leg::init_leg.
LegPipe::LegPipe(std::vector<Leg::Ptr> legs_, bool is_dual_, bool combine_cstyle_)
  : legs(std::move(legs_))
  , num_legs(static_cast<int64>(legs.size()))
  , combine_cstyle(combine_cstyle_)
{
    assert(num_legs > 0);
    float64 dim_prod = 1.;
    for (auto const& leg : legs) {
        dim_prod *= leg->dim;
    }
    init_leg(legs.at(0)->symmetry, dim_prod, is_dual_, combined_basis_perm(legs, combine_cstyle));
}

void
LegPipe::test_sanity() const
{
    for (auto const& leg : legs) {
        assert(leg->symmetry->equals(*symmetry));
        leg->test_sanity();
    }
    Leg::test_sanity();
}

py::object
LegPipe::as_Space()
{
    auto TensorProduct = py::module_::import("cyten.symmetries.spaces").attr("TensorProduct");
    py::list spaces;
    for (auto const& leg : legs) {
        spaces.append(leg->as_Space());
    }
    return TensorProduct(spaces, py::arg("symmetry") = symmetry);
}

Leg::Ptr
LegPipe::dual_leg() const
{
    std::vector<Leg::Ptr> dual_legs;
    dual_legs.reserve(legs.size());
    for (auto it = legs.rbegin(); it != legs.rend(); ++it) {
        dual_legs.push_back((*it)->dual_leg());
    }
    return std::make_shared<LegPipe>(std::move(dual_legs), !is_dual, !combine_cstyle);
}

bool
LegPipe::is_trivial() const
{
    return std::ranges::all_of(legs, [](Leg::Ptr const& leg) { return leg->is_trivial(); });
}

std::vector<Leg::Ptr>
LegPipe::flat_legs()
{
    std::vector<Leg::Ptr> out;
    for (auto const& leg : legs) {
        auto part = leg->flat_legs();
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::vector<Leg::Ptr>
LegPipe::flat_spaces()
{
    std::vector<Leg::Ptr> out;
    for (auto const& leg : legs) {
        auto part = leg->flat_spaces();
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

int64
LegPipe::num_flat_legs() const
{
    int64 n = 0;
    for (auto const& leg : legs) {
        n += leg->num_flat_legs();
    }
    return n;
}

std::vector<int64>
LegPipe::_flat_leg_permutation(int64 offset) const
{
    if (num_legs == num_flat_legs()) {
        std::vector<int64> perm(static_cast<std::size_t>(num_legs));
        std::iota(perm.begin(), perm.end(), offset);
        if (!combine_cstyle) {
            std::reverse(perm.begin(), perm.end());
        }
        return perm;
    }
    std::vector<Leg::Ptr> ordered = legs;
    if (!combine_cstyle) {
        std::reverse(ordered.begin(), ordered.end());
    }
    std::vector<int64> offsets;
    offsets.reserve(ordered.size());
    int64 running = offset;
    for (auto const& leg : ordered) {
        offsets.push_back(running);
        running += leg->num_flat_legs();
    }
    if (!combine_cstyle) {
        std::reverse(offsets.begin(), offsets.end());
    }
    std::vector<int64> perm;
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto part = legs[i]->_flat_leg_permutation(offsets[i]);
        perm.insert(perm.end(), part.begin(), part.end());
    }
    return perm;
}

void
LegPipe::set_basis_perm(std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error(std::format("Can not set basis_perm for {}.", "LegPipe"));
}

void
LegPipe::set_inverse_basis_perm(std::optional<std::vector<int64>> /*inverse_basis_perm*/)
{
    throw py::type_error(std::format("Can not set basis_perm for {}.", "LegPipe"));
}

std::string
LegPipe::ascii_arrow() const
{
    return "║";
}

bool
LegPipe::operator==(Leg const& other) const
{
    auto const* o = dynamic_cast<LegPipe const*>(&other);
    if (o == nullptr) {
        return false;
    }
    if (is_abelian_leg_pipe() != o->is_abelian_leg_pipe()) {
        return false;
    }
    if (is_dual != o->is_dual) {
        return false;
    }
    if (combine_cstyle != o->combine_cstyle) {
        return false;
    }
    if (num_legs != o->num_legs) {
        return false;
    }
    for (std::size_t i = 0; i < legs.size(); ++i) {
        if (!(*legs[i] == *o->legs[i])) {
            return false;
        }
    }
    return true;
}

Leg::Ptr
LegPipe::operator[](int64 idx) const
{
    auto const n = static_cast<int64>(legs.size());
    auto const i = to_valid_idx(idx, n);
    return legs[static_cast<std::size_t>(i)];
}

std::string
LegPipe::repr(bool show_symmetry, bool one_line) const
{
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "LegPipe";

    if (one_line) {
        if (show_symmetry) {
            auto res = std::format("{}(num_legs={}, is_dual={}, symmetry={}, combine_cstyle={})",
                                   ClsName,
                                   num_legs,
                                   bool_repr(is_dual),
                                   symmetry->repr(),
                                   bool_repr(combine_cstyle));
            if (static_cast<int64>(res.size()) <= linewidth) {
                return res;
            }
            return repr(false, true);
        }
        auto res = std::format("{}(num_legs={}, is_dual={}, combine_cstyle={})",
                               ClsName,
                               num_legs,
                               bool_repr(is_dual),
                               bool_repr(combine_cstyle));
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }
        throw std::runtime_error("LegPipe one-line repr exceeds linewidth");
    }

    for (bool force_children_one_line : { false, true }) {
        std::vector<std::string> lines;
        lines.push_back(std::format("{}([", ClsName));
        for (auto const& leg : legs) {
            py::object leg_obj = py::cast(leg);
            std::string rep;
            try {
                rep =
                  py::str(leg_obj.attr("__repr__")(py::arg("show_symmetry") = false,
                                                   py::arg("one_line") = force_children_one_line));
            } catch (py::error_already_set&) {
                rep = py::str(leg_obj.attr("__repr__")());
            }
            std::istringstream iss(rep);
            std::string line;
            while (std::getline(iss, line)) {
                lines.push_back(indent + line);
            }
        }
        if (show_symmetry) {
            lines.push_back(
              std::format("], is_dual={}, symmetry={})", bool_repr(is_dual), symmetry->repr()));
        } else {
            lines.push_back(std::format("], is_dual={})", bool_repr(is_dual)));
        }
        bool maxlines_ok = static_cast<int64>(lines.size()) <= maxlines;
        bool linewidth_ok = std::ranges::all_of(
          lines, [&](std::string const& l) { return static_cast<int64>(l.size()) < linewidth; });
        if (maxlines_ok && linewidth_ok) {
            std::string out = lines[0];
            for (std::size_t i = 1; i < lines.size(); ++i) {
                out += '\n';
                out += lines[i];
            }
            return out;
        }
    }
    return repr(show_symmetry, true);
}

namespace {

/// The Python ``no_symmetry``, i.e. the product symmetry with a single ``NoSymmetry`` factor.
[[nodiscard]] Symmetry::Ptr
no_symmetry_product()
{
    return std::make_shared<Symmetry>(
      std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() });
}

/// ``_sort_sectors``: lexsort the `sectors`, applying the same permutation to `multiplicities`.
[[nodiscard]] std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
sort_sectors(SectorArray const& sectors, std::vector<int64> const& multiplicities)
{
    auto [sorted, perm] = sectors.sorted();
    std::vector<int64> mults(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        mults[i] = multiplicities[perm[i]];
    }
    return { std::move(sorted), std::move(mults), std::move(perm) };
}

/// ``np.concatenate([[0], np.cumsum(values)])``, i.e. the ``values.size() + 1`` slice boundaries.
[[nodiscard]] std::vector<int64>
slice_boundaries(std::vector<int64> const& values)
{
    std::vector<int64> out(values.size() + 1, 0);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out[i + 1] = out[i] + values[i];
    }
    return out;
}

/// ``cyten.tools.misc.rank_data``, i.e. ``argsort(argsort(a))`` with stable sorting.
[[nodiscard]] std::vector<int64>
rank_data(std::vector<int64> const& a)
{
    std::vector<std::size_t> order(a.size());
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    std::ranges::stable_sort(order, [&a](std::size_t i, std::size_t j) { return a[i] < a[j]; });
    std::vector<int64> ranks(a.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        ranks[order[i]] = static_cast<int64>(i);
    }
    return ranks;
}

/// ``symmetry.batch_sector_dim(sectors) * multiplicities``, the number of states per sector.
[[nodiscard]] std::vector<int64>
num_states_per_sector(Symmetry const& symmetry,
                      SectorArray const& sectors,
                      std::vector<int64> const& multiplicities)
{
    auto num_states = py_array_to_i64(symmetry.batch_sector_dim(sectors));
    for (std::size_t i = 0; i < num_states.size(); ++i) {
        num_states[i] *= multiplicities[i];
    }
    return num_states;
}

/// ``_parse_inputs_drop_symmetry``. ``nullopt`` means ``'all'``, both on input and output.
[[nodiscard]] std::pair<std::optional<std::vector<int64>>, Symmetry::Ptr>
parse_inputs_drop_symmetry(std::optional<std::vector<int64>> const& which,
                           Symmetry const& symmetry)
{
    if (!which) {
        return { std::nullopt, no_symmetry_product() };
    }
    auto const num_factors = static_cast<int64>(symmetry.num_factors());
    std::vector<int64> valid;
    valid.reserve(which->size());
    for (auto i : *which) {
        valid.push_back(to_valid_idx(i, num_factors));
    }
    if (static_cast<int64>(valid.size()) == num_factors) {
        return { std::nullopt, no_symmetry_product() };
    }
    std::vector<SymmetryFactor::Ptr> remaining;
    for (int64 i = 0; i < num_factors; ++i) {
        if (std::ranges::find(valid, i) == valid.end()) {
            remaining.push_back(symmetry.factors[static_cast<std::size_t>(i)]);
        }
    }
    return { std::move(valid), std::make_shared<Symmetry>(std::move(remaining)) };
}

[[nodiscard]] py::object
slices_to_py(std::optional<std::vector<std::array<int64, 2>>> const& slices)
{
    if (!slices) {
        return py::none();
    }
    py::array_t<int64> arr({ static_cast<py::ssize_t>(slices->size()), py::ssize_t{ 2 } });
    auto buf = arr.mutable_unchecked<2>();
    for (std::size_t i = 0; i < slices->size(); ++i) {
        buf(static_cast<py::ssize_t>(i), 0) = (*slices)[i][0];
        buf(static_cast<py::ssize_t>(i), 1) = (*slices)[i][1];
    }
    return arr;
}

[[nodiscard]] py::object
optional_perm_to_py(std::optional<std::vector<int64>> const& perm)
{
    if (!perm) {
        return py::none();
    }
    return vector_to_array(*perm);
}

[[nodiscard]] std::optional<std::vector<int64>>
optional_perm_from_py(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py_array_to_i64(py::array::ensure(obj));
}

} // namespace

// note: Leg is a virtual base and can therefore not be initialized here, see Leg::init_leg.
// This is also convenient, since the dim is only computed by the Space constructor.
ElementarySpace::ElementarySpace(Symmetry::Ptr symmetry_,
                                 SectorArray defining_sectors_,
                                 std::optional<std::vector<int64>> multiplicities_,
                                 bool is_dual_,
                                 std::optional<std::vector<int64>> basis_perm_)
  : Space(symmetry_,
          is_dual_ ? symmetry_->dual_sectors(defining_sectors_) : defining_sectors_,
          std::move(multiplicities_),
          is_dual_ ? std::optional<std::string>{ "dual_sorted" }
                   : std::optional<std::string>{ "sorted" })
  , defining_sectors(std::move(defining_sectors_))
{
    assert(symmetry_->are_valid_sectors(defining_sectors));
    init_leg(Space::symmetry, Space::dim, is_dual_, std::move(basis_perm_));
}

void
ElementarySpace::test_sanity() const
{
    assert(static_cast<int64>(defining_sectors.size()) == num_sectors);
    assert(defining_sectors.sector_ind_len() == Space::symmetry->sector_ind_len);
    if (is_dual) {
        assert(sector_order == "dual_sorted");
    } else {
        assert(sector_order == "sorted");
    }
    Space::test_sanity();
    Leg::test_sanity();
}

ElementarySpace::Ptr
ElementarySpace::from_basis(Symmetry::Ptr symmetry, SectorArray sectors_of_basis)
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("from_basis is meaningless for {}.", symmetry->str()));
    }
    // note: the lexsort is stable, i.e. it preserves the order of equal keys.
    auto const basis_perm = sectors_of_basis.lexsort_indices();
    auto const sorted = sectors_of_basis.take(basis_perm);
    auto const diffs = sorted.find_row_differences(/*include_len=*/true);
    // [:-1] to exclude len
    auto sectors = sorted.take(std::span<const std::size_t>(diffs.data(), diffs.size() - 1));
    auto const dims = py_array_to_i64(symmetry->batch_sector_dim(sectors));
    std::vector<int64> multiplicities(sectors.size());
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        // how often the sector appears in the input sectors_of_basis
        auto const num_occurrences = static_cast<int64>(diffs[i + 1] - diffs[i]);
        if (num_occurrences % dims[i] != 0) {
            throw std::invalid_argument(
              "Sectors must appear in whole multiplets, i.e. a number of times that is an "
              "integer multiple of their dimension.");
        }
        multiplicities[i] = num_occurrences / dims[i];
    }
    return std::make_shared<ElementarySpace>(
      std::move(symmetry),
      std::move(sectors),
      std::move(multiplicities),
      false,
      std::vector<int64>(basis_perm.begin(), basis_perm.end()));
}

ElementarySpace::Ptr
ElementarySpace::from_independent_symmetries(std::vector<Ptr> const& independent_descriptions)
{
    // OPTIMIZE this can be implemented better. if many consecutive basis elements have the same
    //          resulting sector, we can skip over all of them.
    assert(!independent_descriptions.empty());
    auto const dim = independent_descriptions[0]->Space::dim;
    assert(std::ranges::all_of(independent_descriptions,
                               [dim](Ptr const& s) { return s->Space::dim == dim; }));
    // ignore those with no_symmetry
    auto const no_sym = no_symmetry_product();
    std::vector<Ptr> descriptions;
    for (auto const& s : independent_descriptions) {
        if (!s->Space::symmetry->equals(*no_sym)) {
            descriptions.push_back(s);
        }
    }
    if (descriptions.empty()) {
        // all descriptions had no_symmetry
        return from_trivial_sector(static_cast<int64>(dim));
    }
    std::vector<SymmetryFactor::Ptr> factors;
    for (auto const& s : descriptions) {
        auto const& own = s->Space::symmetry->factors;
        factors.insert(factors.end(), own.begin(), own.end());
    }
    auto symmetry = std::make_shared<Symmetry>(std::move(factors));
    if (!symmetry->can_be_dropped()) {
        // TODO is there a way to define this? the straight-forward picture works only if we have
        //      a vector space and can identify states.
        //      note: this interface is more general than it needs to be. The use case in
        //            GroupedSite would allow us to specialize, if that is easier. A given state
        //            is in the trivial sector for all but one of the independent_descriptions.
        throw SymmetryError(
          std::format("from_independent_symmetries is not supported for {}.", symmetry->str()));
    }
    // concatenate the sectors_of_basis of all descriptions along the sector axis
    std::vector<SectorArray> parts;
    parts.reserve(descriptions.size());
    for (auto const& s : descriptions) {
        parts.push_back(s->sectors_of_basis());
    }
    auto const num_basis_states = dim_as_size(dim);
    SectorArray sectors_of_basis(num_basis_states, symmetry->sector_ind_len);
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        auto sector = Sector::zeros(symmetry->sector_ind_len);
        std::size_t offset = 0;
        for (auto const& part : parts) {
            auto const& row = part[i];
            for (std::uint8_t k = 0; k < row.len(); ++k) {
                sector[offset++] = row[k];
            }
        }
        sectors_of_basis[i] = sector;
    }
    return from_basis(std::move(symmetry), std::move(sectors_of_basis));
}

ElementarySpace::Ptr
ElementarySpace::from_largest_common_subspace(std::vector<Space::Ptr> const& spaces, bool is_dual)
{
    if (spaces.empty()) {
        throw std::invalid_argument("Need at least one space");
    }
    if (spaces.size() == 1) {
        return spaces[0]->as_ElementarySpace(is_dual).cast<Ptr>();
    }
    if (spaces.size() > 2) {
        // OPTIMIZE directly implement for many
        auto pair = from_largest_common_subspace({ spaces[0], spaces[1] });
        std::vector<Space::Ptr> remaining{ std::static_pointer_cast<Space>(pair) };
        remaining.insert(remaining.end(), spaces.begin() + 2, spaces.end());
        return from_largest_common_subspace(remaining, is_dual);
    }
    auto const& sp1 = *spaces[0];
    auto const& sp2 = *spaces[1];
    SectorArray sectors = SectorArray::empty(sp1.symmetry->sector_ind_len);
    std::vector<int64> mults;
    if (sp1.sector_order == "sorted" && sp2.sector_order == "sorted") {
        SectorArray::iter_common_sorted(
          sp1.sector_decomposition,
          sp2.sector_decomposition,
          /*a_strict=*/true,
          /*b_strict=*/true,
          [&](std::ptrdiff_t i, std::ptrdiff_t j) {
              sectors.push_back(sp1.sector_decomposition[static_cast<std::size_t>(i)]);
              mults.push_back(std::min(sp1.multiplicities[static_cast<std::size_t>(i)],
                                       sp2.multiplicities[static_cast<std::size_t>(j)]));
          });
    } else {
        // OPTIMIZE implementation for mixed orders?
        for (std::size_t i = 0; i < sp1.sector_decomposition.size(); ++i) {
            auto const& sector = sp1.sector_decomposition[i];
            auto const j = sp2.sector_decomposition_where(sector);
            if (!j) {
                continue;
            }
            sectors.push_back(sector);
            mults.push_back(
              std::min(sp1.multiplicities[i], sp2.multiplicities[static_cast<std::size_t>(*j)]));
        }
    }
    auto res = from_sector_decomposition(
      sp1.symmetry, std::move(sectors), std::move(mults), is_dual, std::nullopt, true);
    // from_sector_decomposition potentially introduces a meaningless basis_perm,
    // which we want to ignore here.
    // OPTIMIZE (JU) then dont compute it in the first place?
    res->Leg::set_basis_perm(std::nullopt);
    return res;
}

ElementarySpace::Ptr
ElementarySpace::from_null_space(Symmetry::Ptr symmetry, bool is_dual)
{
    auto sectors = symmetry->empty_sector_array;
    return std::make_shared<ElementarySpace>(
      std::move(symmetry), std::move(sectors), std::vector<int64>{}, is_dual, std::nullopt);
}

ElementarySpace::Ptr
ElementarySpace::from_defining_sectors(Symmetry::Ptr symmetry,
                                       SectorArray defining_sectors,
                                       std::optional<std::vector<int64>> multiplicities_,
                                       bool is_dual,
                                       std::optional<std::vector<int64>> basis_perm,
                                       bool unique_sectors,
                                       std::vector<std::size_t>* return_sorting_perm)
{
    std::vector<int64> multiplicities =
      multiplicities_.value_or(std::vector<int64>(defining_sectors.size(), 1));
    assert(multiplicities.size() == defining_sectors.size());

    // sort sectors
    std::vector<std::size_t> sort;
    if (symmetry->can_be_dropped()) {
        auto const num_states = num_states_per_sector(*symmetry, defining_sectors, multiplicities);
        auto const basis_slices = slice_boundaries(num_states);
        std::tie(defining_sectors, multiplicities, sort) =
          sort_sectors(defining_sectors, multiplicities);
        if (defining_sectors.size() == 0) {
            basis_perm = std::vector<int64>{};
        } else {
            if (!basis_perm) {
                basis_perm = arange(static_cast<std::size_t>(basis_slices.back()));
            }
            std::vector<int64> sorted_perm;
            sorted_perm.reserve(basis_perm->size());
            for (auto const i : sort) {
                for (auto k = basis_slices[i]; k < basis_slices[i + 1]; ++k) {
                    sorted_perm.push_back((*basis_perm)[static_cast<std::size_t>(k)]);
                }
            }
            basis_perm = std::move(sorted_perm);
        }
    } else {
        std::tie(defining_sectors, multiplicities, sort) =
          sort_sectors(defining_sectors, multiplicities);
        assert(!basis_perm);
    }
    // combine duplicate sectors (does not affect basis_perm)
    if (!unique_sectors) {
        auto const mult_slices = slice_boundaries(multiplicities);
        auto const diffs = defining_sectors.find_row_differences(/*include_len=*/true);
        // the convention is that for sectors with dim > 1, all copies of the first
        // state appear, then all copies of the second state, etc. At this point,
        // this order is not yet fully respected
        if (basis_perm && !symmetry->is_abelian()) {
            // updated basis_slices after sorting defining_sectors
            auto const num_states =
              num_states_per_sector(*symmetry, defining_sectors, multiplicities);
            auto const basis_slices = slice_boundaries(num_states);
            for (std::size_t i = 0; i + 1 < diffs.size(); ++i) {
                auto const sector_dim = symmetry->sector_dim(defining_sectors[diffs[i]]);
                if (sector_dim == 1) {
                    continue;
                }
                std::vector<int64> const mults(
                  multiplicities.begin() + static_cast<std::ptrdiff_t>(diffs[i]),
                  multiplicities.begin() + static_cast<std::ptrdiff_t>(diffs[i + 1]));
                std::vector<int64> offsets(mults.size() + 1, 0);
                for (std::size_t j = 0; j < mults.size(); ++j) {
                    offsets[j + 1] = offsets[j] + mults[j] * sector_dim;
                }
                auto const start = static_cast<std::size_t>(basis_slices[diffs[i]]);
                auto const stop = static_cast<std::size_t>(basis_slices[diffs[i + 1]]);
                // take the basis_perm associated with the first states and make them contiguous,
                // then go to the second state, etc.
                std::vector<int64> new_perm;
                new_perm.reserve(stop - start);
                for (int64 k = 0; k < sector_dim; ++k) {
                    for (std::size_t j = 0; j < mults.size(); ++j) {
                        auto const mult = mults[j];
                        for (int64 t = 0; t < mult; ++t) {
                            new_perm.push_back(
                              (*basis_perm)[start +
                                            static_cast<std::size_t>(offsets[j] + k * mult + t)]);
                        }
                    }
                }
                assert(new_perm.size() == stop - start);
                std::ranges::copy(new_perm,
                                  basis_perm->begin() + static_cast<std::ptrdiff_t>(start));
            }
        }
        std::vector<int64> unique_mults(diffs.size() - 1);
        for (std::size_t i = 0; i + 1 < diffs.size(); ++i) {
            unique_mults[i] = mult_slices[diffs[i + 1]] - mult_slices[diffs[i]];
        }
        // [:-1] to exclude len
        defining_sectors =
          defining_sectors.take(std::span<const std::size_t>(diffs.data(), diffs.size() - 1));
        multiplicities = std::move(unique_mults);
    }
    auto res = std::make_shared<ElementarySpace>(std::move(symmetry),
                                                 std::move(defining_sectors),
                                                 std::move(multiplicities),
                                                 is_dual,
                                                 std::move(basis_perm));
    if (return_sorting_perm != nullptr) {
        *return_sorting_perm = std::move(sort);
    }
    return res;
}

ElementarySpace::Ptr
ElementarySpace::from_sector_decomposition(Symmetry::Ptr symmetry,
                                           SectorArray sector_decomposition,
                                           std::optional<std::vector<int64>> multiplicities,
                                           bool is_dual,
                                           std::optional<std::vector<int64>> basis_perm,
                                           bool unique_sectors)
{
    auto defining_sectors =
      is_dual ? symmetry->dual_sectors(sector_decomposition) : std::move(sector_decomposition);
    return from_defining_sectors(std::move(symmetry),
                                 std::move(defining_sectors),
                                 std::move(multiplicities),
                                 is_dual,
                                 std::move(basis_perm),
                                 unique_sectors);
}

ElementarySpace::Ptr
ElementarySpace::from_trivial_sector(int64 dim,
                                     Symmetry::Ptr symmetry,
                                     bool is_dual,
                                     std::optional<std::vector<int64>> basis_perm)
{
    if (!symmetry) {
        symmetry = no_symmetry_product();
    }
    if (dim == 0) {
        return from_null_space(std::move(symmetry), is_dual);
    }
    auto sectors = SectorArray::from_sector(symmetry->trivial_sector);
    return std::make_shared<ElementarySpace>(std::move(symmetry),
                                             std::move(sectors),
                                             std::vector<int64>{ dim },
                                             is_dual,
                                             std::move(basis_perm));
}

ElementarySpace::Ptr
ElementarySpace::shared_es() const
{
    return std::const_pointer_cast<ElementarySpace>(
      std::dynamic_pointer_cast<const ElementarySpace>(shared_from_this()));
}

SectorArray
ElementarySpace::sectors_of_basis() const
{
    if (!Space::symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("sectors_of_basis is meaningless for {}.", Space::symmetry->str()));
    }
    // build in internal basis, then permute
    SectorArray res(dim_as_size(Space::dim), Space::symmetry->sector_ind_len);
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_sectors); ++i) {
        auto const& sector = sector_decomposition[i];
        for (auto k = (*slices)[i][0]; k < (*slices)[i][1]; ++k) {
            res[static_cast<std::size_t>(k)] = sector;
        }
    }
    if (!_inverse_basis_perm) {
        return res;
    }
    std::vector<std::size_t> perm(_inverse_basis_perm->begin(), _inverse_basis_perm->end());
    return res.take(perm);
}

std::string
ElementarySpace::repr(bool show_symmetry, bool one_line) const
{
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "ElementarySpace";

    struct Options
    {
        bool full_sectors;
        bool summarized_sectors;
        bool symmetry;
    };
    // try to show everything, then less and less
    std::array<Options, 4> const options{ { { true, false, show_symmetry },
                                            { false, true, show_symmetry },
                                            { false, false, show_symmetry },
                                            { false, false, false } } };
    for (auto const& opt : options) {
        if (opt.full_sectors && 3 * static_cast<int64>(defining_sectors.size()) *
                                    static_cast<int64>(defining_sectors.sector_ind_len()) >
                                  linewidth) {
            // there is no chance to print all sectors in one line
            continue;
        }

        std::vector<std::string> items;
        if (opt.symmetry) {
            items.push_back(std::format("symmetry={}", Space::symmetry->repr()));
        }
        if (opt.full_sectors) {
            py::list def_sector_strs;
            for (auto const& a : defining_sectors) {
                def_sector_strs.append(Space::symmetry->sector_str(a));
            }
            py::list sector_dec_strs;
            for (auto const& a : sector_decomposition) {
                sector_dec_strs.append(Space::symmetry->sector_str(a));
            }
            items.push_back(std::format("defining_sectors={}", format_like_list(def_sector_strs)));
            items.push_back(
              std::format("sector_decomposition={}", format_like_list(sector_dec_strs)));
            items.push_back(
              std::format("multiplicities={}", format_like_list(py::cast(multiplicities))));
            if (_basis_perm) {
                items.push_back(
                  std::format("basis_perm={}", format_like_list(py::cast(*_basis_perm))));
            }
        }
        if (opt.summarized_sectors) {
            items.push_back(std::format("num_sectors={}", num_sectors));
            if (_basis_perm) {
                items.emplace_back("basis_perm=[...]");
            }
        }
        items.push_back(std::format("is_dual={}", bool_repr(is_dual)));

        // try one line
        std::string res = ClsName + "(";
        for (std::size_t i = 0; i < items.size(); ++i) {
            if (i > 0) {
                res += ", ";
            }
            res += items[i];
        }
        res += ")";
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }

        if (!one_line) {
            // try multi line
            bool const maxlines_ok = static_cast<int64>(items.size()) + 2 <= maxlines;
            bool const linewidth_ok = std::ranges::all_of(items, [&](std::string const& item) {
                return static_cast<int64>(indent.size() + item.size() + 1) < linewidth;
            });
            if (maxlines_ok && linewidth_ok) {
                std::string out = ClsName + "(\n";
                for (auto const& item : items) {
                    out += indent + indent + item + ",\n";
                }
                out += ")";
                return out;
            }
        }
    }
    // one of the above returns should have triggered
    throw std::runtime_error("ElementarySpace repr: no suitable format found");
}

bool
ElementarySpace::operator==(Leg const& other) const
{
    auto const* o = dynamic_cast<ElementarySpace const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return equals_es(*o);
}

bool
ElementarySpace::operator==(Space const& other) const
{
    auto const* o = dynamic_cast<ElementarySpace const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return equals_es(*o);
}

bool
ElementarySpace::equals_es(ElementarySpace const& other) const
{
    if (is_dual != other.is_dual) {
        return false;
    }
    if (!Space::symmetry->equals(*other.Space::symmetry)) {
        return false;
    }
    // check this first to safely compare later
    if (num_sectors != other.num_sectors) {
        return false;
    }
    if (multiplicities != other.multiplicities) {
        return false;
    }
    if (!(defining_sectors == other.defining_sectors)) {
        return false;
    }
    if (_basis_perm || other._basis_perm) {
        if (basis_perm() != other.basis_perm()) {
            return false;
        }
    }
    // else: both permutations are trivial, thus equal
    return true;
}

py::object
ElementarySpace::as_ElementarySpace(bool is_dual_)
{
    if (is_dual_ == is_dual) {
        return py::cast(shared_es());
    }
    return py::cast(with_opposite_duality());
}

ElementarySpace::Ptr
ElementarySpace::as_ket_space()
{
    if (!is_dual) {
        return shared_es();
    }
    return with_opposite_duality();
}

ElementarySpace::Ptr
ElementarySpace::as_bra_space()
{
    if (is_dual) {
        return shared_es();
    }
    return with_opposite_duality();
}

py::object
ElementarySpace::change_symmetry(Symmetry::Ptr symmetry, SectorMapFn sector_map, bool injective)
{
    return py::cast(from_defining_sectors(std::move(symmetry),
                                          sector_map(defining_sectors),
                                          multiplicities,
                                          is_dual,
                                          _basis_perm,
                                          injective));
}

ElementarySpace::Ptr
ElementarySpace::direct_sum(std::vector<Ptr> const& others) const
{
    if (others.empty()) {
        return shared_es();
    }
    assert(std::ranges::all_of(
      others, [this](Ptr const& o) { return o->Space::symmetry->equals(*Space::symmetry); }));
    assert(std::ranges::all_of(others, [this](Ptr const& o) { return o->is_dual == is_dual; }));
    std::optional<std::vector<int64>> basis_perm_;
    if (Space::symmetry->can_be_dropped()) {
        auto perm = basis_perm();
        auto offset = static_cast<int64>(Space::dim);
        for (auto const& other : others) {
            for (auto const idx : other->basis_perm()) {
                perm.push_back(idx + offset);
            }
            offset += static_cast<int64>(other->Space::dim);
        }
        basis_perm_ = std::move(perm);
    }
    auto sectors = defining_sectors;
    auto mults = multiplicities;
    for (auto const& other : others) {
        sectors = sectors.concat(other->defining_sectors);
        mults.insert(mults.end(), other->multiplicities.begin(), other->multiplicities.end());
    }
    return from_defining_sectors(
      Space::symmetry, std::move(sectors), std::move(mults), is_dual, std::move(basis_perm_));
}

py::object
ElementarySpace::drop_symmetry(std::optional<std::vector<int64>> which)
{
    auto const [which_factors, remaining_symmetry] =
      parse_inputs_drop_symmetry(which, *Space::symmetry);
    if (!which_factors) {
        return py::cast(from_trivial_sector(
          static_cast<int64>(Space::dim), remaining_symmetry, is_dual, _basis_perm));
    }
    // the sector components that are kept
    std::vector<bool> mask(Space::symmetry->sector_ind_len, true);
    for (auto const i : *which_factors) {
        auto const idx = static_cast<std::size_t>(i);
        for (auto k = Space::symmetry->sector_slices[idx];
             k < Space::symmetry->sector_slices[idx + 1];
             ++k) {
            mask[k] = false;
        }
    }
    std::vector<std::size_t> keep;
    for (std::size_t k = 0; k < mask.size(); ++k) {
        if (mask[k]) {
            keep.push_back(k);
        }
    }
    SectorMapFn sector_map = [keep](SectorArray const& sectors) {
        SectorArray res(sectors.size(), static_cast<std::uint8_t>(keep.size()));
        for (std::size_t i = 0; i < sectors.size(); ++i) {
            auto sector = Sector::zeros(static_cast<std::uint8_t>(keep.size()));
            for (std::size_t k = 0; k < keep.size(); ++k) {
                sector[k] = sectors[i][keep[k]];
            }
            res[i] = sector;
        }
        return res;
    };
    return change_symmetry(remaining_symmetry, std::move(sector_map));
}

Space::Ptr
ElementarySpace::dual_space() const
{
    return dual_es();
}

Leg::Ptr
ElementarySpace::dual_leg() const
{
    return dual_es();
}

ElementarySpace::Ptr
ElementarySpace::dual_es() const
{
    return std::make_shared<ElementarySpace>(
      Space::symmetry, defining_sectors, multiplicities, !is_dual, _basis_perm);
}

std::pair<int64, int64>
ElementarySpace::parse_index(int64 idx) const
{
    if (!Space::symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("parse_index is meaningless for {}.", Space::symmetry->str()));
    }
    idx = to_valid_idx(idx, static_cast<int64>(Space::dim));
    if (_inverse_basis_perm) {
        idx = (*_inverse_basis_perm)[static_cast<std::size_t>(idx)];
    }
    // bisect the (increasing) starts of the slices
    auto const& sl = *slices;
    std::size_t lo = 0;
    std::size_t hi = sl.size();
    while (lo < hi) {
        auto const mid = lo + (hi - lo) / 2;
        if (sl[mid][0] <= idx) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    auto const sector_idx = static_cast<int64>(lo) - 1;
    assert(sector_idx >= 0);
    auto const multiplicity_idx = idx - sl[static_cast<std::size_t>(sector_idx)][0];
    return { sector_idx, multiplicity_idx };
}

Sector
ElementarySpace::idx_to_sector(int64 idx) const
{
    auto const [sector_idx, _] = parse_index(idx);
    return sector_decomposition[static_cast<std::size_t>(sector_idx)];
}

ElementarySpace::Ptr
ElementarySpace::take_slice(py::array blockmask) const
{
    if (!Space::symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("take_slice is meaningless for {}.", Space::symmetry->str()));
    }
    auto casted = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(blockmask);
    if (!casted || casted.ndim() != 1) {
        throw py::type_error("blockmask must be a 1D array of bool");
    }
    auto const public_mask = casted.unchecked<1>();
    auto const num_basis_states = dim_as_size(Space::dim);
    if (static_cast<std::size_t>(public_mask.shape(0)) != num_basis_states) {
        throw std::invalid_argument("blockmask has wrong length");
    }
    // note: mask is in the internal basis order from here on, i.e. we applied the basis_perm.
    std::vector<bool> mask(num_basis_states);
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        auto const public_idx = _basis_perm ? (*_basis_perm)[i] : static_cast<int64>(i);
        mask[i] = public_mask(static_cast<py::ssize_t>(public_idx));
    }
    SectorArray sectors = SectorArray::empty(Space::symmetry->sector_ind_len);
    std::vector<int64> mults;
    for (std::size_t i = 0; i < static_cast<std::size_t>(num_sectors); ++i) {
        auto const d_a = (*sector_dims)[i];
        auto const [start, stop] = (*slices)[i];
        int64 num_kept = 0;
        for (auto k = start; k < stop; k += d_a) {
            // multiplets need to be kept or discarded as a whole
            bool const keep = mask[static_cast<std::size_t>(k)];
            for (int64 t = 1; t < d_a; ++t) {
                if (mask[static_cast<std::size_t>(k + t)] != keep) {
                    throw std::invalid_argument(
                      "Multiplets need to be kept or discarded as a whole.");
                }
            }
            if (keep) {
                num_kept += d_a;
            }
        }
        auto const mult = num_kept / d_a;
        if (mult > 0) {
            sectors.push_back(defining_sectors[i]);
            mults.push_back(mult);
        }
    }
    // build basis_perm for small leg.
    // it is determined by demanding
    //    a) that the following diagram commutes
    //
    //        (self, public) ---- self.basis_perm ---->  (self, internal)
    //         |                                           |
    //         v public_blockmask                          v projection_internal
    //         |                                           |
    //        (res, public) ----- small_leg_perm ----->  (res, internal)
    //
    //    b) that projection_internal is also just a mask (i.e it preserves ordering)
    //       which is given by public_blockmask[self.basis_perm]
    //
    // this allows us to internally (e.g. in the abelian backend) store only 1D boolean masks
    // as blocks.
    //
    // note mask is in the private basis order.
    auto const perm = basis_perm();
    std::vector<int64> kept_perm;
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        if (mask[i]) {
            kept_perm.push_back(perm[i]);
        }
    }
    return std::make_shared<ElementarySpace>(
      Space::symmetry, std::move(sectors), std::move(mults), is_dual, rank_data(kept_perm));
}

ElementarySpace::Ptr
ElementarySpace::with_opposite_duality() const
{
    SectorArray dual_defining_sectors;
    if (is_dual) {
        // already have the symmetry->dual_sectors(defining_sectors)
        dual_defining_sectors = sector_decomposition;
    } else {
        dual_defining_sectors = Space::symmetry->dual_sectors(defining_sectors);
    }
    // note: dual_defining_sectors are not sorted, but they are unique.
    return from_defining_sectors(Space::symmetry,
                                 std::move(dual_defining_sectors),
                                 multiplicities,
                                 !is_dual,
                                 _basis_perm,
                                 /*unique_sectors=*/true);
}

ElementarySpace::Ptr
ElementarySpace::with_is_dual(bool is_dual_) const
{
    if (is_dual_ == is_dual) {
        return shared_es();
    }
    return with_opposite_duality();
}

py::object
ElementarySpace::as_Space()
{
    return py::cast(shared_es());
}

bool
ElementarySpace::is_trivial() const
{
    return Space::is_trivial();
}

std::string
ElementarySpace::ascii_arrow() const
{
    return is_dual ? "^" : "v";
}

void
ElementarySpace::save_hdf5(py::object hdf5_saver,
                           py::object h5gr,
                           std::string const& subpath) const
{
    auto save = hdf5_saver.attr("save");
    save(py::cast(defining_sectors), subpath + "defining_sectors");
    save(py::cast(sector_decomposition), subpath + "sector_decomposition");
    save(sector_order ? py::cast(*sector_order) : py::none(), subpath + "sector_order");
    save(optional_perm_to_py(_basis_perm), subpath + "_basis_perm");
    save(optional_perm_to_py(_inverse_basis_perm), subpath + "_inverse_basis_perm");
    save(vector_to_array(multiplicities), subpath + "multiplicities");
    save(py::cast(Space::symmetry), subpath + "symmetry");
    save(py::int_(static_cast<long long>(Space::dim)), subpath + "dim");
    save(py::int_(num_sectors), subpath + "num_sectors");
    save(slices_to_py(slices), subpath + "slices");
    save(sector_dims ? py::object(vector_to_array(*sector_dims)) : py::none(),
         subpath + "sector_dims");

    h5gr.attr("attrs")["is_dual"] = is_dual;
}

ElementarySpace::Ptr
ElementarySpace::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto load = hdf5_loader.attr("load");
    auto symmetry = load(subpath + "symmetry").cast<Symmetry::Ptr>();
    auto defining_sectors = load(subpath + "defining_sectors").cast<SectorArray>();
    auto multiplicities = py_array_to_i64(py::array::ensure(load(subpath + "multiplicities")));
    auto basis_perm = optional_perm_from_py(load(subpath + "_basis_perm"));
    auto const is_dual = hdf5_loader.attr("get_attr")(h5gr, "is_dual").cast<bool>();
    auto obj = std::make_shared<ElementarySpace>(std::move(symmetry),
                                                 std::move(defining_sectors),
                                                 std::move(multiplicities),
                                                 is_dual,
                                                 std::move(basis_perm));
    py::object py_obj = py::cast(obj);
    hdf5_loader.attr("memorize_load")(h5gr, py_obj);
    return obj;
}

namespace {

/// The symmetry of a :class:`TensorProduct` factor, i.e. of a :class:`Space` or :class:`Leg`.
[[nodiscard]] Symmetry::Ptr
factor_symmetry(py::handle factor)
{
    if (py::isinstance<Space>(factor)) {
        return factor.cast<Space*>()->symmetry;
    }
    if (py::isinstance<Leg>(factor)) {
        return factor.cast<Leg*>()->symmetry;
    }
    return factor.attr("symmetry").cast<Symmetry::Ptr>();
}

/// ``factor.flat_spaces`` if `spaces`, else ``factor.flat_legs``.
[[nodiscard]] std::vector<Leg::Ptr>
factor_flat(py::handle factor, bool spaces)
{
    if (py::isinstance<TensorProduct>(factor)) {
        auto const* product = factor.cast<TensorProduct const*>();
        return spaces ? product->flat_spaces() : product->flat_legs();
    }
    if (py::isinstance<Leg>(factor)) {
        auto leg = factor.cast<Leg::Ptr>();
        return spaces ? leg->flat_spaces() : leg->flat_legs();
    }
    // pure Python factor, e.g. an AbelianLegPipe: the flattened parts must still be C++ Legs
    std::vector<Leg::Ptr> out;
    for (py::handle item : factor.attr(spaces ? "flat_spaces" : "flat_legs")) {
        out.push_back(item.cast<Leg::Ptr>());
    }
    return out;
}

[[nodiscard]] int64
factor_num_flat_legs(py::handle factor)
{
    if (py::isinstance<TensorProduct>(factor)) {
        return factor.cast<TensorProduct const*>()->num_flat_legs();
    }
    if (py::isinstance<Leg>(factor)) {
        return factor.cast<Leg*>()->num_flat_legs();
    }
    return factor.attr("num_flat_legs").cast<int64>();
}

/// The :class:`Space` described by a (already flattened) leg.
[[nodiscard]] Space::Ptr
leg_as_space(Leg::Ptr const& leg)
{
    if (auto space = std::dynamic_pointer_cast<Space>(leg)) {
        return space;
    }
    return leg->as_Space().cast<Space::Ptr>();
}

/// ``factor.change_symmetry(symmetry, sector_map, injective)``.
[[nodiscard]] py::object
factor_change_symmetry(py::handle factor,
                       Symmetry::Ptr const& symmetry,
                       SectorMapFn const& sector_map,
                       bool injective)
{
    if (py::isinstance<Space>(factor)) {
        return factor.cast<Space*>()->change_symmetry(symmetry, sector_map, injective);
    }
    auto py_sector_map = py::cpp_function([sector_map](py::object sectors) {
        return py::cast(sector_map(sectors.cast<SectorArray>()));
    });
    return factor.attr("change_symmetry")(symmetry, py_sector_map, injective);
}

/// ``factor.drop_symmetry(which)``, where ``nullopt`` means ``'all'``.
[[nodiscard]] py::object
factor_drop_symmetry(py::handle factor, std::optional<std::vector<int64>> const& which)
{
    if (py::isinstance<Space>(factor)) {
        return factor.cast<Space*>()->drop_symmetry(which);
    }
    py::object which_obj = which ? py::cast(*which) : py::object(py::str("all"));
    return factor.attr("drop_symmetry")(which_obj);
}

/// ``factor.__repr__(show_symmetry=..., one_line=...)``, with fallbacks.
[[nodiscard]] std::string
factor_repr(py::handle factor, bool show_symmetry, bool one_line)
{
    // the C++ classes bind the parametrized version as ``repr``, the Python ones as ``__repr__``
    for (char const* name : { "repr", "__repr__" }) {
        if (!py::hasattr(factor, name)) {
            continue;
        }
        try {
            return py::str(factor.attr(name)(py::arg("show_symmetry") = show_symmetry,
                                             py::arg("one_line") = one_line));
        } catch (py::error_already_set&) {
            // the attribute does not accept these arguments; fall through
        }
    }
    return py::str(factor.attr("__repr__")());
}

/// ``np.prod(values)``, i.e. ``1`` for an empty input.
[[nodiscard]] int64
product(std::vector<int64> const& values)
{
    int64 res = 1;
    for (auto const v : values) {
        res *= v;
    }
    return res;
}

/// ``all(a == b for a, b in zip(sectors, other))``, i.e. ignoring surplus entries.
[[nodiscard]] bool
sectors_match(SectorArray const& sectors, SectorArray const& other)
{
    auto const n = std::min(sectors.size(), other.size());
    for (std::size_t i = 0; i < n; ++i) {
        if (!(sectors[i] == other[i])) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string
join(std::vector<std::string> const& parts, std::string const& sep)
{
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            out += sep;
        }
        out += parts[i];
    }
    return out;
}

[[nodiscard]] py::object
dim_to_py(float64 dim)
{
    if (std::floor(dim) == dim) {
        return py::int_(static_cast<long long>(dim));
    }
    return py::float_(dim);
}

/// ``TensorProduct._calc_sectors``, for factors that are already flattened to spaces.
[[nodiscard]] std::pair<SectorArray, std::vector<int64>>
calc_sectors_of_spaces(Symmetry const& symmetry, std::span<const Space::Ptr> spaces)
{
    if (spaces.empty()) {
        return { SectorArray::from_sector(symmetry.trivial_sector), std::vector<int64>{ 1 } };
    }

    if (spaces.size() == 1) {
        auto const& space = *spaces.front();
        if (space.sector_order == "sorted") {
            return { space.sector_decomposition, space.multiplicities };
        }
        auto const perm = space.sector_decomposition.lexsort_indices();
        return { space.sector_decomposition.take(perm),
                 gather_or_all(space.multiplicities, perm) };
    }

    if (symmetry.is_abelian()) {
        // all combinations of one sector per space, ordered like ``make_grid(_, cstyle=False)``
        std::vector<std::size_t> num_sectors(spaces.size());
        std::size_t num_combinations = 1;
        for (std::size_t n = 0; n < spaces.size(); ++n) {
            num_sectors[n] = static_cast<std::size_t>(spaces[n]->num_sectors);
            num_combinations *= num_sectors[n];
        }
        std::vector<SectorArray> uncoupled;
        uncoupled.reserve(spaces.size());
        std::vector<int64> multiplicities(num_combinations, 1);
        std::size_t stride = 1;
        for (std::size_t n = 0; n < spaces.size(); ++n) {
            SectorArray column(num_combinations, symmetry.sector_ind_len);
            for (std::size_t m = 0; m < num_combinations; ++m) {
                auto const i = (m / stride) % num_sectors[n];
                column[m] = spaces[n]->sector_decomposition[i];
                multiplicities[m] *= spaces[n]->multiplicities[i];
            }
            uncoupled.push_back(std::move(column));
            stride *= num_sectors[n];
        }
        auto const sectors = symmetry.multiple_fusion_broadcast(uncoupled);
        auto [unique, mults, perm] = sectors.unique_sorted(multiplicities);
        (void)perm;
        return { std::move(unique), std::move(mults) };
    }

    // define recursively
    auto const [sectors, mults] =
      calc_sectors_of_spaces(symmetry, spaces.first(spaces.size() - 1));
    auto const& last = *spaces.back();
    SectorArray combined = SectorArray::empty(symmetry.sector_ind_len);
    std::vector<int64> combined_mults;
    for (std::size_t j = 0; j < last.sector_decomposition.size(); ++j) {
        auto const s2 = last.sector_decomposition[j];
        auto const m2 = last.multiplicities[j];
        for (std::size_t i = 0; i < sectors.size(); ++i) {
            auto const s1 = sectors[i];
            auto const m12 = mults[i] * m2;
            for (auto const& c : symmetry.fusion_outcomes(s1, s2)) {
                combined.push_back(c);
                // OPTIMIZE support batched N symbol?
                combined_mults.push_back(
                  symmetry.has_unique_fusion() ? m12 : m12 * symmetry._n_symbol(s1, s2, c));
            }
        }
    }
    auto [unique, unique_mults, perm] = combined.unique_sorted(combined_mults);
    (void)perm;
    return { std::move(unique), std::move(unique_mults) };
}

/// ``TensorProduct._calc_sectors``.
[[nodiscard]] std::pair<SectorArray, std::vector<int64>>
calc_sectors_of_factors(Symmetry const& symmetry, std::vector<py::object> const& factors)
{
    // LegPipes do not have sectors -> flatten them for the purpose of calculating sectors
    std::vector<Space::Ptr> spaces;
    for (auto const& factor : factors) {
        for (auto const& leg : factor_flat(factor, /*spaces=*/true)) {
            // need the sector decomposition of each factor. easiest way: convert to Space
            // OPTIMIZE is this optimal? should we store the as_Space() for later use?
            spaces.push_back(leg_as_space(leg));
        }
    }
    return calc_sectors_of_spaces(symmetry, spaces);
}

} // namespace

TensorProduct::Prepared
TensorProduct::prepare(std::vector<py::object> const& factors,
                       Symmetry::Ptr symmetry,
                       std::optional<SectorArray> sector_decomposition,
                       std::optional<std::vector<int64>> multiplicities)
{
    if (!symmetry) {
        if (factors.empty()) {
            throw std::invalid_argument("If spaces is empty, the symmetry arg is required.");
        }
        symmetry = factor_symmetry(factors.front());
    }
    for (auto const& factor : factors) {
        if (!factor_symmetry(factor)->equals(*symmetry)) {
            throw SymmetryError("Incompatible symmetries.");
        }
    }
    if (!sector_decomposition || !multiplicities) {
        if (sector_decomposition || multiplicities) {
            PyErr_WarnEx(PyExc_UserWarning,
                         "Need both _sectors and _multiplicities to skip recomputation. "
                         "Got just one.",
                         1);
        }
        auto [sectors, mults] = calc_sectors_of_factors(*symmetry, factors);
        sector_decomposition = std::move(sectors);
        multiplicities = std::move(mults);
    }
    return { std::move(symmetry), std::move(*sector_decomposition), std::move(*multiplicities) };
}

TensorProduct::TensorProduct(std::vector<py::object> factors_, Prepared prepared)
  : Space(std::move(prepared.symmetry),
          std::move(prepared.sector_decomposition),
          std::move(prepared.multiplicities),
          "sorted")
  , factors(std::move(factors_))
  , num_factors(static_cast<int64>(factors.size()))
{
}

TensorProduct::TensorProduct(std::vector<py::object> factors_,
                             Symmetry::Ptr symmetry_,
                             std::optional<SectorArray> sector_decomposition_,
                             std::optional<std::vector<int64>> multiplicities_)
  : TensorProduct(factors_,
                  prepare(factors_,
                          std::move(symmetry_),
                          std::move(sector_decomposition_),
                          std::move(multiplicities_)))
{
}

void
TensorProduct::test_sanity() const
{
    assert(static_cast<int64>(factors.size()) == num_factors);
    for (auto const& factor : factors) {
        factor.attr("test_sanity")();
    }
    Space::test_sanity();
}

TensorProduct::Ptr
TensorProduct::from_partial_products(std::vector<Ptr> const& factors)
{
    if (factors.empty()) {
        throw std::invalid_argument("Need at least one TensorProduct");
    }
    auto spaces = factors.front()->factors;
    auto symmetry = factors.front()->symmetry;
    std::vector<py::object> partial;
    partial.reserve(factors.size());
    partial.push_back(py::cast(factors.front()));
    for (std::size_t i = 1; i < factors.size(); ++i) {
        spaces.insert(spaces.end(), factors[i]->factors.begin(), factors[i]->factors.end());
        if (!factors[i]->symmetry->equals(*symmetry)) {
            throw SymmetryError("Mismatched symmetries");
        }
        partial.push_back(py::cast(factors[i]));
    }
    // forming isomorphic performs the fusion on the partially fused factors
    auto const isomorphic = std::make_shared<TensorProduct>(std::move(partial), symmetry);
    return std::make_shared<TensorProduct>(std::move(spaces),
                                           std::move(symmetry),
                                           isomorphic->sector_decomposition,
                                           isomorphic->multiplicities);
}

Space::Ptr
TensorProduct::dual_space() const
{
    auto const dual = symmetry->dual_sectors(sector_decomposition);
    auto [sectors, mults, perm] = sort_sectors(dual, multiplicities);
    (void)perm;
    std::vector<py::object> dual_factors;
    dual_factors.reserve(factors.size());
    for (auto it = factors.rbegin(); it != factors.rend(); ++it) {
        dual_factors.push_back(it->attr("dual"));
    }
    return std::make_shared<TensorProduct>(
      std::move(dual_factors), symmetry, std::move(sectors), std::move(mults));
}

int64
TensorProduct::block_size(std::variant<int64, Sector> coupled) const
{
    if (auto const* idx = std::get_if<int64>(&coupled)) {
        return multiplicities[static_cast<std::size_t>(to_valid_idx(*idx, num_sectors))];
    }
    return sector_multiplicity(std::get<Sector>(coupled));
}

py::object
TensorProduct::change_symmetry(Symmetry::Ptr symmetry_, SectorMapFn sector_map, bool injective)
{
    auto sectors = sector_map(sector_decomposition);
    auto mults = multiplicities;
    std::vector<std::size_t> perm;
    if (!injective) {
        std::tie(sectors, mults, perm) = sectors.unique_sorted(mults);
    } else {
        std::tie(sectors, mults, perm) = sort_sectors(sectors, mults);
    }
    std::vector<py::object> new_factors;
    new_factors.reserve(factors.size());
    for (auto const& factor : factors) {
        new_factors.push_back(factor_change_symmetry(factor, symmetry_, sector_map, injective));
    }
    // note: unlike the Python version, which passes ``self.symmetry``, we pass the *new*
    // symmetry here. Otherwise the constructor rejects the new factors.
    return py::cast(std::make_shared<TensorProduct>(
      std::move(new_factors), std::move(symmetry_), std::move(sectors), std::move(mults)));
}

py::object
TensorProduct::drop_symmetry(std::optional<std::vector<int64>> which)
{
    auto const [which_factors, remaining_symmetry] = parse_inputs_drop_symmetry(which, *symmetry);
    SectorArray sectors;
    std::vector<int64> mults;
    if (!which_factors) {
        // note: unlike the Python version, we use the trivial sector of the *remaining*
        // symmetry, which is the only one with the right sector_ind_len.
        sectors = SectorArray::from_sector(remaining_symmetry->trivial_sector);
        mults = { static_cast<int64>(dim) };
    } else {
        // the sector components that are kept
        std::vector<bool> mask(symmetry->sector_ind_len, true);
        for (auto const i : *which_factors) {
            auto const idx = static_cast<std::size_t>(i);
            for (auto k = symmetry->sector_slices[idx]; k < symmetry->sector_slices[idx + 1];
                 ++k) {
                mask[k] = false;
            }
        }
        std::vector<std::size_t> keep;
        for (std::size_t k = 0; k < mask.size(); ++k) {
            if (mask[k]) {
                keep.push_back(k);
            }
        }
        SectorArray kept(sector_decomposition.size(), static_cast<std::uint8_t>(keep.size()));
        for (std::size_t i = 0; i < sector_decomposition.size(); ++i) {
            auto sector = Sector::zeros(static_cast<std::uint8_t>(keep.size()));
            for (std::size_t k = 0; k < keep.size(); ++k) {
                sector[k] = sector_decomposition[i][keep[k]];
            }
            kept[i] = sector;
        }
        std::vector<std::size_t> perm;
        std::tie(sectors, mults, perm) = kept.unique_sorted(multiplicities);
    }
    std::vector<py::object> new_factors;
    new_factors.reserve(factors.size());
    for (auto const& factor : factors) {
        new_factors.push_back(factor_drop_symmetry(factor, which_factors));
    }
    return py::cast(std::make_shared<TensorProduct>(
      std::move(new_factors), remaining_symmetry, std::move(sectors), std::move(mults)));
}

bool
TensorProduct::has_pipes() const
{
    return std::ranges::any_of(factors,
                               [](py::object const& f) { return py::isinstance<LegPipe>(f); });
}

std::vector<Leg::Ptr>
TensorProduct::flat_legs() const
{
    std::vector<Leg::Ptr> out;
    for (auto const& factor : factors) {
        auto part = factor_flat(factor, /*spaces=*/false);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

std::vector<Leg::Ptr>
TensorProduct::flat_spaces() const
{
    std::vector<Leg::Ptr> out;
    for (auto const& factor : factors) {
        auto part = factor_flat(factor, /*spaces=*/true);
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

int64
TensorProduct::num_flat_legs() const
{
    int64 n = 0;
    for (auto const& factor : factors) {
        n += factor_num_flat_legs(factor);
    }
    return n;
}

std::vector<std::vector<int64>>
TensorProduct::flat_legs_nesting() const
{
    int64 i = 0;
    std::vector<std::vector<int64>> res;
    res.reserve(factors.size());
    for (auto const& factor : factors) {
        auto const num = factor_num_flat_legs(factor);
        std::vector<int64> idcs(static_cast<std::size_t>(num));
        std::iota(idcs.begin(), idcs.end(), i);
        res.push_back(std::move(idcs));
        i += num;
    }
    return res;
}

std::vector<int64>
TensorProduct::flat_leg_idcs(int64 i) const
{
    i = to_valid_idx(i, num_factors);
    int64 start = 0;
    for (int64 k = 0; k < i; ++k) {
        start += factor_num_flat_legs(factors[static_cast<std::size_t>(k)]);
    }
    auto const num = factor_num_flat_legs(factors[static_cast<std::size_t>(i)]);
    std::vector<int64> res(static_cast<std::size_t>(num));
    std::iota(res.begin(), res.end(), start);
    return res;
}

int64
TensorProduct::forest_block_size(SectorArray const& uncoupled, Sector coupled) const
{
    // OPTIMIZE ?
    auto const num_trees = static_cast<int64>(fusion_trees(symmetry, uncoupled, coupled).size());
    return num_trees * tree_block_size(uncoupled);
}

IndexSlice
TensorProduct::forest_block_slice(SectorArray const& uncoupled, Sector coupled) const
{
    int64 offset = 0;
    bool found = false;
    for (auto const& item : iter_uncoupled()) {
        if (sectors_match(item.uncoupled, uncoupled)) {
            found = true;
            break;
        }
        auto const tree_block = product(item.multiplicities);
        auto const num_trees =
          static_cast<int64>(fusion_trees(symmetry, item.uncoupled, coupled).size());
        offset += num_trees * tree_block;
    }
    if (!found) {
        throw std::invalid_argument("Uncoupled sectors incompatible");
    }
    auto const size = forest_block_size(uncoupled, coupled);
    return { offset, offset + size };
}

TensorProduct::Ptr
TensorProduct::insert_multiply(py::object other, int64 pos) const
{
    auto const self_ptr = std::const_pointer_cast<TensorProduct>(
      std::dynamic_pointer_cast<TensorProduct const>(shared_from_this()));
    auto const isomorphic =
      std::make_shared<TensorProduct>(std::vector<py::object>{ py::cast(self_ptr), other });
    // Python uses list slicing, i.e. ``factors[:pos] + [other] + factors[pos:]``.
    // In particular, ``pos == -1`` (as used by right_multiply) inserts before the last factor.
    auto const n = static_cast<int64>(factors.size());
    auto const at =
      static_cast<std::ptrdiff_t>(pos < 0 ? std::max<int64>(0, n + pos) : std::min(pos, n));
    std::vector<py::object> new_factors;
    new_factors.reserve(factors.size() + 1);
    new_factors.insert(new_factors.end(), factors.begin(), factors.begin() + at);
    new_factors.push_back(std::move(other));
    new_factors.insert(new_factors.end(), factors.begin() + at, factors.end());
    return std::make_shared<TensorProduct>(std::move(new_factors),
                                           symmetry,
                                           isomorphic->sector_decomposition,
                                           isomorphic->multiplicities);
}

std::vector<TreeBlockItem>
TensorProduct::iter_tree_blocks(SectorArray const& coupled) const
{
    // OPTIMIZE some users in FTBackend ignore some of the yielded values.
    //          is that ok performance wise or should we have special case iterators?
    std::vector<std::uint8_t> are_dual;
    for (auto const& leg : flat_legs()) {
        are_dual.push_back(leg->is_dual ? 1 : 0);
    }
    auto const uncoupled_items = iter_uncoupled();
    std::vector<TreeBlockItem> out;
    for (std::size_t i = 0; i < coupled.size(); ++i) {
        int64 start = 0; // start index of the current tree block within the block
        for (auto const& item : uncoupled_items) {
            auto const tree_block = product(item.multiplicities);
            for (auto const& tree :
                 fusion_trees(symmetry, item.uncoupled, coupled[i], are_dual).all_trees()) {
                out.push_back({ tree,
                                { start, start + tree_block },
                                item.multiplicities,
                                static_cast<int64>(i) });
                start += tree_block;
            }
        }
    }
    return out;
}

std::vector<ForestBlockItem>
TensorProduct::iter_forest_blocks(SectorArray const& coupled) const
{
    auto const uncoupled_items = iter_uncoupled();
    std::vector<ForestBlockItem> out;
    for (std::size_t i = 0; i < coupled.size(); ++i) {
        int64 start = 0;
        for (auto const& item : uncoupled_items) {
            auto const tree_block = product(item.multiplicities);
            auto const num_trees =
              static_cast<int64>(fusion_trees(symmetry, item.uncoupled, coupled[i]).size());
            auto const width = num_trees * tree_block;
            if (width == 0) {
                continue;
            }
            out.push_back({ item.uncoupled, { start, start + width }, static_cast<int64>(i) });
            start += width;
        }
    }
    return out;
}

std::vector<UncoupledItem>
TensorProduct::iter_uncoupled(bool yield_slices) const
{
    auto const legs = flat_legs();
    std::vector<UncoupledItem> out;

    if (legs.empty()) {
        // note: for a TensorProduct of zero spaces we *do* yield once, with empty arrays.
        UncoupledItem item{ symmetry->empty_sector_array, {}, std::nullopt };
        if (yield_slices) {
            item.slices = std::vector<IndexSlice>{};
        }
        out.push_back(std::move(item));
        return out;
    }

    std::vector<Space::Ptr> spaces;
    spaces.reserve(legs.size());
    for (auto const& leg : legs) {
        spaces.push_back(leg_as_space(leg));
    }
    // ``it.product``, i.e. the last index varies the fastest
    std::vector<std::size_t> strides(spaces.size());
    std::size_t total = 1;
    for (std::size_t n = spaces.size(); n-- > 0;) {
        strides[n] = total;
        total *= static_cast<std::size_t>(spaces[n]->num_sectors);
    }
    out.reserve(total);
    for (std::size_t m = 0; m < total; ++m) {
        SectorArray uncoupled(spaces.size(), symmetry->sector_ind_len);
        std::vector<int64> mults(spaces.size());
        std::optional<std::vector<IndexSlice>> slices_;
        if (yield_slices) {
            slices_.emplace(spaces.size());
        }
        for (std::size_t n = 0; n < spaces.size(); ++n) {
            auto const i = (m / strides[n]) % static_cast<std::size_t>(spaces[n]->num_sectors);
            uncoupled[n] = spaces[n]->sector_decomposition[i];
            mults[n] = spaces[n]->multiplicities[i];
            if (yield_slices) {
                auto const& slc = (*spaces[n]->slices)[i];
                (*slices_)[n] = IndexSlice{ slc[0], slc[1] };
            }
        }
        out.push_back({ std::move(uncoupled), std::move(mults), std::move(slices_) });
    }
    return out;
}

TensorProduct::Ptr
TensorProduct::left_multiply(py::object other) const
{
    return insert_multiply(std::move(other), 0);
}

TensorProduct::Ptr
TensorProduct::permuted(std::vector<int64> const& perm) const
{
    if (static_cast<int64>(perm.size()) != num_factors) {
        throw std::invalid_argument("perm has wrong length");
    }
    std::vector<bool> seen(perm.size(), false);
    std::vector<py::object> new_factors;
    new_factors.reserve(perm.size());
    for (auto const i : perm) {
        auto const idx = static_cast<std::size_t>(to_valid_idx(i, num_factors));
        if (seen[idx]) {
            throw std::invalid_argument("perm is not a permutation");
        }
        seen[idx] = true;
        new_factors.push_back(factors[idx]);
    }
    return std::make_shared<TensorProduct>(
      std::move(new_factors), symmetry, sector_decomposition, multiplicities);
}

TensorProduct::Ptr
TensorProduct::right_multiply(py::object other) const
{
    return insert_multiply(std::move(other), -1);
}

int64
TensorProduct::tree_block_size(SectorArray const& uncoupled) const
{
    // OPTIMIZE ?
    auto const legs = flat_legs();
    auto const n = std::min(legs.size(), uncoupled.size());
    int64 res = 1;
    for (std::size_t i = 0; i < n; ++i) {
        res *= leg_as_space(legs[i])->sector_multiplicity(uncoupled[i]);
    }
    return res;
}

IndexSlice
TensorProduct::tree_block_slice(FusionTree const& tree) const
{
    // OPTIMIZE ?
    int64 start = 0;
    int64 tree_block = 1;
    bool found = false;
    for (auto const& item : iter_uncoupled()) {
        tree_block = product(item.multiplicities);
        if (sectors_match(item.uncoupled, tree.uncoupled)) {
            found = true;
            break;
        }
        auto const num_trees =
          static_cast<int64>(fusion_trees(symmetry, item.uncoupled, tree.coupled).size());
        start += num_trees * tree_block;
    }
    if (!found) {
        throw std::invalid_argument("Uncoupled sectors incompatible");
    }
    auto const tree_idx =
      fusion_trees(symmetry, tree.uncoupled, tree.coupled, tree.are_dual).index(tree);
    start += tree_block * static_cast<int64>(tree_idx);
    return { start, start + tree_block };
}

bool
TensorProduct::operator==(Space const& other) const
{
    auto const* o = dynamic_cast<TensorProduct const*>(&other);
    if (o == nullptr) {
        return false;
    }
    if (num_factors != o->num_factors) {
        return false;
    }
    if (!symmetry->equals(*o->symmetry)) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!factors[i].equal(o->factors[i])) {
            return false;
        }
    }
    return true;
}

py::object
TensorProduct::operator[](int64 idx) const
{
    return factors[static_cast<std::size_t>(to_valid_idx(idx, num_factors))];
}

std::string
TensorProduct::repr(bool show_symmetry, bool one_line) const
{
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "TensorProduct";

    struct Options
    {
        bool full_sectors;
        bool summarized_sectors;
        bool show_all_factors;
        bool symmetry;
    };
    std::array<Options, 6> const options{ { { true, false, true, show_symmetry },
                                            { false, true, true, show_symmetry },
                                            { true, false, false, show_symmetry },
                                            { false, true, false, show_symmetry },
                                            { false, false, false, show_symmetry },
                                            { false, false, false, false } } };
    for (auto const& opt : options) {
        if (opt.full_sectors && 3 * static_cast<int64>(sector_decomposition.size()) *
                                    static_cast<int64>(sector_decomposition.sector_ind_len()) >
                                  linewidth) {
            // there is no chance to print all sectors in one line
            continue;
        }

        // populate two lists; one intended for single line, one for multiline
        std::vector<std::string> one_line_items;
        std::vector<std::string> lines{ ClsName + "(" };
        if (opt.symmetry) {
            one_line_items.push_back(std::format("symmetry={}", symmetry->repr()));
            lines.push_back(std::format("{}symmetry={},", indent, symmetry->repr()));
        }
        if (opt.show_all_factors) {
            std::vector<std::string> reprs;
            reprs.reserve(factors.size());
            for (auto const& factor : factors) {
                reprs.push_back(factor_repr(factor, /*show_symmetry=*/false, /*one_line=*/true));
            }
            one_line_items.push_back(std::format("factors=[{}]", join(reprs, ", ")));
            lines.push_back(std::format("{}factors=[", indent));
            for (auto const& r : reprs) {
                lines.push_back(std::format("{}{}{},", indent, indent, r));
            }
            lines.push_back(std::format("{}],", indent));
        } else {
            one_line_items.push_back(std::format("num_factors={}", num_factors));
            lines.push_back(std::format("{}num_factors={},", indent, num_factors));
        }
        if (opt.full_sectors) {
            py::list sector_strs;
            for (auto const& a : sector_decomposition) {
                sector_strs.append(symmetry->sector_str(a));
            }
            std::vector<std::string> const new_items{
                std::format("sector_decomposition={}", format_like_list(sector_strs)),
                std::format("multiplicities={}", format_like_list(py::cast(multiplicities)))
            };
            one_line_items.insert(one_line_items.end(), new_items.begin(), new_items.end());
            for (auto const& item : new_items) {
                lines.push_back(indent + item + ",");
            }
        }
        if (opt.summarized_sectors) {
            one_line_items.push_back(std::format("num_sectors={}", num_sectors));
            lines.push_back(std::format("{}num_sectors={},", indent, num_sectors));
        }
        lines.emplace_back(")");

        // try one line
        auto const res = std::format("{}({})", ClsName, join(one_line_items, ", "));
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }

        if (!one_line) {
            // try multi line
            bool const maxlines_ok = static_cast<int64>(lines.size()) <= maxlines;
            bool const linewidth_ok = std::ranges::all_of(lines, [&](std::string const& l) {
                return static_cast<int64>(l.size()) < linewidth;
            });
            if (maxlines_ok && linewidth_ok) {
                return join(lines, "\n");
            }
        }
    }
    // one of the above returns should have triggered
    throw std::runtime_error("TensorProduct repr: no suitable format found");
}

std::pair<SectorArray, std::vector<int64>>
TensorProduct::calc_sectors(std::vector<py::object> const& factors_) const
{
    return calc_sectors_of_factors(*symmetry, factors_);
}

void
TensorProduct::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    (void)h5gr;
    auto save = hdf5_saver.attr("save");
    py::list factor_list;
    for (auto const& factor : factors) {
        factor_list.append(factor);
    }
    save(factor_list, subpath + "factors");
    save(slices_to_py(slices), subpath + "slices");
    save(py::cast(symmetry), subpath + "symmetry");
    save(py::int_(num_sectors), subpath + "num_sectors");
    save(py::int_(num_factors), subpath + "num_factors");
    save(py::cast(sector_decomposition), subpath + "sector_decomposition");
    save(sector_order ? py::cast(*sector_order) : py::none(), subpath + "sector_order");
    save(dim_to_py(dim), subpath + "dim");
    save(vector_to_array(multiplicities), subpath + "multiplicities");
    save(sector_dims ? py::object(vector_to_array(*sector_dims)) : py::none(),
         subpath + "sector_dims");
}

TensorProduct::Ptr
TensorProduct::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto load = hdf5_loader.attr("load");
    auto symmetry = load(subpath + "symmetry").cast<Symmetry::Ptr>();
    std::vector<py::object> factors;
    for (py::handle factor : load(subpath + "factors")) {
        factors.push_back(py::reinterpret_borrow<py::object>(factor));
    }
    auto sector_decomposition = load(subpath + "sector_decomposition").cast<SectorArray>();
    auto multiplicities = py_array_to_i64(py::array::ensure(load(subpath + "multiplicities")));
    auto obj = std::make_shared<TensorProduct>(std::move(factors),
                                               std::move(symmetry),
                                               std::move(sector_decomposition),
                                               std::move(multiplicities));
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

namespace {

/// ``cyten.tools.misc.make_stride``: the strides of a C- (or F-) style array of the given shape.
[[nodiscard]] std::vector<int64>
make_stride(std::vector<int64> const& shape, bool cstyle)
{
    auto const L = shape.size();
    std::vector<int64> res(L, 1);
    int64 stride = 1;
    if (cstyle) {
        for (std::size_t a = L; a-- > 1;) {
            stride *= shape[a];
            res[a - 1] = stride;
        }
    } else {
        for (std::size_t a = 0; a + 1 < L; ++a) {
            stride *= shape[a];
            res[a + 1] = stride;
        }
    }
    return res;
}

/// Entry ``n`` of row ``m`` of ``cyten.tools.misc.make_grid(shape, cstyle)``.
///
/// `strides` must be ``make_stride(shape, cstyle)``. Since the grid enumerates all multi-indices
/// exactly once, the row index is just the flat index for those strides.
[[nodiscard]] int64
grid_entry(int64 m,
           std::vector<int64> const& strides,
           std::vector<int64> const& shape,
           std::size_t n)
{
    return (m / strides[n]) % shape[n];
}

/// The legs of an :class:`AbelianLegPipe` must all be :class:`ElementarySpace`\ s.
[[nodiscard]] ElementarySpace::Ptr
as_es_leg(Leg::Ptr const& leg)
{
    auto es = std::dynamic_pointer_cast<ElementarySpace>(leg);
    if (!es) {
        throw py::type_error("The legs of an AbelianLegPipe must be ElementarySpaces.");
    }
    return es;
}

} // namespace

AbelianLegPipe::Prepared
AbelianLegPipe::prepare(std::vector<ElementarySpace::Ptr> const& legs,
                        bool is_dual,
                        bool combine_cstyle)
{
    if (legs.empty()) {
        throw std::invalid_argument("Need at least one leg");
    }
    auto symmetry = legs.front()->Space::symmetry;
    if (!symmetry->is_abelian() || !symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("AbelianLegPipe is not supported for {}.", symmetry->str()));
    }
    auto const num_legs = legs.size();

    std::vector<int64> legs_num_sectors(num_legs);
    float64 dim = 1.;
    for (std::size_t n = 0; n < num_legs; ++n) {
        legs_num_sectors[n] = legs[n]->num_sectors;
        dim *= legs[n]->Space::dim;
    }
    auto sector_strides = make_stride(legs_num_sectors, combine_cstyle);

    // number of blocks in the pipe, ``prod(legs_num_sectors)``. Different from num_sectors.
    int64 nblocks = 1;
    for (auto const num : legs_num_sectors) {
        nblocks *= num;
    }
    auto const num_blocks = static_cast<std::size_t>(nblocks);

    // determine block_ind_map -- it's essentially the grid.
    // block_ind_map[:, :2] and [:, -1] are set later.
    std::vector<std::vector<int64>> block_ind_map(num_blocks, std::vector<int64>(3 + num_legs, 0));
    // the multiplicity for given (i1, i2, ...) is the product of ``multiplicities[il]``
    std::vector<int64> multiplicities(num_blocks, 1);
    std::vector<SectorArray> uncoupled;
    uncoupled.reserve(num_legs);
    for (std::size_t n = 0; n < num_legs; ++n) {
        SectorArray column(num_blocks, symmetry->sector_ind_len);
        for (std::size_t m = 0; m < num_blocks; ++m) {
            auto const i = static_cast<std::size_t>(
              grid_entry(static_cast<int64>(m), sector_strides, legs_num_sectors, n));
            block_ind_map[m][2 + n] = static_cast<int64>(i);
            multiplicities[m] *= legs[n]->multiplicities[i];
            column[m] = legs[n]->sector_decomposition[i];
        }
        uncoupled.push_back(std::move(column));
    }

    // calculate new defining_sectors. At this point, they have duplicates and are not sorted.
    auto sectors = symmetry->multiple_fusion_broadcast(uncoupled);
    if (is_dual) {
        // the above are the future sector_decomposition, but we want to compute
        // (and in particular sort according to) the defining_sectors
        sectors = symmetry->dual_sectors(sectors);
    }

    // sort sectors
    auto const sort = sectors.lexsort_indices();
    std::vector<int64> fusion_outcomes_sort(sort.begin(), sort.end());
    {
        std::vector<std::vector<int64>> sorted_map(num_blocks);
        std::vector<int64> sorted_mults(num_blocks);
        for (std::size_t m = 0; m < num_blocks; ++m) {
            sorted_map[m] = std::move(block_ind_map[sort[m]]);
            sorted_mults[m] = multiplicities[sort[m]];
        }
        block_ind_map = std::move(sorted_map);
        multiplicities = std::move(sorted_mults);
        sectors = sectors.take(sort);
    }

    // compute slices in the whole internal basis (we subtract the start of each block below)
    auto const slices = slice_boundaries(multiplicities);
    for (std::size_t m = 0; m < num_blocks; ++m) {
        block_ind_map[m][0] = slices[m];
        block_ind_map[m][1] = slices[m + 1];
    }

    // bunch sectors with equal sectors together
    auto const diffs = sectors.find_row_differences(/*include_len=*/true);
    std::vector<int64> block_ind_map_slices(diffs.begin(), diffs.end());
    auto const num_unique = diffs.size() - 1;
    std::vector<int64> block_starts(diffs.size());
    for (std::size_t k = 0; k < diffs.size(); ++k) {
        block_starts[k] = slices[diffs[k]];
    }
    std::vector<int64> unique_mults(num_unique);
    for (std::size_t k = 0; k < num_unique; ++k) {
        unique_mults[k] = block_starts[k + 1] - block_starts[k];
    }
    // [:-1] to exclude len
    auto unique_sectors =
      sectors.take(std::span<const std::size_t>(diffs.data(), diffs.size() - 1));

    // the new block index J, plus the slices within blocks (subtract the start of each block)
    for (std::size_t k = 0; k < num_unique; ++k) {
        for (std::size_t m = diffs[k]; m < diffs[k + 1]; ++m) {
            block_ind_map[m][2 + num_legs] = static_cast<int64>(k);
            block_ind_map[m][0] -= block_starts[k];
            block_ind_map[m][1] -= block_starts[k];
        }
    }

    auto basis_perm = calc_basis_perm(legs, combine_cstyle, dim, unique_mults, block_ind_map);

    std::vector<Leg::Ptr> leg_ptrs(legs.begin(), legs.end());
    return { std::move(leg_ptrs),
             std::move(symmetry),
             std::move(unique_sectors),
             std::move(unique_mults),
             std::move(basis_perm),
             std::move(sector_strides),
             std::move(fusion_outcomes_sort),
             std::move(block_ind_map_slices),
             std::move(block_ind_map) };
}

// note: the LegPipe base sets a combined basis_perm, which the ElementarySpace base then
// overwrites with the fusion basis_perm. This matches the order of the Python constructor.
AbelianLegPipe::AbelianLegPipe(Prepared prepared, bool is_dual_, bool combine_cstyle_)
  : LegPipe(prepared.legs, is_dual_, combine_cstyle_)
  , ElementarySpace(prepared.symmetry,
                    prepared.defining_sectors,
                    prepared.multiplicities,
                    is_dual_,
                    prepared.basis_perm)
  , sector_strides(std::move(prepared.sector_strides))
  , fusion_outcomes_sort(std::move(prepared.fusion_outcomes_sort))
  , block_ind_map_slices(std::move(prepared.block_ind_map_slices))
  , block_ind_map(std::move(prepared.block_ind_map))
{
}

AbelianLegPipe::AbelianLegPipe(std::vector<ElementarySpace::Ptr> legs_,
                               bool is_dual_,
                               bool combine_cstyle_)
  : AbelianLegPipe(prepare(legs_, is_dual_, combine_cstyle_), is_dual_, combine_cstyle_)
{
}

std::vector<int64>
AbelianLegPipe::fusion_outcomes_perm(std::vector<ElementarySpace::Ptr> const& legs,
                                     bool combine_cstyle,
                                     float64 dim,
                                     std::vector<int64> const& multiplicities,
                                     std::vector<std::vector<int64>> const& block_ind_map)
{
    auto const num_legs = legs.size();
    std::vector<int64> legs_dims(num_legs);
    for (std::size_t n = 0; n < num_legs; ++n) {
        legs_dims[n] = static_cast<int64>(legs[n]->Space::dim);
    }
    auto const dim_strides = make_stride(legs_dims, combine_cstyle);
    std::vector<int64> perm(dim_as_size(dim));

    // slices_starts is slices[:, 0], but we need to compute it here, since the
    // ElementarySpace base may not be initialized yet at this point
    auto const slices_starts = slice_boundaries(multiplicities);

    std::vector<int64> mult_shape(num_legs);
    std::vector<int64> sector_starts(num_legs);
    for (auto const& row : block_ind_map) {
        // shift the slice start:stop from within the block back to the whole internal basis
        auto const J = static_cast<std::size_t>(row[2 + num_legs]);
        auto const start = row[0] + slices_starts[J];

        // Now for each basis element in start:stop, we construct where it was before sorting.
        // multiplicity_grid :: each row stands for a combination of uncoupled basis elements;
        //                     they are the indices of that basis element *within* the sector.
        // sector_starts[n] is the index of the first basis vector for legs[n] that is in the
        // current sector, namely legs[n].sector_decomposition[idcs[n]]
        int64 count = 1;
        for (std::size_t n = 0; n < num_legs; ++n) {
            auto const idx = static_cast<std::size_t>(row[2 + n]);
            mult_shape[n] = legs[n]->multiplicities[idx];
            sector_starts[n] = (*legs[n]->slices)[idx][0];
            count *= mult_shape[n];
        }
        assert(count == row[1] - row[0]);
        auto const mult_strides = make_stride(mult_shape, combine_cstyle);
        // basis_grid :: each row stands for a combination of uncoupled basis elements; they are
        //               the indices of that basis element within its legs internal basis.
        // Note that the relevant strides are ``dim_strides``, which come from a *different*
        // shape than the multiplicity_grid.
        for (int64 k = 0; k < count; ++k) {
            int64 flat = 0;
            for (std::size_t n = 0; n < num_legs; ++n) {
                flat +=
                  (grid_entry(k, mult_strides, mult_shape, n) + sector_starts[n]) * dim_strides[n];
            }
            perm[static_cast<std::size_t>(start + k)] = flat;
        }
    }
    return perm;
}

std::vector<int64>
AbelianLegPipe::calc_basis_perm(std::vector<ElementarySpace::Ptr> const& legs,
                                bool combine_cstyle,
                                float64 dim,
                                std::vector<int64> const& multiplicities,
                                std::vector<std::vector<int64>> const& block_ind_map)
{
    // see the diagram in the docstring of the Python ``_calc_basis_perm``; we follow the path
    // parallel to ``pipe.basis_perm``: inverse of fusion, basis_perm of each leg, fusion, sort.
    auto const num_legs = legs.size();
    std::vector<int64> legs_dims(num_legs);
    std::vector<std::vector<int64>> perms(num_legs);
    for (std::size_t n = 0; n < num_legs; ++n) {
        legs_dims[n] = static_cast<int64>(legs[n]->Space::dim);
        perms[n] = legs[n]->basis_perm();
    }
    auto const dim_strides = make_stride(legs_dims, combine_cstyle);
    auto const num_basis_states = dim_as_size(dim);

    // ``np.reshape(np.arange(dim), dims, order)[np.ix_(*perms)].reshape(dim, order)``
    std::vector<int64> combined(num_basis_states);
    for (std::size_t m = 0; m < num_basis_states; ++m) {
        int64 flat = 0;
        for (std::size_t n = 0; n < num_legs; ++n) {
            auto const i = static_cast<std::size_t>(
              grid_entry(static_cast<int64>(m), dim_strides, legs_dims, n));
            flat += perms[n][i] * dim_strides[n];
        }
        combined[m] = flat;
    }

    auto const fusion_perm =
      fusion_outcomes_perm(legs, combine_cstyle, dim, multiplicities, block_ind_map);
    std::vector<int64> res(num_basis_states);
    for (std::size_t i = 0; i < num_basis_states; ++i) {
        res[i] = combined[static_cast<std::size_t>(fusion_perm[i])];
    }
    return res;
}

std::vector<ElementarySpace::Ptr>
AbelianLegPipe::es_legs() const
{
    std::vector<ElementarySpace::Ptr> out;
    out.reserve(legs.size());
    for (auto const& leg : legs) {
        out.push_back(as_es_leg(leg));
    }
    return out;
}

std::vector<int64>
AbelianLegPipe::get_fusion_outcomes_perm(std::vector<int64> const& multiplicities_) const
{
    return fusion_outcomes_perm(
      es_legs(), combine_cstyle, Space::dim, multiplicities_, block_ind_map);
}

void
AbelianLegPipe::test_sanity() const
{
    auto const es = es_legs();
    for (auto const& leg : es) {
        if (auto const* nested = dynamic_cast<LegPipe const*>(leg.get()); nested != nullptr) {
            assert(nested->is_abelian_leg_pipe());
        }
        leg->test_sanity();
    }
    auto const n = static_cast<std::size_t>(num_legs);
    // check sector_strides
    assert(sector_strides.size() == n);
    std::vector<int64> legs_num_sectors(n);
    int64 nblocks = 1;
    for (std::size_t i = 0; i < n; ++i) {
        legs_num_sectors[i] = es[i]->num_sectors;
        nblocks *= legs_num_sectors[i];
    }
    assert(sector_strides == make_stride(legs_num_sectors, combine_cstyle));
    // check block_ind_map_slices
    // note: we do not check for full correctness, just for consistency as slices
    assert(block_ind_map_slices.size() == static_cast<std::size_t>(num_sectors) + 1);
    assert(block_ind_map_slices.front() == 0);
    assert(block_ind_map_slices.back() == nblocks);
    assert(std::ranges::is_sorted(block_ind_map_slices));
    // check block_ind_map
    assert(block_ind_map.size() == static_cast<std::size_t>(nblocks));
    // the rows are sorted first by J, then by the i, in C-style order if combine_cstyle
    // (see the class docstring). Equivalently, the keys built below are non-decreasing.
    auto const sort_key = [&](std::vector<int64> const& row) {
        std::vector<int64> key{ row[2 + n] };
        for (std::size_t i = 0; i < n; ++i) {
            key.push_back(combine_cstyle ? row[2 + i] : row[1 + n - i]);
        }
        return key;
    };
    for (std::size_t m = 0; m < block_ind_map.size(); ++m) {
        auto const& row = block_ind_map[m];
        assert(row.size() == 3 + n);
        auto const J = static_cast<std::size_t>(row[2 + n]);
        if (m > 0) {
            auto const prev = sort_key(block_ind_map[m - 1]);
            auto const cur = sort_key(row);
            assert(std::ranges::lexicographical_compare(prev, cur));
        }
        if (m > 0 && row[2 + n] == block_ind_map[m - 1][2 + n]) {
            assert(row[0] == block_ind_map[m - 1][1]);
        } else {
            assert(row[0] == 0);
        }
        std::vector<Sector> uncoupled(n);
        for (std::size_t i = 0; i < n; ++i) {
            uncoupled[i] = es[i]->sector_decomposition[static_cast<std::size_t>(row[2 + i])];
        }
        assert(Space::symmetry->multiple_fusion(uncoupled) == sector_decomposition[J]);
    }
    // call to super class(es)
    LegPipe::test_sanity();
    ElementarySpace::test_sanity();
}

py::object
AbelianLegPipe::as_Space()
{
    return py::cast(shared_es());
}

py::object
AbelianLegPipe::as_ElementarySpace(bool is_dual_)
{
    return py::cast(with_is_dual(is_dual_));
}

Space::Ptr
AbelianLegPipe::dual_space() const
{
    return dual_pipe();
}

Leg::Ptr
AbelianLegPipe::dual_leg() const
{
    return dual_pipe();
}

AbelianLegPipe::Ptr
AbelianLegPipe::dual_pipe() const
{
    std::vector<ElementarySpace::Ptr> dual_legs;
    dual_legs.reserve(legs.size());
    for (auto it = legs.rbegin(); it != legs.rend(); ++it) {
        dual_legs.push_back(as_es_leg((*it)->dual_leg()));
    }
    return std::make_shared<AbelianLegPipe>(std::move(dual_legs), !is_dual, !combine_cstyle);
}

bool
AbelianLegPipe::is_trivial() const
{
    return ElementarySpace::is_trivial();
}

std::vector<Leg::Ptr>
AbelianLegPipe::flat_spaces()
{
    // Unlike the plain LegPipe, we do not need to flatten AbelianLegPipes, if we just
    // want to flatten until we get spaces
    return { shared_leg() };
}

std::string
AbelianLegPipe::ascii_arrow() const
{
    // ``Leg.ascii_arrow`` in Python: a filled arrow for a pipe that is also an ElementarySpace
    return is_dual ? "▲" : "▼";
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_independent_symmetries(std::vector<Ptr> const& independent_descriptions)
{
    assert(!independent_descriptions.empty());
    auto const is_dual = independent_descriptions.front()->is_dual;
    assert(std::ranges::all_of(independent_descriptions,
                               [is_dual](Ptr const& i) { return i->is_dual == is_dual; }));
    auto const num_legs = independent_descriptions.front()->num_legs;
    assert(std::ranges::all_of(independent_descriptions,
                               [num_legs](Ptr const& i) { return i->num_legs == num_legs; }));
    std::vector<ElementarySpace::Ptr> legs;
    legs.reserve(static_cast<std::size_t>(num_legs));
    for (std::size_t k = 0; k < static_cast<std::size_t>(num_legs); ++k) {
        std::vector<ElementarySpace::Ptr> group;
        std::vector<Ptr> pipes;
        group.reserve(independent_descriptions.size());
        for (auto const& description : independent_descriptions) {
            auto leg = as_es_leg(description->legs[k]);
            if (auto pipe = std::dynamic_pointer_cast<AbelianLegPipe>(leg)) {
                pipes.push_back(std::move(pipe));
            }
            group.push_back(std::move(leg));
        }
        if (pipes.size() == group.size()) {
            legs.push_back(from_independent_symmetries(pipes));
        } else {
            legs.push_back(ElementarySpace::from_independent_symmetries(group));
        }
    }
    return std::make_shared<AbelianLegPipe>(std::move(legs), is_dual);
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_basis(Symmetry::Ptr /*symmetry*/, SectorArray /*sectors_of_basis*/)
{
    throw py::type_error("from_basis is not supported for AbelianLegPipe");
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_null_space(Symmetry::Ptr /*symmetry*/, bool /*is_dual*/)
{
    throw py::type_error("from_null_space is not supported for AbelianLegPipe");
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_defining_sectors(Symmetry::Ptr /*symmetry*/,
                                      SectorArray /*defining_sectors*/,
                                      std::optional<std::vector<int64>> /*multiplicities*/,
                                      bool /*is_dual*/,
                                      std::optional<std::vector<int64>> /*basis_perm*/,
                                      bool /*unique_sectors*/,
                                      std::vector<std::size_t>* /*return_sorting_perm*/)
{
    throw py::type_error("from_defining_sectors is not supported for AbelianLegPipe");
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_trivial_sector(int64 /*dim*/,
                                    Symmetry::Ptr /*symmetry*/,
                                    bool /*is_dual*/,
                                    std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error("from_trivial_sector is not supported for AbelianLegPipe");
}

py::object
AbelianLegPipe::change_symmetry(Symmetry::Ptr symmetry_, SectorMapFn sector_map, bool injective)
{
    std::vector<ElementarySpace::Ptr> new_legs;
    new_legs.reserve(legs.size());
    for (auto const& leg : es_legs()) {
        new_legs.push_back(
          leg->change_symmetry(symmetry_, sector_map, injective).cast<ElementarySpace::Ptr>());
    }
    return py::cast(
      std::make_shared<AbelianLegPipe>(std::move(new_legs), is_dual, combine_cstyle));
}

py::object
AbelianLegPipe::drop_symmetry(std::optional<std::vector<int64>> which)
{
    // OPTIMIZE can we avoid recomputation of fusion?
    std::vector<ElementarySpace::Ptr> new_legs;
    new_legs.reserve(legs.size());
    for (auto const& leg : es_legs()) {
        new_legs.push_back(leg->drop_symmetry(which).cast<ElementarySpace::Ptr>());
    }
    return py::cast(
      std::make_shared<AbelianLegPipe>(std::move(new_legs), is_dual, combine_cstyle));
}

void
AbelianLegPipe::set_basis_perm(std::optional<std::vector<int64>> /*basis_perm*/)
{
    throw py::type_error("Can not set basis_perm for AbelianLegPipe.");
}

void
AbelianLegPipe::set_inverse_basis_perm(std::optional<std::vector<int64>> /*inverse_basis_perm*/)
{
    throw py::type_error("Can not set basis_perm for AbelianLegPipe.");
}

ElementarySpace::Ptr
AbelianLegPipe::take_slice(py::array blockmask) const
{
    char const* msg =
      "Using `AbelianLegPipe.take_slice` loses the product (pipe) structure and results in "
      "a plain ElementarySpace. Explicitly convert using `as_ElementarySpace` to suppress "
      "this warning.";
    if (PyErr_WarnEx(PyExc_UserWarning, msg, 2) < 0) {
        throw py::error_already_set();
    }
    // note: unlike the Python version, we call the ElementarySpace implementation directly.
    // Python goes through ``as_ElementarySpace(is_dual=self.is_dual)``, which returns ``self``
    // and therefore recurses infinitely.
    return ElementarySpace::take_slice(std::move(blockmask));
}

ElementarySpace::Ptr
AbelianLegPipe::with_opposite_duality() const
{
    return std::make_shared<AbelianLegPipe>(es_legs(), !is_dual, combine_cstyle);
}

bool
AbelianLegPipe::operator==(Leg const& other) const
{
    // note: LegPipe::operator== already compares combine_cstyle and checks that both sides
    // are (not) AbelianLegPipes.
    return LegPipe::operator==(other);
}

bool
AbelianLegPipe::operator==(Space const& other) const
{
    auto const* o = dynamic_cast<LegPipe const*>(&other);
    if (o == nullptr) {
        return false;
    }
    return LegPipe::operator==(*o);
}

std::string
AbelianLegPipe::repr(bool show_symmetry, bool one_line) const
{
    auto const& cfg = get_config();
    auto const linewidth = cfg.print_linewidth;
    std::string const indent(static_cast<std::size_t>(cfg.print_indent), ' ');
    auto const maxlines = cfg.maxlines_spaces;
    std::string const ClsName = "AbelianLegPipe";

    struct Options
    {
        /// 0=show full arrays, 1=show only nums, 2=dont show
        int sector_mode;
        /// 0=show full, 1=force one-line each, 2=show only num
        int child_mode;
        bool summarize_basis_perm;
        bool symmetry;
    };
    std::array<Options, 7> const options{ { { 0, 0, false, show_symmetry },
                                            { 0, 0, true, show_symmetry },
                                            { 0, 1, true, show_symmetry },
                                            { 0, 2, true, show_symmetry },
                                            { 1, 2, true, show_symmetry },
                                            { 2, 2, true, show_symmetry },
                                            { 2, 2, true, false } } };
    for (auto const& opt : options) {
        if (opt.sector_mode == 0 && 3 * static_cast<int64>(sector_decomposition.size()) *
                                        static_cast<int64>(sector_decomposition.sector_ind_len()) >
                                      linewidth) {
            // there is no chance to print all sectors in one line
            continue;
        }

        // populate two lists; one intended for single line, one for multiline.
        // this is because lines behaves differently when dealing with the children / legs
        std::vector<std::string> one_line_items;
        std::vector<std::string> lines{ ClsName + "(" };

        if (opt.symmetry) {
            one_line_items.push_back(std::format("symmetry={}", Space::symmetry->repr()));
            lines.push_back(std::format("{}symmetry={},", indent, Space::symmetry->repr()));
        }

        if (opt.child_mode < 2) {
            std::vector<std::string> reprs;
            reprs.reserve(legs.size());
            for (auto const& leg : legs) {
                reprs.push_back(factor_repr(
                  py::cast(leg), /*show_symmetry=*/false, /*one_line=*/opt.child_mode > 0));
            }
            one_line_items.push_back(std::format("factors=[{}]", join(reprs, ", ")));
            lines.push_back(std::format("{}factors=[", indent));
            for (auto const& r : reprs) {
                lines.push_back(std::format("{}{}{},", indent, indent, r));
            }
            lines.push_back(std::format("{}],", indent));
        } else {
            one_line_items.push_back(std::format("num_legs={}", num_legs));
            lines.push_back(std::format("{}num_legs={},", indent, num_legs));
        }

        if (opt.sector_mode == 0) {
            py::list sector_dec_strs;
            for (auto const& a : sector_decomposition) {
                sector_dec_strs.append(Space::symmetry->sector_str(a));
            }
            py::list def_sector_strs;
            for (auto const& a : defining_sectors) {
                def_sector_strs.append(Space::symmetry->sector_str(a));
            }
            std::vector<std::string> const new_items{
                std::format("sector_decomposition={}", format_like_list(sector_dec_strs)),
                std::format("defining_sectors={}", format_like_list(def_sector_strs)),
                std::format("multiplicities={}", format_like_list(py::cast(multiplicities)))
            };
            one_line_items.insert(one_line_items.end(), new_items.begin(), new_items.end());
            for (auto const& item : new_items) {
                lines.push_back(indent + item + ",");
            }
        } else if (opt.sector_mode == 1) {
            one_line_items.push_back(std::format("num_sectors={}", num_sectors));
            lines.push_back(std::format("{}num_sectors={},", indent, num_sectors));
        }

        if (_basis_perm) {
            if (opt.summarize_basis_perm) {
                one_line_items.emplace_back("basis_perm=[...]");
                lines.push_back(std::format("{}basis_perm=[...],", indent));
            } else {
                auto const perm = format_like_list(py::cast(*_basis_perm));
                one_line_items.push_back(std::format("basis_perm={}", perm));
                lines.push_back(std::format("{}basis_perm={},", indent, perm));
            }
        }

        one_line_items.push_back(std::format("is_dual={}", bool_repr(is_dual)));
        lines.push_back(std::format("{}is_dual={},", indent, bool_repr(is_dual)));
        lines.emplace_back(")");

        // try one line
        auto const res = std::format("{}({})", ClsName, join(one_line_items, ", "));
        if (static_cast<int64>(res.size()) <= linewidth) {
            return res;
        }

        if (!one_line) {
            // try multi line
            bool const maxlines_ok = static_cast<int64>(lines.size()) <= maxlines;
            bool const linewidth_ok = std::ranges::all_of(lines, [&](std::string const& l) {
                return static_cast<int64>(l.size()) < linewidth;
            });
            if (maxlines_ok && linewidth_ok) {
                return join(lines, "\n");
            }
        }
    }
    // one of the above returns should have triggered
    throw std::runtime_error("AbelianLegPipe repr: no suitable format found");
}

void
AbelianLegPipe::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    // note: the Python class inherits ElementarySpace.save_hdf5, which does not store the pipe
    // structure. We additionally store the legs and combine_cstyle, such that from_hdf5 can
    // reconstruct the pipe.
    ElementarySpace::save_hdf5(hdf5_saver, h5gr, subpath);
    py::list leg_list;
    for (auto const& leg : legs) {
        leg_list.append(py::cast(leg));
    }
    hdf5_saver.attr("save")(leg_list, subpath + "legs");
    h5gr.attr("attrs")["combine_cstyle"] = combine_cstyle;
}

AbelianLegPipe::Ptr
AbelianLegPipe::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    std::vector<ElementarySpace::Ptr> legs;
    for (py::handle item : hdf5_loader.attr("load")(subpath + "legs")) {
        legs.push_back(item.cast<ElementarySpace::Ptr>());
    }
    auto const is_dual = hdf5_loader.attr("get_attr")(h5gr, "is_dual").cast<bool>();
    auto const combine_cstyle = hdf5_loader.attr("get_attr")(h5gr, "combine_cstyle").cast<bool>();
    auto obj = std::make_shared<AbelianLegPipe>(std::move(legs), is_dual, combine_cstyle);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

namespace {

[[nodiscard]] bool
is_plain_leg_pipe(Leg::Ptr const& leg)
{
    // Python: ``not isinstance(leg, ElementarySpace) and isinstance(leg, LegPipe)``.
    // AbelianLegPipe is both, so it takes the ElementarySpace path.
    return static_cast<bool>(std::dynamic_pointer_cast<LegPipe>(leg)) &&
           !static_cast<bool>(std::dynamic_pointer_cast<ElementarySpace>(leg));
}

[[nodiscard]] std::size_t
leg_dim_as_size(Leg const& leg)
{
    assert(leg.dim >= 0.);
    assert(std::floor(leg.dim) == leg.dim);
    return static_cast<std::size_t>(leg.dim);
}

} // namespace

py::array
swap_gate(Leg::Ptr V, Leg::Ptr W)
{
    if (!V || !W) {
        throw py::type_error("swap_gate requires two legs");
    }
    if (!V->symmetry->equals(*W->symmetry)) {
        throw SymmetryError("Incompatible symmetries.");
    }
    if (!V->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("braid can not be written as array for {}.", V->symmetry->str()));
    }
    auto const dV = static_cast<py::ssize_t>(leg_dim_as_size(*V));
    auto const dW = static_cast<py::ssize_t>(leg_dim_as_size(*W));
    auto np = py::module_::import("numpy");

    if (is_plain_leg_pipe(V)) {
        auto pipe = std::dynamic_pointer_cast<LegPipe>(V);
        auto const& legs = pipe->legs;
        py::object res = swap_gate(legs.back(), W);
        int n = 0;
        for (auto it = legs.rbegin() + 1; it != legs.rend(); ++it, ++n) {
            py::object sw = swap_gate(*it, W);
            res = np.attr("tensordot")(sw, res, py::make_tuple(2, 0));
            res = np.attr("moveaxis")(res, 2, -2 - n);
        }
        char const* order = pipe->combine_cstyle ? "C" : "F";
        return np.attr("reshape")(res, py::make_tuple(dW, dV, dW, dV), py::arg("order") = order);
    }
    if (is_plain_leg_pipe(W)) {
        auto pipe = std::dynamic_pointer_cast<LegPipe>(W);
        auto const& legs = pipe->legs;
        py::object res = swap_gate(V, legs.front());
        for (std::size_t n = 1; n < legs.size(); ++n) {
            py::object sw = swap_gate(V, legs[n]);
            res = np.attr("tensordot")(res, sw, py::make_tuple(static_cast<int>(n), -1));
            py::list axes;
            for (std::size_t i = 0; i < n; ++i) {
                axes.append(static_cast<int>(i));
            }
            axes.append(-3);
            axes.append(-2);
            for (std::size_t i = n; i < 2 * n; ++i) {
                axes.append(static_cast<int>(i));
            }
            axes.append(-1);
            axes.append(-4);
            res = np.attr("transpose")(res, axes);
        }
        char const* order = pipe->combine_cstyle ? "C" : "F";
        return np.attr("reshape")(res, py::make_tuple(dW, dV, dW, dV), py::arg("order") = order);
    }

    auto Ves = std::dynamic_pointer_cast<ElementarySpace>(V);
    auto Wes = std::dynamic_pointer_cast<ElementarySpace>(W);
    if (!Ves || !Wes) {
        throw py::type_error("swap_gate expects ElementarySpace or LegPipe legs");
    }

    py::object res = np.attr("zeros")(py::make_tuple(dW, dV, dW, dV));
    int64 i = 0;
    for (std::size_t ia = 0; ia < Ves->defining_sectors.size(); ++ia) {
        auto const& a = Ves->defining_sectors[ia];
        auto const ma = Ves->multiplicities[ia];
        auto const da = static_cast<int64>(Ves->Space::symmetry->sector_dim(a));
        int64 j = 0;
        for (std::size_t ib = 0; ib < Wes->defining_sectors.size(); ++ib) {
            auto const& b = Wes->defining_sectors[ib];
            auto const mb = Wes->multiplicities[ib];
            py::array swap = Ves->Space::symmetry->swap_gate(a, b);
            auto const db = static_cast<int64>(Wes->Space::symmetry->sector_dim(b));
            int64 i2 = i;
            for (int64 na = 0; na < ma; ++na) {
                int64 j2 = j;
                for (int64 nb = 0; nb < mb; ++nb) {
                    res[py::make_tuple(py::slice(j2, j2 + db, 1),
                                       py::slice(i2, i2 + da, 1),
                                       py::slice(j2, j2 + db, 1),
                                       py::slice(i2, i2 + da, 1))] = swap;
                    j2 += db;
                }
                i2 += da;
            }
            j += db * mb;
        }
        i += da * ma;
    }
    auto Winv = vector_to_array(Wes->inverse_basis_perm());
    auto Vinv = vector_to_array(Ves->inverse_basis_perm());
    return res[np.attr("ix_")(Winv, Vinv, Winv, Vinv)];
}

py::array
twist_gate_diag(Leg::Ptr V)
{
    if (!V) {
        throw py::type_error("twist_gate_diag requires a leg");
    }
    if (!V->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("twist can not be written as array for {}.", V->symmetry->str()));
    }
    auto np = py::module_::import("numpy");

    if (is_plain_leg_pipe(V)) {
        auto pipe = std::dynamic_pointer_cast<LegPipe>(V);
        char const* order = pipe->combine_cstyle ? "C" : "F";
        py::object res = twist_gate_diag(pipe->legs.front());
        auto newaxis = np.attr("newaxis");
        for (std::size_t n = 1; n < pipe->legs.size(); ++n) {
            py::object next = twist_gate_diag(pipe->legs[n]);
            res = np.attr("reshape")(res[py::make_tuple(py::slice(), newaxis)] *
                                       next[py::make_tuple(newaxis, py::slice())],
                                     -1,
                                     py::arg("order") = order);
        }
        return res;
    }

    // ElementarySpace or AbelianLegPipe
    auto Ves = std::dynamic_pointer_cast<ElementarySpace>(V);
    if (!Ves) {
        throw py::type_error("twist_gate_diag expects ElementarySpace or LegPipe");
    }
    auto const dV = static_cast<py::ssize_t>(leg_dim_as_size(*Ves));
    py::object res_diag = np.attr("zeros")(dV);
    if (!Ves->slices) {
        throw SymmetryError(
          std::format("twist can not be written as array for {}.", Ves->Space::symmetry->str()));
    }
    for (std::size_t n = 0; n < Ves->sector_decomposition.size(); ++n) {
        auto const& a = Ves->sector_decomposition[n];
        auto const i = (*Ves->slices)[n][0];
        auto const j = (*Ves->slices)[n][1];
        complex128 const twist = Ves->Space::symmetry->topological_twist(a);
        // Assign as float when real so ``np.zeros(dV)`` (float64) stays real for
        // symmetries with real twists (e.g. U(1)); matches Python historical dtype.
        if (twist.imag() == 0.0) {
            res_diag[py::slice(i, j, 1)] = twist.real();
        } else {
            res_diag[py::slice(i, j, 1)] = twist;
        }
    }
    return res_diag[vector_to_array(Ves->inverse_basis_perm())];
}

py::array
twist_gate(Leg::Ptr V)
{
    if (!V) {
        throw py::type_error("twist_gate requires a leg");
    }
    if (!V->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("twist can not be written as array for {}.", V->symmetry->str()));
    }
    return py::module_::import("numpy").attr("diag")(twist_gate_diag(std::move(V)));
}

std::vector<int64>
flat_leg_permutation(std::vector<Leg::Ptr> const& legs)
{
    std::vector<int64> offsets(legs.size(), 0);
    int64 running = 0;
    for (std::size_t i = 0; i < legs.size(); ++i) {
        offsets[i] = running;
        running += legs[i]->num_flat_legs();
    }
    std::vector<int64> perm;
    perm.reserve(static_cast<std::size_t>(running));
    for (std::size_t i = 0; i < legs.size(); ++i) {
        auto part = legs[i]->_flat_leg_permutation(offsets[i]);
        perm.insert(perm.end(), part.begin(), part.end());
    }
    return perm;
}

std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
unique_sorted_sectors(SectorArray const& unsorted_sectors,
                      std::vector<int64> const& unsorted_multiplicities)
{
    auto [sectors, mults, perm] = unsorted_sectors.unique_sorted(unsorted_multiplicities);
    return { std::move(sectors), std::vector<int64>(mults.begin(), mults.end()), std::move(perm) };
}

std::tuple<SectorArray, std::vector<int64>, std::vector<std::size_t>>
sort_sectors_public(SectorArray const& sectors, std::vector<int64> const& multiplicities)
{
    return sort_sectors(sectors, multiplicities);
}

std::pair<std::optional<std::vector<int64>>, Symmetry::Ptr>
parse_inputs_drop_symmetry_public(std::optional<std::vector<int64>> which, Symmetry::Ptr symmetry)
{
    if (!symmetry) {
        throw py::type_error("parse_inputs_drop_symmetry requires a symmetry");
    }
    return parse_inputs_drop_symmetry(std::move(which), *symmetry);
}

} // namespace cyten
