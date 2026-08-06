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

// CHECKME: the following was generated by .cursor/skills/pybind11-codegen/pybind11_codegen.py gen_cpp_definition --py-name Space --header-file include/cyten/symmetries/spaces.h --src-file src/symmetries/spaces.cpp

Space::Space(
    Symmetry::Ptr symmetry,
    SectorArray sector_decomposition,
    std::optional<std::vector<int64>> multiplicities,
    std::optional<std::string> sector_order
) {
    /* CHECKME: converted from following python code:
     * self.symmetry = symmetry = symmetry.as_Symmetry()
     *         self.sector_decomposition = sector_decomposition = as_sector_array(
     *             sector_decomposition, sector_ind_len=symmetry.sector_ind_len
     *         )
     * self.sector_order = sector_order
     *         if sector_decomposition.shape[1] != symmetry.sector_ind_len:
     *             msg = f'Wrong sectors.shape: Expected (*, {symmetry.sector_ind_len}), got {sector_decomposition.shape}.'
     *             raise ValueError(msg)
     * assert sector_decomposition.shape[1] == symmetry.sector_ind_len
     * self.num_sectors = num_sectors = len(sector_decomposition)
     *         if multiplicities is None:
     *             self.multiplicities = multiplicities = np.ones((num_sectors,), dtype=int)
     *         else:
     *             self.multiplicities = multiplicities = np.asarray(multiplicities, dtype=int)
     *             assert multiplicities.shape == (num_sectors,)
     *         if symmetry.can_be_dropped:
     *             self.sector_dims = sector_dims = symmetry.batch_sector_dim(sector_decomposition)
     *             self.sector_qdims = sector_dims
     *             slices = np.zeros((len(sector_decomposition), 2), dtype=np.intp)
     *             slices[:, 1] = slice_ends = np.cumsum(multiplicities * sector_dims)
     *             slices[1:, 0] = slice_ends[:-1]  # slices[0, 0] remains 0, which is correct
     *             self.slices = slices
     *             self.dim = np.sum(sector_dims * multiplicities).item()
     *         else:
     *             self.sector_dims = None
     *             self.sector_qdims = sector_qdims = symmetry.batch_qdim(sector_decomposition)
     *             self.slices = None
     *             self.dim = np.sum(sector_qdims * multiplicities).item()
     */
    /* FIXME: multiple assignment: self.symmetry = symmetry = symmetry.as_Symmetry() */
    /* FIXME: multiple assignment:         self.sector_decomposition = sector_decomposition = as_sector_array(
            sector_decomposition, sector_ind_len=symmetry.sector_ind_len
        ) */
    sector_order = sector_order;
    if (sector_decomposition.shape[1] != symmetry.sector_ind_len) {
        auto /* CHECKME: type? */ msg = std::format("Wrong sectors.shape: Expected (*, {}), got {}.", symmetry.sector_ind_len, sector_decomposition.shape);
        throw std::invalid_argument(msg);
    }
    /* FIXME: Assert: assert sector_decomposition.shape[1] == symmetry.sector_ind_len */
    /* FIXME: multiple assignment: self.num_sectors = num_sectors = len(sector_decomposition) */
    if (multiplicities == py::none()) {
        /* FIXME: multiple assignment: self.multiplicities = multiplicities = np.ones((num_sectors,), dtype=int) */
    } else {
        /* FIXME: multiple assignment: self.multiplicities = multiplicities = np.asarray(multiplicities, dtype=int) */
        /* FIXME: Assert: assert multiplicities.shape == (num_sectors,) */
    }
    if (symmetry.can_be_dropped) {
        /* FIXME: multiple assignment: self.sector_dims = sector_dims = symmetry.batch_sector_dim(sector_decomposition) */
        sector_qdims = sector_dims;
        auto /* CHECKME: type? */ slices = np.zeros(std::make_tuple(len(sector_decomposition), 2), py::arg({keyword.key}) = val) /* CHECKME: keywords ['dtype'] */;
        /* FIXME: multiple assignment: slices[:, 1] = slice_ends = np.cumsum(multiplicities * sector_dims) */
        /* Multidimensional slice using std::gslice
 * NOTE: Requires slices_strides to be defined as std::valarray<size_t>
 *       containing the memory strides for each dimension.
 * For a row-major 2D array with shape (n0, n1, ..., n1):
 *   slices_strides = {n1, 1};
 * Alternative: Consider using xtensor, Eigen, or C++23 mdspan
 */
slices[std::gslice(1 * slices_strides[0], std::valarray<std::size_t>{(slices.shape(0) - 1), 1}, std::valarray<std::size_t>{slices_strides[0], slices_strides[1]})] = slice_ends[std::slice(0, -1, 1)];
        /* slices[0, 0] remains 0, which is correct */
        slices = slices;
        dim = np.sum(sector_dims * multiplicities).item();
    } else {
        sector_dims = py::none();
        /* FIXME: multiple assignment: self.sector_qdims = sector_qdims = symmetry.batch_qdim(sector_decomposition) */
        slices = py::none();
        dim = np.sum(sector_qdims * multiplicities).item();
    }
}

/// If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.
/// property getter
bool Space::is_trivial() {
    /* CHECKME: converted from following python code:
     *         if self.num_sectors > 1:
     *             return False
     *         if self.multiplicities[0] > 1:
     *             return False
     * return self.sector_decomposition[0] == self.symmetry.trivial_sector
     */
    if (num_sectors > 1) {
        return false;
    }
    if (multiplicities[0] > 1) {
        return false;
    }
    return sector_decomposition[0] == symmetry.trivial_sector;
}

/// Perform sanity checks.
void Space::test_sanity() {
    /* CHECKME: converted from following python code:
     * assert self.dim >= 0
     * # sectors
     *         if self.sector_decomposition.shape != (self.num_sectors, self.symmetry.sector_ind_len):
     *             raise AssertionError('wrong sectors.shape')
     * assert self.symmetry.are_valid_sectors(self.sector_decomposition), 'invalid sectors'
     * unique, _, _ = self.sector_decomposition.unique_sorted()
     * assert len(unique) == self.num_sectors, 'duplicate sectors'
     *         if self.sector_order == 'sorted':
     *             assert np.all(self.sector_decomposition.lexsort_indices() == np.arange(self.num_sectors)), (
     *                 'wrong sector order'
     *             )
     *         elif self.sector_order == 'dual_sorted':
     *             expect_sorted = self.symmetry.dual_sectors(self.sector_decomposition)
     *             assert np.all(expect_sorted.lexsort_indices() == np.arange(self.num_sectors)), 'wrong sector order'
     *         elif self.sector_order is None:
     *             pass  # nothing to check
     *         else:
     *             raise AssertionError(f'Invalid sector_order: {self.sector_order}')
     * # multiplicities
     * assert np.all(self.multiplicities > 0)
     * assert self.multiplicities.shape == (self.num_sectors,)
     *         if self.symmetry.can_be_dropped:
     *             # slices
     *             assert self.slices.shape == (self.num_sectors, 2)
     *             slice_diffs = self.slices[:, 1] - self.slices[:, 0]
     *             assert np.all(self.sector_dims == self.symmetry.batch_sector_dim(self.sector_decomposition))
     *             expect_diffs = self.sector_dims * self.multiplicities
     *             assert np.all(slice_diffs == expect_diffs)
     *             # slices should be consecutive
     *             if self.num_sectors > 0:
     *                 assert self.slices[0, 0] == 0
     *                 assert np.all(self.slices[1:, 0] == self.slices[:-1, 1])
     *                 assert self.slices[-1, 1] == self.dim
     */
    /* FIXME: Assert: assert self.dim >= 0 */
    // sectors
    if (sector_decomposition.shape != std::make_tuple(num_sectors, symmetry.sector_ind_len)) {
        throw std::runtime_error("wrong sectors.shape");
    }
    /* FIXME: Assert: assert self.symmetry.are_valid_sectors(self.sector_decomposition), 'invalid sectors' */
    std::make_tuple(unique, _, _) = sector_decomposition.unique_sorted();
    /* FIXME: Assert: assert len(unique) == self.num_sectors, 'duplicate sectors' */
    if (sector_order == "sorted") {
        /* FIXME: Assert:             assert np.all(self.sector_decomposition.lexsort_indices() == np.arange(self.num_sectors)), (
                'wrong sector order'
            ) */
    } else {
        if (sector_order == "dual_sorted") {
            auto /* CHECKME: type? */ expect_sorted = symmetry.dual_sectors(sector_decomposition);
            /* FIXME: Assert: assert np.all(expect_sorted.lexsort_indices() == np.arange(self.num_sectors)), 'wrong sector order' */
        } else {
            if (sector_order == py::none()) {
                // pass
                /* nothing to check */
            } else {
                throw std::runtime_error(std::format("Invalid sector_order: {}", sector_order));
            }
        }
    }
    // multiplicities
    /* FIXME: Assert: assert np.all(self.multiplicities > 0) */
    /* FIXME: Assert: assert self.multiplicities.shape == (self.num_sectors,) */
    if (symmetry.can_be_dropped) {
        // slices
        /* FIXME: Assert: assert self.slices.shape == (self.num_sectors, 2) */
        auto /* CHECKME: type? */ slice_diffs = /* Multidimensional slice using std::gslice
 * NOTE: Requires slices_strides to be defined as std::valarray<size_t>
 *       containing the memory strides for each dimension.
 * For a row-major 2D array with shape (n0, n1, ..., n1):
 *   slices_strides = {n1, 1};
 * Alternative: Consider using xtensor, Eigen, or C++23 mdspan
 */
slices[std::gslice(1 * slices_strides[1], std::valarray<std::size_t>{slices.shape(0), 1}, std::valarray<std::size_t>{slices_strides[0], slices_strides[1]})] - /* Multidimensional slice using std::gslice
 * NOTE: Requires slices_strides to be defined as std::valarray<size_t>
 *       containing the memory strides for each dimension.
 * For a row-major 2D array with shape (n0, n1, ..., n1):
 *   slices_strides = {n1, 1};
 * Alternative: Consider using xtensor, Eigen, or C++23 mdspan
 */
slices[std::gslice(0, std::valarray<std::size_t>{slices.shape(0), 1}, std::valarray<std::size_t>{slices_strides[0], slices_strides[1]})];
        /* FIXME: Assert: assert np.all(self.sector_dims == self.symmetry.batch_sector_dim(self.sector_decomposition)) */
        auto /* CHECKME: type? */ expect_diffs = sector_dims * multiplicities;
        /* FIXME: Assert: assert np.all(slice_diffs == expect_diffs) */
        // slices should be consecutive
        if (num_sectors > 0) {
            /* FIXME: Assert: assert self.slices[0, 0] == 0 */
            /* FIXME: Assert: assert np.all(self.slices[1:, 0] == self.slices[:-1, 1]) */
            /* FIXME: Assert: assert self.slices[-1, 1] == self.dim */
        }
    }
}

/// If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.
/// property getter
bool Space::is_trivial() {
    /* CHECKME: converted from following python code:
     *         if self.num_sectors > 1:
     *             return False
     *         if self.multiplicities[0] > 1:
     *             return False
     * return self.sector_decomposition[0] == self.symmetry.trivial_sector
     */
    if (num_sectors > 1) {
        return false;
    }
    if (multiplicities[0] > 1) {
        return false;
    }
    return sector_decomposition[0] == symmetry.trivial_sector;
}

/// If the two spaces are isomorphic, i.e. have the same :attr:`sector_decomposition`.
bool Space::is_isomorphic_to(const Space & other) {
    /* CHECKME: converted from following python code:
     *         if self.symmetry != other.symmetry:
     *             raise SymmetryError('Incompatible symmetries')
     *         if self.num_sectors != other.num_sectors:
     *             return False
     * # find perm1 and perm2 such that ``self.sector_decomposition[perm1]`` and ``other.sector_decomposition[perm2]``
     * # have the same sorting convention and can be directly compared
     *         if self.sector_order is None:
     *             if other.sector_order == 'sorted':
     *                 perm1 = self.sector_decomposition.lexsort_indices()
     *                 perm2 = slice(None, None, None)
     *             elif other.sector_order == 'dual_sorted':
     *                 perm1 = self.symmetry.dual_sectors(self.sector_decomposition).lexsort_indices()
     *                 perm2 = slice(None, None, None)
     *             else:
     *                 perm1 = self.sector_decomposition.lexsort_indices()
     *                 perm2 = other.sector_decomposition.lexsort_indices()
     *         elif other.sector_order is None:
     *             if self.sector_order == 'sorted':
     *                 perm1 = slice(None, None, None)
     *                 perm2 = other.sector_decomposition.lexsort_indices()
     *             elif self.sector_order == 'dual_sorted':
     *                 perm1 = slice(None, None, None)
     *                 perm2 = self.symmetry.dual_sectors(other.sector_decomposition).lexsort_indices()
     *             else:
     *                 raise RuntimeError  # case should have been covered above
     *         elif self.sector_order == other.sector_order:
     *             perm1 = perm2 = slice(None, None, None)
     *         elif self.sector_order == 'sorted':
     *             perm1 = slice(None, None, None)
     *             perm2 = other.sector_decomposition.lexsort_indices()
     *         elif other.sector_order == 'sorted':
     *             perm1 = self.sector_decomposition.lexsort_indices()
     *             perm2 = slice(None, None, None)
     *         else:
     *             raise RuntimeError  # all cases should have been covered.
     *         if not np.all(self.multiplicities[perm1] == other.multiplicities[perm2]):
     *             return False
     * return self.sector_decomposition[perm1] == other.sector_decomposition[perm2]
     */
    if (symmetry != other.symmetry) {
        throw std::runtime_error("Incompatible symmetries");
    }
    if (num_sectors != other.num_sectors) {
        return false;
    }
    // find perm1 and perm2 such that ``self.sector_decomposition[perm1]`` and ``other.sector_decomposition[perm2]``
    // have the same sorting convention and can be directly compared
    if (sector_order == py::none()) {
        if (other.sector_order == "sorted") {
            auto /* CHECKME: type? */ perm1 = sector_decomposition.lexsort_indices();
            auto /* CHECKME: type? */ perm2 = slice(py::none(), py::none(), py::none());
        } else {
            if (other.sector_order == "dual_sorted") {
                perm1 = symmetry.dual_sectors(sector_decomposition).lexsort_indices();
                perm2 = slice(py::none(), py::none(), py::none());
            } else {
                perm1 = sector_decomposition.lexsort_indices();
                perm2 = other.sector_decomposition.lexsort_indices();
            }
        }
    } else {
        if (other.sector_order == py::none()) {
            if (sector_order == "sorted") {
                perm1 = slice(py::none(), py::none(), py::none());
                perm2 = other.sector_decomposition.lexsort_indices();
            } else {
                if (sector_order == "dual_sorted") {
                    perm1 = slice(py::none(), py::none(), py::none());
                    perm2 = symmetry.dual_sectors(other.sector_decomposition).lexsort_indices();
                } else {
                    throw std::runtime_error("");
                    /* case should have been covered above */
                }
            }
        } else {
            if (sector_order == other.sector_order) {
                /* FIXME: multiple assignment: perm1 = perm2 = slice(None, None, None) */
            } else {
                if (sector_order == "sorted") {
                    perm1 = slice(py::none(), py::none(), py::none());
                    perm2 = other.sector_decomposition.lexsort_indices();
                } else {
                    if (other.sector_order == "sorted") {
                        perm1 = sector_decomposition.lexsort_indices();
                        perm2 = slice(py::none(), py::none(), py::none());
                    } else {
                        throw std::runtime_error("");
                        /* all cases should have been covered. */
                    }
                }
            }
        }
    }
    if (!(np.all(multiplicities[perm1] == other.multiplicities[perm2]))) {
        return false;
    }
    return sector_decomposition[perm1] == other.sector_decomposition[perm2];
}

/// Whether self is (isomorphic to) a subspace of other.
bool Space::is_subspace_of(const Space & other) {
    /* CHECKME: converted from following python code:
     *         if not self.symmetry.is_equivalent_to(other.symmetry):
     *             return False
     *         if self.num_sectors == 0:
     *             return True
     *         if self.sector_order == 'sorted' == other.sector_order:
     *             # sectors are sorted, so we can just iterate over both of them
     *             n_self = 0
     *             for other_sector, other_mult in zip(other.sector_decomposition, other.multiplicities):
     *                 if self.sector_decomposition[n_self] == other_sector:
     *                     if self.multiplicities[n_self] > other_mult:
     *                         return False
     *                     n_self += 1
     *                 if n_self == self.num_sectors:
     *                     # have checked all sectors of self
     *                     return True
     *             # reaching this line means self has sectors which other does not have
     *             return False
     * # OPTIMIZE sort once instead of looking up each time
     * num_sectors_checked = 0
     *         for sector, mult in zip(other.sector_decomposition, other.multiplicities):
     *             m = self.sector_multiplicity(sector)
     *             if m == 0:
     *                 continue
     *             if m > mult:
     *                 return False
     *             num_sectors_checked += 1
     *         if num_sectors_checked < self.num_sectors:
     *             # this means self has some sectors that other doesn't have
     *             return False
     * return True
     */
    if (!symmetry.is_equivalent_to(other.symmetry)) {
        return false;
    }
    if (num_sectors == 0) {
        return true;
    }
    if (sector_order == "sorted" == other.sector_order) {
        // sectors are sorted, so we can just iterate over both of them
        auto /* CHECKME: type? */ n_self = 0;
        /* NOTE: Using C++23 std::views::zip (requires -std=c++23)
         * For C++20, use std::ranges::views::zip or implement custom zip.
         * Alternative libraries: range-v3 (ranges::views::zip)
         */
        for (auto [other_sector, other_mult] : std::views::zip(other.sector_decomposition, other.multiplicities)) {
            if (sector_decomposition[n_self] == other_sector) {
                if (multiplicities[n_self] > other_mult) {
                    return false;
                }
                n_self += 1;
            }
            if (n_self == num_sectors) {
                // have checked all sectors of self
                return true;
            }
        }
        // reaching this line means self has sectors which other does not have
        return false;
    }
    // OPTIMIZE sort once instead of looking up each time
    auto /* CHECKME: type? */ num_sectors_checked = 0;
    /* NOTE: Using C++23 std::views::zip (requires -std=c++23)
     * For C++20, use std::ranges::views::zip or implement custom zip.
     * Alternative libraries: range-v3 (ranges::views::zip)
     */
    for (auto [sector, mult] : std::views::zip(other.sector_decomposition, other.multiplicities)) {
        auto /* CHECKME: type? */ m = sector_multiplicity(sector);
        if (m == 0) {
            continue;
        }
        if (m > mult) {
            return false;
        }
        num_sectors_checked += 1;
    }
    if (num_sectors_checked < num_sectors) {
        // this means self has some sectors that other doesn't have
        return false;
    }
    return true;
}

/// Convert to an isomorphic :class:`ElementarySpace`.
py::object Space::as_ElementarySpace(bool is_dual) {
    /* CHECKME: converted from following python code:
     *         if is_dual:
     *             defining_sectors = self.symmetry.dual_sectors(self.sector_decomposition)
     *             is_sorted = self.sector_order == 'dual_sorted'
     *         else:
     *             defining_sectors = self.sector_decomposition
     *             is_sorted = self.sector_order == 'sorted'
     *         if is_sorted:
     *             return ElementarySpace(
     *                 symmetry=self.symmetry,
     *                 defining_sectors=defining_sectors,
     *                 multiplicities=self.multiplicities,
     *                 is_dual=is_dual,
     *             )
     *         return ElementarySpace.from_defining_sectors(
     *             symmetry=self.symmetry,
     *             defining_sectors=defining_sectors,
     *             multiplicities=self.multiplicities,
     *             is_dual=is_dual,
     *             unique_sectors=True,
     *         )
     */
    if (is_dual) {
        auto /* CHECKME: type? */ defining_sectors = symmetry.dual_sectors(sector_decomposition);
        auto /* CHECKME: type? */ is_sorted = sector_order == "dual_sorted";
    } else {
        defining_sectors = sector_decomposition;
        is_sorted = sector_order == "sorted";
    }
    if (is_sorted) {
        return ElementarySpace(py::arg({keyword.key}) = val, py::arg({keyword.key}) = val, py::arg({keyword.key}) = val, py::arg({keyword.key}) = val) /* CHECKME: keywords ['symmetry', 'defining_sectors', 'multiplicities', 'is_dual'] */;
    }
    return ElementarySpace.from_defining_sectors(py::arg({keyword.key}) = val, py::arg({keyword.key}) = val, py::arg({keyword.key}) = val, py::arg({keyword.key}) = val, py::arg({keyword.key}) = val) /* CHECKME: keywords ['symmetry', 'defining_sectors', 'multiplicities', 'is_dual', 'unique_sectors'] */;
}

Ptr Space::as_Space() {
    /* CHECKME: converted from following python code:
     * return self
     */
    return self  /* FIXME: standalone self reference */;
}

/// Find the index of a given sector in the :attr:`sector_decomposition`.
std::optional<int64> Space::sector_decomposition_where(Sector sector) {
    /* CHECKME: converted from following python code:
     * # OPTIMIZE : if sector_order allows it, use that sectors are sorted to speed up the lookup
     * return self.sector_decomposition.row_where(as_sector(sector))
     */
    // OPTIMIZE : if sector_order allows it, use that sectors are sorted to speed up the lookup
    return sector_decomposition.row_where(as_sector(sector));
}

/// The multiplicity of a given sector in the :attr:`sector_decomposition`.
int64 Space::sector_multiplicity(Sector sector) {
    /* CHECKME: converted from following python code:
     * idx = self.sector_decomposition_where(sector)
     *         if idx is None:
     *             return 0
     * return self.multiplicities[idx]
     */
    auto /* CHECKME: type? */ idx = sector_decomposition_where(sector);
    if (idx == py::none()) {
        return 0;
    }
    return multiplicities[idx];
}

} // namespace cyten
