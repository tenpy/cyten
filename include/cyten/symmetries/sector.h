#pragma once

#include "../cyten.h"

#include <array>
#include <cassert>
#include <compare>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

/// Maximum number of integers in a Sector (product / SUN length cap).
inline constexpr std::size_t max_sector_ind_len = 7;

/// Owning sector: fixed capacity, runtime length. Fits in 16 bytes (128 bit).
///
/// Exposed to Python as ``cyten.Sector``. Factor helpers should view storage via
/// ``as_span()`` / ``subspan()`` with a compile-time length, not via a separate owning
/// fixed-N type.
///
/// ``len()`` is always in ``[0, max_sector_ind_len]`` and is fixed at construction
/// (default, initializer list, or ``from_span``).
class Sector
{
  public:
    std::array<int16_t, max_sector_ind_len> q{};

    Sector() = default;

    Sector(std::initializer_list<int16_t> values)
    {
        if (values.size() > max_sector_ind_len) {
            throw std::invalid_argument("Sector length exceeds max_sector_ind_len");
        }
        len_ = static_cast<std::uint8_t>(values.size());
        std::size_t i = 0;
        for (int16_t v : values) {
            q[i++] = v;
        }
    }

    /// Copy ``values`` into a new sector. Throws if ``values.size() > max_sector_ind_len``.
    static Sector from_span(std::span<const int16_t> values)
    {
        if (values.size() > max_sector_ind_len) {
            throw std::invalid_argument("Sector length exceeds max_sector_ind_len");
        }
        Sector s;
        s.len_ = static_cast<std::uint8_t>(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            s.q[i] = values[i];
        }
        return s;
    }

    /// Zero-filled sector of the given length.
    static Sector zeros(std::uint8_t len)
    {
        if (len > max_sector_ind_len) {
            throw std::invalid_argument("Sector length exceeds max_sector_ind_len");
        }
        Sector s;
        s.len_ = len;
        return s;
    }

    /// Current number of components; always ``<= max_sector_ind_len``.
    [[nodiscard]] std::uint8_t len() const noexcept { return len_; }

    std::span<int16_t> span() { return { q.data(), len_ }; }
    std::span<const int16_t> span() const { return { q.data(), len_ }; }

    template<std::size_t N>
    std::span<int16_t, N> as_span()
    {
        assert(len_ == N);
        return std::span<int16_t, N>{ q.data(), N };
    }

    template<std::size_t N>
    std::span<const int16_t, N> as_span() const
    {
        assert(len_ == N);
        return std::span<const int16_t, N>{ q.data(), N };
    }

    /// View ``N`` consecutive components starting at ``offset`` (e.g. a factor slice).
    template<std::size_t N>
    std::span<int16_t, N> subspan(std::size_t offset)
    {
        assert(offset + N <= len_);
        return std::span<int16_t, N>{ q.data() + offset, N };
    }

    template<std::size_t N>
    std::span<const int16_t, N> subspan(std::size_t offset) const
    {
        assert(offset + N <= len_);
        return std::span<const int16_t, N>{ q.data() + offset, N };
    }

    int16_t& operator[](std::size_t i)
    {
        assert(i < len_);
        return q[i];
    }

    int16_t operator[](std::size_t i) const
    {
        assert(i < len_);
        return q[i];
    }

    friend bool operator==(Sector const& a, Sector const& b) noexcept
    {
        if (a.len_ != b.len_) {
            return false;
        }
        for (std::uint8_t i = 0; i < a.len_; ++i) {
            if (a.q[i] != b.q[i]) {
                return false;
            }
        }
        return true;
    }

    friend std::strong_ordering operator<=>(Sector const& a, Sector const& b) noexcept
    {
        auto const n = a.len_ < b.len_ ? a.len_ : b.len_;
        for (std::uint8_t i = 0; i < n; ++i) {
            if (auto cmp = a.q[i] <=> b.q[i]; cmp != 0) {
                return cmp;
            }
        }
        return a.len_ <=> b.len_;
    }

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    /// A batch of sectors with shape ``(num_sectors, sector_ind_len)``.
    static Sector from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);

  private:
    std::uint8_t len_ = 0;
};

static_assert(sizeof(Sector) == 16);
static_assert(alignof(Sector) <= 16);

/// Batch of sectors with a shared ``sector_ind_len`` (including when empty).
///
/// Subclasses ``std::vector<Sector>`` so ``operator[]`` returns ``Sector&``.
/// All sectors must have the same ``len()``; this is checked at construction /
/// when appending via the typed mutators below.
class SectorArray : public std::vector<Sector>
{
  public:
    using Base = std::vector<Sector>;

    SectorArray() = default;

    /// ``n`` zero-filled sectors of length ``sector_ind_len_``.
    SectorArray(std::size_t n, std::uint8_t sector_ind_len_);

    /// Construct from an existing list of sectors (must share the same length).
    explicit SectorArray(std::vector<Sector> sectors);

    /// Empty array that still remembers ``sector_ind_len`` (hides ``vector::empty()``).
    static SectorArray empty(std::uint8_t sector_ind_len_);

    static SectorArray from_sector(Sector const& sector);

    static SectorArray repeat(Sector const& sector, std::size_t n);

    [[nodiscard]] std::uint8_t sector_ind_len() const noexcept { return sector_ind_len_; }

    /// Hide base ``resize`` that would default-construct ``Sector{}`` (len 0).
    void resize(std::size_t n);
    void resize(std::size_t n, Sector const& fill);

    void push_back(Sector const& s);
    void push_back(Sector&& s);

    [[nodiscard]] std::vector<std::size_t> lexsort_indices() const;

    /// ``(sorted_sectors, permutation)`` with ``sorted = (*this)[perm]``.
    [[nodiscard]] std::pair<SectorArray, std::vector<std::size_t>> sorted() const;

    [[nodiscard]] std::vector<std::size_t> find_row_differences(bool include_len = false) const;

    /// Sort then merge duplicate rows; sum multiplicities.
    /// Returns ``(unique_sorted_sectors, multiplicities, perm)``.
    [[nodiscard]] std::tuple<SectorArray, std::vector<std::int64_t>, std::vector<std::size_t>>
    unique_sorted(std::vector<std::int64_t> const& multiplicities) const;

    [[nodiscard]] std::optional<std::size_t> row_where(Sector const& sector) const;

    [[nodiscard]] SectorArray concat(SectorArray const& other) const;

    [[nodiscard]] SectorArray take(std::span<const std::size_t> indices) const;

    [[nodiscard]] SectorArray take_mask(std::vector<bool> const& mask) const;

    [[nodiscard]] SectorArray slice(std::size_t start, std::size_t stop) const;

    /// Merge walk of two lex-sorted SectorArrays (``np.lexsort`` order).
    static void iter_common_sorted(
      SectorArray const& a,
      SectorArray const& b,
      bool a_strict,
      bool b_strict,
      std::function<void(std::ptrdiff_t, std::ptrdiff_t)> const& yield);

    friend bool operator==(SectorArray const& a, SectorArray const& b) noexcept
    {
        return a.sector_ind_len_ == b.sector_ind_len_ &&
               static_cast<Base const&>(a) == static_cast<Base const&>(b);
    }

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    static SectorArray from_hdf5(py::object hdf5_loader,
                                 py::object h5gr,
                                 std::string const& subpath);

  private:
    std::uint8_t sector_ind_len_ = 0;

    void check_sector_len(Sector const& s) const;

    /// Lex compare of rows matching ``np.lexsort(sectors.T)`` (last column primary).
    static int cmp_lexsort(SectorArray const& a,
                           std::size_t i,
                           SectorArray const& b,
                           std::size_t j) noexcept;
};

} // namespace cyten

template<>
struct std::hash<cyten::Sector>
{
    std::size_t operator()(cyten::Sector const& s) const noexcept
    {
        std::size_t h = s.len();
        for (std::uint8_t i = 0; i < s.len(); ++i) {
            h ^= static_cast<std::size_t>(static_cast<std::uint16_t>(s.q[i]) + 0x9e3779b9u +
                                          (h << 6) + (h >> 2));
        }
        return h;
    }
};
