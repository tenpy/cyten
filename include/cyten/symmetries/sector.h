#pragma once

#include "../cyten.h"

#include <array>
#include <cassert>
#include <compare>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyten {

/// Maximum number of integers in a Sector (product / SUN length cap).
inline constexpr std::size_t max_sector_ind_len = 7;

/// Owning sector: fixed capacity, runtime length. Fits in 16 bytes (128 bit).
///
/// Exposed to Python as ``cyten.Sector``. Factor helpers should view storage via
/// ``as_span<N>()`` / ``subspan<N>()``, not via a separate owning fixed-N type.
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

    static Sector from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);

  private:
    std::uint8_t len_ = 0;
};

static_assert(sizeof(Sector) == 16);
static_assert(alignof(Sector) <= 16);

/// Contiguous batch of sectors: shape ``(num_sectors, sector_ind_len)``, row-major ``int16_t``.
///
/// Prefer this over ``std::vector<Sector>`` for batch ops and NumPy interoperability.
struct SectorArray
{
    std::vector<int16_t> data;
    std::size_t num_sectors = 0;
    std::uint8_t sector_ind_len = 0;

    SectorArray() = default;

    SectorArray(std::size_t num_sectors_, std::uint8_t sector_ind_len_)
      : data(num_sectors_ * static_cast<std::size_t>(sector_ind_len_), 0)
      , num_sectors(num_sectors_)
      , sector_ind_len(sector_ind_len_)
    {
        if (sector_ind_len_ > max_sector_ind_len) {
            throw std::invalid_argument("SectorArray sector_ind_len exceeds max_sector_ind_len");
        }
    }

    static SectorArray empty(std::uint8_t sector_ind_len_)
    {
        return SectorArray(0, sector_ind_len_);
    }

    std::span<int16_t> row(std::size_t i)
    {
        assert(i < num_sectors);
        auto const off = i * static_cast<std::size_t>(sector_ind_len);
        return { data.data() + off, sector_ind_len };
    }

    std::span<const int16_t> row(std::size_t i) const
    {
        assert(i < num_sectors);
        auto const off = i * static_cast<std::size_t>(sector_ind_len);
        return { data.data() + off, sector_ind_len };
    }

    template<std::size_t N>
    std::span<int16_t, N> row_as_span(std::size_t i)
    {
        assert(sector_ind_len == N);
        assert(i < num_sectors);
        auto const off = i * N;
        return std::span<int16_t, N>{ data.data() + off, N };
    }

    template<std::size_t N>
    std::span<const int16_t, N> row_as_span(std::size_t i) const
    {
        assert(sector_ind_len == N);
        assert(i < num_sectors);
        auto const off = i * N;
        return std::span<const int16_t, N>{ data.data() + off, N };
    }

    Sector operator[](std::size_t i) const
    {
        assert(i < num_sectors);
        return Sector::from_span(row(i));
    }

    void set(std::size_t i, Sector const& s)
    {
        assert(i < num_sectors);
        assert(s.len() == sector_ind_len);
        auto r = row(i);
        for (std::uint8_t j = 0; j < sector_ind_len; ++j) {
            r[j] = s.q[j];
        }
    }

    friend bool operator==(SectorArray const& a, SectorArray const& b) noexcept
    {
        return a.num_sectors == b.num_sectors && a.sector_ind_len == b.sector_ind_len &&
               a.data == b.data;
    }

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    static SectorArray from_hdf5(py::object hdf5_loader,
                                 py::object h5gr,
                                 std::string const& subpath);
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
