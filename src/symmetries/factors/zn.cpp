#include <cyten/symmetries/zn.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace cyten {

namespace {

/// Python-style non-negative remainder for ``x % N`` with ``N > 0``.
int16_t
mod_n(int32_t x, int N)
{
    int r = static_cast<int>(x % N);
    if (r < 0) {
        r += N;
    }
    return static_cast<int16_t>(r);
}

std::string
subscript_digits(int N)
{
    static constexpr char const* map[] = { "₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉" };
    std::string out;
    for (char c : std::to_string(N)) {
        if (c >= '0' && c <= '9') {
            out += map[c - '0'];
        }
    }
    return out;
}

} // namespace

ZN::ZN(int N_, std::optional<std::string> descriptive_name, bool trivial_shift)
  : AbelianGroup(Sector{ 0 },
                 std::string("ℤ") + subscript_digits(N_),
                 static_cast<float64>(N_),
                 std::move(descriptive_name),
                 trivial_shift)
  , N(N_)
{
    if (N <= 1) {
        throw std::invalid_argument("invalid ZN(N=" + std::to_string(N) + ")");
    }
}

bool
ZN::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] < N;
}

bool
ZN::are_valid_sectors(SectorArray const& sectors) const
{
    // Intentional fix vs Python bug ``np.all(0 < self.N)``: require ``0 <= q < N``.
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto q = sectors[i][0];
        if (q < 0 || q >= N) {
            return false;
        }
    }
    return true;
}

SectorArray
ZN::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 1);
    SectorArray bb(1, 1);
    aa[0] = a;
    bb[0] = b;
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
ZN::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    SectorArray out(a.size(), 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i][0] = mod_n(static_cast<int32_t>(a[i][0]) + b[i][0], N);
    }
    return out;
}

SectorArray
ZN::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i][0] = mod_n(static_cast<int32_t>(out[i][0]) + sectors[s][i][0], N);
        }
    }
    return out;
}

Sector
ZN::dual_sector(Sector a) const
{
    return Sector{ mod_n(-static_cast<int32_t>(a.q[0]), N) };
}

SectorArray
ZN::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.size(), 1);
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        out[i][0] = mod_n(-static_cast<int32_t>(sectors[i][0]), N);
    }
    return out;
}

SectorArray
ZN::all_sectors() const
{
    SectorArray out(static_cast<std::size_t>(N), 1);
    for (int i = 0; i < N; ++i) {
        out[static_cast<std::size_t>(i)][0] = static_cast<int16_t>(i);
    }
    return out;
}

std::string
ZN::repr() const
{
    if (!descriptive_name.has_value()) {
        return "ZNSymmetry(" + std::to_string(N) + ")";
    }
    return "ZNSymmetry(" + std::to_string(N) + ", \"" + *descriptive_name + "\")";
}

bool
ZN::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* zn = dynamic_cast<ZN const*>(&other)) {
        return zn->N == N;
    }
    return false;
}

void
ZN::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(N, subpath + "N");
}

ZN::Ptr
ZN::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    int N = hdf5_loader.attr("load")(subpath + "N").cast<int>();
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    bool trivial_shift = trivial_shift_from_hdf5(hdf5_loader, subpath);
    auto obj = std::make_shared<ZN>(N, name, trivial_shift);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
