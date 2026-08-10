#include <cyten/symmetries/factors/zn_anyon_category.h>

#include <cyten/symmetries/topo_ones.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyten {

namespace {

complex128
make_phase(int n, int N)
{
    return std::exp(complex128{ 0.0,
                                2.0 * std::numbers::pi_v<float64> * static_cast<float64>(n) /
                                  static_cast<float64>(N) });
}

complex128
phase_pow(complex128 phase, int exp)
{
    return std::pow(phase, static_cast<float64>(exp));
}

FusionSymbol
phase_r_symbol(complex128 val)
{
    return FusionSymbol::scalar1d(val, Dtype::Complex128);
}

FusionSymbol
phase_c_symbol(complex128 val)
{
    return topo_ones::one_4D() * val;
}

} // namespace

ZNAnyonCategory::ZNAnyonCategory(int N_, int n_, std::optional<std::string> descriptive_name)
  : SymmetryFactor(FusionStyle::single,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
                   "ℤ_" + std::to_string(N_) + "^" + std::to_string(n_ % N_) + " anyon category",
                   static_cast<float64>(N_),
                   (n_ % N_) > 0,
                   std::move(descriptive_name))
  , N(N_)
  , n(n_ % N_)
  , _phase(make_phase(n, N))
{
    if (N <= 1) {
        throw std::invalid_argument("invalid ZNAnyonCategory(N=" + std::to_string(N) + ")");
    }
}

bool
ZNAnyonCategory::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] < N;
}

bool
ZNAnyonCategory::are_valid_sectors(SectorArray const& sectors) const
{
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
ZNAnyonCategory::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 1);
    SectorArray bb(1, 1);
    aa[0] = a;
    bb[0] = b;
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
ZNAnyonCategory::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    SectorArray out(a.size(), 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i][0] = topo_ones::mod_n(static_cast<int32_t>(a[i][0]) + b[i][0], N);
    }
    return out;
}

SectorArray
ZNAnyonCategory::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i][0] = topo_ones::mod_n(static_cast<int32_t>(out[i][0]) + sectors[s][i][0], N);
        }
    }
    return out;
}

int64
ZNAnyonCategory::sector_dim(Sector /*a*/) const
{
    return 1;
}

std::vector<int64>
ZNAnyonCategory::batch_sector_dim(SectorArray const& a) const
{
    return std::vector<int64>(a.size(), 1);
}

std::vector<float64>
ZNAnyonCategory::batch_qdim(SectorArray const& a) const
{
    return std::vector<float64>(a.size(), 1.0);
}

std::string
ZNAnyonCategory::repr() const
{
    if (!descriptive_name.has_value()) {
        return "ZNAnyonCategory(" + std::to_string(N) + ", " + std::to_string(n) + ")";
    }
    return "ZNAnyonCategory(" + std::to_string(N) + ", " + std::to_string(n) + ", \"" +
           *descriptive_name + "\")";
}

bool
ZNAnyonCategory::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* cat = dynamic_cast<ZNAnyonCategory const*>(&other)) {
        return cat->N == N && cat->n == n;
    }
    return false;
}

Sector
ZNAnyonCategory::dual_sector(Sector a) const
{
    return Sector{ topo_ones::mod_n(-static_cast<int32_t>(a.q[0]), N) };
}

SectorArray
ZNAnyonCategory::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.size(), 1);
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        out[i][0] = topo_ones::mod_n(-static_cast<int32_t>(sectors[i][0]), N);
    }
    return out;
}

int64
ZNAnyonCategory::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

FusionSymbol
ZNAnyonCategory::_f_symbol(Sector /*a*/,
                           Sector /*b*/,
                           Sector /*c*/,
                           Sector /*d*/,
                           Sector /*e*/,
                           Sector /*f*/) const
{
    return topo_ones::one_4D();
}

int64
ZNAnyonCategory::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
ZNAnyonCategory::qdim(Sector /*a*/) const
{
    return 1.0;
}

FusionSymbol
ZNAnyonCategory::_r_symbol(Sector a, Sector b, Sector /*c*/) const
{
    int const exp = static_cast<int>(a.q[0]) * static_cast<int>(b.q[0]);
    return phase_r_symbol(phase_pow(_phase, exp));
}

FusionSymbol
ZNAnyonCategory::_c_symbol(Sector /*a*/,
                           Sector b,
                           Sector c,
                           Sector /*d*/,
                           Sector /*e*/,
                           Sector /*f*/) const
{
    int const exp = static_cast<int>(b.q[0]) * static_cast<int>(c.q[0]);
    return phase_c_symbol(phase_pow(_phase, exp));
}

SectorArray
ZNAnyonCategory::all_sectors() const
{
    SectorArray out(static_cast<std::size_t>(N), 1);
    for (int i = 0; i < N; ++i) {
        out[static_cast<std::size_t>(i)][0] = static_cast<int16_t>(i);
    }
    return out;
}

void
ZNAnyonCategory::save_hdf5(py::object hdf5_saver,
                           py::object h5gr,
                           std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(N, subpath + "N");
    hdf5_saver.attr("save")(n, subpath + "n");
}

ZNAnyonCategory::Ptr
ZNAnyonCategory::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    int N = hdf5_loader.attr("load")(subpath + "N").cast<int>();
    int n = hdf5_loader.attr("load")(subpath + "n").cast<int>();
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    auto obj = std::make_shared<ZNAnyonCategory>(N, n, name);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
