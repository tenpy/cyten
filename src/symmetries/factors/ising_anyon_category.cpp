#include <cyten/symmetries/factors/ising_anyon_category.h>

#include <cyten/symmetries/topo_ones.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

Sector const IsingAnyonCategory::vacuum{ 0 };
Sector const IsingAnyonCategory::sigma{ 1 };
Sector const IsingAnyonCategory::psi{ 2 };

namespace {

Sector
sector1(int16_t q)
{
    return Sector{ q };
}

FusionSymbol
default_c_symbol(BaseSymmetry const& sym,
                 Sector a,
                 Sector b,
                 Sector c,
                 Sector d,
                 Sector e,
                 Sector f)
{
    auto R1 = sym._r_symbol(e, c, d);
    auto F = sym._f_symbol(c, a, b, d, e, f);
    auto R2 = sym._r_symbol(a, c, f).conj();
    auto R1e = R1.reshaped(4, FusionSymbol::Shape{ { 1, R1.size(), 1, 1 } });
    auto R2e = R2.reshaped(4, FusionSymbol::Shape{ { 1, 1, R2.size(), 1 } });
    return R1e.multiply(F).multiply(R2e);
}

FusionSymbol
scaled_one_4D(complex128 factor)
{
    return topo_ones::one_4D() * factor;
}

std::array<int64, 3>
make_frobenius_array(int nu)
{
    int64_t const exp = (static_cast<int64_t>(nu) * nu - 1) / 8;
    int64_t const fs1 = (exp % 2 == 0) ? 1 : -1;
    return { 1, fs1, 1 };
}

FusionSymbol
make_f_table(std::array<int64, 3> const& frobenius)
{
    auto const fs1 = static_cast<float64>(frobenius[1]);
    auto const inv_sqrt2 = 1.0 / std::sqrt(2.0);
    return FusionSymbol::from_float64(
      1,
      FusionSymbol::Shape{ { 5, 1, 1, 1 } },
      { 1.0 * fs1 * inv_sqrt2, 0.0, 1.0 * fs1 * inv_sqrt2, 0.0, -1.0 * fs1 * inv_sqrt2 });
}

FusionSymbol
make_r_table(int nu, std::array<int64, 3> const& frobenius)
{
    auto const fs1 = static_cast<float64>(frobenius[1]);
    auto const pi = std::numbers::pi_v<float64>;
    std::vector<complex128> vals{
        std::pow(complex128{ 0.0, -1.0 }, nu),
        complex128{ -1.0, 0.0 },
        std::exp(complex128{ 0.0, 3.0 * nu * pi / 8.0 }) * fs1,
        std::exp(complex128{ 0.0, -static_cast<float64>(nu) * pi / 8.0 }) * fs1,
        complex128{ 0.0, 0.0 },
    };
    return FusionSymbol::from_complex128(
      1, FusionSymbol::Shape{ { 5, 1, 1, 1 } }, std::move(vals));
}

SectorArray
fusion_map_lookup(int16_t a, int16_t b)
{
    switch (a * a + b * b) {
        case 0: {
            SectorArray out(1, 1);
            out[0][0] = 0;
            return out;
        }
        case 1: {
            SectorArray out(1, 1);
            out[0][0] = 1;
            return out;
        }
        case 2: {
            SectorArray out(2, 1);
            out[0][0] = 0;
            out[1][0] = 2;
            return out;
        }
        case 4: {
            SectorArray out(1, 1);
            out[0][0] = 2;
            return out;
        }
        case 5: {
            SectorArray out(1, 1);
            out[0][0] = 1;
            return out;
        }
        case 8: {
            SectorArray out(1, 1);
            out[0][0] = 0;
            return out;
        }
        default:
            throw std::invalid_argument("invalid Ising fusion inputs");
    }
}

bool
all_sigma(Sector b, Sector c)
{
    return b.q[0] == 1 && c.q[0] == 1;
}

bool
all_nontrivial(Sector a, Sector b)
{
    return a.q[0] != 0 && b.q[0] != 0;
}

bool
sectors_are(Sector a, Sector b, Sector c, Sector d, int16_t va, int16_t vb, int16_t vc, int16_t vd)
{
    return a.q[0] == va && b.q[0] == vb && c.q[0] == vc && d.q[0] == vd;
}

FusionSymbol
zero_c_entry()
{
    return FusionSymbol::zeros(4, FusionSymbol::Shape{ { 1, 1, 1, 1 } }, Dtype::Complex128);
}

} // namespace

IsingAnyonCategory::IsingAnyonCategory(int nu_)
  : SymmetryFactor(FusionStyle::multiple_unique,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
                   // --- hints from Python IsingAnyonCategory.__init__ ---
                   // nontrivial F-symbols
                   // nontrivial R-symbols
                   // nontrivial C-symbols
                   // ---
                   "IsingAnyonCategory",
                   3.0,
                   true)
  , nu(((nu_ % 16) + 16) % 16)
  , frobenius(make_frobenius_array(nu))
  , _f(make_f_table(frobenius))
  , _r(make_r_table(nu, frobenius))
{
    if (nu_ % 2 == 0) {
        throw std::invalid_argument("IsingAnyonCategory nu must be odd");
    }
    auto const phase = std::pow(complex128{ 0.0, -1.0 }, nu);
    auto const neg_phase = -phase;
    _c = {
        scaled_one_4D(phase),
        scaled_one_4D(neg_phase),
        default_c_symbol(
          *this, sector1(0), sector1(1), sector1(1), sector1(0), sector1(1), sector1(1)),
        default_c_symbol(
          *this, sector1(0), sector1(1), sector1(1), sector1(2), sector1(1), sector1(1)),
        default_c_symbol(
          *this, sector1(1), sector1(1), sector1(1), sector1(1), sector1(0), sector1(0)),
        default_c_symbol(
          *this, sector1(1), sector1(1), sector1(1), sector1(1), sector1(0), sector1(2)),
        default_c_symbol(
          *this, sector1(1), sector1(1), sector1(1), sector1(1), sector1(2), sector1(2)),
        zero_c_entry(),
        default_c_symbol(
          *this, sector1(2), sector1(1), sector1(1), sector1(0), sector1(1), sector1(1)),
        default_c_symbol(
          *this, sector1(2), sector1(1), sector1(1), sector1(2), sector1(1), sector1(1)),
        scaled_one_4D(complex128{ -1.0, 0.0 }),
    };
}

bool
IsingAnyonCategory::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] < 3;
}

bool
IsingAnyonCategory::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto q = sectors[i][0];
        if (q < 0 || q >= 3) {
            return false;
        }
    }
    return true;
}

SectorArray
IsingAnyonCategory::fusion_outcomes(Sector a, Sector b) const
{
    return fusion_map_lookup(a.q[0], b.q[0]);
}

std::string
IsingAnyonCategory::sector_str(Sector a) const
{
    if (a.q[0] == 1) {
        return "sigma";
    }
    return a.q[0] == 0 ? "vacuum" : "psi";
}

std::string
IsingAnyonCategory::repr() const
{
    return "IsingAnyonCategory(nu=" + std::to_string(nu) + ")";
}

bool
IsingAnyonCategory::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* cat = dynamic_cast<IsingAnyonCategory const*>(&other)) {
        return cat->nu == nu;
    }
    return false;
}

Sector
IsingAnyonCategory::dual_sector(Sector a) const
{
    return a;
}

SectorArray
IsingAnyonCategory::dual_sectors(SectorArray const& sectors) const
{
    return sectors;
}

int64
IsingAnyonCategory::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

FusionSymbol
IsingAnyonCategory::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (sectors_are(a, b, c, d, 1, 1, 1, 1)) {
        auto idx = static_cast<std::size_t>(e.q[0] + f.q[0]);
        return FusionSymbol::full(
          4, FusionSymbol::Shape{ { 1, 1, 1, 1 } }, _f.get_complex(idx), Dtype::Float64);
    }
    if (sectors_are(a, b, c, d, 2, 1, 2, 1)) {
        return scaled_one_4D(complex128{ -1.0, 0.0 });
    }
    if (sectors_are(a, b, c, d, 1, 2, 1, 2)) {
        return scaled_one_4D(complex128{ -1.0, 0.0 });
    }
    return topo_ones::one_4D();
}

int64
IsingAnyonCategory::frobenius_schur(Sector a) const
{
    return frobenius.at(static_cast<std::size_t>(a.q[0]));
}

float64
IsingAnyonCategory::qdim(Sector a) const
{
    return a.q[0] == 1 ? std::sqrt(2.0) : 1.0;
}

std::vector<float64>
IsingAnyonCategory::batch_qdim(SectorArray const& a) const
{
    std::vector<float64> out(a.size());
    auto const sqrt2 = std::sqrt(2.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = (a[i][0] == 1) ? sqrt2 : 1.0;
    }
    return out;
}

FusionSymbol
IsingAnyonCategory::_r_symbol(Sector a, Sector b, Sector c) const
{
    if (all_nontrivial(a, b)) {
        // Match NumPy negative indexing used by the original formula.
        auto row = static_cast<std::ptrdiff_t>((a.q[0] + b.q[0]) * (c.q[0] - 1));
        auto const n = static_cast<std::ptrdiff_t>(_r.extent(0));
        if (row < 0) {
            row += n;
        }
        return FusionSymbol::scalar1d(_r.get_complex(static_cast<std::size_t>(row)),
                                      Dtype::Complex128);
    }
    return topo_ones::one_1D();
}

FusionSymbol
IsingAnyonCategory::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    // --- hints from Python IsingAnyonCategory._c_symbol ---
    // = 0 if σ and ψ or σ and ψ, 1 otherwise
    // ---
    if (all_sigma(b, c)) {
        int64_t factor = -1 * (b.q[0] - c.q[0] - 1) * (b.q[0] - c.q[0] + 1);
        factor *= (1 - a.q[0] / 2 - d.q[0] / 2 + 9 * (b.q[0] - 1) +
                   (2 - b.q[0]) * ((e.q[0] + f.q[0]) / 2 + d.q[0] / 2 + 3 * a.q[0]));
        auto const idx = static_cast<std::size_t>(factor + a.q[0] / 2 + d.q[0] / 2);
        return _c.at(idx);
    }
    return topo_ones::one_4D();
}

SectorArray
IsingAnyonCategory::all_sectors() const
{
    SectorArray out(3, 1);
    out[0][0] = 0;
    out[1][0] = 1;
    out[2][0] = 2;
    return out;
}

void
IsingAnyonCategory::save_hdf5(py::object hdf5_saver,
                              py::object h5gr,
                              std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(nu, subpath + "nu");
}

IsingAnyonCategory::Ptr
IsingAnyonCategory::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    int nu = hdf5_loader.attr("load")(subpath + "nu").cast<int>();
    auto obj = std::make_shared<IsingAnyonCategory>(nu);
    obj->descriptive_name = descriptive_name_from_hdf5_attrs(h5gr);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
