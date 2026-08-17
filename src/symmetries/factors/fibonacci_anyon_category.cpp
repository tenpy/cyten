#include <cyten/symmetries/factors/fibonacci_anyon_category.h>

#include <cyten/symmetries/topo_ones.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

Sector const FibonacciAnyonCategory::vacuum{ 0 };
Sector const FibonacciAnyonCategory::tau{ 1 };

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

float64
golden_ratio()
{
    return 0.5 * (1.0 + std::sqrt(5.0));
}

FusionSymbol
make_f_table(float64 phi)
{
    return FusionSymbol::from_float64(
      1,
      FusionSymbol::Shape{ { 3, 1, 1, 1 } },
      { std::pow(phi, -1.0), std::pow(phi, -0.5), -std::pow(phi, -1.0) });
}

FusionSymbol
make_r_table(std::string const& handedness)
{
    auto const pi = std::numbers::pi_v<float64>;
    std::vector<complex128> vals{ std::exp(complex128{ 0.0, -4.0 * pi / 5.0 }),
                                  std::exp(complex128{ 0.0, 3.0 * pi / 5.0 }) };
    auto arr =
      FusionSymbol::from_complex128(1, FusionSymbol::Shape{ { 2, 1, 1, 1 } }, std::move(vals));
    if (handedness == "right") {
        return arr.conj();
    }
    return arr;
}

SectorArray
fusion_map_lookup(int16_t a, int16_t b)
{
    switch (a + b) {
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
            out[1][0] = 1;
            return out;
        }
        default:
            throw std::invalid_argument("invalid Fibonacci fusion inputs");
    }
}

bool
all_tau(Sector a, Sector b)
{
    return a.q[0] == 1 && b.q[0] == 1;
}

bool
all_tau4(Sector a, Sector b, Sector c, Sector d)
{
    return a.q[0] == 1 && b.q[0] == 1 && c.q[0] == 1 && d.q[0] == 1;
}

FusionSymbol
zero_c_entry()
{
    return FusionSymbol::zeros(4, FusionSymbol::Shape{ { 1, 1, 1, 1 } }, Dtype::Complex128);
}

} // namespace

FibonacciAnyonCategory::FibonacciAnyonCategory(std::string handedness_)
  : SymmetryFactor(FusionStyle::multiple_unique,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
                   // --- hints from Python FibonacciAnyonCategory.__init__ ---
                   // nontrivial C-symbols
                   // ---
                   "FibonacciAnyonCategory",
                   2.0,
                   true)
  , handedness(std::move(handedness_))
  , _phi(golden_ratio())
  , _f(make_f_table(_phi))
  , _r(make_r_table(handedness))
{
    if (handedness != "left" && handedness != "right") {
        throw std::invalid_argument("FibonacciAnyonCategory handedness must be 'left' or 'right'");
    }
    _c = {
        default_c_symbol(
          *this, sector1(0), sector1(1), sector1(1), sector1(0), sector1(1), sector1(1)),
        zero_c_entry(),
        zero_c_entry(),
        default_c_symbol(
          *this, sector1(0), sector1(1), sector1(1), sector1(1), sector1(1), sector1(1)),
        zero_c_entry(),
        zero_c_entry(),
        default_c_symbol(
          *this, sector1(1), sector1(1), sector1(1), sector1(0), sector1(1), sector1(1)),
        default_c_symbol(
          *this, sector1(1), sector1(1), sector1(1), sector1(1), sector1(0), sector1(0)),
        default_c_symbol(
          *this, sector1(1), sector1(1), sector1(1), sector1(1), sector1(1), sector1(0)),
        default_c_symbol(
          *this, sector1(1), sector1(1), sector1(1), sector1(1), sector1(1), sector1(1)),
    };
}

bool
FibonacciAnyonCategory::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] < 2;
}

bool
FibonacciAnyonCategory::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto q = sectors[i][0];
        if (q < 0 || q >= 2) {
            return false;
        }
    }
    return true;
}

SectorArray
FibonacciAnyonCategory::fusion_outcomes(Sector a, Sector b) const
{
    return fusion_map_lookup(a.q[0], b.q[0]);
}

std::string
FibonacciAnyonCategory::sector_str(Sector a) const
{
    return a.q[0] == 0 ? "vacuum" : "tau";
}

std::string
FibonacciAnyonCategory::repr() const
{
    return "FibonacciAnyonCategory(handedness=" + handedness + ")";
}

bool
FibonacciAnyonCategory::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* cat = dynamic_cast<FibonacciAnyonCategory const*>(&other)) {
        return cat->handedness == handedness;
    }
    return false;
}

Sector
FibonacciAnyonCategory::dual_sector(Sector a) const
{
    return a;
}

SectorArray
FibonacciAnyonCategory::dual_sectors(SectorArray const& sectors) const
{
    return sectors;
}

int64
FibonacciAnyonCategory::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

FusionSymbol
FibonacciAnyonCategory::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (all_tau4(a, b, c, d)) {
        auto idx = static_cast<std::size_t>(e.q[0] + f.q[0]);
        return FusionSymbol::full(
          4, FusionSymbol::Shape{ { 1, 1, 1, 1 } }, _f.get_complex(idx), Dtype::Float64);
    }
    return topo_ones::one_4D();
}

int64
FibonacciAnyonCategory::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
FibonacciAnyonCategory::qdim(Sector a) const
{
    return a.q[0] == 0 ? 1.0 : _phi;
}

std::vector<float64>
FibonacciAnyonCategory::batch_qdim(SectorArray const& a) const
{
    std::vector<float64> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = (a[i][0] == 1) ? _phi : 1.0;
    }
    return out;
}

FusionSymbol
FibonacciAnyonCategory::_r_symbol(Sector a, Sector b, Sector c) const
{
    if (all_tau(a, b)) {
        return FusionSymbol::scalar1d(_r.get_complex(static_cast<std::size_t>(c.q[0])),
                                      Dtype::Complex128);
    }
    return topo_ones::one_1D();
}

FusionSymbol
FibonacciAnyonCategory::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (all_tau(b, c)) {
        auto const idx = static_cast<std::size_t>(6 * a.q[0] + 3 * d.q[0] + e.q[0] + f.q[0] - 2);
        return _c.at(idx);
    }
    return topo_ones::one_4D();
}

SectorArray
FibonacciAnyonCategory::all_sectors() const
{
    SectorArray out(2, 1);
    out[0][0] = 0;
    out[1][0] = 1;
    return out;
}

void
FibonacciAnyonCategory::save_hdf5(py::object hdf5_saver,
                                  py::object h5gr,
                                  std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(handedness, subpath + "handedness");
}

FibonacciAnyonCategory::Ptr
FibonacciAnyonCategory::from_hdf5(py::object hdf5_loader,
                                  py::object h5gr,
                                  std::string const& subpath)
{
    std::string handedness = hdf5_loader.attr("load")(subpath + "handedness").cast<std::string>();
    auto obj = std::make_shared<FibonacciAnyonCategory>(handedness);
    obj->descriptive_name = descriptive_name_from_hdf5_attrs(h5gr);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
