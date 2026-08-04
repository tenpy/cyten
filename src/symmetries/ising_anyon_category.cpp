#include <cyten/symmetries/ising_anyon_category.h>

#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/topo_ones.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

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

py::array
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
    auto R2 = sym._r_symbol(a, c, f);
    auto np = topo_ones::numpy();
    return (R1.attr("reshape")(py::make_tuple(1, -1, 1, 1)) * F *
            np.attr("conj")(R2).attr("reshape")(py::make_tuple(1, 1, -1, 1)))
      .cast<py::array>();
}

py::array
c_entry_to_array(py::object const& entry)
{
    if (py::isinstance<py::array>(entry)) {
        return entry.cast<py::array>();
    }
    return topo_ones::numpy().attr("array")(entry).cast<py::array>();
}

py::array
scaled_one_4D(py::object factor)
{
    return (factor * topo_ones::one_4D()).cast<py::array>();
}

py::array
make_frobenius_array(int nu)
{
    auto np = topo_ones::numpy();
    int64_t const exp = (static_cast<int64_t>(nu) * nu - 1) / 8;
    int64_t const fs1 = (exp % 2 == 0) ? 1 : -1;
    py::list vals;
    vals.append(1);
    vals.append(fs1);
    vals.append(1);
    return np.attr("array")(vals).cast<py::array>();
}

py::array
make_f_table(py::array const& frobenius)
{
    auto np = topo_ones::numpy();
    auto const fs1 = frobenius.attr("__getitem__")(1);
    py::list vals;
    vals.append(1);
    vals.append(0);
    vals.append(1);
    vals.append(0);
    vals.append(-1);
    return np
      .attr("expand_dims")((np.attr("array")(vals) * fs1).attr("__truediv__")(std::sqrt(2.0)),
                           py::make_tuple(1, 2, 3, 4))
      .cast<py::array>();
}

py::array
make_r_table(int nu, py::array const& frobenius)
{
    auto np = topo_ones::numpy();
    auto const fs1 = frobenius.attr("__getitem__")(1);
    auto const pi = std::numbers::pi_v<float64>;
    py::list vals;
    vals.append(np.attr("power")(py::cast(complex128{ 0.0, -1.0 }), nu));
    vals.append(-1);
    vals.append(np.attr("exp")(py::cast(complex128{ 0.0, 3.0 * nu * pi / 8.0 })) * fs1);
    vals.append(np.attr("exp")(py::cast(complex128{ 0.0, -static_cast<float64>(nu) * pi / 8.0 })) *
                fs1);
    vals.append(0);
    return np.attr("expand_dims")(vals, 1).cast<py::array>();
}

SectorArray
fusion_map_lookup(int16_t a, int16_t b)
{
    switch (a * a + b * b) {
        case 0: {
            SectorArray out(1, 1);
            out.row(0)[0] = 0;
            return out;
        }
        case 1: {
            SectorArray out(1, 1);
            out.row(0)[0] = 1;
            return out;
        }
        case 2: {
            SectorArray out(2, 1);
            out.row(0)[0] = 0;
            out.row(1)[0] = 2;
            return out;
        }
        case 4: {
            SectorArray out(1, 1);
            out.row(0)[0] = 2;
            return out;
        }
        case 5: {
            SectorArray out(1, 1);
            out.row(0)[0] = 1;
            return out;
        }
        case 8: {
            SectorArray out(1, 1);
            out.row(0)[0] = 0;
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
    // Match Python ``np.all(np.concatenate([a, b]))``: both charges nonzero.
    return a.q[0] != 0 && b.q[0] != 0;
}

bool
sectors_are(Sector a, Sector b, Sector c, Sector d, int16_t va, int16_t vb, int16_t vc, int16_t vd)
{
    return a.q[0] == va && b.q[0] == vb && c.q[0] == vc && d.q[0] == vd;
}

} // namespace

IsingAnyonCategory::IsingAnyonCategory(int nu_)
  : SymmetryFactor(FusionStyle::multiple_unique,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
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
    auto np = topo_ones::numpy();
    auto const phase = np.attr("power")(py::cast(complex128{ 0.0, -1.0 }), nu);
    auto const neg_phase = py::cast(-1) * phase;
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
        py::int_(0),
        default_c_symbol(
          *this, sector1(2), sector1(1), sector1(1), sector1(0), sector1(1), sector1(1)),
        default_c_symbol(
          *this, sector1(2), sector1(1), sector1(1), sector1(2), sector1(1), sector1(1)),
        scaled_one_4D(py::cast(-1)),
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
    if (sectors.sector_ind_len != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        auto q = sectors.row(i)[0];
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

py::array
IsingAnyonCategory::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (sectors_are(a, b, c, d, 1, 1, 1, 1)) {
        auto idx = static_cast<py::ssize_t>(e.q[0] + f.q[0]);
        return _f.attr("__getitem__")(idx).cast<py::array>();
    }
    if (sectors_are(a, b, c, d, 2, 1, 2, 1)) {
        return scaled_one_4D(py::cast(-1));
    }
    if (sectors_are(a, b, c, d, 1, 2, 1, 2)) {
        return scaled_one_4D(py::cast(-1));
    }
    return topo_ones::one_4D();
}

int64
IsingAnyonCategory::frobenius_schur(Sector a) const
{
    return frobenius.attr("__getitem__")(a.q[0]).cast<int64>();
}

float64
IsingAnyonCategory::qdim(Sector a) const
{
    return a.q[0] == 1 ? std::sqrt(2.0) : 1.0;
}

py::array
IsingAnyonCategory::batch_qdim(SectorArray const& a) const
{
    auto np = topo_ones::numpy();
    return np.attr("where")(sector_array_to_numpy(a).attr("__eq__")(1), std::sqrt(2.0), 1)
      .attr("flatten")()
      .cast<py::array>();
}

py::array
IsingAnyonCategory::_r_symbol(Sector a, Sector b, Sector c) const
{
    if (all_nontrivial(a, b)) {
        auto const row = static_cast<py::ssize_t>((a.q[0] + b.q[0]) * (c.q[0] - 1));
        return _r.attr("__getitem__")(py::make_tuple(row, py::slice())).cast<py::array>();
    }
    return topo_ones::one_1D();
}

py::array
IsingAnyonCategory::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (all_sigma(b, c)) {
        int64_t factor = -1 * (b.q[0] - c.q[0] - 1) * (b.q[0] - c.q[0] + 1);
        factor *= (1 - a.q[0] / 2 - d.q[0] / 2 + 9 * (b.q[0] - 1) +
                   (2 - b.q[0]) * ((e.q[0] + f.q[0]) / 2 + d.q[0] / 2 + 3 * a.q[0]));
        auto const idx = static_cast<std::size_t>(factor + a.q[0] / 2 + d.q[0] / 2);
        return c_entry_to_array(_c.at(idx));
    }
    return topo_ones::one_4D();
}

SectorArray
IsingAnyonCategory::all_sectors() const
{
    SectorArray out(3, 1);
    out.row(0)[0] = 0;
    out.row(1)[0] = 1;
    out.row(2)[0] = 2;
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
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
