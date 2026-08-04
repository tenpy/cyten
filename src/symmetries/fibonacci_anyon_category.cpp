#include <cyten/symmetries/fibonacci_anyon_category.h>

#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/topo_ones.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace cyten {

Sector const FibonacciAnyonCategory::vacuum{ 0 };
Sector const FibonacciAnyonCategory::tau{ 1 };

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

float64
golden_ratio()
{
    return 0.5 * (1.0 + std::sqrt(5.0));
}

py::array
make_f_table(float64 phi)
{
    auto np = topo_ones::numpy();
    py::list vals;
    vals.append(std::pow(phi, -1.0));
    vals.append(std::pow(phi, -0.5));
    vals.append(-std::pow(phi, -1.0));
    return np.attr("expand_dims")(vals, py::make_tuple(1, 2, 3, 4)).cast<py::array>();
}

py::array
make_r_table(std::string const& handedness)
{
    auto np = topo_ones::numpy();
    auto const pi = std::numbers::pi_v<float64>;
    py::list vals;
    vals.append(std::exp(complex128{ 0.0, -4.0 * pi / 5.0 }));
    vals.append(std::exp(complex128{ 0.0, 3.0 * pi / 5.0 }));
    auto arr = np.attr("expand_dims")(vals, 1).cast<py::array>();
    if (handedness == "right") {
        arr = np.attr("conj")(arr).cast<py::array>();
    }
    return arr;
}

SectorArray
fusion_map_lookup(int16_t a, int16_t b)
{
    switch (a + b) {
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
            out.row(1)[0] = 1;
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

} // namespace

FibonacciAnyonCategory::FibonacciAnyonCategory(std::string handedness_)
  : SymmetryFactor(FusionStyle::multiple_unique,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
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
        py::int_(0),
        py::int_(0),
        default_c_symbol(
          *this, sector1(0), sector1(1), sector1(1), sector1(1), sector1(1), sector1(1)),
        py::int_(0),
        py::int_(0),
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
    if (sectors.sector_ind_len != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        auto q = sectors.row(i)[0];
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

py::array
FibonacciAnyonCategory::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (all_tau4(a, b, c, d)) {
        auto idx = static_cast<py::ssize_t>(e.q[0] + f.q[0]);
        return _f.attr("__getitem__")(idx).cast<py::array>();
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

py::array
FibonacciAnyonCategory::batch_qdim(SectorArray const& a) const
{
    auto np = topo_ones::numpy();
    return np.attr("where")(sector_array_to_numpy(a).attr("__eq__")(1), _phi, 1)
      .attr("flatten")()
      .cast<py::array>();
}

py::array
FibonacciAnyonCategory::_r_symbol(Sector a, Sector b, Sector c) const
{
    if (all_tau(a, b)) {
        return _r.attr("__getitem__")(py::make_tuple(c.q[0], py::slice())).cast<py::array>();
    }
    return topo_ones::one_1D();
}

py::array
FibonacciAnyonCategory::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (all_tau(b, c)) {
        auto const idx = static_cast<std::size_t>(6 * a.q[0] + 3 * d.q[0] + e.q[0] + f.q[0] - 2);
        return c_entry_to_array(_c.at(idx));
    }
    return topo_ones::one_4D();
}

SectorArray
FibonacciAnyonCategory::all_sectors() const
{
    SectorArray out(2, 1);
    out.row(0)[0] = 0;
    out.row(1)[0] = 1;
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
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
