#include <cyten/symmetries/quantum_double_zn_anyon_category.h>

#include <cyten/symmetries/topo_ones.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace cyten {

namespace {

complex128
make_unit_phase(int N)
{
    return std::exp(
      complex128{ 0.0, 2.0 * std::numbers::pi_v<float64> / static_cast<float64>(N) });
}

complex128
phase_pow(complex128 phase, int exp)
{
    return std::pow(phase, static_cast<float64>(exp));
}

py::array
phase_r_symbol(complex128 val)
{
    return topo_ones::numpy().attr("array")(py::make_tuple(py::cast(val))).cast<py::array>();
}

py::array
phase_c_symbol(complex128 val)
{
    return topo_ones::numpy()
      .attr("multiply")(py::cast(val), topo_ones::one_4D())
      .cast<py::array>();
}

Sector
mod_sector(Sector a, int N)
{
    return Sector{ topo_ones::mod_n(static_cast<int32_t>(a.q[0]), N),
                   topo_ones::mod_n(static_cast<int32_t>(a.q[1]), N) };
}

} // namespace

QuantumDoubleZNAnyonCategory::QuantumDoubleZNAnyonCategory(
  int N_,
  std::optional<std::string> descriptive_name)
  : SymmetryFactor(FusionStyle::single,
                   BraidingStyle::anyonic,
                   Sector{ 0, 0 },
                   "D(ℤ_" + std::to_string(N_) + ")",
                   static_cast<float64>(N_) * static_cast<float64>(N_),
                   N_ > 2,
                   std::move(descriptive_name))
  , N(N_)
  , _phase(make_unit_phase(N))
{
    if (N <= 1) {
        throw std::invalid_argument("invalid QuantumDoubleZNAnyonCategory(N=" + std::to_string(N) +
                                    ")");
    }
}

bool
QuantumDoubleZNAnyonCategory::is_valid_sector(Sector a) const
{
    return a.len() == 2 && a.q[0] >= 0 && a.q[0] < N && a.q[1] >= 0 && a.q[1] < N;
}

bool
QuantumDoubleZNAnyonCategory::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != 2) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto row = sectors[i];
        if (row[0] < 0 || row[0] >= N || row[1] < 0 || row[1] >= N) {
            return false;
        }
    }
    return true;
}

SectorArray
QuantumDoubleZNAnyonCategory::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 2);
    SectorArray bb(1, 2);
    aa[0] = a;
    bb[0] = b;
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
QuantumDoubleZNAnyonCategory::fusion_outcomes_broadcast(SectorArray const& a,
                                                        SectorArray const& b) const
{
    SectorArray out(a.size(), 2);
    for (std::size_t i = 0; i < a.size(); ++i) {
        auto ar = a[i];
        auto br = b[i];
        out[i][0] = topo_ones::mod_n(static_cast<int32_t>(ar[0]) + br[0], N);
        out[i][1] = topo_ones::mod_n(static_cast<int32_t>(ar[1]) + br[1], N);
    }
    return out;
}

SectorArray
QuantumDoubleZNAnyonCategory::_multiple_fusion_broadcast(
  std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i][0] = topo_ones::mod_n(static_cast<int32_t>(out[i][0]) + sectors[s][i][0], N);
            out[i][1] = topo_ones::mod_n(static_cast<int32_t>(out[i][1]) + sectors[s][i][1], N);
        }
    }
    return out;
}

int64
QuantumDoubleZNAnyonCategory::sector_dim(Sector /*a*/) const
{
    return 1;
}

py::array
QuantumDoubleZNAnyonCategory::batch_sector_dim(SectorArray const& a) const
{
    return topo_ones::numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.size())),
                    py::arg("dtype") = topo_ones::numpy().attr("intp"))
      .cast<py::array>();
}

py::array
QuantumDoubleZNAnyonCategory::batch_qdim(SectorArray const& a) const
{
    return topo_ones::numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.size())),
                    py::arg("dtype") = topo_ones::numpy().attr("intp"))
      .cast<py::array>();
}

std::string
QuantumDoubleZNAnyonCategory::repr() const
{
    if (!descriptive_name.has_value()) {
        return "QuantumDoubleZNAnyonCategory(" + std::to_string(N) + ")";
    }
    return "QuantumDoubleZNAnyonCategory(" + std::to_string(N) + ", \"" + *descriptive_name +
           "\")";
}

bool
QuantumDoubleZNAnyonCategory::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* cat = dynamic_cast<QuantumDoubleZNAnyonCategory const*>(&other)) {
        return cat->N == N;
    }
    return false;
}

Sector
QuantumDoubleZNAnyonCategory::dual_sector(Sector a) const
{
    return mod_sector(Sector{ static_cast<int16_t>(-a.q[0]), static_cast<int16_t>(-a.q[1]) }, N);
}

SectorArray
QuantumDoubleZNAnyonCategory::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.size(), 2);
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto row = sectors[i];
        out[i][0] = topo_ones::mod_n(-static_cast<int32_t>(row[0]), N);
        out[i][1] = topo_ones::mod_n(-static_cast<int32_t>(row[1]), N);
    }
    return out;
}

int64
QuantumDoubleZNAnyonCategory::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

py::array
QuantumDoubleZNAnyonCategory::_f_symbol(Sector /*a*/,
                                        Sector /*b*/,
                                        Sector /*c*/,
                                        Sector /*d*/,
                                        Sector /*e*/,
                                        Sector /*f*/) const
{
    return topo_ones::one_4D();
}

int64
QuantumDoubleZNAnyonCategory::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
QuantumDoubleZNAnyonCategory::qdim(Sector /*a*/) const
{
    return 1.0;
}

py::array
QuantumDoubleZNAnyonCategory::_r_symbol(Sector a, Sector b, Sector /*c*/) const
{
    int const exp = static_cast<int>(a.q[0]) * static_cast<int>(b.q[1]);
    return phase_r_symbol(phase_pow(_phase, exp));
}

py::array
QuantumDoubleZNAnyonCategory::_c_symbol(Sector /*a*/,
                                        Sector b,
                                        Sector c,
                                        Sector /*d*/,
                                        Sector /*e*/,
                                        Sector /*f*/) const
{
    int const exp = static_cast<int>(b.q[0]) * static_cast<int>(c.q[1]);
    return phase_c_symbol(phase_pow(_phase, exp));
}

SectorArray
QuantumDoubleZNAnyonCategory::all_sectors() const
{
    SectorArray out(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 2);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            auto idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(N) +
                       static_cast<std::size_t>(j);
            out[idx][0] = static_cast<int16_t>(j);
            out[idx][1] = static_cast<int16_t>(i);
        }
    }
    return out;
}

void
QuantumDoubleZNAnyonCategory::save_hdf5(py::object hdf5_saver,
                                        py::object h5gr,
                                        std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(N, subpath + "N");
}

QuantumDoubleZNAnyonCategory::Ptr
QuantumDoubleZNAnyonCategory::from_hdf5(py::object hdf5_loader,
                                        py::object h5gr,
                                        std::string const& subpath)
{
    int N = hdf5_loader.attr("load")(subpath + "N").cast<int>();
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    auto obj = std::make_shared<QuantumDoubleZNAnyonCategory>(N, name);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
