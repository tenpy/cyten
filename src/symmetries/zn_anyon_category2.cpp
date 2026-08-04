#include <cyten/symmetries/zn_anyon_category2.h>

#include <cyten/symmetries/topo_ones.h>

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>

namespace cyten {

namespace {

complex128
make_phase(int n, int N)
{
    return std::exp(complex128{ 0.0,
                                2.0 * std::numbers::pi_v<float64> *
                                  (static_cast<float64>(n) + 0.5) / static_cast<float64>(N) });
}

complex128
phase_pow(complex128 phase, int exp)
{
    return std::pow(phase, static_cast<float64>(exp));
}

int
parity_sign(int exp)
{
    return (exp % 2 == 0) ? 1 : -1;
}

py::array
signed_one_4D(int sign)
{
    return topo_ones::numpy().attr("multiply")(sign, topo_ones::one_4D()).cast<py::array>();
}

} // namespace

ZNAnyonCategory2::ZNAnyonCategory2(int N_, int n_, std::optional<std::string> descriptive_name)
  : SymmetryFactor(FusionStyle::single,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
                   "ℤ_" + std::to_string(N_) + "^(" + std::to_string(n_ % N_) +
                     "+1/2) anyon category",
                   static_cast<float64>(N_),
                   true,
                   std::move(descriptive_name))
  , N(N_)
  , n(n_ % N_)
  , _phase(make_phase(n, N))
{
    if (N <= 1) {
        PyErr_SetString(PyExc_AssertionError,
                        ("invalid ZNAnyonCategory2(N=" + std::to_string(N) + ")").c_str());
        throw py::error_already_set();
    }
    if (N % 2 != 0) {
        PyErr_SetString(PyExc_AssertionError,
                        ("ZNAnyonCategory2 requires even N, got N=" + std::to_string(N)).c_str());
        throw py::error_already_set();
    }
}

bool
ZNAnyonCategory2::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] < N;
}

bool
ZNAnyonCategory2::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        auto q = sectors.row(i)[0];
        if (q < 0 || q >= N) {
            return false;
        }
    }
    return true;
}

SectorArray
ZNAnyonCategory2::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 1);
    SectorArray bb(1, 1);
    aa.set(0, a);
    bb.set(0, b);
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
ZNAnyonCategory2::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    SectorArray out(a.num_sectors, 1);
    for (std::size_t i = 0; i < a.num_sectors; ++i) {
        out.row(i)[0] = topo_ones::mod_n(static_cast<int32_t>(a.row(i)[0]) + b.row(i)[0], N);
    }
    return out;
}

SectorArray
ZNAnyonCategory2::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.num_sectors; ++i) {
            out.row(i)[0] =
              topo_ones::mod_n(static_cast<int32_t>(out.row(i)[0]) + sectors[s].row(i)[0], N);
        }
    }
    return out;
}

int64
ZNAnyonCategory2::sector_dim(Sector /*a*/) const
{
    return 1;
}

py::array
ZNAnyonCategory2::batch_sector_dim(SectorArray const& a) const
{
    return topo_ones::numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.num_sectors)),
                    py::arg("dtype") = topo_ones::numpy().attr("intp"))
      .cast<py::array>();
}

py::array
ZNAnyonCategory2::batch_qdim(SectorArray const& a) const
{
    return topo_ones::numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.num_sectors)),
                    py::arg("dtype") = topo_ones::numpy().attr("intp"))
      .cast<py::array>();
}

std::string
ZNAnyonCategory2::repr() const
{
    if (!descriptive_name.has_value()) {
        return "ZNAnyonCategory2(" + std::to_string(N) + ", " + std::to_string(n) + ")";
    }
    return "ZNAnyonCategory2(" + std::to_string(N) + ", " + std::to_string(n) + ", \"" +
           *descriptive_name + "\")";
}

bool
ZNAnyonCategory2::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* cat = dynamic_cast<ZNAnyonCategory2 const*>(&other)) {
        return cat->N == N && cat->n == n;
    }
    return false;
}

Sector
ZNAnyonCategory2::dual_sector(Sector a) const
{
    return Sector{ topo_ones::mod_n(-static_cast<int32_t>(a.q[0]), N) };
}

SectorArray
ZNAnyonCategory2::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.num_sectors, 1);
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        out.row(i)[0] = topo_ones::mod_n(-static_cast<int32_t>(sectors.row(i)[0]), N);
    }
    return out;
}

int64
ZNAnyonCategory2::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

py::array
ZNAnyonCategory2::_f_symbol(Sector a, Sector b, Sector c, Sector /*d*/, Sector /*e*/, Sector /*f*/)
  const
{
    int const exp =
      static_cast<int>(a.q[0]) * ((static_cast<int>(b.q[0]) + static_cast<int>(c.q[0])) / N);
    return signed_one_4D(parity_sign(exp));
}

int64
ZNAnyonCategory2::frobenius_schur(Sector a) const
{
    return parity_sign(static_cast<int>(a.q[0]));
}

float64
ZNAnyonCategory2::qdim(Sector /*a*/) const
{
    return 1.0;
}

py::array
ZNAnyonCategory2::_r_symbol(Sector a, Sector b, Sector /*c*/) const
{
    int const exp = static_cast<int>(a.q[0]) * static_cast<int>(b.q[0]);
    auto val = phase_pow(_phase, exp);
    return topo_ones::numpy()
      .attr("multiply")(py::cast(val), topo_ones::one_1D())
      .cast<py::array>();
}

py::array
ZNAnyonCategory2::_c_symbol(Sector /*a*/,
                            Sector b,
                            Sector c,
                            Sector /*d*/,
                            Sector /*e*/,
                            Sector /*f*/) const
{
    int const exp = static_cast<int>(b.q[0]) * static_cast<int>(c.q[0]);
    auto val = phase_pow(_phase, exp);
    return topo_ones::numpy()
      .attr("multiply")(py::cast(val), topo_ones::one_4D())
      .cast<py::array>();
}

SectorArray
ZNAnyonCategory2::all_sectors() const
{
    SectorArray out(static_cast<std::size_t>(N), 1);
    for (int i = 0; i < N; ++i) {
        out.row(static_cast<std::size_t>(i))[0] = static_cast<int16_t>(i);
    }
    return out;
}

void
ZNAnyonCategory2::save_hdf5(py::object hdf5_saver,
                            py::object h5gr,
                            std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(N, subpath + "N");
    hdf5_saver.attr("save")(n, subpath + "n");
}

ZNAnyonCategory2::Ptr
ZNAnyonCategory2::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    int N = hdf5_loader.attr("load")(subpath + "N").cast<int>();
    int n = hdf5_loader.attr("load")(subpath + "n").cast<int>();
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    auto obj = std::make_shared<ZNAnyonCategory2>(N, n, name);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
