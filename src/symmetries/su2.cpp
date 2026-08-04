#include <cyten/symmetries/su2.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>

namespace cyten {

Sector const SU2::spin_zero{ 0 };
Sector const SU2::spin_half{ 1 };
Sector const SU2::spin_one{ 2 };

namespace {

py::module_
su2data()
{
    return py::module_::import("cyten.symmetries._su2data");
}

py::module_
numpy()
{
    return py::module_::import("numpy");
}

} // namespace

SU2::SU2(std::optional<std::string> descriptive_name)
  : Group(FusionStyle::multiple_unique,
          Sector{ 0 },
          "SU(2)",
          std::numeric_limits<float64>::infinity(),
          /*has_complex_topological_data=*/false,
          std::move(descriptive_name),
          /*trivial_shift=*/true)
{
    fusion_tensor_dtype = Dtype::Float64;
}

bool
SU2::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0;
}

bool
SU2::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        if (sectors.row(i)[0] < 0) {
            return false;
        }
    }
    return true;
}

SectorArray
SU2::fusion_outcomes(Sector a, Sector b) const
{
    auto const aa = a.q[0];
    auto const bb = b.q[0];
    auto const jj_min = static_cast<int16_t>(std::abs(aa - bb));
    auto const jj_max = static_cast<int16_t>(aa + bb);
    auto const n = static_cast<std::size_t>((jj_max - jj_min) / 2 + 1);
    SectorArray out(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        out.row(i)[0] = static_cast<int16_t>(jj_min + static_cast<int16_t>(2 * i));
    }
    return out;
}

bool
SU2::can_fuse_to(Sector a, Sector b, Sector c) const
{
    auto const aa = a.q[0];
    auto const bb = b.q[0];
    auto const cc = c.q[0];
    return (cc <= aa + bb) && (aa <= bb + cc) && (bb <= cc + aa) && ((aa + bb + cc) % 2 == 0);
}

int64
SU2::sector_dim(Sector a) const
{
    return static_cast<int64>(a.q[0]) + 1;
}

py::array
SU2::batch_sector_dim(SectorArray const& a) const
{
    py::array_t<int64> out(static_cast<py::ssize_t>(a.num_sectors));
    auto r = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < a.num_sectors; ++i) {
        r(static_cast<py::ssize_t>(i)) = static_cast<int64>(a.row(i)[0]) + 1;
    }
    return out;
}

std::string
SU2::sector_str(Sector a) const
{
    auto const jj = a.q[0];
    std::string j_str =
      (jj % 2 == 0) ? std::to_string(jj / 2) : (std::to_string(jj) + "/2");
    return std::to_string(jj) + " (J=" + j_str + ")";
}

std::string
SU2::repr() const
{
    if (!descriptive_name.has_value()) {
        return "SU2Symmetry()";
    }
    return std::string("SU2Symmetry(\"") + *descriptive_name + "\")";
}

bool
SU2::_is_equivalent_factor(SymmetryFactor const& other) const
{
    return dynamic_cast<SU2 const*>(&other) != nullptr;
}

Sector
SU2::dual_sector(Sector a) const
{
    return a;
}

SectorArray
SU2::dual_sectors(SectorArray const& sectors) const
{
    return sectors;
}

int64
SU2::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

py::array
SU2::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    return su2data()
      .attr("f_symbol")(a.q[0], b.q[0], c.q[0], d.q[0], e.q[0], f.q[0])
      .cast<py::array>();
}

int64
SU2::frobenius_schur(Sector a) const
{
    return 1 - 2 * (static_cast<int64>(a.q[0]) % 2);
}

float64
SU2::qdim(Sector a) const
{
    return static_cast<float64>(a.q[0]) + 1.0;
}

py::array
SU2::_r_symbol(Sector a, Sector b, Sector c) const
{
    // Shape ``(1,)``: +1 if ``(a+b-c)%4==0``, else -1 (when ==2).
    auto const val = static_cast<int64>(1 - ((a.q[0] + b.q[0] - c.q[0]) % 4));
    py::array_t<int64> out(1);
    out.mutable_at(0) = val;
    return out;
}

py::array
SU2::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    auto np = numpy();
    py::object X = su2data().attr("fusion_tensor")(a.q[0], b.q[0], c.q[0]);
    if (Z_a && Z_b) {
        X = np.attr("tensordot")(X, Z_iso(dual_sector(a)), py::make_tuple(1, 0));
        X = np.attr("tensordot")(X, Z_iso(dual_sector(b)), py::make_tuple(1, 0));
        X = np.attr("transpose")(X, py::make_tuple(0, 2, 3, 1));
    } else if (Z_a) {
        X = np.attr("tensordot")(X, Z_iso(dual_sector(a)), py::make_tuple(1, 0));
        X = np.attr("transpose")(X, py::make_tuple(0, 3, 1, 2));
    } else if (Z_b) {
        X = np.attr("tensordot")(X, Z_iso(dual_sector(b)), py::make_tuple(2, 0));
        X = np.attr("transpose")(X, py::make_tuple(0, 1, 3, 2));
    }
    return X.cast<py::array>();
}

py::array
SU2::Z_iso(Sector a) const
{
    return su2data().attr("Z_iso")(a.q[0]).cast<py::array>();
}

} // namespace cyten
