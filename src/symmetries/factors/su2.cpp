#include <cyten/symmetries/factors/su2.h>

#include <cyten/block_backend/numpy.h>
#include <cyten/symmetries/fusion_symbol.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

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
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        if (sectors[i][0] < 0) {
            return false;
        }
    }
    return true;
}

SectorArray
SU2::fusion_outcomes(Sector a, Sector b) const
{
    // --- hints from Python SU2.fusion_outcomes ---
    // J_tot = |J1 - J2|, ..., J1 + J2
    // ---
    auto const aa = a.q[0];
    auto const bb = b.q[0];
    auto const jj_min = static_cast<int16_t>(std::abs(aa - bb));
    auto const jj_max = static_cast<int16_t>(aa + bb);
    auto const n = static_cast<std::size_t>((jj_max - jj_min) / 2 + 1);
    SectorArray out(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        out[i][0] = static_cast<int16_t>(jj_min + static_cast<int16_t>(2 * i));
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
    // --- hints from Python SU2.sector_dim ---
    // dim = 2 * J + 1 = jj + 1
    // ---
    return static_cast<int64>(a.q[0]) + 1;
}

std::vector<int64>
SU2::batch_sector_dim(SectorArray const& a) const
{
    // --- hints from Python SU2.batch_sector_dim ---
    // dim = 2 * J + 1 = jj + 1
    // ---
    std::vector<int64> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = static_cast<int64>(a[i][0]) + 1;
    }
    return out;
}

std::string
SU2::sector_str(Sector a) const
{
    auto const jj = a.q[0];
    std::string j_str = (jj % 2 == 0) ? std::to_string(jj / 2) : (std::to_string(jj) + "/2");
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
    // --- hints from Python SU2.dual_sector ---
    // all sectors are self-dual
    // ---
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

FusionSymbol
SU2::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    // --- hints from Python SU2._f_symbol ---
    // OPTIMIZE: jutho has a special case if all sectors are trivial ...?
    // ---
    return fusion_symbol_from_numpy(
      su2data()
        .attr("f_symbol")(a.q[0], b.q[0], c.q[0], d.q[0], e.q[0], f.q[0])
        .cast<py::array>());
}

int64
SU2::frobenius_schur(Sector a) const
{
    // --- hints from Python SU2.frobenius_schur ---
    // +1 for integer spin (i.e. even `a`), -1 for half integer
    // ---
    return 1 - 2 * (static_cast<int64>(a.q[0]) % 2);
}

float64
SU2::qdim(Sector a) const
{
    return static_cast<float64>(a.q[0]) + 1.0;
}

FusionSymbol
SU2::_r_symbol(Sector a, Sector b, Sector c) const
{
    // --- hints from Python SU2._r_symbol ---
    // R symbol is +1 if ``j_sum = (j_a + j_b - j_c)`` is even, -1 otherwise.
    // Note that (j_a + j_b - j_c) is integer by fusion rule and that e.g. ``a == 2 * j_a``.
    // For even (odd) j_sum, we get that ``(a + b - c) % 4`` is 0 (2),
    // such that ``1 - (a + b - c) % 4`` is 1 (-1). It has shape ``(1,)``.
    // ---
    // Shape ``(1,)``: +1 if ``(a+b-c)%4==0``, else -1 (when ==2).
    auto const val = static_cast<float64>(1 - ((a.q[0] + b.q[0] - c.q[0]) % 4));
    return FusionSymbol::scalar1d(val);
}

FusionSymbol
SU2::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    // --- hints from Python SU2._fusion_tensor ---
    // [µ, m_a, m_b, m_c] @ [m_a, m_abar*] -> [µ, m_b, m_c, m_abar*]
    // [µ, m_b, m_c, m_abar*] @ [m_b, m_bbar*] -> [µ, m_c, m_abar*, m_bbar*]
    // [µ, m_a, m_b, m_c] @ [m_b, m_bbar*] -> [µ, m_a, m_c, m_bbar*]
    // ---
    auto X = fusion_symbol_from_numpy(
      su2data().attr("fusion_tensor")(a.q[0], b.q[0], c.q[0]).cast<py::array>());
    if (!Z_a && !Z_b) {
        return X;
    }

    auto& be = *static_cast<BlockBackend*>(NumpyBlockBackend::from_factory("cpu"));
    auto Xb = block_from_fusion_symbol(be, X);
    if (Z_a && Z_b) {
        auto Za = block_from_fusion_symbol(be, Z_iso(dual_sector(a)));
        auto Zb = block_from_fusion_symbol(be, Z_iso(dual_sector(b)));
        Xb = be.tdot(Xb, Za, { 1 }, { 0 });
        Xb = be.tdot(Xb, Zb, { 1 }, { 0 });
        Xb = be.permute_axes(Xb, { 0, 2, 3, 1 });
    } else if (Z_a) {
        auto Za = block_from_fusion_symbol(be, Z_iso(dual_sector(a)));
        Xb = be.tdot(Xb, Za, { 1 }, { 0 });
        Xb = be.permute_axes(Xb, { 0, 3, 1, 2 });
    } else {
        auto Zb = block_from_fusion_symbol(be, Z_iso(dual_sector(b)));
        Xb = be.tdot(Xb, Zb, { 2 }, { 0 });
        Xb = be.permute_axes(Xb, { 0, 1, 3, 2 });
    }
    return fusion_symbol_from_block(Xb);
}

FusionSymbol
SU2::Z_iso(Sector a) const
{
    return fusion_symbol_from_numpy(su2data().attr("Z_iso")(a.q[0]).cast<py::array>());
}

SU2::Ptr
SU2::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& /*subpath*/)
{
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    auto obj = std::make_shared<SU2>(name);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
