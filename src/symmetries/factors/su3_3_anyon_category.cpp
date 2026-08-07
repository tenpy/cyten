#include <cyten/symmetries/su3_3_anyon_category.h>

#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/topo_ones.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace cyten {

Sector const SU3_3AnyonCategory::one_irrep{ 0 };
Sector const SU3_3AnyonCategory::eight_irrep{ 1 };
Sector const SU3_3AnyonCategory::ten_irrep{ 2 };
Sector const SU3_3AnyonCategory::ten_bar_irrep{ 3 };

namespace {

SectorArray
single_sector(int16_t q)
{
    SectorArray out(1, 1);
    out[0][0] = q;
    return out;
}

py::array
default_c_symbol(SU3_3AnyonCategory const& sym,
                 Sector a,
                 Sector b,
                 Sector c,
                 Sector d,
                 Sector e,
                 Sector f)
{
    auto np = topo_ones::numpy();
    py::array R1 = sym._r_symbol(e, c, d);
    // Match ``_default_c_symbol``: F^{c a b}_{d e f}, not F^{a b c}_{d e f}.
    py::array F = sym._f_symbol(c, a, b, d, e, f);
    py::array R2 = sym._r_symbol(a, c, f);
    return R1.attr("reshape")(py::make_tuple(1, -1, 1, 1)) * F *
           np.attr("conj")(R2).attr("reshape")(py::make_tuple(1, 1, -1, 1));
}

py::array
neg_one_4D()
{
    return topo_ones::numpy().attr("negative")(topo_ones::one_4D()).cast<py::array>();
}

std::pair<int, int>
f4_slices(int charge)
{
    switch (charge) {
        case 0:
            return { 0, 1 };
        case 1:
            return { 1, 5 };
        case 2:
            return { 5, 6 };
        default:
            return { 6, 7 };
    }
}

bool
all_non_trivial(std::array<int, 4> const& charges)
{
    return charges[0] != 0 && charges[1] != 0 && charges[2] != 0 && charges[3] != 0;
}

} // namespace

SectorArray
SU3_3AnyonCategory::fusion_map(int key)
{
    switch (key) {
        case 0:
            return single_sector(0);
        case 1:
            return single_sector(1);
        case 2: {
            SectorArray out(4, 1);
            for (int i = 0; i < 4; ++i) {
                out[static_cast<std::size_t>(i)][0] = static_cast<int16_t>(i);
            }
            return out;
        }
        case 4:
            return single_sector(2);
        case 5:
        case 10:
            return single_sector(1);
        case 8:
        case 9:
            return single_sector(3);
        case 13:
            return single_sector(0);
        case 18:
            return single_sector(2);
        default:
            return SectorArray(0, 1);
    }
}

Sector
SU3_3AnyonCategory::dual_map(int j)
{
    switch (j) {
        case 2:
            return Sector{ 3 };
        case 3:
            return Sector{ 2 };
        default:
            return Sector{ static_cast<int16_t>(j) };
    }
}

py::array
SU3_3AnyonCategory::_f1()
{
    return topo_ones::numpy().attr("identity")(2).cast<py::array>();
}

py::array
SU3_3AnyonCategory::_f2()
{
    auto np = topo_ones::numpy();
    auto sqrt3 = std::sqrt(3.0);
    py::list row0;
    row0.append(-0.5);
    row0.append(-sqrt3 / 2.0);
    py::list row1;
    row1.append(sqrt3 / 2.0);
    row1.append(-0.5);
    py::list rows;
    rows.append(row0);
    rows.append(row1);
    return np.attr("array")(rows).cast<py::array>();
}

py::array
SU3_3AnyonCategory::_f3()
{
    return _f2().attr("T").cast<py::array>();
}

py::array
SU3_3AnyonCategory::_f4()
{
    auto np = topo_ones::numpy();
    auto sqrt3 = std::sqrt(3.0);
    auto sqrt12 = std::sqrt(12.0);
    auto f4 = np.attr("zeros")(py::make_tuple(7, 7)).cast<py::array>();
    auto r = f4.mutable_unchecked<double, 2>();

    r(0, 0) = r(5, 5) = r(6, 5) = r(5, 6) = r(6, 6) = 1.0 / 3.0;
    r(0, 5) = r(0, 6) = r(5, 0) = r(6, 0) = -1.0 / 3.0;
    r(0, 1) = r(1, 0) = r(0, 4) = r(4, 0) = 1.0 / sqrt3;
    r(2, 2) = r(3, 2) = r(2, 3) = r(3, 3) = r(1, 4) = r(4, 1) = 0.5;
    r(2, 6) = r(6, 3) = r(3, 5) = r(5, 2) = 0.5;
    r(2, 5) = r(5, 3) = r(3, 6) = r(6, 2) = -0.5;
    r(1, 1) = r(4, 4) = -0.5;
    r(1, 5) = r(1, 6) = r(5, 1) = r(6, 1) = 1.0 / sqrt12;
    r(4, 5) = r(4, 6) = r(5, 4) = r(6, 4) = 1.0 / sqrt12;
    return f4;
}

SU3_3AnyonCategory::SU3_3AnyonCategory()
  : SymmetryFactor(FusionStyle::general,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
                   "SU3_3AnyonCategory",
                   4.0,
                   true,
                   std::nullopt)
{
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                for (int d = 0; d < 4; ++d) {
                    for (int e = 0; e < 4; ++e) {
                        for (int f = 0; f < 4; ++f) {
                            FSymKey key{ a, b, c, d, e, f };
                            _fsym_map[key] = _compute_f_symbol(Sector{ static_cast<int16_t>(a) },
                                                               Sector{ static_cast<int16_t>(b) },
                                                               Sector{ static_cast<int16_t>(c) },
                                                               Sector{ static_cast<int16_t>(d) },
                                                               Sector{ static_cast<int16_t>(e) },
                                                               Sector{ static_cast<int16_t>(f) });
                        }
                    }
                }
            }
        }
    }

    // C-symbol fusion conditions (not the F-symbol ones):
    // a ⊗ b → e, e ⊗ c → d, a ⊗ c → f, f ⊗ b → d
    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            for (int c = 0; c < 4; ++c) {
                for (int d = 0; d < 4; ++d) {
                    for (int e = 0; e < 4; ++e) {
                        for (int f = 0; f < 4; ++f) {
                            Sector sa{ static_cast<int16_t>(a) };
                            Sector sb{ static_cast<int16_t>(b) };
                            Sector sc{ static_cast<int16_t>(c) };
                            Sector sd{ static_cast<int16_t>(d) };
                            Sector se{ static_cast<int16_t>(e) };
                            Sector sf{ static_cast<int16_t>(f) };
                            if (can_fuse_to(sa, sb, se) && can_fuse_to(se, sc, sd) &&
                                can_fuse_to(sa, sc, sf) && can_fuse_to(sf, sb, sd)) {
                                _c[FSymKey{ a, b, c, d, e, f }] =
                                  default_c_symbol(*this, sa, sb, sc, sd, se, sf);
                            }
                        }
                    }
                }
            }
        }
    }
}

py::array
SU3_3AnyonCategory::_compute_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
  const
{
    if (!can_fuse_to(b, c, e) || !can_fuse_to(a, e, d) || !can_fuse_to(a, b, f) ||
        !can_fuse_to(f, c, d)) {
        return topo_ones::one_4D();
    }

    std::array<int, 4> abcd = { a.q[0], b.q[0], c.q[0], d.q[0] };
    int const n8 =
      static_cast<int>(std::count_if(abcd.begin(), abcd.end(), [](int q) { return q == 1; }));
    auto const shape = py::make_tuple(
      _n_symbol(b, c, e), _n_symbol(a, e, d), _n_symbol(a, b, f), _n_symbol(f, c, d));

    if (n8 == 4) {
        auto [e0, e1] = f4_slices(e.q[0]);
        auto [f0, f1] = f4_slices(f.q[0]);
        return _f4()
          .attr("__getitem__")(py::make_tuple(py::slice(f0, f1, 1), py::slice(e0, e1, 1)))
          .attr("reshape")(shape)
          .cast<py::array>();
    }

    if (n8 == 3) {
        int index = 0;
        for (int i = 0; i < 4; ++i) {
            if (abcd[static_cast<std::size_t>(i)] != 1) {
                index = i;
                break;
            }
        }
        int const not_8 = abcd[static_cast<std::size_t>(index)];
        if (not_8 == 0) {
            return _f1().attr("reshape")(shape).cast<py::array>();
        }
        if ((not_8 == 2 && index != 1) || (not_8 == 3 && index == 1)) {
            return _f2().attr("reshape")(shape).cast<py::array>();
        }
        return _f3().attr("reshape")(shape).cast<py::array>();
    }

    if (n8 == 2 && all_non_trivial(abcd)) {
        int index1 = -1;
        for (int i = 0; i < 4; ++i) {
            if (abcd[static_cast<std::size_t>(i)] == 1) {
                index1 = i;
                break;
            }
        }
        int index2 = -1;
        for (int i = index1 + 1; i < 4; ++i) {
            if (abcd[static_cast<std::size_t>(i)] == 1) {
                index2 = i;
                break;
            }
        }
        if (index2 == index1 + 1 || (index1 == 0 && index2 == 3)) {
            return neg_one_4D();
        }
    }

    if (n8 == 0 && all_non_trivial(abcd)) {
        int n10 = 0;
        for (int q : abcd) {
            if (q == 2) {
                ++n10;
            }
        }
        int index = 1;
        if (n10 == 3) {
            for (int i = 0; i < 4; ++i) {
                if (abcd[static_cast<std::size_t>(i)] != 2) {
                    index = i;
                    break;
                }
            }
        } else if (n10 == 1) {
            for (int i = 0; i < 4; ++i) {
                if (abcd[static_cast<std::size_t>(i)] == 2) {
                    index = i;
                    break;
                }
            }
        }
        if (index == 0 || index == 2) {
            return neg_one_4D();
        }
    }

    return topo_ones::one_4D();
}

bool
SU3_3AnyonCategory::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] < 4;
}

bool
SU3_3AnyonCategory::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto const q = sectors[i][0];
        if (q < 0 || q >= 4) {
            return false;
        }
    }
    return true;
}

SectorArray
SU3_3AnyonCategory::fusion_outcomes(Sector a, Sector b) const
{
    auto const key = static_cast<int>(a.q[0]) * static_cast<int>(a.q[0]) +
                     static_cast<int>(b.q[0]) * static_cast<int>(b.q[0]);
    return fusion_map(key);
}

int64
SU3_3AnyonCategory::sector_dim(Sector /*a*/) const
{
    return 1;
}

py::array
SU3_3AnyonCategory::batch_sector_dim(SectorArray const& a) const
{
    return topo_ones::numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.size())),
                    py::arg("dtype") = topo_ones::numpy().attr("intp"))
      .cast<py::array>();
}

std::string
SU3_3AnyonCategory::sector_str(Sector a) const
{
    switch (a.q[0]) {
        case 1:
            return "eight";
        case 2:
            return "ten";
        case 0:
            return "one";
        default:
            return "ten_bar";
    }
}

std::string
SU3_3AnyonCategory::repr() const
{
    return "SU3_3AnyonCategory()";
}

bool
SU3_3AnyonCategory::_is_equivalent_factor(SymmetryFactor const& other) const
{
    return dynamic_cast<SU3_3AnyonCategory const*>(&other) != nullptr;
}

Sector
SU3_3AnyonCategory::dual_sector(Sector a) const
{
    return dual_map(a.q[0]);
}

SectorArray
SU3_3AnyonCategory::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.size(), 1);
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto const q = sectors[i][0];
        if (q >= 2) {
            out[i][0] = topo_ones::mod_n(-static_cast<int32_t>(q), 5);
        } else {
            out[i][0] = q;
        }
    }
    return out;
}

int64
SU3_3AnyonCategory::_n_symbol(Sector a, Sector b, Sector c) const
{
    return (a.q[0] == 1 && b.q[0] == 1 && c.q[0] == 1) ? 2 : 1;
}

py::array
SU3_3AnyonCategory::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    FSymKey key{ a.q[0], b.q[0], c.q[0], d.q[0], e.q[0], f.q[0] };
    return _fsym_map.at(key);
}

int64
SU3_3AnyonCategory::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
SU3_3AnyonCategory::qdim(Sector a) const
{
    return (a.q[0] == 1) ? 3.0 : 1.0;
}

py::array
SU3_3AnyonCategory::batch_qdim(SectorArray const& a) const
{
    auto np = topo_ones::numpy();
    py::array charges = sector_array_to_numpy(a);
    return np.attr("where")(charges.attr("__eq__")(1), 3, 1).attr("flatten")().cast<py::array>();
}

py::array
SU3_3AnyonCategory::_r_symbol(Sector a, Sector b, Sector c) const
{
    if (a.q[0] == 1 && b.q[0] == 1) {
        if (c.q[0] == 1) {
            py::list vals;
            vals.append(std::complex<float64>{ 0.0, -1.0 });
            vals.append(std::complex<float64>{ 0.0, 1.0 });
            return topo_ones::numpy().attr("array")(vals).cast<py::array>();
        }
        return topo_ones::numpy().attr("negative")(topo_ones::one_1D()).cast<py::array>();
    }
    return topo_ones::one_1D();
}

py::array
SU3_3AnyonCategory::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    FSymKey key{ a.q[0], b.q[0], c.q[0], d.q[0], e.q[0], f.q[0] };
    auto it = _c.find(key);
    if (it != _c.end()) {
        return it->second;
    }
    return topo_ones::one_4D();
}

SectorArray
SU3_3AnyonCategory::all_sectors() const
{
    SectorArray out(4, 1);
    for (int i = 0; i < 4; ++i) {
        out[static_cast<std::size_t>(i)][0] = static_cast<int16_t>(i);
    }
    return out;
}

SU3_3AnyonCategory::Ptr
SU3_3AnyonCategory::from_hdf5(py::object hdf5_loader,
                              py::object h5gr,
                              std::string const& /*subpath*/)
{
    auto obj = std::make_shared<SU3_3AnyonCategory>();
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
