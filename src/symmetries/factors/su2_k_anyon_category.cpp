#include <cyten/symmetries/factors/su2_k_anyon_category.h>

#include <cyten/symmetries/topo_ones.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

Sector const SU2_kAnyonCategory::spin_zero{ 0 };
Sector const SU2_kAnyonCategory::spin_half{ 1 };

namespace {

int
iround(float64 x)
{
    return static_cast<int>(std::llround(x));
}

FusionSymbol
scale_one_1D(complex128 factor)
{
    return FusionSymbol::scalar1d(factor, Dtype::Complex128);
}

FusionSymbol
scale_one_4D(complex128 factor)
{
    return topo_ones::one_4D() * factor;
}

int
argmax(std::vector<int> const& vals)
{
    return static_cast<int>(
      std::distance(vals.begin(), std::max_element(vals.begin(), vals.end())));
}

} // namespace

SU2_kAnyonCategory::SU2_kAnyonCategory(int k_, std::string handedness_)
  : SymmetryFactor(FusionStyle::multiple_unique,
                   BraidingStyle::anyonic,
                   Sector{ 0 },
                   // --- hints from Python SU2_kAnyonCategory.__init__ ---
                   // do not save trivial R-symbols and use symmetry jj1 <-> jj2
                   // ---
                   "SU2_kAnyonCategory",
                   static_cast<float64>(k_ + 1),
                   true,
                   std::nullopt)
  , k(k_)
  , handedness(std::move(handedness_))
  , _q(
      std::exp(complex128{ 0.0, 2.0 * std::numbers::pi_v<float64> / static_cast<float64>(k + 2) }))
{
    if (k < 1) {
        throw std::invalid_argument("SU2_kAnyonCategory requires k >= 1");
    }
    if (handedness != "left" && handedness != "right") {
        throw std::invalid_argument("SU2_kAnyonCategory handedness must be 'left' or 'right'");
    }
    if (k >= 2) {
        spin_one = Sector{ 2 };
    }

    for (int jj1 = 0; jj1 <= k; ++jj1) {
        for (int jj2 = 0; jj2 <= k; ++jj2) {
            for (int jj = 0; jj <= k; ++jj) {
                if (jj > jj1 + jj2 || jj < std::abs(jj1 - jj2) || jj1 * jj2 == 0 || jj1 < jj2) {
                    continue;
                }
                auto const parity = static_cast<float64>((jj - jj1 - jj2) / 2);
                auto factor =
                  std::pow(-1.0, parity) *
                  std::pow(_q,
                           complex128{ static_cast<float64>(jj * (jj + 2) - jj1 * (jj1 + 2) -
                                                            jj2 * (jj2 + 2)) /
                                       8.0 });
                if (handedness == "right") {
                    factor = std::conj(factor);
                }
                _r[RKey{ jj1, jj2, jj }] = scale_one_1D(factor);
            }
        }
    }

    for (int jj1 = 0; jj1 <= k; ++jj1) {
        for (int jj2 = 0; jj2 <= k; ++jj2) {
            for (int jj12 = 0; jj12 <= k; ++jj12) {
                for (int jj3 = 0; jj3 <= k; ++jj3) {
                    for (int jj = 0; jj <= k; ++jj) {
                        for (int jj23 = 0; jj23 <= k; ++jj23) {
                            auto const max6 = std::max({ jj1, jj2, jj3, jj, jj12, jj23 });
                            auto const max4 = std::max({ jj2, jj, jj12, jj23 });
                            if (jj1 != max6 || jj2 != max4) {
                                continue;
                            }
                            auto const jsymbol = _j_symbol(jj1, jj2, jj12, jj3, jj, jj23);
                            if (jsymbol != 0.0) {
                                _6j[SixJKey{ jj1, jj2, jj12, jj3, jj, jj23 }] = jsymbol;
                            }
                        }
                    }
                }
            }
        }
    }
}

float64
SU2_kAnyonCategory::_n_q(int n) const
{
    auto const half_n = 0.5 * static_cast<float64>(n);
    auto const q_half = std::pow(_q, 0.5);
    auto const q_minus_half = std::pow(_q, -0.5);
    auto const num = std::pow(_q, half_n) - std::pow(_q, -half_n);
    auto const den = q_half - q_minus_half;
    return std::real(num / den);
}

float64
SU2_kAnyonCategory::_n_q_fac(int n) const
{
    float64 fac = 1.0;
    for (int i = 0; i < n; ++i) {
        fac *= _n_q(i + 1);
    }
    return fac;
}

float64
SU2_kAnyonCategory::_delta(int jj1, int jj2, int jj3) const
{
    float64 res = _n_q_fac(iround(-jj1 / 2.0 + jj2 / 2.0 + jj3 / 2.0)) *
                  _n_q_fac(iround(jj1 / 2.0 - jj2 / 2.0 + jj3 / 2.0));
    res *= _n_q_fac(iround(jj1 / 2.0 + jj2 / 2.0 - jj3 / 2.0)) /
           _n_q_fac(iround(jj1 / 2.0 + jj2 / 2.0 + jj3 / 2.0 + 1.0));
    return std::sqrt(res);
}

float64
SU2_kAnyonCategory::_j_symbol(int jj1, int jj2, int jj12, int jj3, int jj, int jj23) const
{
    // --- hints from Python SU2_kAnyonCategory._j_symbol ---
    // runs over all integers for which the factorials have non-negative arguments
    // ---
    int const triads[4][3] = {
        { jj1, jj2, jj12 }, { jj1, jj, jj23 }, { jj3, jj2, jj23 }, { jj3, jj, jj12 }
    };
    for (auto const& triad : triads) {
        if (triad[0] > triad[1] + triad[2] || triad[0] < std::abs(triad[1] - triad[2])) {
            return 0.0;
        }
    }

    int const start =
      std::max({ jj1 + jj2 + jj12, jj12 + jj3 + jj, jj2 + jj3 + jj23, jj1 + jj23 + jj }) / 2;
    int const stop =
      std::min({ jj1 + jj2 + jj3 + jj, jj1 + jj12 + jj3 + jj23, jj2 + jj12 + jj + jj23 }) / 2;

    float64 res = 0.0;
    for (int z = start; z <= stop; ++z) {
        float64 factor = 1.0;
        factor *= _n_q_fac(iround(static_cast<float64>(z) - jj1 / 2.0 - jj2 / 2.0 - jj12 / 2.0));
        factor *= _n_q_fac(iround(static_cast<float64>(z) - jj12 / 2.0 - jj3 / 2.0 - jj / 2.0));
        factor *= _n_q_fac(iround(static_cast<float64>(z) - jj2 / 2.0 - jj3 / 2.0 - jj23 / 2.0));
        factor *= _n_q_fac(iround(static_cast<float64>(z) - jj1 / 2.0 - jj23 / 2.0 - jj / 2.0));
        factor *= _n_q_fac(iround(jj1 / 2.0 + jj2 / 2.0 + jj3 / 2.0 + jj / 2.0 - z));
        factor *= _n_q_fac(iround(jj1 / 2.0 + jj12 / 2.0 + jj3 / 2.0 + jj23 / 2.0 - z));
        factor *= _n_q_fac(iround(jj2 / 2.0 + jj12 / 2.0 + jj / 2.0 + jj23 / 2.0 - z));
        res += std::pow(-1.0, static_cast<float64>(z)) * _n_q_fac(z + 1) / factor;
    }

    return res * _delta(jj1, jj2, jj12) * _delta(jj12, jj3, jj) * _delta(jj2, jj3, jj23) *
           _delta(jj1, jj23, jj);
}

bool
SU2_kAnyonCategory::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] <= k;
}

bool
SU2_kAnyonCategory::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto const q = sectors[i][0];
        if (q < 0 || q > k) {
            return false;
        }
    }
    return true;
}

SectorArray
SU2_kAnyonCategory::fusion_outcomes(Sector a, Sector b) const
{
    auto const aa = static_cast<int>(a.q[0]);
    auto const bb = static_cast<int>(b.q[0]);
    auto const upper_limit = std::min(aa + bb, 2 * k - aa - bb);
    auto const lower = std::abs(aa - bb);
    auto const n = static_cast<std::size_t>((upper_limit - lower) / 2 + 1);
    SectorArray out(n, 1);
    for (std::size_t i = 0; i < n; ++i) {
        out[i][0] = static_cast<int16_t>(lower + static_cast<int>(2 * i));
    }
    return out;
}

std::string
SU2_kAnyonCategory::sector_str(Sector a) const
{
    auto const jj = a.q[0];
    std::string j_str = (jj % 2 == 0) ? std::to_string(jj / 2) : (std::to_string(jj) + "/2");
    return std::to_string(jj) + " (j=" + j_str + ")";
}

std::string
SU2_kAnyonCategory::repr() const
{
    return "SU2_kAnyonCategory(" + std::to_string(k) + ", \"" + handedness + "\")";
}

bool
SU2_kAnyonCategory::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* cat = dynamic_cast<SU2_kAnyonCategory const*>(&other)) {
        return cat->k == k && cat->handedness == handedness;
    }
    return false;
}

Sector
SU2_kAnyonCategory::dual_sector(Sector a) const
{
    return a;
}

SectorArray
SU2_kAnyonCategory::dual_sectors(SectorArray const& sectors) const
{
    return sectors;
}

int64
SU2_kAnyonCategory::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

FusionSymbol
SU2_kAnyonCategory::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    // --- hints from Python SU2_kAnyonCategory._f_symbol ---
    // The q-deformed 6j symbols have the same symmetries as the usual SU(2) 6j symbols.
    // We can get all f symbols from the cases 6j symbols for
    // a == np.max([a, b, c, d, e, f]) and b == np.max([b, c, e, f]).
    // I.e., we need to exchange the charges accordingly
    // need to compute before exchanging charges
    // nontrivial F-symbols
    // ---
    int ja = a.q[0];
    int jb = b.q[0];
    int jc = c.q[0];
    int jd = d.q[0];
    int je = e.q[0];
    int jf = f.q[0];

    auto factor = std::sqrt(_n_q(je + 1) * _n_q(jf + 1));
    factor *= std::pow(-1.0, static_cast<float64>((ja + jb + jc + jd) / 2));

    int argm = argmax({ ja, jc, jb, jd, jf, je });
    if (argm > 1) {
        if (argm / 2 == 1) {
            std::swap(ja, jb);
            std::swap(jc, jd);
        } else {
            std::swap(ja, jf);
            std::swap(jc, je);
        }
    }

    int argm_ = argmax({ jb, jd, jf, je });
    if (argm_ > 1) {
        std::swap(jb, jf);
        std::swap(jd, je);
    }

    if (argm % 2 == 1 && argm_ % 2 == 1) {
        std::swap(ja, jc);
        std::swap(jb, jd);
    } else if (argm % 2 == 1) {
        std::swap(ja, jc);
        std::swap(jf, je);
    } else if (argm_ % 2 == 1) {
        std::swap(jb, jd);
        std::swap(jf, je);
    }

    auto it = _6j.find(SixJKey{ ja, jb, jf, jc, jd, je });
    if (it != _6j.end()) {
        return scale_one_4D(static_cast<complex128>(factor * it->second));
    }
    return topo_ones::one_4D();
}

int64
SU2_kAnyonCategory::frobenius_schur(Sector a) const
{
    return (a.q[0] % 2 == 1) ? -1 : 1;
}

float64
SU2_kAnyonCategory::qdim(Sector a) const
{
    auto const denom = std::sin(std::numbers::pi_v<float64> / static_cast<float64>(k + 2));
    return std::sin(static_cast<float64>(a.q[0] + 1) * std::numbers::pi_v<float64> /
                    static_cast<float64>(k + 2)) /
           denom;
}

std::vector<float64>
SU2_kAnyonCategory::batch_qdim(SectorArray const& a) const
{
    std::vector<float64> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i] = qdim(a[i]);
    }
    return out;
}

FusionSymbol
SU2_kAnyonCategory::_r_symbol(Sector a, Sector b, Sector c) const
{
    // --- hints from Python SU2_kAnyonCategory._r_symbol ---
    // nontrivial R-symbols
    // ---
    int ja = a.q[0];
    int jb = b.q[0];
    if (ja < jb) {
        std::swap(ja, jb);
    }
    auto it = _r.find(RKey{ ja, jb, c.q[0] });
    if (it != _r.end()) {
        return it->second;
    }
    return topo_ones::one_1D();
}

SectorArray
SU2_kAnyonCategory::all_sectors() const
{
    SectorArray out(static_cast<std::size_t>(k + 1), 1);
    for (int i = 0; i <= k; ++i) {
        out[static_cast<std::size_t>(i)][0] = static_cast<int16_t>(i);
    }
    return out;
}

void
SU2_kAnyonCategory::save_hdf5(py::object hdf5_saver,
                              py::object h5gr,
                              std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(k, subpath + "k");
    hdf5_saver.attr("save")(handedness, subpath + "handedness");
}

SU2_kAnyonCategory::Ptr
SU2_kAnyonCategory::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    int k = hdf5_loader.attr("load")(subpath + "k").cast<int>();
    std::string handedness = hdf5_loader.attr("load")(subpath + "handedness").cast<std::string>();
    auto obj = std::make_shared<SU2_kAnyonCategory>(k, handedness);
    obj->descriptive_name = descriptive_name_from_hdf5_attrs(h5gr);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
