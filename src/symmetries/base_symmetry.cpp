#include <cyten/symmetries/base_symmetry.h>

#include <cyten/config.h>

#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace cyten {

namespace {

py::array
zeros_like_int64_1d(py::ssize_t n)
{
    return py::array_t<int64>({n});
}

} // namespace

BaseSymmetry::BaseSymmetry(FusionStyle fusion_style_,
                           BraidingStyle braiding_style_,
                           Sector trivial_sector_,
                           float64 num_sectors_,
                           bool has_complex_topological_data_,
                           bool trivial_shift_)
    : fusion_style(fusion_style_)
    , braiding_style(braiding_style_)
    , trivial_sector(trivial_sector_)
    , num_sectors(num_sectors_)
    , sector_ind_len(trivial_sector_.len)
    , empty_sector_array(SectorArray::empty(trivial_sector_.len))
    , has_complex_topological_data(has_complex_topological_data_)
    , trivial_shift(trivial_shift_)
{
    if (trivial_sector_.len == 0 || trivial_sector_.len > max_sector_ind_len) {
        throw std::invalid_argument("BaseSymmetry: invalid trivial_sector length");
    }
}

bool
BaseSymmetry::can_be_dropped() const
{
    return has_symmetric_braid();
}

bool
BaseSymmetry::has_symmetric_braid() const
{
    return braiding_style <= BraidingStyle::fermionic;
}

bool
BaseSymmetry::has_trivial_braid() const
{
    return braiding_style == BraidingStyle::bosonic;
}

bool
BaseSymmetry::is_abelian() const
{
    return fusion_style == FusionStyle::single;
}

bool
BaseSymmetry::has_unique_fusion() const
{
    return fusion_style <= FusionStyle::multiple_unique;
}

py::array
BaseSymmetry::_fusion_tensor(Sector /*a*/, Sector /*b*/, Sector /*c*/, bool /*Z_a*/, bool /*Z_b*/)
  const
{
    if (!can_be_dropped()) {
        throw SymmetryError("fusion tensor can not be written as array for this symmetry");
    }
    throw std::runtime_error("BaseSymmetry::_fusion_tensor should be implemented by subclass");
}

py::array
BaseSymmetry::swap_gate(Sector /*a*/, Sector /*b*/) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("braid can not be written as array for this symmetry");
    }
    throw std::runtime_error("BaseSymmetry::swap_gate should be implemented by subclass");
}

py::array
BaseSymmetry::Z_iso(Sector a) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("Z iso can not be written as array for this symmetry");
    }
    // fallback: sqrt(d_a) * conj(X)[0, :, :, 0].T
    auto X = fusion_tensor(a, dual_sector(a), trivial_sector);
    auto Xc = py::reinterpret_borrow<py::array>(X.attr("conj")());
    auto slice = py::reinterpret_steal<py::object>(
      Xc.attr("__getitem__")(py::make_tuple(0, py::ellipsis(), 0)));
    auto transposed = slice.attr("T");
    auto np = py::module_::import("numpy");
    return py::reinterpret_steal<py::array>(np.attr("multiply")(sqrt_qdim(a), transposed));
}

SectorArray
BaseSymmetry::all_sectors() const
{
    if (!std::isfinite(num_sectors)) {
        throw SymmetryError("symmetry has infinitely many sectors.");
    }
    throw std::runtime_error("BaseSymmetry::all_sectors should be implemented in subclass");
}

int64
BaseSymmetry::n_symbol(Sector a, Sector b, Sector c) const
{
    if (!can_fuse_to(a, b, c)) {
        return 0;
    }
    return _n_symbol(a, b, c);
}

py::array
BaseSymmetry::f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (get_config().check_fusion) {
        bool ok = can_fuse_to(b, c, e) && can_fuse_to(a, e, d) && can_fuse_to(a, b, f)
                  && can_fuse_to(f, c, d);
        if (!ok) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _f_symbol(a, b, c, d, e, f);
}

py::array
BaseSymmetry::b_symbol(Sector a, Sector b, Sector c) const
{
    if (get_config().check_fusion) {
        if (!can_fuse_to(a, b, c)) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _b_symbol(a, b, c);
}

py::array
BaseSymmetry::r_symbol(Sector a, Sector b, Sector c) const
{
    if (get_config().check_fusion) {
        if (!can_fuse_to(a, b, c)) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _r_symbol(a, b, c);
}

py::array
BaseSymmetry::c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    if (get_config().check_fusion) {
        bool ok = can_fuse_to(a, b, e) && can_fuse_to(e, c, d) && can_fuse_to(a, c, f)
                  && can_fuse_to(f, b, d);
        if (!ok) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _c_symbol(a, b, c, d, e, f);
}

py::array
BaseSymmetry::fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    if (get_config().check_fusion) {
        if (!can_fuse_to(a, b, c)) {
            throw SymmetryError("Sectors are not consistent with fusion rules.");
        }
    }
    return _fusion_tensor(a, b, c, Z_a, Z_b);
}

bool
BaseSymmetry::are_valid_sectors(SectorArray const& sectors) const
{
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        if (!is_valid_sector(sectors[i])) {
            return false;
        }
    }
    return true;
}

SectorArray
BaseSymmetry::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    assert(is_abelian());
    assert(a.num_sectors == b.num_sectors);
    assert(a.sector_ind_len == sector_ind_len);
    SectorArray out(a.num_sectors, sector_ind_len);
    for (std::size_t i = 0; i < a.num_sectors; ++i) {
        auto outcomes = fusion_outcomes(a[i], b[i]);
        assert(outcomes.num_sectors == 1);
        out.set(i, outcomes[0]);
    }
    return out;
}

Sector
BaseSymmetry::multiple_fusion(std::vector<Sector> const& sectors) const
{
    std::vector<SectorArray> as_arrays;
    as_arrays.reserve(sectors.size());
    for (auto const& s : sectors) {
        SectorArray row(1, s.len);
        row.set(0, s);
        as_arrays.push_back(std::move(row));
    }
    return multiple_fusion_broadcast(as_arrays)[0];
}

SectorArray
BaseSymmetry::multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    if (sectors.empty()) {
        SectorArray out(1, sector_ind_len);
        out.set(0, trivial_sector);
        return out;
    }
    if (sectors.size() == 1) {
        return sectors[0];
    }
    return _multiple_fusion_broadcast(sectors);
}

SectorArray
BaseSymmetry::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    assert(sectors.size() >= 2);
    SectorArray acc = sectors[0];
    for (std::size_t i = 1; i < sectors.size(); ++i) {
        acc = fusion_outcomes_broadcast(acc, sectors[i]);
    }
    return acc;
}

bool
BaseSymmetry::can_fuse_to(Sector a, Sector b, Sector c) const
{
    auto outcomes = fusion_outcomes(a, b);
    for (std::size_t i = 0; i < outcomes.num_sectors; ++i) {
        if (outcomes[i] == c) {
            return true;
        }
    }
    return false;
}

int64
BaseSymmetry::sector_dim(Sector a) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("sector_dim is not supported for this symmetry.");
    }
    // Note: Python incorrectly called qdim() without `a`; use qdim(a).
    return static_cast<int64>(std::llround(qdim(a)));
}

py::array
BaseSymmetry::batch_sector_dim(SectorArray const& a) const
{
    py::array_t<int64> out(static_cast<py::ssize_t>(a.num_sectors));
    auto r = out.mutable_unchecked<1>();
    if (is_abelian()) {
        for (std::size_t i = 0; i < a.num_sectors; ++i) {
            r(static_cast<py::ssize_t>(i)) = 1;
        }
        return out;
    }
    for (std::size_t i = 0; i < a.num_sectors; ++i) {
        r(static_cast<py::ssize_t>(i)) = sector_dim(a[i]);
    }
    return out;
}

py::array
BaseSymmetry::batch_qdim(SectorArray const& a) const
{
    // Python returns dtype=int for abelian; float otherwise. Use float64 always for C++.
    py::array_t<float64> out(static_cast<py::ssize_t>(a.num_sectors));
    auto r = out.mutable_unchecked<1>();
    if (is_abelian()) {
        for (std::size_t i = 0; i < a.num_sectors; ++i) {
            r(static_cast<py::ssize_t>(i)) = 1.0;
        }
        return out;
    }
    for (std::size_t i = 0; i < a.num_sectors; ++i) {
        r(static_cast<py::ssize_t>(i)) = qdim(a[i]);
    }
    return out;
}

std::string
BaseSymmetry::sector_str(Sector a) const
{
    std::ostringstream oss;
    oss << '[';
    for (std::uint8_t i = 0; i < a.len; ++i) {
        if (i != 0) {
            oss << ' ';
        }
        oss << a[i];
    }
    oss << ']';
    return oss.str();
}

SectorArray
BaseSymmetry::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.num_sectors, sectors.sector_ind_len);
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        out.set(i, dual_sector(sectors[i]));
    }
    return out;
}

int64
BaseSymmetry::frobenius_schur(Sector a) const
{
    auto F = _f_symbol(a, dual_sector(a), a, a, trivial_sector, trivial_sector);
    auto val = F.attr("__getitem__")(py::make_tuple(0, 0, 0, 0)).cast<complex128>();
    float64 re = val.real();
    return (re >= 0.0) ? 1 : -1;
}

float64
BaseSymmetry::qdim(Sector a) const
{
    auto F = _f_symbol(a, dual_sector(a), a, a, trivial_sector, trivial_sector);
    auto val = F.attr("__getitem__")(py::make_tuple(0, 0, 0, 0)).cast<complex128>();
    return 1.0 / std::abs(val);
}

float64
BaseSymmetry::sqrt_qdim(Sector a) const
{
    return std::sqrt(qdim(a));
}

float64
BaseSymmetry::inv_sqrt_qdim(Sector a) const
{
    return 1.0 / sqrt_qdim(a);
}

float64
BaseSymmetry::total_qdim() const
{
    auto sectors = all_sectors();
    float64 D2 = 0.0;
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        auto d = qdim(sectors[i]);
        D2 += d * d;
    }
    return std::sqrt(D2);
}

py::array
BaseSymmetry::_b_symbol(Sector a, Sector b, Sector c) const
{
    auto F = _f_symbol(a, b, dual_sector(b), a, trivial_sector, c).attr("conj")();
    // F[0, 0, :, :] * sqrt(d_b)
    auto block = F.attr("__getitem__")(py::make_tuple(0, 0, py::ellipsis()));
    auto np = py::module_::import("numpy");
    return py::reinterpret_steal<py::array>(np.attr("multiply")(sqrt_qdim(b), block));
}

py::array
BaseSymmetry::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    auto R1 = _r_symbol(e, c, d);
    auto F = _f_symbol(c, a, b, d, e, f);
    auto R2 = _r_symbol(a, c, f);
    // R1[None, :, None, None] * F * conj(R2)[None, None, :, None]
    auto np = py::module_::import("numpy");
    auto R1e = R1.attr("reshape")(py::make_tuple(1, -1, 1, 1));
    auto R2e = R2.attr("conj")().attr("reshape")(py::make_tuple(1, 1, -1, 1));
    return py::reinterpret_steal<py::array>(np.attr("multiply")(np.attr("multiply")(R1e, F), R2e));
}

complex128
BaseSymmetry::topological_twist(Sector a) const
{
    if (has_trivial_braid()) {
        return complex128{+1.0, 0.0};
    }
    complex128 res{0.0, 0.0};
    auto outcomes = fusion_outcomes(a, a);
    for (std::size_t i = 0; i < outcomes.num_sectors; ++i) {
        Sector b = outcomes[i];
        auto r = _r_symbol(a, a, b);
        auto sum_r = r.attr("sum")().cast<complex128>();
        res += qdim(b) * sum_r;
    }
    res /= qdim(a);
    if (has_symmetric_braid()) {
        float64 re = res.real();
        return complex128{(re < 0.0) ? -1.0 : +1.0, 0.0};
    }
    return res;
}

complex128
BaseSymmetry::s_matrix_element(Sector a, Sector b) const
{
    complex128 S{0.0, 0.0};
    auto outcomes = fusion_outcomes(a, b);
    for (std::size_t i = 0; i < outcomes.num_sectors; ++i) {
        Sector c = outcomes[i];
        S += static_cast<float64>(_n_symbol(a, b, c)) * qdim(c) * topological_twist(c);
    }
    S /= topological_twist(a) * topological_twist(b) * total_qdim();
    // real_if_close: if imag part tiny, drop it
    if (std::abs(S.imag()) < 1e-12) {
        return complex128{S.real(), 0.0};
    }
    return S;
}

py::array
BaseSymmetry::s_matrix() const
{
    auto sectors = all_sectors();
    auto n = static_cast<py::ssize_t>(sectors.num_sectors);
    py::array_t<complex128> S({n, n});
    auto r = S.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            r(i, j) = 0.0;
        }
    }
    float64 D = total_qdim();
    std::vector<complex128> inv_twist(static_cast<std::size_t>(n));
    for (py::ssize_t i = 0; i < n; ++i) {
        inv_twist[static_cast<std::size_t>(i)] = complex128{1.0, 0.0} / topological_twist(sectors[static_cast<std::size_t>(i)]);
    }
    for (py::ssize_t ia = 0; ia < n; ++ia) {
        Sector a = sectors[static_cast<std::size_t>(ia)];
        for (py::ssize_t ib = 0; ib < n; ++ib) {
            Sector b = sectors[static_cast<std::size_t>(ib)];
            complex128 Sab{0.0, 0.0};
            auto outcomes = fusion_outcomes(a, b);
            for (std::size_t k = 0; k < outcomes.num_sectors; ++k) {
                Sector c = outcomes[k];
                Sab += static_cast<float64>(_n_symbol(a, b, c)) * qdim(c) * topological_twist(c);
            }
            Sab *= inv_twist[static_cast<std::size_t>(ia)] * inv_twist[static_cast<std::size_t>(ib)] / D;
            if (std::abs(Sab.imag()) < 1e-12) {
                Sab = complex128{Sab.real(), 0.0};
            }
            r(ia, ib) = Sab;
        }
    }
    return S;
}

} // namespace cyten
