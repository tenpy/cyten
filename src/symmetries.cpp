

#include <cassert>
#include <stdexcept>
#include <ranges>

#include "cyten/symmetries.h"

#ifndef DO_FUSION_INPUT_CHECKS
#define DO_FUSION_INPUT_CHECKS 1
#endif

namespace cyten {

Symmetry::Symmetry(FusionStyle fusion, BraidingStyle braiding)
    : fusion_style(fusion), braiding_style(braiding), _num_sectors(INFINITE_NUM_SECTORS)
{
    np = py::module_::import("numpy");
}

bool Symmetry::can_be_dropped() const {
    return false;
}

bool Symmetry::is_abelian() const {
    return fusion_style == FusionStyle::single;
}

size_t Symmetry::num_sectors() const {
    return _num_sectors;
}

size_t Symmetry::sector_ind_len() const {
    return _sector_ind_len;
}

bool Symmetry::has_symmetric_braid() const {
    return braiding_style <= BraidingStyle::fermionic;
}

cyten_int Symmetry::n_symbol(Sector a, Sector b, Sector c) const {
    if (! can_fuse_to(a, b, c))
        return 0;
    return _n_symbol(a, b, c);
}

py::array Symmetry::f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const {
#if DO_FUSION_INPUT_CHECKS
    if (!(
        can_fuse_to(b, c, e) &&
        can_fuse_to(a, e, d) &&
        can_fuse_to(a, b, f) &&
        can_fuse_to(f, c, d)
    ))
        throw SymmetryError("Sectors are not consistent with fusion rules.");
#endif
    return _f_symbol(a, b, c, d, e, f);
}

py::array Symmetry::r_symbol(Sector a, Sector b, Sector c) const {
#if DO_FUSION_INPUT_CHECKS
    if (!can_fuse_to(a, b, c))
        throw SymmetryError("Sectors are not consistent with fusion rules.");
#endif
    return _r_symbol(a, b, c);
}


py::array Symmetry::fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const {
#if DO_FUSION_INPUT_CHECKS
    if (!can_fuse_to(a, b, c))
        throw SymmetryError("Sectors are not consistent with fusion rules.");
#endif
    return _fusion_tensor(a, b, c, Z_a, Z_b);
}

py::array Symmetry::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    if (! can_be_dropped())
        throw SymmetryError("fusion_tensor not possible: can't drop symmetry for " + group_name());
    throw SymmetryError("not implemented for " + group_name());
}

SectorArray Symmetry::all_sectors() const
{
    if (_num_sectors == INFINITE_NUM_SECTORS)
        throw SymmetryError("Infinite Sectors in " + group_name());
    throw SymmetryError("all_sectors() not implemented in " + group_name());
}

// FALLBACK IMPLEMENTATIONS (might want to override)

Sector Symmetry::compress_sector(std::vector<Sector> const & decompressed) const {
    // default implementation for _sector_ind_len = 1
    if (_sector_ind_len != 1)
        throw SymmetryError("compress_sector() for _sector_ind_len != 1 needs to be implemented");
    assert(decompressed.size() == 1);
    return decompressed[0];
}

std::vector<Sector> Symmetry::decompress_sector(Sector compressed) const {
    // default implementation for _sector_ind_len = 1
    if (_sector_ind_len != 1)
        throw SymmetryError("decompress_sector() for _sector_ind_len != 1 needs to be implemented");
    return {compressed};
}


SectorArray Symmetry::compress_sectorarray(py::array_t<Sector> sectors) const {
    assert(sectors.ndim() == 1);
    std::vector<Sector> decompressed(sector_ind_len());
    SectorArray result(sectors.shape()[0]);
    for(size_t i = 0; i < sectors.shape()[0]; ++i) {
        for (size_t j = 0; j < sector_ind_len(); ++j)
            decompressed[j] = sectors.at(i, j);
        result[i] = compress_sector(decompressed);
    }
    return result;
}

py::array_t<Sector> Symmetry::decompress_sectorarray(SectorArray sectors) const {
    py::array_t<Sector> result({sectors.size(), sector_ind_len()});
    for(size_t i = 0; i < sectors.size(); ++i) {
        std::vector<Sector> decompressed = decompress_sector(sectors[i]);
        for (size_t j = 0; j < sector_ind_len(); ++j)
            result.mutable_at(i, j) = decompressed[j];
    }
    return result;
}


bool Symmetry::are_valid_sectors(SectorArray const& sectors) const {
    for (auto const& s : sectors)
        if (! is_valid_sector(s))
            return false;
    return true;
}

SectorArray Symmetry::fusion_outcomes_broadcast(SectorArray const & a, SectorArray const & b) const {
    assert(a.size() == b.size());
    SectorArray result(a.size());
    for (auto [c, a_entry, b_entry] : std::views::zip(result, a, b))
        c = fusion_outcomes(a_entry, b_entry)[0];
    return result;
}

Sector Symmetry::multiple_fusion(std::vector<Sector> const & sectors) const {
    // fall back to multiple_fusion_broadcast implementation
    std::vector<SectorArray> sector_arrays;
    for (auto & s : sectors)
        sector_arrays.push_back({s});
    return multiple_fusion_broadcast(sector_arrays)[0];
}

SectorArray Symmetry::multiple_fusion_broadcast(std::vector<SectorArray> const & sectors) const {
    assert(is_abelian());
    if (sectors.size() == 0)
        return {trivial_sector()};
    if (sectors.size() == 1)
        return *sectors.begin();
    return _multiple_fusion_broadcast(sectors);
}

SectorArray Symmetry::_multiple_fusion_broadcast(std::vector<SectorArray> const & sectors) const {
    assert(sectors.size() > 1); // can assume sectors.size() > 2
    auto it = sectors.begin();
    SectorArray result = fusion_outcomes_broadcast(*it++, *it++);
    for (; it != sectors.end(); ++it)
        result = fusion_outcomes_broadcast(result, *it);
    return result;
}


bool Symmetry::can_fuse_to(Sector a, Sector b, Sector c) const {
    for (Sector s : fusion_outcomes(a, b))
        if (s == c)
            return true;
    return false;
}

cyten_int Symmetry::sector_dim(Sector a) const {
    if (! can_be_dropped())
        throw SymmetryError("sector_dim is not supported for " + group_name());
    return (cyten_int)(qdim(a) + 0.5);
}

std::vector<cyten_int> Symmetry::sector_dim(SectorArray const & a) const {
    std::vector<cyten_int> result;
    result.reserve(a.size());
    for (Sector b : a)
        result.push_back(sector_dim(b));
    return result;
    // return a | std::views::transform(sector_dim);
}

cyten_float Symmetry::qdim(Sector a) const {
    py::array f = _f_symbol(a, dual_sector(a), a, a, trivial_sector(), trivial_sector());
    cyten_complex c = * static_cast<cyten_complex const*>(f.data(0, 0, 0, 0)); // TODO make f_symbol return py::array_t
    return 1./std::abs(c);
}

cyten_int Symmetry::frobenius_schur(Sector a) const {
    py::array f = _f_symbol(a, dual_sector(a), a, a, trivial_sector(), trivial_sector());
    cyten_complex c = f[py::make_tuple(0, 0, 0, 0)].cast<cyten_complex>();
    return 1 ? (c.real() > 0) : -1;
}

std::vector<cyten_float> Symmetry::qdim(SectorArray const& a) const {
    std::vector<cyten_float> result;
    result.reserve(a.size());
    for (Sector b : a)
        result.push_back(qdim(b));
    return result;
}


std::string Symmetry::sector_str(Sector a) const {
    return std::to_string(a);
}

cyten_float Symmetry::sqrt_qdim(Sector a) const {
    return std::sqrt(qdim(a));
}

cyten_float Symmetry::inv_sqrt_qdim(Sector a) const {
    return 1./std::sqrt(qdim(a));
}

cyten_float Symmetry::total_qdim() const {
    cyten_int D = 0;
    for (Sector a : all_sectors()) {
        cyten_int d = qdim(a);
        D += d*d;
    }
    return std::sqrt((cyten_float) D);
}


py::array Symmetry::b_symbol(Sector a, Sector b, Sector c) const {
#if DO_FUSION_INPUT_CHECKS
    if (!can_fuse_to(a, b, c))
        throw SymmetryError("Sectors are not consistent with fusion rules.");
#endif
    return _b_symbol(a, b, c);
}

py::array Symmetry::c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const {
#if DO_FUSION_INPUT_CHECKS
    if (!(can_fuse_to(a, b, e) &&
          can_fuse_to(e, c, d) &&
          can_fuse_to(a, c, f) &&
          can_fuse_to(f, b, d)))
        throw SymmetryError("Sectors are not consistent with fusion rules.");
#endif
    return _c_symbol(a, b, c, d, e, f);
}


py::array Symmetry::_b_symbol(Sector a, Sector b, Sector c) const {
    py::array F = _f_symbol(a, b, dual_sector(b), a, trivial_sector(), c).attr("conj")();
    return py::float_(sqrt_qdim(b)) * F[py::make_tuple(0, 0, py::slice(), py::slice())]; // TODO: without python wrapping?
}

py::array Symmetry::_c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const {
    // axis [mu, nu, kap, lam] ; R symbols are diagonal
    py::object R1 = np.attr("array")(_r_symbol(e, c, d))[py::make_tuple(py::none(), py::slice(), py::none(), py::none())];
    py::object F = np.attr("array")(_f_symbol(c, a, b, d, e, f));
    py::object R2 = np.attr("conj")(_r_symbol(a, c, f))[py::make_tuple(py::none(), py::none(), py::slice(), py::none())];
    return R1 * F * R2;  // TODO: without python wrapping?
}

cyten_complex Symmetry::topological_twist(Sector a) const {
    return static_cast<cyten_float>(frobenius_schur(a)) * std::conj(*static_cast<cyten_complex const*>(_r_symbol(dual_sector(a), a, trivial_sector()).data(0)));
}

cyten_complex Symmetry::s_matrix_element(Sector a, Sector b) const {
    cyten_complex S = 0;
    for (Sector c : fusion_outcomes(a, b))
        S += _n_symbol(a, b, c) * qdim(c) * topological_twist(c);
    S /= topological_twist(a) * topological_twist(b) * total_qdim();
    return S;
}

py::array Symmetry::s_matrix(Sector a, Sector b) const {
    py::array_t<cyten_complex> result({num_sectors(), num_sectors()});
    auto res = result.mutable_unchecked<2>();
    auto all_sectors_ = all_sectors();
    auto total_qdim_ = total_qdim();
    for(size_t i = 0; i < num_sectors(); ++i) {
        Sector a = all_sectors_[i];
        for(size_t j = 0; j < num_sectors(); ++j) {
            Sector b = all_sectors_[j];
            cyten_complex entry = 0;
            for (Sector c : fusion_outcomes(a, b))
                entry += _n_symbol(a, b, c) * qdim(c) * topological_twist(c);
            res(i, j) = entry / topological_twist(a) / topological_twist(b) / total_qdim_;
        }
    }
    return result;
}


// CONCRETE IMPLEMENTATIONS

SectorArray Symmetry::dual_sectors(SectorArray const & sectors) const {
    SectorArray res;
    res.reserve(sectors.size());
    for (Sector a : sectors)
        res.push_back(dual_sector(a));
    return res;
}

std::string Symmetry::__str__() const {
    if (descriptive_name.size() > 0)
        return group_name() + "(" + descriptive_name + ")";
    return group_name();
}

bool Symmetry::operator==(Symmetry const & rhs) const {
    if (descriptive_name != rhs.descriptive_name)
        return false;
    return is_same_symmetry(rhs);
}

GroupSymmetry::GroupSymmetry(FusionStyle fusion)
    :Symmetry(fusion, BraidingStyle::bosonic)
{
}

bool GroupSymmetry::can_be_dropped() const {
    return true;
}



Sector _compress_sector(std::vector<Sector> decompressed, std::vector<int> const &bit_lengths) {
    const int N = bit_lengths.size();
    // rest is same code as in _compress_sector_fixed<N>(std::array<Sector, N> const &, std::array<int, N> const &)
    Sector compressed = 0;
    int shift = 0;
    const Sector sign_bit = Sector(1) << 63;
    for (int i = 0; i < N ; ++i) {
        //keep least significant bit_lengths[i] bits
        // note: negative values start with bitstrings of 1, so most significant bit kept is new sign bit.
        Sector mask_last_bits = (Sector(1) << bit_lengths[i]) - 1;
        Sector compress_i = decompressed[i] & mask_last_bits;
        if (decompressed[i] >> 63 & 1) {
            if ((decompressed[i] | ((Sector(1) << (bit_lengths[i] - 1)) - 1)) != Sector(-1))
                throw std::overflow_error("discarding bits in compress_Sector");
        } else {
            if ((decompressed[i] & ~((Sector(1) << (bit_lengths[i] - 1)) - 1)) != Sector(0))
                throw std::overflow_error("discarding bits in compress_Sector");
        }
        compressed |= compress_i << shift;
        shift += bit_lengths[i];
    }
    assert(shift <= 64);
    return compressed;
}

std::vector<Sector> _decompress_sector(Sector compressed, std::vector<int> const &bit_lengths) {
    const int N = bit_lengths.size();
    std::vector<Sector> decompressed(bit_lengths.size());
    // rest is same code as in _decompress_sector_fixed<N>(Sector, std::array<int, N> const &)
    int shift = 0;
    const Sector sign_bit = Sector(1) << 63;
    for (size_t i = 0; i < N ; ++i) {
        // kept sign bit + bit_lengths[i]-1 bits from right of decompressed[i]
        Sector mask_last_bits = (Sector(1) << bit_lengths[i]) - 1;
        Sector decomp = (compressed >> shift) & mask_last_bits;
        if (decomp >> (bit_lengths[i] - 1))
            decomp |= ~mask_last_bits;
        decompressed[i] = decomp;
        shift += bit_lengths[i];
    }
    assert(shift <= 64);
    return decompressed;
}

} // namespace cyten
