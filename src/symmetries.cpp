

#include <cassert>
#include <stdexcept>

#include "cyten/symmetries.h"

namespace cyten {
    
Symmetry::Symmetry(FusionStyle fusion, BraidingStyle braiding)
    : fusion_style(fusion), braiding_style(braiding), _num_sectors(INFINITE_NUM_SECTORS)
{

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

SectorArray Symmetry::compress_sectorarray(py::array_t<Sector> sectors) {
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

py::array_t<Sector> Symmetry::decompress_sectorarray(SectorArray sectors) {
    py::array_t<Sector> result({sectors.size(), sector_ind_len()});
    for(size_t i = 0; i < sectors.size(); ++i) {
        std::vector<Sector> decompressed = decompress_sector(sectors[i]);
        for (size_t j = 0; j < sector_ind_len(); ++j)
            result.mutable_at(i, j) = decompressed[j];
    }
    return result;
}

bool Symmetry::has_symmetric_braid() {
    return braiding_style <= BraidingStyle::fermionic;
}

py::array Symmetry::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b)
{
    throw SymmetryError("Not defined");
}

SectorArray Symmetry::all_sectors()
{
    return SectorArray();
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
