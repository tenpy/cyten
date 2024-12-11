#pragma once

#include <vector>
#include <cstdint>
#include <string>
#include <array>

#include <stdexcept>
#include <cassert>

namespace cyten {

enum class FusionStyle {
    single = 0,  // only one resulting sector, a ⊗ b = c, e.g. abelian symmetry groups
    multiple_unique = 10,  // every sector appears at most once in pairwise fusion, N^{ab}_c \in {0,1}
    general = 20,  // assumptions N^{ab}_c = 0, 1, 2, ...
};

enum class BraidingStyle { 
    bosonic = 0,  // symmetric braiding with trivial twist; v ⊗ w ↦ w ⊗ v
    fermionic = 10,  // symmetric braiding with non-trivial twist; v ⊗ w ↦ (-1)^p(v,w) w ⊗ v
    anyonic = 20,  // non-symmetric braiding
    no_braiding = 30,  // braiding is not defined
};


// representation of a Sector by a single 64-bit int
// Product symmetries limit the size of each of the Sectors to a fixed bit-length such that total bit-length <= 64.
typedef int64_t Sector;


Sector compress_Sector_dynamic(std::vector<Sector> decompressed, std::vector<int> const & bit_lengths);

template<int N>
Sector compress_Sector(std::array<Sector, N> decompressed, std::array<int, N> const & bit_lengths);

std::vector<Sector> decompress_Sector_dynamic(Sector compressed, std::vector<int> const & bit_lengths);

template<int N>
std::array<Sector, N> decompress_Sector(Sector compressed, std::array<int, N> const & bit_lengths);


typedef std::vector<Sector> SectorArray;

class Symmetry {
    public:
        Symmetry(FusionStyle fusion, BraidingStyle braiding, bool can_be_dropped);
        const FusionStyle fusion_style;
        const BraidingStyle braiding_style;
        const bool can_be_dropped;
        const Sector trivial_sector;       
};



class U1Symmetry: public Symmetry {
    
};

// class ProductSymmetry : public Symmetry {
//     const std::vector<Symmetry const *> factors;
// }


template<int N>
Sector compress_Sector(std::array<Sector, N> decompressed, std::array<int, N> const & bit_lengths)
{
    Sector compressed = 0;
    int shift = 0;
    const Sector sign_bit = Sector(1) << 63;
    for (int i = 0; i < N ; ++i) 
    {
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

template<int N>
std::array<Sector, N> decompress_Sector(Sector compressed, std::array<int, N> const & bit_lengths)
{
    std::array<Sector, N> decompressed;
    int shift = 0;
    const Sector sign_bit = Sector(1) << 63;
    for (size_t i = 0; i < N ; ++i) 
    {
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

}