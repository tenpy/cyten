
#include "cyten/symmetries.h"

#include <cassert>
#include <stdexcept>

namespace cyten {
    
Sector compress_sector(std::vector<Sector> decompressed, std::vector<int> const &bit_lengths)
{
    const int N = bit_lengths.size();
    // rest is same code as in compress_sector_fixed<N>(std::array<Sector, N> const &, std::array<int, N> const &)
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

std::vector<Sector> decompress_sector(Sector compressed, std::vector<int> const &bit_lengths)
{
    const int N = bit_lengths.size();
    std::vector<Sector> decompressed(bit_lengths.size());
    // rest is same code as in decompress_Sector_fixed<N>(Sector, std::array<int, N> const &)
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

    
} // namespace cyten
