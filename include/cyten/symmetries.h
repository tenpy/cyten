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

}

#include "./internal/symmetries.hpp"
