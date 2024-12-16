#pragma once

#include "./cyten.h"


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


/// @brief representation of a Sector by a single 64-bit int
/// 
/// Product symmetries limit the size of each of the Sectors to a fixed bit-length such that sum of bit_lengths <= 64.
typedef int64_t Sector;

Sector compress_sector(std::vector<Sector> decompressed, std::vector<int> const & bit_lengths);

template<int N>
Sector compress_sector_fixed(std::array<Sector, N> decompressed, std::array<int, N> const & bit_lengths);

std::vector<Sector> decompress_sector(Sector compressed, std::vector<int> const & bit_lengths);

template<int N>
std::array<Sector, N> decompress_sector_fixed(Sector compressed, std::array<int, N> const & bit_lengths);


typedef std::vector<Sector> SectorArray;

class Symmetry {
    public:
        Symmetry(FusionStyle fusion, BraidingStyle braiding, bool can_be_dropped);
        const FusionStyle fusion_style;
        const BraidingStyle braiding_style;
        std::string descriptive_name;
    protected:
        size_t _num_sectors; // -1 for infinite
        Sector _trivial_sector;       
        // subclasses might have a bitlength array/vector.
        size_t sector_ind_len;   // how many ints does the decompressed sector split into?
        size_t max_sector_bits;  // how many bits are needed to represent biggest sector in this int?
                                 // i.e. how many bits the FactorSymmetry needs to reserve for this.
    public:
        Symmetry(FusionStyle fusion, BraidingStyle braiding);
        virtual bool can_be_dropped() const;
        virtual std::string group_name() const = 0;
        bool is_abelian() const;
        Sector trivial_sector() const;
        virtual Sector compress_sector(std::vector<Sector> decompressed) = 0;
        virtual Sector compress_sector_recursive(std::vector<Sector> decompressed);
        virtual std::vector<Sector> decompress_sector(Sector compressed) = 0;
        virtual std::vector<Sector> decompress_sector_recursive(Sector compressed);
        py::array_t<Sector> as_2D_SectorArray(SectorArray sectors);
    public:
        std::vector<Sector> decompress_Sector(Sector compressed);
};

class ProductSymmetry : public Symmetry {
    public: 
        const std::vector<std::shared_ptr<Symmetry *>> factors; 
    public:
        ProductSymmetry(std::vector<std::shared_ptr<Symmetry *>> const & factors);
};


// class ProductSymmetry : public Symmetry {
//     const std::vector<Symmetry const *> factors;
// }

}

#include "./internal/symmetries.hpp"
