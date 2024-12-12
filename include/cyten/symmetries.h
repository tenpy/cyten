#pragma once

#include "./cyten.h"




namespace cyten {

class SymmetryError : public std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};
    
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

Sector _compress_sector(std::vector<Sector> decompressed, std::vector<int> const & bit_lengths);

template<int N>
Sector _compress_sector_fixed(std::array<Sector, N> decompressed, std::array<int, N> const & bit_lengths);

std::vector<Sector> _decompress_sector(Sector compressed, std::vector<int> const & bit_lengths);

template<int N>
std::array<Sector, N> _decompress_sector_fixed(Sector compressed, std::array<int, N> const & bit_lengths);


typedef std::vector<Sector> SectorArray;

class Symmetry {
    public:
        const FusionStyle fusion_style;
        const BraidingStyle braiding_style;
        std::string descriptive_name;
    protected:
        size_t _num_sectors;
        const static size_t INFINITE_NUM_SECTORS = -1;
        Sector _trivial_sector;       
        // subclasses might have a bitlength array/vector.
        size_t _sector_ind_len;   // how many ints does the decompressed sector split into?
        size_t _max_sector_bits;  // how many bits are needed to represent biggest sector in this int?
                                 // i.e. how many bits the FactorSymmetry needs to reserve for this.
    public:
        Symmetry(FusionStyle fusion, BraidingStyle braiding);
        virtual bool can_be_dropped() const;
        virtual std::string group_name() const = 0;
        bool is_abelian() const;
        size_t num_sectors() const;
        virtual Sector trivial_sector() const = 0;
        size_t sector_ind_len() const;
        virtual Sector compress_sector(std::vector<Sector> const & decompressed) = 0;
        virtual std::vector<Sector> decompress_sector(Sector compressed) = 0;
        SectorArray compress_sectorarray(py::array_t<Sector> sectors);
        py::array_t<Sector> decompress_sectorarray(SectorArray sectors);
        
        virtual bool is_valid_sector(Sector sector) const = 0;
        virtual SectorArray fusion_outcomes(Sector a, Sector b) = 0;

        // Convention: valid syntax for the constructor, i.e. "ClassName(..., name='...')"
        virtual std::string __repr__() const = 0; // TODO maybe easier in python?

        /// Whether self and other describe the same mathematical structure.
        virtual bool is_same_symmetry(Symmetry * other) const = 0;        
        
        /// The sector dual to a, such that N^{a,dual(a)}_u = 1.
        virtual Sector dual_sector(Sector a) const = 0;

        /// Optimized version of n_symbol() that assumes that c is a valid fusion outcome.
        /// If it is not, the results may be nonsensical. We do this for optimization purposes
        virtual cyten_int _n_symbol(Sector a, Sector b, Sector c) = 0;
        /// @brief Internal implementation of :meth:`f_symbol`. Can assume that inputs are valid."""
        virtual py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) = 0;
        /// @brief Internal implementation of :meth:`r_symbol`. Can assume that inputs are valid.
        virtual py::array _r_symbol(Sector a, Sector b, Sector c) = 0;
        
        bool has_symmetric_braid();
        
        virtual py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b);
        virtual py::array Z_iso(Sector a) = 0;
        virtual SectorArray all_sectors() ;
        
        // TODO: continue with FALLBACK IMPLEMENTATIONS in py
};


// class ProductSymmetry : public Symmetry {
//     public: 
//         const std::vector<std::shared_ptr<Symmetry *>> factors; 
//     public:
//         ProductSymmetry(std::vector<std::shared_ptr<Symmetry *>> const & factors);
// };


class GroupSymmetry : public Symmetry {
    public:
        GroupSymmetry(FusionStyle fusion);
        virtual bool can_be_dropped() const override;
    protected:
        virtual py::array _fusion_tensor(Sector a, Sector b, Sector c) = 0;
        virtual py::array _Z_iso(Sector a) = 0;
              
        

};

class AbelianSymmetry: public GroupSymmetry {
    
};




} // namespace cyten

#include "./internal/symmetries.hpp"
