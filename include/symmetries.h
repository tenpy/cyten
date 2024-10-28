#pragma once

#include <vector>

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

typedef signed int charge;

typedef std::vector<charge> Sector;


/// @brief vector of fixed-length Sectors given during intialization
class SectorArray {
    private:
        std::vector<charge> data; 
    public:
        const size_t sector_len;
        SectorArray(size_t sector_ind_len);
        size_t size() const;
        Sector operator[](size_t i);
        const Sector & operator[](size_t i) const; 
        void push_back(Sector const & s);
        charge * raw_pointer();
};


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