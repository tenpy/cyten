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

class ProductSymmetry;  // forward declaration

class Symmetry {
    public:
        const FusionStyle fusion_style;
        const BraidingStyle braiding_style;
        std::string descriptive_name;
        size_t _num_sectors;
        const static size_t INFINITE_NUM_SECTORS = -1;
        // subclasses might have a bitlength array/vector.
        size_t _sector_ind_len = 1;   // how many ints does the decompressed sector split into?
        size_t _max_sector_bits = -1;  // how many bits are needed to represent biggest sector in this int?
                                 // i.e. how many bits the FactorSymmetry needs to reserve for this.
    protected:
        py::module_ np;
    // constructor / attribute accesss
    public:
        Symmetry(FusionStyle fusion, BraidingStyle braiding);
        virtual bool can_be_dropped() const;
        bool is_abelian() const;
        size_t num_sectors() const;
        size_t sector_ind_len() const;
        bool has_symmetric_braid() const;

    // ABSTRACT METHODS (virtual methods need to be overwritten)
    public:
        virtual std::string group_name() const = 0;
        virtual Sector trivial_sector() const = 0;

        virtual bool is_valid_sector(Sector sector) const = 0;
        virtual SectorArray fusion_outcomes(Sector a, Sector b) const = 0;

        // Convention: valid syntax for the constructor, i.e. "ClassName(..., name='...')"
        virtual std::string __repr__() const = 0;

        /// Whether self and other describe the same mathematical structure.
        virtual bool is_same_symmetry(Symmetry const & other) const = 0;

        /// The sector dual to a, such that N^{a,dual(a)}_u = 1.
        virtual Sector dual_sector(Sector a) const = 0;
       // element-wise dual_sector(a)
        SectorArray dual_sectors(SectorArray const& sectors) const;

        cyten_int n_symbol(Sector a, Sector b, Sector c) const;
        /// @brief Optimized version of n_symbol() that assumes that c is a valid fusion outcome.
        /// If it is not, the results may be nonsensical. We do this for optimization purposes
        virtual cyten_int _n_symbol(Sector a, Sector b, Sector c) const = 0;
        py::array f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
        /// @brief Internal implementation of :meth:`f_symbol`. Can assume that inputs are valid.
        virtual py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const = 0;
        py::array r_symbol(Sector a, Sector b, Sector c) const;
        /// @brief Internal implementation of :meth:`r_symbol`. Can assume that inputs are valid.
        virtual py::array _r_symbol(Sector a, Sector b, Sector c) const = 0;
        py::array fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const;
        /// @brief Internal implementation of :meth:`fusion_tensor`. Can assume that inputs are valid.
        virtual py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const;
        virtual py::array Z_iso(Sector a) = 0;
        virtual SectorArray all_sectors() const ;

    // FALLBACK IMPLEMENTATIONS (might want to override virtual methods)
    public:
        // default implementation for _sector_ind_len = 1
        virtual Sector compress_sector(std::vector<Sector> const & decompressed) const;
        // default implementation for _sector_ind_len = 1
        virtual std::vector<Sector> decompress_sector(Sector compressed) const;
        SectorArray compress_sectorarray(py::array_t<Sector> sectors) const;
        py::array_t<Sector> decompress_sectorarray(SectorArray sectors) const;

        virtual bool are_valid_sectors(SectorArray const & sectors) const;
        virtual SectorArray fusion_outcomes_broadcast(SectorArray const & a, SectorArray const & b) const;

        Sector multiple_fusion(std::vector<Sector> const & sectors) const;
        SectorArray multiple_fusion_broadcast(std::vector<SectorArray> const & sectors) const;
        virtual SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const & sectors) const;
    public:

        virtual bool can_fuse_to(Sector a, Sector b, Sector c) const;

        /// @brief The dimension of a sector, as an unstructured space (i.e. if we drop the symmetry).
        /// For bosonic braiding style, e.g. for group symmetries, this coincides with the quantum dimension computed by :meth:`qdim`.
        /// For other braiding styles,
        virtual cyten_int sector_dim(Sector a) const;
        // batch_sector_dim in python
        virtual std::vector<cyten_int> sector_dim(SectorArray const& a) const;

        /// The quantum dimension ``Tr(id_a)`` of a sector
        virtual cyten_float qdim(Sector a) const;
        virtual std::vector<cyten_float> qdim(SectorArray const& a) const;
        cyten_float sqrt_qdim(Sector a) const;
        cyten_float inv_sqrt_qdim(Sector a) const;
        // Total quantum dimension, :math:`D = \sqrt{\sum_a d_a^2}`.
        cyten_float total_qdim() const;

        /// Short and readable string for the sector. Is used in __str__ of symmetry-related objects.
        virtual std::string sector_str(Sector a) const;


        /// The Frobenius Schur indicator of a sector.
        virtual cyten_int frobenius_schur(Sector a) const;

        py::array b_symbol(Sector a, Sector b, Sector c) const;
        py::array c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
        virtual py::array _b_symbol(Sector a, Sector b, Sector c) const;
        virtual py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
    public:
        cyten_complex topological_twist(Sector a) const;
        cyten_complex s_matrix_element(Sector a, Sector b) const;
        py::array s_matrix(Sector a, Sector b) const;


    // COMMON IMPLEMENTATIONS
    public:

        std::string __str__() const;

        // TODO __mul__ equivelent operator*
        // ProductSymmetry operator* (Symmetry & rhs);
        bool operator== (Symmetry const & rhs) const;

};


class ProductSymmetry : public Symmetry {
    public:
        const std::vector<std::shared_ptr<Symmetry *>> factors;
    public:
        ProductSymmetry(std::vector<std::shared_ptr<Symmetry *>> const & factors);
};



class GroupSymmetry : public Symmetry {
    public:
        GroupSymmetry(FusionStyle fusion);
        virtual bool can_be_dropped() const override;
        virtual py::array _fusion_tensor(Sector a, Sector b, Sector c) = 0;
        virtual py::array _Z_iso(Sector a) = 0;



};

class AbelianSymmetry: public GroupSymmetry {

};




} // namespace cyten

#include "./internal/symmetries.hpp"
