#pragma once

#include "../cyten.h"
#include "exceptions.h"
#include "sector.h"
#include "styles.h"

#include <memory>
#include <string>
#include <vector>

namespace cyten {

class Symmetry; // product symmetry; defined later

/// Common method implementations for both :class:`SymmetryFactor` and :class:`Symmetry`.
///
/// This contains the fallback implementations of e.g. :meth:`qdim` in terms of F symbols.
class BaseSymmetry : public std::enable_shared_from_this<BaseSymmetry>
{
  public:
    using Ptr = std::shared_ptr<BaseSymmetry>;
    using CPtr = std::shared_ptr<const BaseSymmetry>;

    FusionStyle fusion_style;
    BraidingStyle braiding_style;
    Sector trivial_sector;
    /// Number of sectors; ``+inf`` if infinite (matches Python ``int | float``).
    float64 num_sectors;
    std::uint8_t sector_ind_len = 0;
    SectorArray empty_sector_array;
    bool has_complex_topological_data = false;
    bool trivial_shift = true;

    BaseSymmetry(FusionStyle fusion_style,
                 BraidingStyle braiding_style,
                 Sector trivial_sector,
                 float64 num_sectors,
                 bool has_complex_topological_data,
                 bool trivial_shift);
    virtual ~BaseSymmetry() = default;

    // --- properties (Python @property) ---

    /// If the symmetry supports converting tensors to/from numpy.
    virtual bool can_be_dropped() const;
    virtual bool has_symmetric_braid() const;
    virtual bool has_trivial_braid() const;
    /// If FusionStyle.single (all sectors one-dimensional). Not necessarily a group.
    virtual bool is_abelian() const;
    /// If N symbols are only 0 or 1.
    virtual bool has_unique_fusion() const;

    // --- abstract / must override ---

    /// The sector dual to a, such that N^{a,dual(a)}_u = 1.
    virtual Sector dual_sector(Sector a) const = 0;
    /// Optimized n_symbol assuming c is a valid fusion outcome.
    virtual int64 _n_symbol(Sector a, Sector b, Sector c) const = 0;
    /// Internal F symbol; inputs assumed valid.
    virtual py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const = 0;
    /// Internal R symbol; inputs assumed valid.
    virtual py::array _r_symbol(Sector a, Sector b, Sector c) const = 0;
    /// Wrap as a product :class:`Symmetry` (identity if already a product).
    virtual std::shared_ptr<Symmetry> as_Symmetry() = 0;
    virtual bool is_valid_sector(Sector a) const = 0;
    virtual SectorArray fusion_outcomes(Sector a, Sector b) const = 0;

    // --- defaults (may override) ---

    /// Internal fusion tensor; inputs assumed valid.
    virtual py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const;
    /// Swap gate (numpy braid) of single sectors.
    virtual py::array swap_gate(Sector a, Sector b) const;
    /// Z isomorphism :math:`Z_{\bar{a}} : \bar{a}^* \to a`.
    virtual py::array Z_iso(Sector a) const;
    /// All sectors if finitely many.
    virtual SectorArray all_sectors() const;

    int64 n_symbol(Sector a, Sector b, Sector c) const;
    py::array f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
    py::array b_symbol(Sector a, Sector b, Sector c) const;
    py::array r_symbol(Sector a, Sector b, Sector c) const;
    py::array c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
    py::array fusion_tensor(Sector a,
                            Sector b,
                            Sector c,
                            bool Z_a = false,
                            bool Z_b = false) const;

    virtual bool are_valid_sectors(SectorArray const& sectors) const;
    /// Element-wise fusion for FusionStyle.single.
    virtual SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                                  SectorArray const& b) const;
    Sector multiple_fusion(std::vector<Sector> const& sectors) const;
    SectorArray multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const;
    /// Internal; may assume ``sectors.size() >= 2``.
    virtual SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const;
    virtual bool can_fuse_to(Sector a, Sector b, Sector c) const;
    /// Dimension as unstructured space (if symmetry can be dropped).
    virtual int64 sector_dim(Sector a) const;
    virtual py::array batch_sector_dim(SectorArray const& a) const;
    virtual py::array batch_qdim(SectorArray const& a) const;
    virtual std::string sector_str(Sector a) const;
    virtual SectorArray dual_sectors(SectorArray const& sectors) const;
    virtual int64 frobenius_schur(Sector a) const;
    virtual float64 qdim(Sector a) const;
    virtual float64 sqrt_qdim(Sector a) const;
    virtual float64 inv_sqrt_qdim(Sector a) const;
    float64 total_qdim() const;
    virtual py::array _b_symbol(Sector a, Sector b, Sector c) const;
    virtual py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const;
    virtual complex128 topological_twist(Sector a) const;
    complex128 s_matrix_element(Sector a, Sector b) const;
    py::array s_matrix() const;
};

} // namespace cyten
