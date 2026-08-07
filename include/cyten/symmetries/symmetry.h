#pragma once

#include "../block_backend/dtypes.h"
#include "symmetry_factor.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Product symmetry: composition of one or more :class:`SymmetryFactor`s.
///
/// Always use this product structure, even for zero or one factors.
class Symmetry : public BaseSymmetry
{
  public:
    using Ptr = std::shared_ptr<Symmetry>;
    using CPtr = std::shared_ptr<const Symmetry>;

    /// Individual factors (no nesting: never contains another :class:`Symmetry`).
    std::vector<SymmetryFactor::Ptr> factors;
    /// Cumulative factor offsets into a product sector; length ``num_factors + 1``.
    /// Slice ``sector_slices[i]:sector_slices[i+1]`` is the component for ``factors[i]``.
    std::vector<std::uint8_t> sector_slices;
    /// Dtype of fusion tensors, or nullopt if any factor lacks fusion tensors.
    std::optional<Dtype> fusion_tensor_dtype;

    /// Build from factors; nested :class:`Symmetry` instances are flattened.
    explicit Symmetry(std::vector<SymmetryFactor::Ptr> factors);

    ~Symmetry() override = default;

    std::size_t num_factors() const { return factors.size(); }

    /// Index of the first factor with ``descriptive_name == name``; throws if missing.
    std::size_t factor_where(std::string const& descriptive_name) const;

    /// Whether ``other`` is among the factors (instance or type check via Python binding).
    bool has_factor(SymmetryFactor const& other) const;

    bool is_equivalent_to(Symmetry const& other, bool strict_ordering = false) const;

    py::object as_Symmetry() override;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;

    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    SectorArray all_sectors() const override;

    int64 sector_dim(Sector a) const override;
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
    std::vector<float64> batch_qdim(SectorArray const& a) const override;
    float64 qdim(Sector a) const override;
    std::string sector_str(Sector a) const override;

    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    FusionSymbol swap_gate(Sector a, Sector b) const override;
    FusionSymbol Z_iso(Sector a) const override;

    std::string repr() const;
    std::string str() const;
    bool equals(Symmetry const& other) const;
    /// Product with another factor or product → new :class:`Symmetry`.
    Ptr mul(SymmetryFactor::Ptr other) const;
    Ptr mul(Symmetry const& other) const;

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);

    /// Slice of product sector belonging to factor ``i``.
    Sector factor_sector(Sector const& a, std::size_t i) const;
    SectorArray factor_sectors(SectorArray const& a, std::size_t i) const;

  private:
    static FusionStyle max_fusion_style(std::vector<SymmetryFactor::Ptr> const& factors);
    static BraidingStyle max_braiding_style(std::vector<SymmetryFactor::Ptr> const& factors);
    static Sector concat_trivial_sectors(std::vector<SymmetryFactor::Ptr> const& factors);
    static float64 prod_num_sectors(std::vector<SymmetryFactor::Ptr> const& factors);
};

} // namespace cyten
