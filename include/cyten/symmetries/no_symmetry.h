#pragma once

#include "abelian_group.h"

#include <vector>

namespace cyten {

/// Trivial symmetry group that doesn't do anything.
///
/// The only allowed sector is ``[0]``.
class NoSymmetry : public AbelianGroup
{
  public:
    using Ptr = std::shared_ptr<NoSymmetry>;
    using CPtr = std::shared_ptr<const NoSymmetry>;

    NoSymmetry();
    ~NoSymmetry() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    std::string sector_str(Sector a) const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    SectorArray all_sectors() const override;
};

} // namespace cyten
