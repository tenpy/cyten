#pragma once

#include "abelian_group.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// U(1) symmetry.
///
/// Allowed sectors are 1D arrays with a single integer entry:
/// ``…, [-2], [-1], [0], [1], [2], …``.
class U1 : public AbelianGroup
{
  public:
    using Ptr = std::shared_ptr<U1>;
    using CPtr = std::shared_ptr<const U1>;

    explicit U1(std::optional<std::string> descriptive_name = std::nullopt,
                bool trivial_shift = true);
    ~U1() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
};

} // namespace cyten
