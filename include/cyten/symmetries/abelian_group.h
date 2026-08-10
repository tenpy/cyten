#pragma once

#include "../block_backend/dtypes.h"
#include "group.h"

#include <optional>
#include <string>

namespace cyten {

/// Base-class for abelian symmetry groups.
class AbelianGroup : public Group
{
  public:
    using Ptr = std::shared_ptr<AbelianGroup>;
    using CPtr = std::shared_ptr<const AbelianGroup>;

    AbelianGroup(Sector trivial_sector,
                 std::string group_name,
                 float64 num_sectors,
                 std::optional<std::string> descriptive_name = std::nullopt,
                 bool trivial_shift = true);
    ~AbelianGroup() override = default;

    std::string sector_str(Sector a) const override;
    int64 sector_dim(Sector a) const override;
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    float64 sqrt_qdim(Sector a) const override;
    float64 inv_sqrt_qdim(Sector a) const override;
    FusionSymbol _b_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    FusionSymbol Z_iso(Sector a) const override;
};

} // namespace cyten
