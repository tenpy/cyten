#include <cyten/symmetries/group.h>

#include <utility>
#include <vector>

namespace cyten {

Group::Group(FusionStyle fusion_style,
             Sector trivial_sector,
             std::string group_name,
             float64 num_sectors,
             bool has_complex_topological_data,
             std::optional<std::string> descriptive_name,
             bool trivial_shift)
  : SymmetryFactor(fusion_style,
                   BraidingStyle::bosonic,
                   trivial_sector,
                   std::move(group_name),
                   num_sectors,
                   has_complex_topological_data,
                   std::move(descriptive_name),
                   trivial_shift)
{
}

FusionSymbol
Group::swap_gate(Sector a, Sector b) const
{
    // [b, a, b*, a*] = eye(dim_a)[None, :, None, :] * eye(dim_b)[:, None, :, None]
    auto const da = static_cast<std::size_t>(sector_dim(a));
    auto const db = static_cast<std::size_t>(sector_dim(b));
    FusionSymbol out(4, FusionSymbol::Shape{ { db, da, db, da } }, Dtype::Float64);
    for (std::size_t ib = 0; ib < db; ++ib) {
        for (std::size_t ia = 0; ia < da; ++ia) {
            out.set(ib, ia, ib, ia, complex128{ 1.0, 0.0 });
        }
    }
    return out;
}

float64
Group::qdim(Sector a) const
{
    return static_cast<float64>(sector_dim(a));
}

std::vector<float64>
Group::batch_qdim(SectorArray const& a) const
{
    auto dims = batch_sector_dim(a);
    return std::vector<float64>(dims.begin(), dims.end());
}

complex128
Group::topological_twist(Sector /*a*/) const
{
    return complex128{ 1.0, 0.0 };
}

} // namespace cyten
