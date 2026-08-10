#include <cyten/symmetries/abelian_group.h>

#include <cyten/symmetries/topo_ones.h>

#include <utility>
#include <vector>

namespace cyten {

AbelianGroup::AbelianGroup(Sector trivial_sector,
                           std::string group_name,
                           float64 num_sectors,
                           std::optional<std::string> descriptive_name,
                           bool trivial_shift)
  : Group(FusionStyle::single,
          trivial_sector,
          std::move(group_name),
          num_sectors,
          /*has_complex_topological_data=*/false,
          std::move(descriptive_name),
          trivial_shift)
{
    fusion_tensor_dtype = Dtype::Float64;
}

std::string
AbelianGroup::sector_str(Sector a) const
{
    // --- hints from Python AbelianGroup.sector_str ---
    // we know sectors are labelled by a single number
    // ---
    // Sectors labelled by a single number.
    if (a.len() == 0) {
        return "";
    }
    return std::to_string(a.q[0]);
}

int64
AbelianGroup::sector_dim(Sector /*a*/) const
{
    return 1;
}

std::vector<int64>
AbelianGroup::batch_sector_dim(SectorArray const& a) const
{
    return std::vector<int64>(a.size(), 1);
}

int64
AbelianGroup::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

FusionSymbol
AbelianGroup::_f_symbol(Sector /*a*/,
                        Sector /*b*/,
                        Sector /*c*/,
                        Sector /*d*/,
                        Sector /*e*/,
                        Sector /*f*/) const
{
    return topo_ones::one_4D();
}

int64
AbelianGroup::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
AbelianGroup::qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
AbelianGroup::sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
AbelianGroup::inv_sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

FusionSymbol
AbelianGroup::_b_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return topo_ones::one_2D();
}

FusionSymbol
AbelianGroup::_r_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    // --- hints from Python AbelianGroup._r_symbol ---
    // For abelian groups, the R symbol is always 1.
    // ---
    // For abelian groups, the R symbol is always 1.
    return topo_ones::one_1D();
}

FusionSymbol
AbelianGroup::_c_symbol(Sector /*a*/,
                        Sector /*b*/,
                        Sector /*c*/,
                        Sector /*d*/,
                        Sector /*e*/,
                        Sector /*f*/) const
{
    return topo_ones::one_4D();
}

FusionSymbol
AbelianGroup::_fusion_tensor(Sector /*a*/, Sector /*b*/, Sector /*c*/, bool /*Z_a*/, bool /*Z_b*/)
  const
{
    return topo_ones::one_4D_float();
}

FusionSymbol
AbelianGroup::Z_iso(Sector /*a*/) const
{
    return topo_ones::one_2D_float();
}

} // namespace cyten
