#include <cyten/symmetries/factors/no_symmetry.h>

namespace cyten {

NoSymmetry::NoSymmetry()
  : AbelianGroup(Sector{ 0 },
                 "no_symmetry",
                 1.0,
                 /*descriptive_name=*/std::nullopt,
                 /*trivial_shift=*/true)
{
}

bool
NoSymmetry::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] == 0;
}

bool
NoSymmetry::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        if (sectors[i][0] != 0) {
            return false;
        }
    }
    return true;
}

SectorArray
NoSymmetry::fusion_outcomes(Sector a, Sector /*b*/) const
{
    SectorArray out(1, 1);
    out[0] = a;
    return out;
}

SectorArray
NoSymmetry::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& /*b*/) const
{
    return a;
}

SectorArray
NoSymmetry::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    return sectors[0];
}

Sector
NoSymmetry::dual_sector(Sector a) const
{
    return a;
}

SectorArray
NoSymmetry::dual_sectors(SectorArray const& sectors) const
{
    return sectors;
}

std::string
NoSymmetry::sector_str(Sector /*a*/) const
{
    return "0";
}

std::string
NoSymmetry::repr() const
{
    return "NoSymmetry()";
}

bool
NoSymmetry::_is_equivalent_factor(SymmetryFactor const& other) const
{
    return dynamic_cast<NoSymmetry const*>(&other) != nullptr;
}

SectorArray
NoSymmetry::all_sectors() const
{
    SectorArray out(1, 1);
    out[0] = trivial_sector;
    return out;
}

NoSymmetry::Ptr
NoSymmetry::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& /*subpath*/)
{
    auto obj = std::make_shared<NoSymmetry>();
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
