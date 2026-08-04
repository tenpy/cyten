#include <cyten/symmetries/u1.h>

#include <cmath>
#include <limits>
#include <utility>

namespace cyten {

U1::U1(std::optional<std::string> descriptive_name, bool trivial_shift)
  : AbelianGroup(Sector{ 0 },
                 "U(1)",
                 std::numeric_limits<float64>::infinity(),
                 std::move(descriptive_name),
                 trivial_shift)
{
}

bool
U1::is_valid_sector(Sector a) const
{
    return a.len() == 1;
}

bool
U1::are_valid_sectors(SectorArray const& sectors) const
{
    return sectors.sector_ind_len == 1;
}

SectorArray
U1::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 1);
    SectorArray bb(1, 1);
    aa.set(0, a);
    bb.set(0, b);
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
U1::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    SectorArray out(a.num_sectors, 1);
    for (std::size_t i = 0; i < a.num_sectors; ++i) {
        out.row(i)[0] = static_cast<int16_t>(a.row(i)[0] + b.row(i)[0]);
    }
    return out;
}

SectorArray
U1::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.num_sectors; ++i) {
            out.row(i)[0] = static_cast<int16_t>(out.row(i)[0] + sectors[s].row(i)[0]);
        }
    }
    return out;
}

Sector
U1::dual_sector(Sector a) const
{
    return Sector{ static_cast<int16_t>(-a.q[0]) };
}

SectorArray
U1::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.num_sectors, 1);
    for (std::size_t i = 0; i < sectors.num_sectors; ++i) {
        out.row(i)[0] = static_cast<int16_t>(-sectors.row(i)[0]);
    }
    return out;
}

std::string
U1::repr() const
{
    if (!descriptive_name.has_value()) {
        return "U1Symmetry()";
    }
    return std::string("U1Symmetry(\"") + *descriptive_name + "\")";
}

bool
U1::_is_equivalent_factor(SymmetryFactor const& other) const
{
    return dynamic_cast<U1 const*>(&other) != nullptr;
}

U1::Ptr
U1::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    bool trivial_shift = trivial_shift_from_hdf5(hdf5_loader, subpath);
    auto obj = std::make_shared<U1>(name, trivial_shift);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
