#pragma once

#include "../abelian_group.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Z_N symmetry.
///
/// Allowed sectors are 1D arrays with a single integer entry between ``0`` and ``N-1``.
class ZN : public AbelianGroup
{
  public:
    using Ptr = std::shared_ptr<ZN>;
    using CPtr = std::shared_ptr<const ZN>;

    int N;

    explicit ZN(int N,
                std::optional<std::string> descriptive_name = std::nullopt,
                bool trivial_shift = true);
    ~ZN() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    SectorArray all_sectors() const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;

    void save_hdf5(py::object hdf5_saver,
                   py::object h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
