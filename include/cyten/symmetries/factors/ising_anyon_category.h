#pragma once

#include "../symmetry_factor.h"

#include <array>
#include <string>
#include <vector>

namespace cyten {

/// Category describing Ising anyons.
///
/// Allowed sectors are 1D arrays with a single entry of either ``0`` (vacuum), ``1`` (sigma), or
/// ``2`` (psi).
class IsingAnyonCategory : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<IsingAnyonCategory>;
    using CPtr = std::shared_ptr<const IsingAnyonCategory>;

    static Sector const vacuum;
    static Sector const sigma;
    static Sector const psi;

    int nu;
    std::array<int64, 3> frobenius;
    FusionSymbol _f;
    FusionSymbol _r;
    std::vector<FusionSymbol> _c;

    explicit IsingAnyonCategory(int nu = 1);
    ~IsingAnyonCategory() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    std::string sector_str(Sector a) const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    std::vector<float64> batch_qdim(SectorArray const& a) const override;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    SectorArray all_sectors() const override;

    void save_hdf5(py::object hdf5_saver,
                   py::object h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
