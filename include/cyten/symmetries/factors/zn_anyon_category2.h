#pragma once

#include "../symmetry.h"

#include <complex>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Abelian anyon category @f$ Z_N^{(n+1/2)} @f$ (``N`` must be even).
///
/// Allowed sectors are 1D arrays with a single integer in ``[0, N-1]``.
class ZNAnyonCategory2 : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<ZNAnyonCategory2>;
    using CPtr = std::shared_ptr<const ZNAnyonCategory2>;

    int N;
    int n;
    complex128 _phase;

    ZNAnyonCategory2(int N, int n, std::optional<std::string> descriptive_name = std::nullopt);
    ~ZNAnyonCategory2() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;
    int64 sector_dim(Sector a) const override;
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
    std::vector<float64> batch_qdim(SectorArray const& a) const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    FusionSymbol _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    SectorArray all_sectors() const override;

    void save_hdf5(py::object hdf5_saver,
                   py::object h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

/// Semion category @f$ Z_2^{(1/2)} @f$ as a product `Symmetry`.
[[nodiscard]] inline Symmetry::Ptr
semion_category()
{
    return std::make_shared<Symmetry>(
      std::vector<SymmetryFactor::Ptr>{ std::make_shared<ZNAnyonCategory2>(2, 0) });
}

/// Double-semion category @f$ Z_2^{(1/2)} \times Z_2^{(3/2)} @f$ as a product `Symmetry`.
[[nodiscard]] inline Symmetry::Ptr
double_semion_category()
{
    return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{
      std::make_shared<ZNAnyonCategory2>(2, 0),
      std::make_shared<ZNAnyonCategory2>(2, 1),
    });
}

} // namespace cyten
