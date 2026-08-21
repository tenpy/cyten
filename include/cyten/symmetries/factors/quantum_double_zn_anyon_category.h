#pragma once

#include "../symmetry_factor.h"

#include <complex>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Doubled abelian anyon category @f$ D(Z_N) @f$ (fusion rules of ``Z_N × Z_N``).
///
/// Allowed sectors are 1D arrays with two integers in ``[0, N-1]``:
/// ``[0, 0]``, ``[0, 1]``, ..., ``[N-1, N-1]``.
/// This is not a simple product of two `ZNAnyonCategory` s; there are nontrivial R-symbols.
class QuantumDoubleZNAnyonCategory : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<QuantumDoubleZNAnyonCategory>;
    using CPtr = std::shared_ptr<const QuantumDoubleZNAnyonCategory>;

    int N;
    complex128 _phase;

    explicit QuantumDoubleZNAnyonCategory(
      int N,
      std::optional<std::string> descriptive_name = std::nullopt);
    ~QuantumDoubleZNAnyonCategory() override = default;

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

} // namespace cyten
