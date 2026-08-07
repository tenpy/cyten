#pragma once

#include "../symmetry_factor.h"

#include <complex>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Doubled abelian anyon category :math:`D(Z_N)` (fusion rules of ``Z_N × Z_N``).
///
/// Allowed sectors are 1D arrays with two integers in ``[0, N-1]``.
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
    py::array batch_sector_dim(SectorArray const& a) const override;
    py::array batch_qdim(SectorArray const& a) const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    py::array _r_symbol(Sector a, Sector b, Sector c) const override;
    py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override;
    SectorArray all_sectors() const override;

    void save_hdf5(py::object hdf5_saver,
                   py::object h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
