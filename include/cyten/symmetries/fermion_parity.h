#pragma once

#include "symmetry_factor.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Fermionic parity (Z_2 fusion, fermionic braid).
class FermionParity : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<FermionParity>;
    using CPtr = std::shared_ptr<const FermionParity>;

    static Sector const even;
    static Sector const odd;

    explicit FermionParity(std::optional<std::string> descriptive_name = std::nullopt,
                           bool trivial_shift = true);
    ~FermionParity() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override;
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override;
    int64 sector_dim(Sector a) const override;
    py::array batch_sector_dim(SectorArray const& a) const override;
    py::array batch_qdim(SectorArray const& a) const override;
    std::string sector_str(Sector a) const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    float64 sqrt_qdim(Sector a) const override;
    float64 inv_sqrt_qdim(Sector a) const override;
    py::array _b_symbol(Sector a, Sector b, Sector c) const override;
    py::array _r_symbol(Sector a, Sector b, Sector c) const override;
    py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override;
    py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    py::array swap_gate(Sector a, Sector b) const override;
    complex128 topological_twist(Sector a) const override;
    SectorArray all_sectors() const override;
    py::array Z_iso(Sector a) const override;
    std::string repr() const override;
};

} // namespace cyten
