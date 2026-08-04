#pragma once

#include "symmetry_factor.h"

#include <string>
#include <vector>

namespace cyten {

/// Category describing Fibonacci anyons.
///
/// Allowed sectors are 1D arrays with a single entry of either ``0`` (vacuum) or ``1`` (tau
/// anyon).
class FibonacciAnyonCategory : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<FibonacciAnyonCategory>;
    using CPtr = std::shared_ptr<const FibonacciAnyonCategory>;

    static Sector const vacuum;
    static Sector const tau;

    std::string handedness;
    float64 _phi;
    py::array _f;
    py::array _r;
    std::vector<py::object> _c;

    explicit FibonacciAnyonCategory(std::string handedness = "left");
    ~FibonacciAnyonCategory() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    std::string sector_str(Sector a) const override;
    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    Sector dual_sector(Sector a) const override;
    SectorArray dual_sectors(SectorArray const& sectors) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;
    py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override;
    int64 frobenius_schur(Sector a) const override;
    float64 qdim(Sector a) const override;
    py::array batch_qdim(SectorArray const& a) const override;
    py::array _r_symbol(Sector a, Sector b, Sector c) const override;
    py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override;
    SectorArray all_sectors() const override;
};

} // namespace cyten
