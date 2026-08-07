#pragma once

#include "../symmetry_factor.h"

#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cyten {

/// :math:`SU(3)_3` anyon category.
///
/// Allowed sectors are 1D arrays ``[j]`` with ``j = 0, 1, 2, 3`` (``1``, ``8``, ``10``,
/// ``\bar{10}``).
class SU3_3AnyonCategory : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<SU3_3AnyonCategory>;
    using CPtr = std::shared_ptr<const SU3_3AnyonCategory>;

    static Sector const one_irrep;
    static Sector const eight_irrep;
    static Sector const ten_irrep;
    static Sector const ten_bar_irrep;

    SU3_3AnyonCategory();
    ~SU3_3AnyonCategory() override = default;

    bool is_valid_sector(Sector a) const override;
    bool are_valid_sectors(SectorArray const& sectors) const override;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;
    int64 sector_dim(Sector a) const override;
    std::vector<int64> batch_sector_dim(SectorArray const& a) const override;
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

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);

  private:
    using FSymKey = std::tuple<int, int, int, int, int, int>;

    std::map<FSymKey, FusionSymbol> _fsym_map;
    std::map<FSymKey, FusionSymbol> _c;

    FusionSymbol _compute_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const;

    static FusionSymbol _f1();
    static FusionSymbol _f2();
    static FusionSymbol _f3();
    static FusionSymbol _f4();
    static SectorArray fusion_map(int key);
    static Sector dual_map(int j);
};

} // namespace cyten
