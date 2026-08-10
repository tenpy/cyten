#pragma once

#include "../symmetry_factor.h"

#include <complex>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cyten {

/// :math:`SU(2)_k` anyon category.
///
/// Allowed sectors are 1D arrays ``[jj]`` with ``jj = 0, 1, …, k`` (spin ``jj/2``, cutoff at
/// ``k/2``).
class SU2_kAnyonCategory : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<SU2_kAnyonCategory>;
    using CPtr = std::shared_ptr<const SU2_kAnyonCategory>;

    static Sector const spin_zero;
    static Sector const spin_half;

    int k;
    std::string handedness;
    std::optional<Sector> spin_one;
    complex128 _q;

    SU2_kAnyonCategory(int k, std::string handedness = "left");
    ~SU2_kAnyonCategory() override = default;

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
    SectorArray all_sectors() const override;

    void save_hdf5(py::object hdf5_saver,
                   py::object h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);

  private:
    using RKey = std::tuple<int, int, int>;
    using SixJKey = std::tuple<int, int, int, int, int, int>;

    std::map<RKey, FusionSymbol> _r;
    std::map<SixJKey, float64> _6j;

    float64 _n_q(int n) const;
    float64 _n_q_fac(int n) const;
    float64 _delta(int jj1, int jj2, int jj3) const;
    float64 _j_symbol(int jj1, int jj2, int jj12, int jj3, int jj, int jj23) const;
};

} // namespace cyten
