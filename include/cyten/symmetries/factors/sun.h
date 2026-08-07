#pragma once

#include "../group.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// SU(N) group symmetry.
///
/// Sectors are length-``N`` arrays (first rows of normalized Gelfand–Tsetlin patterns).
/// Clebsch–Gordan / F / R data are loaded from HDF5 files (``h5py.File`` objects).
class SUN : public Group
{
  public:
    using Ptr = std::shared_ptr<SUN>;
    using CPtr = std::shared_ptr<const SUN>;

    int N;
    /// HDF5 handles (``h5py.File``).
    py::object CGfile;
    py::object Ffile;
    py::object Rfile;

    SUN(int N,
        py::object CGfile,
        py::object Ffile,
        py::object Rfile,
        std::optional<std::string> descriptive_name = std::nullopt);
    ~SUN() override = default;

    bool is_valid_sector(Sector a) const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;
    int64 sector_dim(Sector a) const override;
    std::string repr() const override;
    Sector dual_sector(Sector a) const override;

    int64 hweight_from_CG_hdf5() const;
    int64 hweight_from_F_hdf5() const;
    int64 hweight_from_R_hdf5() const;

    bool can_fuse_to(Sector a, Sector b, Sector c) const override;
    int64 _n_symbol(Sector a, Sector b, Sector c) const override;

    int64 S_index_irrep_weight(Sector a) const;
    Sector highest_irrep_in_decomp(Sector a, Sector b) const;
    SectorArray fusion_outcomes(Sector a, Sector b) const override;

    py::dict dims_of_irreps(Sector a, Sector b) const;
    py::dict outer_multiplicity_from_CG(Sector a, Sector b) const;

    float64 clebschgordan(Sector a, int64 q_a, Sector b, int64 q_b, Sector c, int64 q_c, int64 mu)
      const;

    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
    FusionSymbol _f_symbol_from_CG(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
    FusionSymbol _r_symbol_from_CG(Sector a, Sector b, Sector c) const;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    int64 frobenius_schur(Sector a) const override;

    bool has_data_in_group(py::object group) const;
    void sanity_check_hdf5(py::object file) const;

    void save_hdf5(py::object hdf5_saver,
                   py::object h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
