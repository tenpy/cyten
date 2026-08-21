#pragma once

#include "../group.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// SU(N) group symmetry.
///
/// Sectors are arrays of length `N` which correspond to first rows of normalized
/// Gelfand–Tsetlin patterns (see https://arxiv.org/pdf/1009.0437).
/// E.g. for SU(3) the 8-dimensional irrep is ``[2, 1, 0]``.
/// Clebsch–Gordan coefficients and F/R symbols need to be calculated with the
/// ``clebsch_gordan_coefficients`` package and exported as HDF5 files.
///
/// @param CGfile HDF5 file containing the Clebsch–Gordan coefficients.
/// @param Ffile HDF5 file containing the F symbols.
/// @param Rfile HDF5 file containing the R symbols.
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

/// Returns a dictionary with the outer multiplicities for the irreps in the decomposition of a x b.
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

/// Evaluate a single Clebsch-Gordan coefficient.
///
/// @param a, b, c Sector for the fusion @f$ a \otimes b \mapsto c @f$.
/// @param q_a, q_b, q_c Indices of the Gelfand Tsetlin pattern
/// @param mu multiplicity index 1 <= mu
/// @returns The CG coefficient for the given input
    float64 clebschgordan(Sector a, int64 q_a, Sector b, int64 q_b, Sector c, int64 q_c, int64 mu)
      const;

    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override;
/// Returns the F symbol for the specified input irreps calculated from CG coefficients.
///
/// a,b,c,d,e,f are irrep labels, i.e. first rows of GT patterns
/// output is the conjugated F symbol [F^{abc}_{def}]^*_{mu,nu,kappa, lambda}
/// where a x b = mu c, c x d =nu e, b x d= kappa f and a x f =lambda e
///
/// @param a, b, c, d, e, f Irreps specifying the CG coefficient.
    FusionSymbol _f_symbol_from_CG(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const;
    FusionSymbol _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f)
      const override;
/// Returns the R symbol for the specified input irreps calculated from CG coefficients.
///
/// @param a, b, c Irreps specifying the R symbol.
    FusionSymbol _r_symbol_from_CG(Sector a, Sector b, Sector c) const;
    FusionSymbol _r_symbol(Sector a, Sector b, Sector c) const override;
    int64 frobenius_schur(Sector a) const override;

    bool has_data_in_group(py::object group) const;
/// Sanity check for Hdf5 files containing CG-coefficients, F-symbols or R-symbols.
///
/// This method takes a Hdf5 file and checks if it has the required structure and if
/// the necessary data has been saved to it. This excludes the possibility of using incompletely generated files,
/// but cannot guarantee completeness of the file and correctness of the data in the file.
/// In particular, consistency of the data in the file should be checked by the cyten tests for SU(N) symmetry.
    void sanity_check_hdf5(py::object file) const;

    void save_hdf5(py::object hdf5_saver,
                   py::object h5gr,
                   std::string const& subpath) const override;
    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
