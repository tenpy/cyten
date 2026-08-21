#pragma once

#include "../block_backend/dtypes.h"
#include "base_symmetry.h"

#include <optional>
#include <string>

namespace cyten {

/// Base class for symmetries that impose a block-structure on tensors (single factor).
///
/// Attributes:
///
/// can_be_dropped : bool
///     If the symmetry could be dropped to `NoSymmetry` while preserving the structure.
///     This is e.g. the case for group symmetries.
///     This means that there is a well-defined notion of a basis of graded vector spaces and of
///     dense array representations of symmetric Tensor. See notes below.
/// trivial_sector : Sector
///     The trivial sector of the symmetry.
///     For a group this is the "symmetric" sector, where the group acts trivially.
///     For a general category, this is the monoidal unit.
/// group_name : str
///     A readable name for the symmetry, purely as a mathematical structure, e.g. ``'U(1)'``.
/// descriptive_name : str or None
///     Optionally, an additional name for the group, indicating e.g. how it arises.
///     Could be e.g. ``'Sz'`` for the U(1) symmetry that conserves magnetization.
/// num_sectors : int or float
///     The number of sectors of the symmetry. An integer if finite, otherwise infinity.
/// sector_ind_len : int
///     Valid sectors are arrays with shape ``(sector_ind_len,)``.
/// empty_sector_array
///     A SectorArray with no sectors, shape ``(0, sector_ind_len)``.
/// has_complex_topological_data : bool
///     If any of the topological data (F, R, C, B symbols, twist) for any sectors is complex.
///     If so, tensors with that symmetry must have a complex dtype (except DiagonalTensor or Mask),
///     since real blocks become complex under leg manipulations.
///     Note: for a group (and for fermions), the topo data must be real if the fusion tensors
///     are real. This is because the associator, the braid, and the cup are all real for groups.
///
/// Notes:
///
/// Some symmetries can be dropped to `NoSymmetry`, see `can_be_dropped`.
/// It implies that all operations that may be carried out on symmetric objects have a corresponding
/// operation on a non-symmetric counterpart. For example, a symmetric space @f$ A @f$ has a
/// corresponding space @f$ \mathbb{C}^{n_A} @f$, without further structure.
/// It "corresponds" to @f$ A @f$ in the sense that it has the same properties, e.g. same dimension,
/// and that there are compatible operations (tensor product, direct sum, ...) such that::
///
///     symmetric A  -------- (operation) --->   symmetric B
///             |                                         |
///          (drop symm)                               (drop symm)
///             |                                         |
///             v                                         v
///     C^{n_A}  --- (operation) --->   C^{n_B}
///
/// commutes.
/// The same goes for tensors, i.e. for symmetric tensors there are corresponding non-symmetric
/// tensors which we may manipulate instead. This means that if *and only if* the symmetry has this
/// property does it make sense to convert between symmetric tensors and e.g. numpy arrays, which we
/// can think of as tensors with `NoSymmetry`. Additionally, the concept of a basis only makes
/// sense in exactly these cases.
class SymmetryFactor : public BaseSymmetry
{
  public:
    using Ptr = std::shared_ptr<SymmetryFactor>;
    using CPtr = std::shared_ptr<const SymmetryFactor>;

    std::string group_name;
    std::optional<std::string> descriptive_name;
    /// Dtype of fusion tensors, or nullopt if fusion tensors are not defined.
    std::optional<Dtype> fusion_tensor_dtype;

    SymmetryFactor(FusionStyle fusion_style,
                   BraidingStyle braiding_style,
                   Sector trivial_sector,
                   std::string group_name,
                   float64 num_sectors,
                   bool has_complex_topological_data,
                   std::optional<std::string> descriptive_name = std::nullopt,
                   bool trivial_shift = true);
    ~SymmetryFactor() override = default;

    /// Convention: valid syntax for the constructor, e.g. ``ClassName(..., name='...')``.
    virtual std::string repr() const = 0;

    /// Whether self and other describe the same mathematical structure (ignore descriptive_name).
/// Whether self and other describe the same mathematical structure.
///
/// In particular, `descriptive_name` is ignored.
    virtual bool _is_equivalent_factor(SymmetryFactor const& other) const = 0;

    bool is_equivalent_to(BaseSymmetry const& other) const;

    /// Convert to a product `Symmetry` with this single factor (via Python until
    /// converted).
/// Convert any `SymmetryFactor` to a `Symmetry` with that single factor.
    py::object as_Symmetry() override;

    std::string str() const;

    /// Product with another factor or product symmetry → Python ``Symmetry``.
    py::object mul(py::object other);

    bool equals(SymmetryFactor const& other) const;

    virtual void save_hdf5(py::object hdf5_saver,
                           py::object h5gr,
                           std::string const& subpath) const;
    /// Reconstruct into an existing instance (used by concrete from_hdf5).
    void load_hdf5_common(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

/// Helpers for concrete ``from_hdf5`` implementations.
std::optional<std::string> descriptive_name_from_hdf5_attrs(py::object h5gr);
bool trivial_shift_from_hdf5(py::object hdf5_loader, std::string const& subpath);

} // namespace cyten
