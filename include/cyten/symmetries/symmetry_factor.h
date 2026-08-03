#pragma once

#include "../block_backend/dtypes.h"
#include "base_symmetry.h"

#include <optional>
#include <string>

namespace cyten {

/// Base class for symmetries that impose a block-structure on tensors (single factor).
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
    virtual bool _is_equivalent_factor(SymmetryFactor const& other) const = 0;

    bool is_equivalent_to(BaseSymmetry const& other) const;

    /// Convert to a product :class:`Symmetry` with this single factor (via Python until converted).
    py::object as_Symmetry() override;

    std::string str() const;

    /// Product with another factor or product symmetry → Python ``Symmetry``.
    py::object mul(py::object other) const;

    bool equals(SymmetryFactor const& other) const;

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;
    /// Reconstruct into an existing instance (used by concrete from_hdf5).
    void load_hdf5_common(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
