#pragma once

#include "quantum_double_zn_anyon_category.h"

#include <optional>
#include <string>

namespace cyten {

/// Toric code anyon category (:math:`D(Z_2)`).
class ToricCodeCategory : public QuantumDoubleZNAnyonCategory
{
  public:
    using Ptr = std::shared_ptr<ToricCodeCategory>;
    using CPtr = std::shared_ptr<const ToricCodeCategory>;

    static Sector const vacuum;
    static Sector const electric_charge;
    static Sector const magnetic_flux;
    static Sector const fermion;

    explicit ToricCodeCategory(std::optional<std::string> descriptive_name = std::nullopt);
    ~ToricCodeCategory() override = default;

    std::string repr() const override;
    bool _is_equivalent_factor(SymmetryFactor const& other) const override;

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
