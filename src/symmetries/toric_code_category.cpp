#include <cyten/symmetries/toric_code_category.h>

#include <utility>

namespace cyten {

Sector const ToricCodeCategory::vacuum{ 0, 0 };
Sector const ToricCodeCategory::electric_charge{ 0, 1 };
Sector const ToricCodeCategory::magnetic_flux{ 1, 0 };
Sector const ToricCodeCategory::fermion{ 1, 1 };

ToricCodeCategory::ToricCodeCategory(std::optional<std::string> descriptive_name)
  : QuantumDoubleZNAnyonCategory(2, std::move(descriptive_name))
{
}

std::string
ToricCodeCategory::repr() const
{
    if (!descriptive_name.has_value()) {
        return "ToricCodeCategory()";
    }
    return "ToricCodeCategory(\"" + *descriptive_name + "\")";
}

bool
ToricCodeCategory::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* qd = dynamic_cast<QuantumDoubleZNAnyonCategory const*>(&other)) {
        return qd->N == 2;
    }
    return false;
}

} // namespace cyten
