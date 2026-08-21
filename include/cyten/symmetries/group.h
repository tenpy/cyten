#pragma once

#include "symmetry_factor.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Base-class for symmetries that are described by a group.
///
/// The symmetry is given via a faithful representation on the Hilbert space.
/// Notable counter-examples are fermionic parity or anyonic grading.
class Group : public SymmetryFactor
{
  public:
    using Ptr = std::shared_ptr<Group>;
    using CPtr = std::shared_ptr<const Group>;

    Group(FusionStyle fusion_style,
          Sector trivial_sector,
          std::string group_name,
          float64 num_sectors,
          bool has_complex_topological_data,
          std::optional<std::string> descriptive_name = std::nullopt,
          bool trivial_shift = true);
    ~Group() override = default;

/// The quantum dimension ``Tr(id_a)`` of a sector
    FusionSymbol _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override =
      0;

    FusionSymbol swap_gate(Sector a, Sector b) const override;
    float64 qdim(Sector a) const override;
/// Quantum dimension of every sector (row) in `a`
    std::vector<float64> batch_qdim(SectorArray const& a) const override;
/// The prefactor that relates the twist on a single sector to the identity.
///
/// Graphically::
///
///     |   │   ╭─╮                |
///     |    ╲ ╱  │                |
///     |     ╱   │   =   theta_a  |
///     |    ╱ ╲  │                |
///     |   │   ╰─╯                |
///     |   a                      a
///
/// Notes:
///
/// For a twist with opposite chirality, the prefactor is conjugated.
///
///     |   │   ╭─╮                      |
///     |    ╲ ╱  │                      |
///     |     ╲   │   =   conj(theta_a)  |
///     |    ╱ ╲  │                      |
///     |   │   ╰─╯                      |
///     |   a                            a
    complex128 topological_twist(Sector a) const override;
};

} // namespace cyten
