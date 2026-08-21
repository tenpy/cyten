#pragma once

#include <cstdint>

namespace cyten {

/// Describes properties of fusion, i.e. of the tensor product.
///
/// - `single`: fusing sectors results in a single sector ``a ⊗ b = c``, e.g. abelian groups.
/// - `multiple_unique`: every sector appears at most once in pairwise fusion, ``N_symbol in [0, 1]``.
/// - `general`: no assumptions, ``N_symbol in [0, 1, 2, 3, ...]``.
enum class FusionStyle : std::uint8_t
{
    single = 0,           ///< a ⊗ b = c (unique), e.g. abelian groups
    multiple_unique = 10, ///< each sector appears at most once; N in {0,1}
    general = 20,         ///< N in {0,1,2,...}
};

/// Describes properties of braiding.
///
/// - `bosonic`: symmetric braiding with trivial twist.
/// - `fermionic`: symmetric braiding with non-trivial twist.
/// - `anyonic`: general, non-symmetric braiding.
/// - `no_braiding`: braiding is not defined.
enum class BraidingStyle : std::uint8_t
{
    bosonic = 0,      ///< symmetric braid, trivial twist
    fermionic = 10,   ///< symmetric braid, non-trivial twist
    anyonic = 20,     ///< non-symmetric braid
    no_braiding = 30, ///< braiding not defined
};

} // namespace cyten
