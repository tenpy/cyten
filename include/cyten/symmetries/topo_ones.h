#pragma once

/// Trivial ones FusionSymbols and small helpers formerly backed by NumPy.

#include "fusion_symbol.h"

namespace cyten::topo_ones {

inline FusionSymbol
one_1D()
{
    return FusionSymbol::one_1D();
}

inline FusionSymbol
one_2D()
{
    return FusionSymbol::one_2D();
}

inline FusionSymbol
one_2D_float()
{
    return FusionSymbol::one_2D();
}

inline FusionSymbol
one_4D()
{
    return FusionSymbol::one_4D();
}

inline FusionSymbol
one_4D_float()
{
    return FusionSymbol::one_4D();
}

inline int16_t
mod_n(int32_t x, int N)
{
    int r = static_cast<int>(x % N);
    if (r < 0) {
        r += N;
    }
    return static_cast<int16_t>(r);
}

} // namespace cyten::topo_ones
