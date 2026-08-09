#include <cyten/tools/mappings.h>

namespace cyten {

// Explicit instantiations for the key types used by the FusionTreeBackend mapping stack.
template class SparseMapping<FusionTree, complex128>;
template class SparseMapping<std::pair<FusionTree, FusionTree>, complex128>;
template class IdentityMapping<FusionTree, complex128>;
template class IdentityMapping<std::pair<FusionTree, FusionTree>, complex128>;

} // namespace cyten
