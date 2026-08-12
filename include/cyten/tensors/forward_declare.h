#pragma once

#include <memory>

namespace cyten {

/// Incomplete tensor types for backend headers.
///
/// Backend headers must not include complete tensor headers (those include
/// ``tensor_backend.h`` again). Backend ``.cpp`` files include the real headers
/// and may access members.

class Tensor;
class SymmetricTensor;
class DiagonalTensor;
class Identity;
class Mask;
class ChargedTensor;

using TensorPtr = std::shared_ptr<Tensor>;
using TensorCPtr = std::shared_ptr<const Tensor>;

using SymmetricTensorPtr = std::shared_ptr<SymmetricTensor>;
using SymmetricTensorCPtr = std::shared_ptr<const SymmetricTensor>;

using DiagonalTensorPtr = std::shared_ptr<DiagonalTensor>;
using DiagonalTensorCPtr = std::shared_ptr<const DiagonalTensor>;

using IdentityPtr = std::shared_ptr<Identity>;
using IdentityCPtr = std::shared_ptr<const Identity>;

using MaskPtr = std::shared_ptr<Mask>;
using MaskCPtr = std::shared_ptr<const Mask>;

using ChargedTensorPtr = std::shared_ptr<ChargedTensor>;
using ChargedTensorCPtr = std::shared_ptr<const ChargedTensor>;

} // namespace cyten
