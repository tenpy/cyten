#pragma once

#include <memory>

namespace cyten {

/// Incomplete tensor types for backend headers.
///
/// Backend headers must not include complete tensor headers (those include
/// ``tensor_backend.h`` again). Backend ``.cpp`` files include the real headers
/// and may access members.

class VectorLike;
class DirectSum;
class Tensor;
class SymmetricTensor;
class DiagonalTensor;
class Identity;
class Mask;
class ChargedTensor;
class HiddenLegTensor;

using VectorLikePtr = std::shared_ptr<VectorLike>;
using VectorLikeCPtr = std::shared_ptr<const VectorLike>;

using DirectSumPtr = std::shared_ptr<DirectSum>;
using DirectSumCPtr = std::shared_ptr<const DirectSum>;

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

using HiddenLegTensorPtr = std::shared_ptr<HiddenLegTensor>;
using HiddenLegTensorCPtr = std::shared_ptr<const HiddenLegTensor>;

} // namespace cyten
