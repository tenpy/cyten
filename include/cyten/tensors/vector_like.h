#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>

#include <memory>
#include <string>

namespace cyten {

/// Abstract vector for Krylov / sparse linear algebra.
///
/// A :class:`VectorLike` supports copy, inner product, norm, scalar multiplication and addition.
/// :class:`Tensor` is a single-tensor vector; :class:`DirectSum` is a list of tensors treated as
/// one vector in a direct-sum Hilbert space.
class VectorLike
{
  public:
    using Ptr = std::shared_ptr<VectorLike>;
    using CPtr = std::shared_ptr<const VectorLike>;

    virtual ~VectorLike();

    /// Independent copy of this vector.
    [[nodiscard]] virtual Ptr clone() const = 0;

    [[nodiscard]] virtual Dtype vector_dtype() const = 0;
    [[nodiscard]] virtual std::string vector_device() const = 0;
    [[nodiscard]] virtual TensorBackend::Ptr vector_backend() const = 0;

    /// Frobenius (or direct-sum Frobenius) norm.
    [[nodiscard]] virtual BlockBackend::Scalar vector_norm() const = 0;

    /// Inner product ``<self|other>``. If `do_dagger`, conjugate `self` first.
    [[nodiscard]] virtual BlockBackend::Scalar vector_inner(CPtr other,
                                                            bool do_dagger = true) const = 0;

    /// Scalar multiplication ``a * self``.
    [[nodiscard]] virtual Ptr scaled(BlockBackend::Scalar const& a) const = 0;

    /// BLAS-style axpy: ``a * self + other``.
    [[nodiscard]] virtual Ptr axpy(BlockBackend::Scalar const& a, CPtr other) const = 0;

    /// Whether `other` lives in the same vector space (same type and matching structure).
    [[nodiscard]] virtual bool compatible_with(CPtr other) const = 0;
};

} // namespace cyten
