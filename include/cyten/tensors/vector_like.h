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
/// Supports copy, inner product, norm, scalar multiplication and addition.
/// `Tensor` is a single-tensor vector; `DirectSum` is a list of tensors treated as
/// one vector in a direct-sum Hilbert space.
class VectorLike
{
  public:
    using Ptr = std::shared_ptr<VectorLike>;
    using CPtr = std::shared_ptr<const VectorLike>;

    virtual ~VectorLike();

    /// Independent copy of this vector.
    ///
    /// Exposed to Python as `copy`.
    [[nodiscard]] virtual Ptr clone() const = 0;

    /// Dtype of the vector components.
    [[nodiscard]] virtual Dtype vector_dtype() const = 0;
    /// Device string of the vector components.
    [[nodiscard]] virtual std::string vector_device() const = 0;
    /// Tensor backend of the vector components.
    [[nodiscard]] virtual TensorBackend::Ptr vector_backend() const = 0;

    /// Frobenius (or direct-sum Frobenius) norm.
    [[nodiscard]] virtual BlockBackend::Scalar vector_norm() const = 0;

    /// Inner product `<self|other>`. If `do_dagger`, conjugate `self` first.
    ///
    /// @param other Other vector in the same space.
    /// @param do_dagger Whether to dagger `self` before the product.
    [[nodiscard]] virtual BlockBackend::Scalar vector_inner(CPtr other,
                                                            bool do_dagger = true) const = 0;

    /// Scalar multiplication `a * self`.
    ///
    /// @param a Scalar factor.
    [[nodiscard]] virtual Ptr scaled(BlockBackend::Scalar const& a) const = 0;

    /// BLAS-style axpy: `a * self + other`.
    ///
    /// @param a Scalar factor for `self`.
    /// @param other Vector to add.
    [[nodiscard]] virtual Ptr axpy(BlockBackend::Scalar const& a, CPtr other) const = 0;

    /// Whether `other` lives in the same vector space (same type and matching structure).
    ///
    /// @param other Candidate vector.
    [[nodiscard]] virtual bool compatible_with(CPtr other) const = 0;
};

} // namespace cyten
