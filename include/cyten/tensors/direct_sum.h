#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>
#include <cyten/tensors/forward_declare.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tensors/vector_like.h>

#include <memory>
#include <string>
#include <vector>

namespace cyten {

/// Direct-sum vector: a list of tensors treated as one Krylov / LinearOperator vector.
///
/// Addition and scalar multiplication are defined componentwise (zip over `components`).
/// Components need not share legs (they may live on different spaces / sites).
/// Nested `DirectSum` is not supported.
///
/// Inner product and norm are the standard direct-sum Hilbert-space inner product
/// @f$\langle X \mid Y \rangle = \sum_i \langle X_i \mid Y_i \rangle@f$ and
/// @f$\|X\| = \sqrt{\sum_i \|X_i\|^2}@f$, where the sum runs over components and each
/// @f$\langle X_i \mid Y_i \rangle@f$ is the usual (Frobenius) tensor inner product on that
/// component.
///
/// Intended for Krylov / `LinearOperator` algorithms. An example is a VUMPS calculation or
/// tangent-space excitations with a multi-site unit cell: after gauge-fixing, an orthonormal
/// parametrization of the ground state / excitations has one tensor per site in the unit cell,
/// corresponding to one `component` entry.
///
/// This class **assumes** that cross terms of the inner product between different components
/// are zero. That is only sensible with an orthonormal parametrization of vectors in the full
/// (many-body) Hilbert space.
class DirectSum : public VectorLike
{
  public:
    using Ptr = std::shared_ptr<DirectSum>;
    using CPtr = std::shared_ptr<const DirectSum>;

    /// Build from a non-empty list of tensors.
    ///
    /// @param components Non-empty sequence of tensors.
    explicit DirectSum(std::vector<TensorPtr> components);

    /// The component tensors.
    [[nodiscard]] std::vector<TensorPtr> const& components() const { return _components; }
    /// Number of components.
    [[nodiscard]] std::size_t size() const { return _components.size(); }
    /// Component at index `i`.
    [[nodiscard]] TensorPtr at(int64 i) const;

    [[nodiscard]] Dtype dtype() const { return _dtype; }
    [[nodiscard]] std::string const& device() const { return _device; }
    [[nodiscard]] TensorBackend::Ptr backend() const { return _backend; }

    /// Copy all component tensors.
    ///
    /// @param deep If true, deep-copy components; otherwise share them.
    [[nodiscard]] Ptr copy(bool deep = true) const;

    [[nodiscard]] VectorLike::Ptr clone() const override;
    [[nodiscard]] Dtype vector_dtype() const override;
    [[nodiscard]] std::string vector_device() const override;
    [[nodiscard]] TensorBackend::Ptr vector_backend() const override;
    [[nodiscard]] BlockBackend::Scalar vector_norm() const override;
    [[nodiscard]] BlockBackend::Scalar vector_inner(VectorLike::CPtr other,
                                                    bool do_dagger = true) const override;
    [[nodiscard]] VectorLike::Ptr scaled(BlockBackend::Scalar const& a) const override;
    [[nodiscard]] VectorLike::Ptr axpy(BlockBackend::Scalar const& a,
                                       VectorLike::CPtr other) const override;
    [[nodiscard]] bool compatible_with(VectorLike::CPtr other) const override;

  private:
    std::vector<TensorPtr> _components;
    Dtype _dtype = Dtype::Float64;
    std::string _device;
    TensorBackend::Ptr _backend;

    [[nodiscard]] DirectSumCPtr as_direct_sum(VectorLike::CPtr other, char const* op) const;
};

} // namespace cyten
