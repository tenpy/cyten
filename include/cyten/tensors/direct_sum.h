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
/// Addition, scalar multiplication, inner product and norm are componentwise /
/// :math:`\sum_i` over the components. Components need not share legs (they live on
/// different sites / spaces). Nested :class:`DirectSum` is not supported.
class DirectSum : public VectorLike
{
  public:
    using Ptr = std::shared_ptr<DirectSum>;
    using CPtr = std::shared_ptr<const DirectSum>;

    /// Build from a non-empty list of tensors.
    explicit DirectSum(std::vector<TensorPtr> components);

    [[nodiscard]] std::vector<TensorPtr> const& components() const { return _components; }
    [[nodiscard]] std::size_t size() const { return _components.size(); }
    [[nodiscard]] TensorPtr at(int64 i) const;

    [[nodiscard]] Dtype dtype() const { return _dtype; }
    [[nodiscard]] std::string const& device() const { return _device; }
    [[nodiscard]] TensorBackend::Ptr backend() const { return _backend; }

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
