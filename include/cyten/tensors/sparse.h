#pragma once

#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/direct_sum.h>
#include <cyten/tensors/forward_declare.h>
#include <cyten/tensors/vector_like.h>

#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

using VectorLabels = std::optional<LegLabels>;
inline constexpr float64 kGramSchmidtDefaultRcond = 1.0e-14;

bool same_legs(std::vector<Leg::Ptr> const& legs1, std::vector<Leg::Ptr> const& legs2);

/// Base class for a linear operator acting on VectorLike objects.
class LinearOperator
{
  public:
    using Ptr = std::shared_ptr<LinearOperator>;
    using CPtr = std::shared_ptr<const LinearOperator>;

    std::vector<Leg::Ptr> vector_legs;
    VectorLabels vector_labels;
    Dtype dtype = Dtype::Float64;

    static py::object acts_on;

    LinearOperator(std::vector<Leg::Ptr> vector_legs,
                   Dtype dtype,
                   VectorLabels vector_labels = std::nullopt);
    virtual ~LinearOperator() = default;

    [[nodiscard]] virtual VectorLike::Ptr matvec(VectorLike::CPtr vec) = 0;
    [[nodiscard]] virtual TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) = 0;
    [[nodiscard]] TensorPtr to_matrix(TensorBackend::Ptr backend = nullptr);
    [[nodiscard]] virtual Ptr adjoint();
};

/// Linear operator represented by contraction with a two-leg symmetric tensor.
class TensorLinearOperator : public LinearOperator
{
  public:
    using Ptr = std::shared_ptr<TensorLinearOperator>;

    SymmetricTensorPtr tensor;
    int64 which_leg = 1;
    int64 other_leg = 0;

    TensorLinearOperator(SymmetricTensorPtr tensor,
                         std::variant<int64, std::string> which_leg = int64(-1));

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// Base class for wrappers around another LinearOperator.
class LinearOperatorWrapper : public LinearOperator
{
  public:
    using Ptr = std::shared_ptr<LinearOperatorWrapper>;

    LinearOperator::Ptr original_operator;

    explicit LinearOperatorWrapper(LinearOperator::Ptr original_operator);

    [[nodiscard]] LinearOperator::Ptr unwrapped(bool recursive = true) const;
    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

class SumLinearOperator : public LinearOperatorWrapper
{
  public:
    using Ptr = std::shared_ptr<SumLinearOperator>;

    std::vector<LinearOperator::Ptr> more_operators;

    SumLinearOperator(LinearOperator::Ptr original_operator,
                      std::vector<LinearOperator::Ptr> more_operators = {});

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

class ShiftedLinearOperator : public LinearOperatorWrapper
{
  public:
    using Ptr = std::shared_ptr<ShiftedLinearOperator>;

    complex128 shift = complex128(0.);

    ShiftedLinearOperator(LinearOperator::Ptr original_operator, complex128 shift);

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

class ProjectedLinearOperator : public LinearOperatorWrapper
{
  public:
    using Ptr = std::shared_ptr<ProjectedLinearOperator>;

    std::vector<VectorLike::Ptr> ortho_vecs;
    bool project_operator = true;
    std::optional<complex128> penalty = std::nullopt;

    ProjectedLinearOperator(LinearOperator::Ptr original_operator,
                            std::vector<VectorLike::Ptr> ortho_vecs,
                            bool project_operator = true,
                            std::optional<complex128> penalty = std::nullopt);

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
    [[nodiscard]] LinearOperator::Ptr adjoint() override;
};

/// Gram-Schmidt orthonormalization of a list of vectors.
std::vector<VectorLike::Ptr> gram_schmidt(std::vector<VectorLike::Ptr> const& vecs,
                                          float64 rcond = kGramSchmidtDefaultRcond);

} // namespace cyten
