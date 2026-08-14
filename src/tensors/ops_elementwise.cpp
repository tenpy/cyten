#include <cyten/tensors/ops_elementwise.h>

#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/helpers.h>
#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tools.h>

#include <cmath>
#include <format>
#include <memory>
#include <stdexcept>
#include <vector>

namespace cyten {

namespace {

py::object
numpy()
{
    return py::module_::import("numpy");
}

DiagonalTensorPtr
elementwise_on_diagonal(DiagonalTensorCPtr x, BlockUnaryFn func, bool maps_zero_to_zero)
{
    auto mut = std::const_pointer_cast<DiagonalTensor>(x);
    return mut->_elementwise_unary(std::move(func), maps_zero_to_zero);
}

BlockBackend::Scalar
numpy_unary_scalar(BlockBackend::Scalar const& x, char const* name)
{
    return numpy().attr(name)(py::cast(x)).cast<BlockBackend::Scalar>();
}

} // namespace

DiagonalTensorPtr
angle(DiagonalTensorCPtr x)
{
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x, [bb](BlockBackend::BlockPtr const& b) { return bb->angle(b); }, true);
}

BlockBackend::Scalar
angle(BlockBackend::Scalar const& x)
{
    return numpy_unary_scalar(x, "angle");
}

DiagonalTensorPtr
cutoff_inverse(DiagonalTensorCPtr x, float64 cutoff)
{
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x,
      [bb, cutoff](BlockBackend::BlockPtr const& b) { return bb->cutoff_inverse(b, cutoff); },
      true);
}

BlockBackend::Scalar
cutoff_inverse(BlockBackend::Scalar const& x, float64 cutoff)
{
    py::object pyx = py::cast(x);
    py::object abs_x = py::module_::import("builtins").attr("abs")(pyx);
    if (abs_x.cast<float64>() < cutoff) {
        return py::int_(0).cast<BlockBackend::Scalar>();
    }
    return (py::float_(1.0) / pyx).cast<BlockBackend::Scalar>();
}

DiagonalTensorPtr
complex_conj(DiagonalTensorCPtr x)
{
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x, [bb](BlockBackend::BlockPtr const& b) { return bb->conj(b); }, true);
}

BlockBackend::Scalar
complex_conj(BlockBackend::Scalar const& x)
{
    return numpy_unary_scalar(x, "conj");
}

DiagonalTensorPtr
imag(DiagonalTensorCPtr x)
{
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x, [bb](BlockBackend::BlockPtr const& b) { return bb->imag(b); }, true);
}

BlockBackend::Scalar
imag(BlockBackend::Scalar const& x)
{
    return numpy_unary_scalar(x, "imag");
}

DiagonalTensorPtr
real(DiagonalTensorCPtr x)
{
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x, [bb](BlockBackend::BlockPtr const& b) { return bb->real(b); }, true);
}

BlockBackend::Scalar
real(BlockBackend::Scalar const& x)
{
    return numpy_unary_scalar(x, "real");
}

DiagonalTensorPtr
real_if_close(DiagonalTensorCPtr x, float64 tol)
{
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x, [bb, tol](BlockBackend::BlockPtr const& b) { return bb->real_if_close(b, tol); }, true);
}

BlockBackend::Scalar
real_if_close(BlockBackend::Scalar const& x, float64 tol)
{
    return numpy()
      .attr("real_if_close")(py::cast(x), py::arg("tol") = tol)
      .cast<BlockBackend::Scalar>();
}

DiagonalTensorPtr
sqrt(DiagonalTensorCPtr x)
{
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x, [bb](BlockBackend::BlockPtr const& b) { return bb->sqrt(b); }, true);
}

BlockBackend::Scalar
sqrt(BlockBackend::Scalar const& x)
{
    return numpy_unary_scalar(x, "sqrt");
}

DiagonalTensorPtr
stable_log(DiagonalTensorCPtr x, float64 cutoff)
{
    if (!(cutoff > 0)) {
        throw std::runtime_error("cutoff must be > 0");
    }
    auto bb = x->backend->block_backend;
    return elementwise_on_diagonal(
      x,
      [bb, cutoff](BlockBackend::BlockPtr const& b) { return bb->stable_log(b, cutoff); },
      true);
}

BlockBackend::Scalar
stable_log(BlockBackend::Scalar const& x, float64 cutoff)
{
    if (!(cutoff > 0)) {
        throw std::runtime_error("cutoff must be > 0");
    }
    auto np = numpy();
    py::object pyx = py::cast(x);
    return np.attr("where")(np.attr("greater")(pyx, cutoff), np.attr("log")(pyx), 0.0)
      .cast<BlockBackend::Scalar>();
}

TensorPtr
exp(TensorCPtr obj)
{
    // --- hints from Python exp ---
    // OPTIMIZE have the same pipe in domain and codomain. could avoid recomputing?
    // should have considered all tensor types above
    // ---
    if (auto diag = std::dynamic_pointer_cast<DiagonalTensor const>(obj)) {
        auto mut = std::const_pointer_cast<DiagonalTensor>(diag);
        auto bb = diag->backend->block_backend;
        return mut->_elementwise_unary(
          [bb](BlockBackend::BlockPtr const& b) { return bb->exp(b); });
    }
    if (std::dynamic_pointer_cast<ChargedTensor const>(obj)) {
        throw py::type_error("ChargedTensor can not be exponentiated.");
    }
    if (auto sym = std::dynamic_pointer_cast<SymmetricTensor const>(obj)) {
        _check_compatible_legs(std::vector<Space::Ptr>{ sym->domain },
                               std::vector<Space::Ptr>{ sym->codomain });

        auto backend = sym->backend;
        bool combine = (!backend->can_decompose_tensors) && (sym->num_domain_legs() > 1);
        if (combine) {
            int64 J = sym->num_codomain_legs();
            int64 N = sym->num_legs;
            std::vector<LegRef> cod_idcs;
            std::vector<LegRef> dom_idcs;
            for (int64 i = 0; i < J; ++i) {
                cod_idcs.emplace_back(i);
            }
            for (int64 i = J; i < N; ++i) {
                dom_idcs.emplace_back(i);
            }
            auto combined = combine_legs(sym, { std::move(cod_idcs), std::move(dom_idcs) });
            sym = std::dynamic_pointer_cast<SymmetricTensor const>(combined);
        }
        auto bb = backend->block_backend;
        auto data = backend->act_block_diagonal_square_matrix(
          sym,
          [bb](BlockBackend::BlockPtr const& block) { return bb->matrix_exp(block); },
          std::nullopt);
        auto res = std::make_shared<SymmetricTensor>(
          std::move(data), sym->codomain, sym->domain, backend, sym->symmetry, sym->labels());
        if (combine) {
            return split_legs(res, std::vector<LegRef>{ int64{ 0 }, int64{ 1 } });
        }
        return res;
    }
    throw NotImplemented("exp");
}

} // namespace cyten
