#pragma once

#include <cyten/block_backend/dtypes.h>
#include <cyten/cyten.h>
#include <cyten/tensors/sparse.h>
#include <cyten/tensors/vector_like.h>

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cyten {

/// Base class for iterative algorithms building a Krylov basis with cyten tensors.
class KrylovBased
{
  public:
    using Ptr = std::shared_ptr<KrylovBased>;

    LinearOperator::Ptr H;
    VectorLike::Ptr psi0;
    py::dict options;
    int64 N_min = 2;
    int64 N_max = 20;
    int64 N_cache = 20;
    float64 P_tol = 1.0e-14;
    float64 min_gap = 1.0e-12;
    bool reortho = false;
    std::optional<float64> E_shift;
    float64 _cutoff = 0.;
    std::optional<float64> _psi0_norm;
    std::vector<VectorLike::Ptr> _cache;
    /// ``Es[n, :]`` contains Ritz values of the n-step projected operator.
    std::vector<complex128> Es;
    /// H projected onto the Krylov ONB, shape ``(N_max+1, N_max+1)``, row-major.
    std::vector<complex128> _h_krylov;
    /// Result in the Krylov ONB (vector or column-major matrix, see ``_result_krylov_cols``).
    std::vector<complex128> _result_krylov;
    int64 _result_krylov_rows = 0;
    int64 _result_krylov_cols = 0;
    Dtype _dtype_h_krylov = Dtype::Complex128;
    Dtype _dtype_E = Dtype::Complex128;

    KrylovBased(LinearOperator::Ptr H,
                VectorLike::Ptr psi0,
                py::object options,
                Dtype dtype_h_krylov = Dtype::Complex128,
                Dtype dtype_E = Dtype::Complex128);
    virtual ~KrylovBased() = default;

    [[nodiscard]] virtual int64 _build_krylov() = 0;
    virtual void _calc_result_krylov(int64 k) = 0;
    [[nodiscard]] virtual bool _converged(int64 k) = 0;

    /// Clear cached Krylov vectors and the projected Hessenberg matrix.
    void _reset_krylov_state();
    /// Transform ``_result_krylov`` from the Krylov ONB to the original basis (single vector).
    [[nodiscard]] VectorLike::Ptr _calc_result_full(int64 N);
    /// Add psi to cache, keep at most ``N_cache``.
    virtual void _to_cache(VectorLike::Ptr psi);
    /// Rebuild uncached Krylov vectors while accumulating the result (Lanczos).
    [[nodiscard]] virtual VectorLike::Ptr _rebuild_krylov_for_result_full(VectorLike::Ptr psif,
                                                                          int64 N_max);

    [[nodiscard]] complex128 h_krylov(int64 i, int64 j) const;
    void set_h_krylov(int64 i, int64 j, complex128 value);
    [[nodiscard]] complex128 Es_at(int64 i, int64 j) const;
    void set_Es(int64 i, int64 j, complex128 value);
    [[nodiscard]] py::array Es_numpy() const;
    [[nodiscard]] py::array h_krylov_numpy() const;
    [[nodiscard]] py::array result_krylov_numpy() const;

  protected:
    [[nodiscard]] int64 h_stride() const { return N_max + 1; }
};

/// GMRES solver for ``A x = b`` with cyten tensors.
class GMRES
{
  public:
    using Ptr = std::shared_ptr<GMRES>;

    py::dict options;
    int64 N_min = 5;
    int64 N_max = 20;
    int64 restart = 10;
    float64 res = 1.0e-8;
    LinearOperator::Ptr A;
    VectorLike::Ptr b;
    VectorLike::Ptr x;
    std::vector<VectorLike::Ptr> rs;
    std::vector<VectorLike::Ptr> qs;
    float64 b_norm = 0.;
    float64 r_norm = 0.;
    std::vector<std::vector<float64>> total_error;
    std::vector<int64> total_iters;
    std::vector<complex128> sine;
    std::vector<complex128> cosine;
    std::vector<complex128> e1;
    std::vector<complex128> H; // (N_max+1, N_max) row-major
    std::vector<complex128> y;

    GMRES(LinearOperator::Ptr A, VectorLike::Ptr x, VectorLike::Ptr b, py::object options);

    void _init_hessenberg();
    /// Returns ``(x, rel_residual, total_error, total_iters)``.
    std::tuple<VectorLike::Ptr, float64, std::vector<std::vector<float64>>, std::vector<int64>>
    run();
    void arnoldi(int64 k);
    void apply_givens_rotation(int64 k);
    void givens_rotation(int64 k);
    void backsolve(int64 k);
    void reset();

    [[nodiscard]] py::array H_numpy() const;
};

/// Arnoldi method for diagonalizing square, non-hermitian/symmetric matrices.
class Arnoldi : public KrylovBased
{
  public:
    using Ptr = std::shared_ptr<Arnoldi>;

    float64 E_tol = std::numeric_limits<float64>::infinity();
    std::string which = "LM";
    int64 num_ev = 1;

    Arnoldi(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

    /// Find extremal eigenpairs of ``H``. Returns ``(E0s, psis, N)``.
    std::tuple<std::vector<complex128>, std::vector<VectorLike::Ptr>, int64> run();
    int64 _build_krylov() override;
    void _calc_result_krylov(int64 k) override;
    /// Transform Krylov eigenvectors back to the original basis.
    [[nodiscard]] std::vector<VectorLike::Ptr> _calc_result_full_multi(int64 N);
    void _to_cache(VectorLike::Ptr psi) override;
    bool _converged(int64 k) override;
};

/// Compute :math:`exp(\delta H) |\psi_0\rangle` using Arnoldi for non-Hermitian `H`.
class ArnoldiEvolution : public Arnoldi
{
  public:
    using Ptr = std::shared_ptr<ArnoldiEvolution>;

    float64 _result_norm = 1.;
    std::optional<complex128> delta;

    ArnoldiEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

    /// Compute ``expm(delta * H).dot(psi0)``. Returns ``(psi_f, N)``.
    std::tuple<VectorLike::Ptr, int64> run(complex128 delta,
                                           std::optional<bool> normalize = std::nullopt);
    void _calc_result_krylov(int64 k) override;
    bool _converged(int64 k) override;
    [[nodiscard]] VectorLike::Ptr _calc_result_full_evolution(int64 N);
};

/// Lanczos algorithm to find the ground state. Assumes `H` is hermitian.
class LanczosGroundState : public KrylovBased
{
  public:
    using Ptr = std::shared_ptr<LanczosGroundState>;

    float64 E_tol = std::numeric_limits<float64>::infinity();

    LanczosGroundState(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

    /// Find the ground state of H. Returns ``(E0, psi0, N)``.
    std::tuple<float64, VectorLike::Ptr, int64> run();
    int64 _build_krylov() override;
    bool _converged(int64 k) override;
    VectorLike::Ptr _rebuild_krylov_for_result_full(VectorLike::Ptr psif, int64 N_max) override;
    void _calc_result_krylov(int64 k) override;
};

/// Calculate :math:`exp(delta H) |psi0>` using Lanczos.
class LanczosEvolution : public LanczosGroundState
{
  public:
    using Ptr = std::shared_ptr<LanczosEvolution>;

    float64 _result_norm = 1.;
    std::optional<complex128> delta;

    LanczosEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

    /// Calculate ``expm(delta H).dot(psi0)``. Returns ``(psi_f, N)``.
    std::tuple<VectorLike::Ptr, int64> run(complex128 delta,
                                           std::optional<bool> normalize = std::nullopt);
    void _calc_result_krylov(int64 k) override;
    bool _converged(int64 k) override;
};

/// Simple wrapper calling ``LanczosGroundState(H, psi, options).run()``.
std::tuple<float64, VectorLike::Ptr, int64> lanczos(LinearOperator::Ptr H,
                                                    VectorLike::Ptr psi,
                                                    py::object options = py::none());

} // namespace cyten
