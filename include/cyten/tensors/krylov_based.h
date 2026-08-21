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
///
/// Algorithms like `LanczosGroundState` and `Arnoldi`
/// are based on iteratively building an orthonormal basis of the Krylov space spanned by
/// ``|psi0>, H|psi0>, H^2|psi0>, ... H^N |psi0>``, where `N` is the number of iterations
/// performed so far, and ``|psi0>`` is an initial guess and starting vector.
/// During that iteration, the projection of `H` into the Krylov space is built, where it can
/// be solved effectively (with `H` being just a N by N matrix), yielding the "Ritz" eigenvalues/
/// eigenvectors. Finally, the solution can be translated back into the original space using the
/// basis.
///
/// An important strategy is also to (implicitly) restart the algorithm after some number of steps.
/// This is **not** done here: when we use these classes, we usually have an explicit outer loop
/// performed until convergence, e.g., the "sweeps" in DMRG.
///
/// Constructed with a hermitian linear operator `H`, starting vector `psi0`, and an
/// options dict. The algorithm stops if both criteria for `e_tol` and `p_tol` are met
/// or if the maximum number of steps was reached.
///
/// Attributes:
///
/// options : dict_like
///     Optional parameters.
/// H : `LinearOperator`
///     The linear operator used for building the Krylov space.
/// psi0 : `VectorLike`
///     The *normalized* starting vector.
/// N_min, N_max, P_tol, min_gap, _cutoff, E_shift:
///     Parameters as described in the options.
/// Es : ndarray, shape(N_max, N_max)
///     ``Es[n, :]`` contains the energies of ``_h_krylov[:n+1, :n+1]`` in step `n`.
/// _h_krylov : ndarray, shape (N_max + 1, N_max +1)
///     The matrix representing `H` projected onto the orthonormalized Krylov basis.
/// _psi0_norm : float
///     Initial norm of the `psi0` parameter. Note that ``self.psi0`` gets normalized.
/// _cache : list of psi0-like vectors
///     The ONB of the Krylov space generated during the iteration.
///     FIFO (first in first out) cache of at most `N_cache` vectors.
/// _result_krylov : ndarray
///     Result in the ONB of the Krylov space, e.g. the ground state of `_h_krylov`.
///     What exactly this is depends on the subclass.
///
/// Notes:
///
/// The Ritz residual `RitzRes` is computed according to
/// http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual.
/// Given the gap, the Ritz residual gives a bound on the error in the wavefunction,
/// ``err < (RitzRes/gap)**2``. The gap is estimated from the full Lanczos spectrum.
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
///
/// @param A Linear operator. Must implement `matvec`.
/// @param x Initial guess. Copied; the caller's vector is not modified.
/// @param b Right-hand side.
/// @param options Solver options.
///
/// Options:
///
/// N_min : int
///     Minimum number of Arnoldi steps per restart cycle before checking convergence.
/// N_max : int
///     Maximum Krylov dimension per restart cycle.
/// restart : int
///     Maximum number of restart cycles.
/// res : float
///     Relative residual tolerance ``|A x - b| / |b|``.
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
///
/// Generalization of `LanczosGroundState`, allowing general, square matrices.
///
/// Options:
class Arnoldi : public KrylovBased
{
  public:
    using Ptr = std::shared_ptr<Arnoldi>;

    float64 E_tol = std::numeric_limits<float64>::infinity();
    std::string which = "LM";
    int64 num_ev = 1;

    Arnoldi(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

/// Find the ground state of self.H.
///
/// @returns E0s : numpy array Best eigenvalue estimates, `num_ev` entries, sorted according to `which`. psis : list of `Tensor` Corresponding best eigenvectors (estimates). N : int Used dimension of the Krylov space, i.e., how many iterations where performed.
    std::tuple<std::vector<complex128>, std::vector<VectorLike::Ptr>, int64> run();
    int64 _build_krylov() override;
    void _calc_result_krylov(int64 k) override;
    /// Transform Krylov eigenvectors back to the original basis.
    [[nodiscard]] std::vector<VectorLike::Ptr> _calc_result_full_multi(int64 N);
    void _to_cache(VectorLike::Ptr psi) override;
    bool _converged(int64 k) override;
};

/// Compute @f$ exp(\delta H) |\psi_0\rangle @f$ using Arnoldi for non-Hermitian `H`.
///
/// Drop-in replacement for `LanczosEvolution` when `H` is not Hermitian.
/// Builds an upper Hessenberg projection of `H` via full Gram-Schmidt orthogonalization
/// (Arnoldi iteration), then computes the matrix exponential of the small projected matrix
/// via eigendecomposition (``numpy.linalg.eig`` + pointwise scalar exponentials).
///
/// @param H, psi0, options Same as `Arnoldi`. Note that `H` need not be Hermitian.
///
/// Options:
///
/// Attributes:
///
/// delta : float/complex or None
///     Prefactor of H in the exponential.
/// _result_norm : float
///     Norm of the result vector.
class ArnoldiEvolution : public Arnoldi
{
  public:
    using Ptr = std::shared_ptr<ArnoldiEvolution>;

    float64 _result_norm = 1.;
    std::optional<complex128> delta;

    ArnoldiEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

/// Compute ``expm(delta * H).dot(psi0)`` using Arnoldi.
///
/// @param delta Prefactor of H in the exponential. Note that the complex ``i`` is *not* included.
/// @param normalize Whether to normalize the result. Defaults to ``False``. Unlike `LanczosEvolution` (which defaults to ``np.real(delta) == 0``), non-Hermitian evolution does not in general preserve the norm, so normalization would strip physically meaningful decay or growth and is off by default.
/// @returns psi_f : `Tensor` Best approximation for ``expm(delta * H).dot(psi0)``. N : int Krylov space dimension used.
    std::tuple<VectorLike::Ptr, int64> run(complex128 delta,
                                           std::optional<bool> normalize = std::nullopt);
    void _calc_result_krylov(int64 k) override;
    bool _converged(int64 k) override;
    [[nodiscard]] VectorLike::Ptr _calc_result_full_evolution(int64 N);
};

/// Lanczos algorithm to find the ground state.
///
/// **Assumes** that `H` is hermitian.
///
/// Options:
class LanczosGroundState : public KrylovBased
{
  public:
    using Ptr = std::shared_ptr<LanczosGroundState>;

    float64 E_tol = std::numeric_limits<float64>::infinity();

    LanczosGroundState(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

/// Find the ground state of H.
///
/// @returns E0 : float Ground state energy (estimate). psi0 : `VectorLike` Ground state vector (estimate). N : int Used dimension of the Krylov space, i.e., how many iterations where performed.
    std::tuple<float64, VectorLike::Ptr, int64> run();
    int64 _build_krylov() override;
    bool _converged(int64 k) override;
    VectorLike::Ptr _rebuild_krylov_for_result_full(VectorLike::Ptr psif, int64 N_max) override;
    void _calc_result_krylov(int64 k) override;
};

/// Calculate @f$ exp(delta H) |psi0> @f$ using Lanczos.
///
/// It turns out that the Lanczos algorithm is also good for calculating the matrix exponential
/// applied to the starting vector. Instead of diagonalizing the tri-diagonal `h` and taking the
/// ground state, we now calculate ``exp(delta h) e_0`` in the Krylov ONB, where
/// ``e_0 = (1, 0, 0, ...)`` corresponds to ``psi0`` in the original basis.
///
/// @param H, psi0, options Hamiltonian, starting vector and parameters as defined in `LanczosGroundState`. The option :cfg:option`LanczosEvolution.P_tol` defines when convergence is reached, see `_converged` for details.
///
/// Options:
///
/// Attributes:
///
/// delta : float/complex
///     Prefactor of H in the exponential.
/// _result_norm : float
///     Norm of the resulting vector.
class LanczosEvolution : public LanczosGroundState
{
  public:
    using Ptr = std::shared_ptr<LanczosEvolution>;

    float64 _result_norm = 1.;
    std::optional<complex128> delta;

    LanczosEvolution(LinearOperator::Ptr H, VectorLike::Ptr psi0, py::object options);

/// Calculate ``expm(delta H).dot(psi0)`` using Lanczos.
///
/// @param delta Time step by which we should evolve psi0: prefactor of H in the exponential. Note that the complex `i` is *not* included!
/// @param normalize Whether to normalize the resulting state. Defaults to ``np.real(delta) == 0``.
/// @returns psi_f : `Tensor` Best approximation for ``expm(delta H).dot(psi0)``. If `E_shift` is used, it's an approximation for ``expm(delta (H + E_shift)).dot(psi)``. N : int Krylov space dimension used.
    std::tuple<VectorLike::Ptr, int64> run(complex128 delta,
                                           std::optional<bool> normalize = std::nullopt);
    void _calc_result_krylov(int64 k) override;
    bool _converged(int64 k) override;
};

/// Simple wrapper calling ``LanczosGroundState(H, psi, options).run()``.
/// Simple wrapper calling ``LanczosGroundState(H, psi, options).run()``
///
/// @param H, psi, options See `LanczosGroundState`.
/// @returns E0, psi0, N : See `run`.
std::tuple<float64, VectorLike::Ptr, int64> lanczos(LinearOperator::Ptr H,
                                                    VectorLike::Ptr psi,
                                                    py::object options = py::none());

} // namespace cyten
