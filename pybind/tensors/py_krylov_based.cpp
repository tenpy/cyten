#include <cyten/tensors/krylov_based.h>

#include "../py_cyten_pybind11.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <optional>
#include <utility>
#include <vector>

namespace cyten {

namespace {

class PyKrylovBased
  : public KrylovBased
  , public py::trampoline_self_life_support
{
  public:
    using KrylovBased::KrylovBased;

    int64 _build_krylov() override { PYBIND11_OVERRIDE_PURE(int64, KrylovBased, _build_krylov); }

    void _calc_result_krylov(int64 k) override
    {
        PYBIND11_OVERRIDE_PURE(void, KrylovBased, _calc_result_krylov, k);
    }

    bool _converged(int64 k) override { PYBIND11_OVERRIDE_PURE(bool, KrylovBased, _converged, k); }

    void _to_cache(VectorLike::Ptr psi) override
    {
        PYBIND11_OVERRIDE(void, KrylovBased, _to_cache, psi);
    }

    VectorLike::Ptr _rebuild_krylov_for_result_full(VectorLike::Ptr psif, int64 N_max) override
    {
        PYBIND11_OVERRIDE(
          VectorLike::Ptr, KrylovBased, _rebuild_krylov_for_result_full, psif, N_max);
    }
};

py::object
optional_float_to_py(std::optional<float64> const& v)
{
    if (!v.has_value()) {
        return py::none();
    }
    return py::float_(*v);
}

py::object
optional_complex_to_py(std::optional<complex128> const& v)
{
    if (!v.has_value()) {
        return py::none();
    }
    return py::cast(*v);
}

py::array
eigenvalues_to_numpy(std::vector<complex128> const& E0s)
{
    py::array_t<complex128> arr(static_cast<py::ssize_t>(E0s.size()));
    auto r = arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
        r(i) = E0s[static_cast<std::size_t>(i)];
    }
    return arr;
}

} // namespace

void
bind_tensors_krylov_based(py::module_& m)
{
    py::class_<KrylovBased, PyKrylovBased, py::smart_holder> krylov_based(m, "KrylovBased");
    krylov_based.doc() = R"pydoc(
Base class for iterative algorithms building a Krylov basis with cyten tensors.

Algorithms like :class:`LanczosGroundState` and `:class:`ArnoldiDiagonalize`
are based on iteratively building an orthonormal basis of the Krylov space spanned by
``|psi0>, H|psi0>, H^2|psi0>, ... H^N |psi0>``, where `N` is the number of iterations
performed so far, and ``|psi0>`` is an initial guess and starting vector.
During that iteration, the projection of `H` into the Krylov space is built, where it can
be solved effectively (with `H` being just a N by N matrix), yielding the "Ritz" eigenvalues/
eigenvectors. Finally, the solution can be translated back into the original space using the
basis.

An important strategy is also to (implicitly) restart the algorithm after some number of steps.
This is **not** done here: when we use these classes, we usually have an explicit outer loop
performed until convergence, e.g., the "sweeps" in DMRG.

Parameters
----------
H : :class:`~cyten.sparse.LinearOperator`
    A hermitian linear operator.
    In order to use :class:`~cyten.tensors.Tensor`s or other
    :class:`~cyten.tensors.Tensor` types, see :class:`~cyten.sparse.TensorLinearOperator`.
    The operator must map tensors to tensors with the same legs.
psi0 : :class:`~cyten.tensors.VectorLike`
    The starting vector defining the Krylov basis.
    For finding the ground state, this should be the best guess available.
    A :class:`~cyten.tensors.Tensor` of any rank, or a :class:`~cyten.tensors.DirectSum`
    of tensors, is allowed.
options : dict
    Further optional parameters as described in :cfg:config:`Lanczos`.
    The algorithm stops if *both* criteria for `e_tol` and `p_tol` are met
    or if the maximum number of steps was reached.

Options
-------
.. cfg:config :: KrylovBased

    N_min : int
        Minimum number of steps to perform.
    N_max : int
        Maximum number of steps to perform.
    P_tol : float
        Tolerance for the error estimate from the Ritz Residual,
        stop if ``(RitzRes/gap)**2 < P_tol``
    min_gap : float
        Lower cutoff for the gap estimate used in the P_tol criterion.
    cutoff : float
        Cutoff to abort if the norm of the new krylov vector is too small.
        This is necessary if the rank of `H` is smaller than `N_max`, but it's *not* the error
        tolerance for final values!
    E_shift : float
        Shift the energy (=eigenvalues) by that amount *during* the Lanczos run by using the
        :class:`~cyten.sparse.ShiftedLinearOperator`.
        The ground state energy `E0` returned by :meth:`run` is made independent of the shift.
        This option is useful if the :class:`~cyten.sparse.ProjectedLinearOperator`
        is used: the orthogonal vectors are *exact* eigenvectors with eigenvalue 0 independent
        of the shift, so you can use it to ensure that the energy is smaller than zero
        to avoid getting those.
    reortho : bool
        For poorly conditioned matrices, one can quickly loose orthogonality of the
        generated Krylov basis.
        If `reortho` is True, we re-orthogonalize against all the
        vectors kept in cache to avoid that problem.

Attributes
----------
options : dict_like
    Optional parameters.
H : :class:`~cyten.sparse.LinearOperator`
    The linear operator used for building the Krylov space.
psi0 : :class:`~cyten.tensors.VectorLike`
    The *normalized* starting vector.
N_min, N_max, P_tol, min_gap, _cutoff, E_shift:
    Parameters as described in the options.
Es : ndarray, shape(N_max, N_max)
    ``Es[n, :]`` contains the energies of ``_h_krylov[:n+1, :n+1]`` in step `n`.
_h_krylov : ndarray, shape (N_max + 1, N_max +1)
    The matrix representing `H` projected onto the orthonormalized Krylov basis.
_psi0_norm : float
    Initial norm of the `psi0` parameter. Note that ``self.psi0`` gets normalized.
_cache : list of psi0-like vectors
    The ONB of the Krylov space generated during the iteration.
    FIFO (first in first out) cache of at most `N_cache` vectors.
_result_krylov : ndarray
    Result in the ONB of the Krylov space, e.g. the ground state of `_h_krylov`.
    What exactly this is depends on the subclass.

Notes
-----
The Ritz residual `RitzRes` is computed according to
http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual.
Given the gap, the Ritz residual gives a bound on the error in the wavefunction,
``err < (RitzRes/gap)**2``. The gap is estimated from the full Lanczos spectrum.
)pydoc";

    krylov_based
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("H", &KrylovBased::H)
      .def_readwrite("psi0", &KrylovBased::psi0)
      .def_readwrite("options", &KrylovBased::options)
      .def_readwrite("N_min", &KrylovBased::N_min)
      .def_readwrite("N_max", &KrylovBased::N_max)
      .def_readwrite("N_cache", &KrylovBased::N_cache)
      .def_readwrite("P_tol", &KrylovBased::P_tol)
      .def_readwrite("min_gap", &KrylovBased::min_gap)
      .def_readwrite("reortho", &KrylovBased::reortho)
      .def_property(
        "E_shift",
        [](KrylovBased const& self) { return optional_float_to_py(self.E_shift); },
        [](KrylovBased& self, py::object v) {
            if (v.is_none()) {
                self.E_shift = std::nullopt;
            } else {
                self.E_shift = v.cast<float64>();
            }
        })
      .def_readwrite("_cutoff", &KrylovBased::_cutoff)
      .def_property_readonly("Es", &KrylovBased::Es_numpy)
      .def_property_readonly("_h_krylov", &KrylovBased::h_krylov_numpy)
      .def_property_readonly("_result_krylov", &KrylovBased::result_krylov_numpy)
      .def("_reset_krylov_state",
           &KrylovBased::_reset_krylov_state,
           "Clear cached Krylov vectors and the projected Hessenberg matrix.");

    py::class_<GMRES, py::smart_holder> gmres(m, "GMRES");
    gmres.doc() = R"pydoc(
GMRES solver for ``A x = b`` with cyten tensors.

Parameters
----------
A : :class:`~cyten.sparse.LinearOperator`
    Linear operator. Must implement `matvec`.
x : :class:`~cyten.tensors.VectorLike`
    Initial guess. Copied; the caller's vector is not modified.
b : :class:`~cyten.tensors.VectorLike`
    Right-hand side.
options : dict
    Solver options.

Options
-------
N_min : int
    Minimum number of Arnoldi steps per restart cycle before checking convergence.
N_max : int
    Maximum Krylov dimension per restart cycle.
restart : int
    Maximum number of restart cycles.
res : float
    Relative residual tolerance ``|A x - b| / |b|``.
)pydoc";

    gmres
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("A"),
           py::arg("x"),
           py::arg("b"),
           py::arg("options") = py::none())
      .def_readwrite("A", &GMRES::A)
      .def_readwrite("x", &GMRES::x)
      .def_readwrite("b", &GMRES::b)
      .def_readwrite("options", &GMRES::options)
      .def_readwrite("N_min", &GMRES::N_min)
      .def_readwrite("N_max", &GMRES::N_max)
      .def_readwrite("restart", &GMRES::restart)
      .def_readwrite("res", &GMRES::res)
      .def("run", &GMRES::run)
      .def("arnoldi", &GMRES::arnoldi, py::arg("k"))
      .def("apply_givens_rotation", &GMRES::apply_givens_rotation, py::arg("k"))
      .def("givens_rotation", &GMRES::givens_rotation, py::arg("k"))
      .def("backsolve", &GMRES::backsolve, py::arg("k"))
      .def("reset", &GMRES::reset);

    py::class_<Arnoldi, KrylovBased, py::smart_holder> arnoldi(m, "Arnoldi");
    arnoldi.doc() = R"pydoc(
Arnoldi method for diagonalizing square, non-hermitian/symmetric matrices.

Generalization of :class:`LanczosGroundState`, allowing general, square matrices.

Options
-------
.. cfg:config :: Arnoldi
    :include: KrylovBased

    E_tol : float
        Stop if energy difference per step < `E_tol`
    which : ``'LM' | 'LR' | 'SR'``
        Determines which (extremal) eigenvalues to look for, name
        largest magnitude (in absolute value, ``'LM'``), or
        largest or smallest real part (``'LR'`` and ``'SR'``, respectively).
    num_ev : int
        Number of eigenvectors to look for/return in `run`.
)pydoc";

    arnoldi
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("E_tol", &Arnoldi::E_tol)
      .def_readwrite("which", &Arnoldi::which)
      .def_readwrite("num_ev", &Arnoldi::num_ev)
      .def(
        "run",
        [](Arnoldi& self) {
            auto [E0s, psis, N] = self.run();
            return py::make_tuple(eigenvalues_to_numpy(E0s), std::move(psis), N);
        },
        R"pydoc(Find the ground state of self.H.

Returns
-------
E0s : numpy array
    Best eigenvalue estimates, :cfg:option:`Arnoldi.num_ev` entries,
    sorted according to :cfg:option:`Arnoldi.which`.
psis : list of :class:`~cyten.tensors.Tensor`
    Corresponding best eigenvectors (estimates).
N : int
    Used dimension of the Krylov space, i.e., how many iterations where performed.
)pydoc");

    py::class_<ArnoldiEvolution, Arnoldi, py::smart_holder> arnoldi_evolution(m,
                                                                              "ArnoldiEvolution");
    arnoldi_evolution.doc() = R"pydoc(
Compute :math:`exp(\delta H) |\psi_0\rangle` using Arnoldi for non-Hermitian `H`.

Drop-in replacement for :class:`LanczosEvolution` when `H` is not Hermitian.
Builds an upper Hessenberg projection of `H` via full Gram-Schmidt orthogonalization
(Arnoldi iteration), then computes the matrix exponential of the small projected matrix
via eigendecomposition (``numpy.linalg.eig`` + pointwise scalar exponentials).

Parameters
----------
H, psi0, options :
    Same as :class:`Arnoldi`. Note that `H` need not be Hermitian.

Options
-------
.. cfg:config :: ArnoldiEvolution
    :include: Arnoldi

    E_tol, which, num_ev :
        Inherited but ignored.

Attributes
----------
delta : float/complex or None
    Prefactor of H in the exponential.
_result_norm : float
    Norm of the result vector.
)pydoc";

    arnoldi_evolution
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("_result_norm", &ArnoldiEvolution::_result_norm)
      .def_property(
        "delta",
        [](ArnoldiEvolution const& self) { return optional_complex_to_py(self.delta); },
        [](ArnoldiEvolution& self, py::object v) {
            if (v.is_none()) {
                self.delta = std::nullopt;
            } else {
                self.delta = v.cast<complex128>();
            }
        })
      .def("run",
           &ArnoldiEvolution::run,
           py::arg("delta"),
           py::arg("normalize") = py::none(),
           R"pydoc(Compute ``expm(delta * H).dot(psi0)`` using Arnoldi.

Parameters
----------
delta : float/complex
    Prefactor of H in the exponential. Note that the complex ``i`` is *not* included.
normalize : bool
    Whether to normalize the result. Defaults to ``False``.
    Unlike :class:`LanczosEvolution` (which defaults to ``np.real(delta) == 0``),
    non-Hermitian evolution does not in general preserve the norm, so normalization
    would strip physically meaningful decay or growth and is off by default.

Returns
-------
psi_f : :class:`~cyten.tensors.Tensor`
    Best approximation for ``expm(delta * H).dot(psi0)``.
N : int
    Krylov space dimension used.
)pydoc");

    py::class_<LanczosGroundState, KrylovBased, py::smart_holder> lanczos_ground_state(
      m, "LanczosGroundState");
    lanczos_ground_state.doc() = R"pydoc(
Lanczos algorithm to find the ground state.

**Assumes** that `H` is hermitian.


Options
-------
.. cfg:config :: LanczosGroundState
    :include: KrylovBased

    E_tol : float
        Stop if energy difference per step < `E_tol`
    N_cache : int
        The maximum number of `psi` to keep in memory during the first iteration.
        By default, we keep all states (up to N_max).
        Set this to a number >= 2 if you are short on memory.
        The penalty is that one needs another Lanczos iteration to
        determine the ground state in the end, i.e., runtime is large.
)pydoc";

    lanczos_ground_state
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("E_tol", &LanczosGroundState::E_tol)
      .def("run",
           &LanczosGroundState::run,
           R"pydoc(Find the ground state of H.

Returns
-------
E0 : float
    Ground state energy (estimate).
psi0 : :class:`~cyten.tensors.VectorLike`
    Ground state vector (estimate).
N : int
    Used dimension of the Krylov space, i.e., how many iterations where performed.
)pydoc");

    py::class_<LanczosEvolution, LanczosGroundState, py::smart_holder> lanczos_evolution(
      m, "LanczosEvolution");
    lanczos_evolution.doc() = R"pydoc(
Calculate :math:`exp(delta H) |psi0>` using Lanczos.

It turns out that the Lanczos algorithm is also good for calculating the matrix exponential
applied to the starting vector. Instead of diagonalizing the tri-diagonal `h` and taking the
ground state, we now calculate ``exp(delta h) e_0`` in the Krylov ONB, where
``e_0 = (1, 0, 0, ...)`` corresponds to ``psi0`` in the original basis.

Parameters
----------
H, psi0, options :
    Hamiltonian, starting vector and parameters as defined in :class:`LanczosGroundState`.
    The option :cfg:option`LanczosEvolution.P_tol` defines when convergence is reached,
    see :meth:`_converged` for details.

Options
-------
.. cfg:config :: LanczosEvolution
    :include: LanczosGroundState

    E_tol :
        Ignored.
    min_gap :
        Ignored.

Attributes
----------
delta : float/complex
    Prefactor of H in the exponential.
_result_norm : float
    Norm of the resulting vector.
)pydoc";

    lanczos_evolution
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("_result_norm", &LanczosEvolution::_result_norm)
      .def_property(
        "delta",
        [](LanczosEvolution const& self) { return optional_complex_to_py(self.delta); },
        [](LanczosEvolution& self, py::object v) {
            if (v.is_none()) {
                self.delta = std::nullopt;
            } else {
                self.delta = v.cast<complex128>();
            }
        })
      .def("run",
           &LanczosEvolution::run,
           py::arg("delta"),
           py::arg("normalize") = py::none(),
           R"pydoc(Calculate ``expm(delta H).dot(psi0)`` using Lanczos.

Parameters
----------
delta : float/complex
    Time step by which we should evolve psi0: prefactor of H in the exponential.
    Note that the complex `i` is *not* included!
normalize : bool
    Whether to normalize the resulting state.
    Defaults to ``np.real(delta) == 0``.

Returns
-------
psi_f : :class:`~cyten.tensors.Tensor`
    Best approximation for ``expm(delta H).dot(psi0)``.
    If :cfg:option:`Lanczos.E_shift` is used, it's an approximation for
    ``expm(delta (H + E_shift)).dot(psi)``.
N : int
    Krylov space dimension used.
)pydoc");

    m.def("lanczos",
          &lanczos,
          py::arg("H"),
          py::arg("psi"),
          py::arg("options") = py::none(),
          R"pydoc(Simple wrapper calling ``LanczosGroundState(H, psi, options).run()``

Parameters
----------
H, psi, options:
    See :class:`LanczosGroundState`.

Returns
-------
E0, psi0, N :
    See :meth:`LanczosGroundState.run`.
)pydoc");
}

} // namespace cyten
