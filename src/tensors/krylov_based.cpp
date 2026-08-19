#include <cyten/tensors/krylov_based.h>

#include <cyten/block_backend/dtypes.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tools.h>

#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cyten {

namespace {

py::module_
numpy_mod()
{
    return py::module_::import("numpy");
}

py::object
krylov_logger()
{
    return py::module_::import("logging").attr("getLogger")("cyten.tensors.krylov_based");
}

py::dict
parse_options(py::object options)
{
    if (options.is_none()) {
        return py::dict();
    }
    if (py::isinstance<py::dict>(options)) {
        return options.cast<py::dict>();
    }
    return py::dict(options);
}

template<typename T>
T
dict_get(py::dict const& d, char const* key, T const& def)
{
    if (d.contains(key) && !d[key].is_none()) {
        return d[key].cast<T>();
    }
    return def;
}

float64
abs_number(BlockBackend::Scalar const& s)
{
    return std::abs(s.as_complex128());
}

complex128
to_number(BlockBackend::Scalar const& s)
{
    return s.as_complex128();
}

BlockBackend::Scalar
as_scalar(VectorLike const& v, complex128 z)
{
    auto dt = v.vector_dtype();
    if (z.imag() != 0.0 && dtype::is_real(dt)) {
        dt = dtype::to_complex(dt);
    }
    return v.vector_backend()->block_backend->as_scalar(z, dt);
}

BlockBackend::Scalar
as_scalar(VectorLike const& v, float64 x)
{
    return v.vector_backend()->block_backend->as_scalar(x);
}

VectorLike::Ptr
scaled_num(VectorLike::CPtr v, complex128 z)
{
    return v->scaled(as_scalar(*v, z));
}

VectorLike::Ptr
add_vec(VectorLike::Ptr a, VectorLike::CPtr b)
{
    return a->axpy(as_scalar(*a, 1.0), std::move(b));
}

VectorLike::Ptr
sub_vec(VectorLike::Ptr a, VectorLike::CPtr b)
{
    return b->axpy(as_scalar(*b, -1.0), std::move(a));
}

std::size_t
idx2(int64 i, int64 j, int64 cols)
{
    return static_cast<std::size_t>(i * cols + j);
}

std::vector<complex128>
py_array_to_complex(py::object obj)
{
    py::array_t<complex128, py::array::c_style | py::array::forcecast> arr(obj);
    auto req = arr.request();
    auto n = static_cast<std::size_t>(req.size);
    auto* ptr = static_cast<complex128*>(req.ptr);
    return { ptr, ptr + n };
}

py::array
vector_to_numpy_2d(std::vector<complex128> const& data, int64 rows, int64 cols, Dtype dt)
{
    if (dt == Dtype::Float64) {
        py::array_t<float64> arr({ rows, cols });
        auto r = arr.mutable_unchecked<2>();
        for (int64 i = 0; i < rows; ++i) {
            for (int64 j = 0; j < cols; ++j) {
                r(i, j) = data[idx2(i, j, cols)].real();
            }
        }
        return arr;
    }
    py::array_t<complex128> arr({ rows, cols });
    auto r = arr.mutable_unchecked<2>();
    for (int64 i = 0; i < rows; ++i) {
        for (int64 j = 0; j < cols; ++j) {
            r(i, j) = data[idx2(i, j, cols)];
        }
    }
    return arr;
}

py::array
submatrix_numpy(std::vector<complex128> const& data, int64 stride, int64 n, bool as_real)
{
    if (as_real) {
        py::array_t<float64> arr({ n, n });
        auto r = arr.mutable_unchecked<2>();
        for (int64 i = 0; i < n; ++i) {
            for (int64 j = 0; j < n; ++j) {
                r(i, j) = data[idx2(i, j, stride)].real();
            }
        }
        return arr;
    }
    py::array_t<complex128> arr({ n, n });
    auto r = arr.mutable_unchecked<2>();
    for (int64 i = 0; i < n; ++i) {
        for (int64 j = 0; j < n; ++j) {
            r(i, j) = data[idx2(i, j, stride)];
        }
    }
    return arr;
}

std::vector<int64>
argsort_which(std::vector<complex128> const& values, std::string const& which)
{
    std::vector<int64> idx(static_cast<std::size_t>(values.size()));
    std::iota(idx.begin(), idx.end(), int64(0));
    auto key = [&](complex128 z) -> float64 {
        if (which == "LM" || which == "m>") {
            return -std::abs(z);
        }
        if (which == "SM" || which == "m<") {
            return std::abs(z);
        }
        if (which == "LR" || which == ">" || which == "LA") {
            return -z.real();
        }
        if (which == "SR" || which == "<" || which == "SA") {
            return z.real();
        }
        if (which == "LI") {
            return -z.imag();
        }
        if (which == "SI") {
            return z.imag();
        }
        return z.real();
    };
    std::ranges::sort(idx, [&](int64 a, int64 b) {
        return key(values[static_cast<std::size_t>(a)]) < key(values[static_cast<std::size_t>(b)]);
    });
    return idx;
}

} // namespace

complex128
KrylovBased::h_krylov(int64 i, int64 j) const
{
    return _h_krylov[idx2(i, j, h_stride())];
}

void
KrylovBased::set_h_krylov(int64 i, int64 j, complex128 value)
{
    _h_krylov[idx2(i, j, h_stride())] = value;
}

complex128
KrylovBased::Es_at(int64 i, int64 j) const
{
    return Es[idx2(i, j, N_max)];
}

void
KrylovBased::set_Es(int64 i, int64 j, complex128 value)
{
    Es[idx2(i, j, N_max)] = value;
}

py::array
KrylovBased::Es_numpy() const
{
    return vector_to_numpy_2d(Es, N_max, N_max, _dtype_E);
}

py::array
KrylovBased::h_krylov_numpy() const
{
    auto n = h_stride();
    return vector_to_numpy_2d(_h_krylov, n, n, _dtype_h_krylov);
}

py::array
KrylovBased::result_krylov_numpy() const
{
    if (_result_krylov_cols <= 1) {
        if (_dtype_h_krylov == Dtype::Float64) {
            py::array_t<float64> arr(_result_krylov_rows);
            auto r = arr.mutable_unchecked<1>();
            for (int64 i = 0; i < _result_krylov_rows; ++i) {
                r(i) = _result_krylov[static_cast<std::size_t>(i)].real();
            }
            return arr;
        }
        py::array_t<complex128> arr(_result_krylov_rows);
        auto r = arr.mutable_unchecked<1>();
        for (int64 i = 0; i < _result_krylov_rows; ++i) {
            r(i) = _result_krylov[static_cast<std::size_t>(i)];
        }
        return arr;
    }
    return vector_to_numpy_2d(
      _result_krylov, _result_krylov_rows, _result_krylov_cols, _dtype_h_krylov);
}

KrylovBased::KrylovBased(LinearOperator::Ptr H_,
                         VectorLike::Ptr psi0_,
                         py::object options_,
                         Dtype dtype_h_krylov,
                         Dtype dtype_E)
  : H(std::move(H_))
  , psi0(psi0_ ? psi0_->clone() : nullptr)
  , _dtype_h_krylov(dtype_h_krylov)
  , _dtype_E(dtype_E)
{
    if (!H) {
        throw std::invalid_argument("H must not be null");
    }
    if (!psi0) {
        throw std::invalid_argument("psi0 must not be null");
    }
    options = parse_options(std::move(options_));
    N_min = dict_get<int64>(options, "N_min", int64(2));
    N_max = dict_get<int64>(options, "N_max", int64(20));
    N_cache = N_max;
    P_tol = dict_get<float64>(options, "P_tol", 1.0e-14);
    min_gap = dict_get<float64>(options, "min_gap", 1.0e-12);
    reortho = dict_get<bool>(options, "reortho", false);
    if (options.contains("E_shift") && !options["E_shift"].is_none()) {
        E_shift = options["E_shift"].cast<float64>();
    }
    if (N_min < 2) {
        throw std::invalid_argument("Should perform at least 2 steps.");
    }
    _cutoff = dict_get<float64>(options, "cutoff", dtype::eps(psi0->vector_dtype()) * 100);
    if (E_shift.has_value()) {
        if (auto proj = std::dynamic_pointer_cast<ProjectedLinearOperator>(H)) {
            proj->original_operator =
              std::make_shared<ShiftedLinearOperator>(proj->original_operator, *E_shift);
        } else {
            H = std::make_shared<ShiftedLinearOperator>(H, *E_shift);
        }
    }
    Es.assign(static_cast<std::size_t>(N_max * N_max), complex128(0.));
    _h_krylov.assign(static_cast<std::size_t>((N_max + 1) * (N_max + 1)), complex128(0.));
}

void
KrylovBased::_reset_krylov_state()
{
    _cache.clear();
    std::fill(_h_krylov.begin(), _h_krylov.end(), complex128(0.));
    std::fill(Es.begin(), Es.end(), complex128(0.));
}

VectorLike::Ptr
KrylovBased::_calc_result_full(int64 N)
{
    // this implementation assumes there is a single state
    auto const& vf = _result_krylov;
    auto const len_vf = static_cast<int64>(vf.size());
    if (!(N == len_vf && len_vf > 1)) {
        throw std::runtime_error("KrylovBased._calc_result_full: expected N == len(vf) > 1");
    }
    auto psif = scaled_num(psi0, vf[0]);
    auto const len_cache = static_cast<int64>(_cache.size());
    auto const n_loop = std::min(len_cache + 1, N);
    for (int64 k = 1; k < n_loop; ++k) {
        psif = add_vec(std::move(psif),
                       scaled_num(_cache[static_cast<std::size_t>(len_cache - k)],
                                  vf[static_cast<std::size_t>(N - k)]));
    }
    // other vectors are not cached, so we need to restart the Lanczos iteration.
    _cache.clear(); // free memory: we need at least two more vectors

    psif = _rebuild_krylov_for_result_full(std::move(psif), N - len_cache - 1);

    auto psif_norm = abs_number(norm(VectorLikeCPtr(psif)));
    if (std::abs(1.0 - psif_norm) > 1.0e-5) {
        // One reason can be that `H` is not Hermitian
        // Otherwise, the matrix (even if small) might be ill conditioned.
        // If you get this warning, you can try to set the parameters
        // `reortho`=True and `N_cache` >= `N_max`
        krylov_logger().attr("warning")("poorly conditioned H matrix in KrylovBased! |psi_0| = %f",
                                        psif_norm);
    }
    return scalar_multiply(as_scalar(*psif, 1.0 / psif_norm), psif);
}

void
KrylovBased::_to_cache(VectorLike::Ptr psi)
{
    _cache.push_back(std::move(psi));
    if (static_cast<int64>(_cache.size()) > N_cache) {
        _cache.erase(_cache.begin()); // remove *first* entry
    }
}

VectorLike::Ptr
KrylovBased::_rebuild_krylov_for_result_full(VectorLike::Ptr /*psif*/, int64 /*N_max*/)
{
    throw NotImplemented("KrylovBased._rebuild_krylov_for_result_full");
}

GMRES::GMRES(LinearOperator::Ptr A_, VectorLike::Ptr x_, VectorLike::Ptr b_, py::object options_)
  : A(std::move(A_))
  , b(b_ ? b_->clone() : nullptr)
  , x(x_ ? x_->clone() : nullptr)
{
    if (!A) {
        throw std::invalid_argument("A must not be null");
    }
    if (!x || !b) {
        throw std::invalid_argument("x and b must not be null");
    }
    options = parse_options(std::move(options_));
    N_min = dict_get<int64>(options, "N_min", int64(5));
    N_max = dict_get<int64>(options, "N_max", int64(20));
    restart = dict_get<int64>(options, "restart", int64(10));
    res = dict_get<float64>(options, "res", 1.0e-8);

    auto r0 = sub_vec(b->clone(), A->matvec(x));
    rs = { r0 };
    b_norm = abs_number(norm(VectorLikeCPtr(b)));
    r_norm = abs_number(norm(VectorLikeCPtr(r0)));
    auto denom = b_norm != 0.0 ? b_norm : 1.0;
    total_error = { { r_norm / denom } };
    if (r_norm > 0) {
        qs = { scalar_multiply(as_scalar(*r0, 1.0 / r_norm), r0) };
    } else {
        qs = { r0 };
    }
    _init_hessenberg();
}

void
GMRES::_init_hessenberg()
{
    sine.assign(static_cast<std::size_t>(N_max), complex128(0.));
    cosine.assign(static_cast<std::size_t>(N_max), complex128(0.));
    e1.assign(static_cast<std::size_t>(N_max + 1), complex128(0.));
    e1[0] = r_norm;
    H.assign(static_cast<std::size_t>((N_max + 1) * N_max), complex128(0.));
}

py::array
GMRES::H_numpy() const
{
    return vector_to_numpy_2d(H, N_max + 1, N_max, Dtype::Complex128);
}

std::tuple<VectorLike::Ptr, float64, std::vector<std::vector<float64>>, std::vector<int64>>
GMRES::run()
{
    if (total_error[0][0] < res) {
        return { x, total_error[0][0], total_error, total_iters };
    }
    for (int64 /*cycle*/ _ = 0; _ < restart; ++_) {
        bool converged = false;
        int64 k = 0;
        int64 performed = 0;
        for (; k < N_max; ++k) {
            arnoldi(k);
            apply_givens_rotation(k);
            e1[static_cast<std::size_t>(k + 1)] =
              -sine[static_cast<std::size_t>(k)] * e1[static_cast<std::size_t>(k)];
            e1[static_cast<std::size_t>(k)] =
              cosine[static_cast<std::size_t>(k)] * e1[static_cast<std::size_t>(k)];
            // The residual is the last element of the beta vector (see Wikipedia).
            auto denom = b_norm != 0.0 ? b_norm : 1.0;
            auto error = std::abs(e1[static_cast<std::size_t>(k + 1)]) / denom;
            total_error.back().push_back(error);
            performed = k + 1;
            if (error < res && k >= N_min) {
                converged = true;
                break;
            }
        }
        total_iters.push_back(performed);
        backsolve(performed);
        for (int64 i = 0; i < performed; ++i) {
            x =
              add_vec(std::move(x),
                      scaled_num(qs[static_cast<std::size_t>(i)], y[static_cast<std::size_t>(i)]));
        }
        if (!converged) {
            reset();
        } else {
            break;
        }
    }

    auto rel = abs_number(norm(VectorLikeCPtr(sub_vec(A->matvec(x), b))));
    if (b_norm != 0.0) {
        rel = rel / b_norm;
    }
    return { x, rel, total_error, total_iters };
}

void
GMRES::arnoldi(int64 k)
{
    // Iterative build orthogonal Krylov subspace and Hessenberg matrix.
    auto q = A->matvec(qs.back());
    for (int64 i = 0; i < k + 1; ++i) {
        auto hik = to_number(inner(VectorLikeCPtr(qs[static_cast<std::size_t>(i)]), q));
        H[idx2(i, k, N_max)] = hik;
        q = qs[static_cast<std::size_t>(i)]->axpy(as_scalar(*q, -hik), q);
    }
    auto h_next = abs_number(norm(VectorLikeCPtr(q)));
    H[idx2(k + 1, k, N_max)] = h_next;
    if (h_next > 0) { // avoid warning if norm(q)==0, error=0 in that case
        q = scalar_multiply(as_scalar(*q, 1.0 / h_next), q);
    }
    qs.push_back(std::move(q));
}

void
GMRES::apply_givens_rotation(int64 k)
{
    // Apply rotation to H so that it becomes upper triangular.
    for (int64 i = 0; i < k; ++i) {
        auto temp = cosine[static_cast<std::size_t>(i)] * H[idx2(i, k, N_max)] +
                    sine[static_cast<std::size_t>(i)] * H[idx2(i + 1, k, N_max)];
        H[idx2(i + 1, k, N_max)] = -sine[static_cast<std::size_t>(i)] * H[idx2(i, k, N_max)] +
                                   cosine[static_cast<std::size_t>(i)] * H[idx2(i + 1, k, N_max)];
        H[idx2(i, k, N_max)] = temp;
    }

    givens_rotation(k);
    H[idx2(k, k, N_max)] = cosine[static_cast<std::size_t>(k)] * H[idx2(k, k, N_max)] +
                           sine[static_cast<std::size_t>(k)] * H[idx2(k + 1, k, N_max)];
    H[idx2(k + 1, k, N_max)] = 0;
}

void
GMRES::givens_rotation(int64 k)
{
    // Find cosine and sine such that the element below the diagonal of kth column of H is removed.
    auto v1 = H[idx2(k, k, N_max)];
    auto v2 = H[idx2(k + 1, k, N_max)];
    auto t = std::sqrt(v1 * v1 + v2 * v2);
    cosine[static_cast<std::size_t>(k)] = v1 / t;
    sine[static_cast<std::size_t>(k)] = v2 / t;
}

void
GMRES::backsolve(int64 k)
{
    // H is now upper triangular; backsolve to find y exactly.
    y.assign(static_cast<std::size_t>(k), complex128(0.));
    for (int64 i = k - 1; i >= 0; --i) {
        y[static_cast<std::size_t>(i)] = e1[static_cast<std::size_t>(i)];
        for (int64 j = i + 1; j < k; ++j) {
            y[static_cast<std::size_t>(i)] -=
              H[idx2(i, j, N_max)] * y[static_cast<std::size_t>(j)];
        }
        y[static_cast<std::size_t>(i)] /= H[idx2(i, i, N_max)];
    }
}

void
GMRES::reset()
{
    // Restart GMRES using current x as initial guess.
    auto r = sub_vec(b->clone(), A->matvec(x));
    rs.push_back(r);
    r_norm = abs_number(norm(VectorLikeCPtr(r)));
    auto denom = b_norm != 0.0 ? b_norm : 1.0;
    total_error.push_back({ r_norm / denom });
    if (r_norm > 0) {
        qs = { scalar_multiply(as_scalar(*r, 1.0 / r_norm), r) };
    } else {
        qs = { std::move(r) };
    }
    _init_hessenberg();
}

Arnoldi::Arnoldi(LinearOperator::Ptr H_, VectorLike::Ptr psi0_, py::object options_)
  : KrylovBased(std::move(H_), std::move(psi0_), options_)
{
    E_tol = dict_get<float64>(options, "E_tol", std::numeric_limits<float64>::infinity());
    which = dict_get<std::string>(options, "which", std::string("LM"));
    num_ev = dict_get<int64>(options, "num_ev", int64(1));
}

std::tuple<std::vector<complex128>, std::vector<VectorLike::Ptr>, int64>
Arnoldi::run()
{
    if (N_cache < N_max) {
        throw std::runtime_error("Arnoldi requires N_cache >= N_max");
    }
    auto N = _build_krylov();
    std::vector<complex128> E0(static_cast<std::size_t>(num_ev));
    for (int64 i = 0; i < num_ev; ++i) {
        E0[static_cast<std::size_t>(i)] = Es_at(N - 1, i);
    }
    if (E_shift.has_value()) {
        for (auto& e : E0) {
            e -= *E_shift;
        }
    }
    if (N == 1) {
        return { std::move(E0), { psi0 }, N };
    }
    return { std::move(E0), _calc_result_full_multi(N), N };
}

int64
Arnoldi::_build_krylov()
{
    auto w = psi0;
    auto w_norm = abs_number(norm(VectorLikeCPtr(w)));
    psi0 = scaled_num(w, 1.0 / w_norm);
    int64 k = 0;
    int64 performed = 0;
    for (; k < N_max; ++k) {
        w = scalar_multiply(as_scalar(*w, 1.0 / w_norm), w);
        _to_cache(w);
        w = H->matvec(w);
        for (int64 i = 0; i < static_cast<int64>(_cache.size()); ++i) {
            auto ov = inner(VectorLikeCPtr(_cache[static_cast<std::size_t>(i)]), w);
            set_h_krylov(i, k, to_number(ov));
            w = _cache[static_cast<std::size_t>(i)]->axpy(-ov, w);
        }
        w_norm = abs_number(norm(VectorLikeCPtr(w)));
        set_h_krylov(k + 1, k, w_norm);
        _calc_result_krylov(k);
        performed = k + 1;
        if (w_norm < _cutoff || (k + 1 >= N_min && _converged(k))) {
            break;
        }
    }
    return performed;
}

void
Arnoldi::_calc_result_krylov(int64 k)
{
    if (k == 0) {
        set_Es(0, 0, h_krylov(0, 0));
        _result_krylov = { 1. };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto np = numpy_mod();
    auto h = submatrix_numpy(_h_krylov, h_stride(), n, /*as_real=*/false);
    py::tuple ev = np.attr("linalg").attr("eig")(h);
    auto E_kr = py_array_to_complex(ev[0]);
    auto v_kr = py_array_to_complex(ev[1]);
    auto sort = argsort_which(E_kr, which);
    for (int64 j = 0; j < n; ++j) {
        set_Es(k, j, E_kr[static_cast<std::size_t>(sort[static_cast<std::size_t>(j)])]);
    }
    _result_krylov.assign(static_cast<std::size_t>(n * n), complex128(0.));
    _result_krylov_rows = n;
    _result_krylov_cols = n;
    for (int64 col = 0; col < n; ++col) {
        auto src = sort[static_cast<std::size_t>(col)];
        for (int64 row = 0; row < n; ++row) {
            _result_krylov[idx2(row, col, n)] = v_kr[idx2(row, src, n)];
        }
    }
}

std::vector<VectorLike::Ptr>
Arnoldi::_calc_result_full_multi(int64 N)
{
    std::vector<VectorLike::Ptr> psis;
    auto np = numpy_mod();
    auto n_ev = std::min(N, num_ev);
    psis.reserve(static_cast<std::size_t>(n_ev));
    for (int64 i = 0; i < n_ev; ++i) {
        py::array_t<complex128> vf_arr(N);
        {
            auto r = vf_arr.mutable_unchecked<1>();
            for (int64 row = 0; row < N; ++row) {
                r(row) = _result_krylov[idx2(row, i, _result_krylov_cols)];
            }
        }
        // try to convert to real:
        // e.g. the dominant eigenvectors of the MPS transfermatrix should be equivalent to
        // the power method, which will be purely real for H.dtype=float, even if there might
        // be other eigenvectors which are complex
        auto vf = py_array_to_complex(np.attr("real_if_close")(vf_arr));
        if (!(N == static_cast<int64>(vf.size()) && N > 1)) {
            throw std::runtime_error("Arnoldi._calc_result_full: expected N == len(vf) > 1");
        }
        auto const& krylov_basis = _cache;
        if (static_cast<int64>(krylov_basis.size()) < N) {
            throw std::runtime_error("Arnoldi._calc_result_full: Krylov basis shorter than N");
        }
        auto psi = scaled_num(krylov_basis[0], vf[0]);
        for (int64 k = 1; k < N; ++k) {
            psi = add_vec(std::move(psi),
                          scaled_num(krylov_basis[static_cast<std::size_t>(k)],
                                     vf[static_cast<std::size_t>(k)]));
        }

        auto psi_norm = abs_number(norm(VectorLikeCPtr(psi)));
        if (std::abs(1.0 - psi_norm) > 1.0e-5) {
            // One reason can be that `H` is not Hermitian
            // Otherwise, the matrix (even if small) might be ill conditioned.
            // If you get this warning, you can try to set the parameters
            // `reortho`=True and `N_cache` >= `N_max`
            krylov_logger().attr("warning")("poorly conditioned H matrix in Arnoldi! |psi| = %f",
                                            psi_norm);
        }
        psis.push_back(scalar_multiply(as_scalar(*psi, 1.0 / psi_norm), psi));
    }
    return psis;
}

void
Arnoldi::_to_cache(VectorLike::Ptr psi)
{
    _cache.push_back(std::move(psi));
    if (static_cast<int64>(_cache.size()) > N_cache) {
        throw std::runtime_error("Arnoldi cache exceeded N_cache");
    }
}

bool
Arnoldi::_converged(int64 k)
{
    auto v0k = _result_krylov[idx2(k, 0, _result_krylov_cols)];
    auto RitzRes = std::abs(v0k) * std::abs(h_krylov(k + 1, k));
    float64 min_diff = std::numeric_limits<float64>::infinity();
    for (int64 i = 0; i < num_ev; ++i) {
        float64 local = std::numeric_limits<float64>::infinity();
        for (int64 j = i + 1; j < N_max; ++j) {
            local = std::min(local, std::abs(Es_at(k, j) - Es_at(k, i)));
        }
        min_diff = std::min(min_diff, local);
    }
    auto gap = std::max(min_diff, min_gap);
    auto P_err = (RitzRes / gap) * (RitzRes / gap);
    auto Delta_E0 = Es_at(k - 1, 0) - Es_at(k, 0);
    return P_err < P_tol && Delta_E0.real() < E_tol;
}

ArnoldiEvolution::ArnoldiEvolution(LinearOperator::Ptr H_,
                                   VectorLike::Ptr psi0_,
                                   py::object options_)
  : Arnoldi(std::move(H_), psi0_, options_)
{
    _result_norm = 1.0;
    delta = std::nullopt;
    // Arnoldi._build_krylov does not set _psi0_norm; do it here.
    _psi0_norm = abs_number(norm(VectorLikeCPtr(psi0_)));
}

std::tuple<VectorLike::Ptr, int64>
ArnoldiEvolution::run(complex128 delta_, std::optional<bool> normalize)
{
    if (N_cache < N_max) {
        throw std::runtime_error("ArnoldiEvolution requires N_cache >= N_max");
    }
    delta = delta_;
    // Arnoldi._to_cache does not pop old entries, so we must clear state between calls.
    _reset_krylov_state();
    auto N = _build_krylov();
    if (N > 1) {
        auto last = _result_krylov[idx2(N - 1, 0, _result_krylov_cols)];
        krylov_logger().attr("debug")(
          "ArnoldiEvolution N=%d, |result[-1]|=%.3e", N, std::abs(last));
    } else {
        krylov_logger().attr("debug")("ArnoldiEvolution N=1, |h[0,0]|=%.3e",
                                      std::abs(h_krylov(0, 0)));
    }
    VectorLike::Ptr result_full;
    if (N == 1) {
        result_full = scaled_num(psi0, _result_krylov[0]);
    } else {
        result_full = _calc_result_full_evolution(N);
    }
    bool do_normalize = normalize.value_or(false);
    if (do_normalize) {
        return { std::move(result_full), N };
    }
    auto scale = (_psi0_norm.value_or(1.0)) * _result_norm;
    return { scaled_num(result_full, scale), N };
}

void
ArnoldiEvolution::_calc_result_krylov(int64 k)
{
    auto np = numpy_mod();
    auto dlt = *delta;
    if (k == 0) {
        auto exp_dE = std::exp(dlt * h_krylov(0, 0));
        _result_norm = std::abs(exp_dE);
        _result_krylov = { exp_dE / _result_norm };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto h = submatrix_numpy(_h_krylov, h_stride(), n, /*as_real=*/false);
    py::tuple ev = np.attr("linalg").attr("eig")(h);
    py::object v_kr = ev[1];
    py::object E_kr = ev[0];
    // V^{-1} e0 = first column of V^{-1}; use solve for numerical stability
    py::array_t<complex128> e0(n);
    {
        auto r = e0.mutable_unchecked<1>();
        r(0) = 1.0;
        for (int64 i = 1; i < n; ++i) {
            r(i) = 0.0;
        }
    }
    auto coeff = np.attr("linalg").attr("solve")(v_kr, e0);
    auto exp_dH_e0 = np.attr("dot")(v_kr, np.attr("exp")(E_kr * py::cast(dlt)) * coeff);
    _result_norm = np.attr("linalg").attr("norm")(exp_dH_e0).cast<float64>();
    auto vf = py_array_to_complex(exp_dH_e0 / py::cast(_result_norm));
    _result_krylov = std::move(vf);
    _result_krylov_rows = n;
    _result_krylov_cols = 1;
}

bool
ArnoldiEvolution::_converged(int64 k)
{
    return std::abs(_result_krylov[static_cast<std::size_t>(k)]) < P_tol;
}

VectorLike::Ptr
ArnoldiEvolution::_calc_result_full_evolution(int64 N)
{
    auto const& cache = _cache;
    if (static_cast<int64>(cache.size()) < N) {
        throw std::runtime_error("ArnoldiEvolution: Krylov basis shorter than N");
    }
    auto psif = scaled_num(cache[0], _result_krylov[0]);
    for (int64 k = 1; k < N; ++k) {
        psif = add_vec(std::move(psif),
                       scaled_num(cache[static_cast<std::size_t>(k)],
                                  _result_krylov[static_cast<std::size_t>(k)]));
    }
    auto psif_norm = abs_number(norm(VectorLikeCPtr(psif)));
    if (std::abs(1.0 - psif_norm) > 1.0e-5) {
        krylov_logger().attr("warning")("poorly conditioned H in ArnoldiEvolution! |psi|=%f",
                                        psif_norm);
    }
    return scalar_multiply(as_scalar(*psif, 1.0 / psif_norm), psif);
}

LanczosGroundState::LanczosGroundState(LinearOperator::Ptr H_,
                                       VectorLike::Ptr psi0_,
                                       py::object options_)
  : KrylovBased(std::move(H_), std::move(psi0_), options_, Dtype::Float64, Dtype::Float64)
{
    E_tol = dict_get<float64>(options, "E_tol", std::numeric_limits<float64>::infinity());
    N_cache = dict_get<int64>(options, "N_cache", N_max);
    if (N_cache < 2) {
        throw std::invalid_argument("Need to cache at least two vectors.");
    }
}

std::tuple<float64, VectorLike::Ptr, int64>
LanczosGroundState::run()
{
    auto N = _build_krylov();
    auto E0 = Es_at(N - 1, 0).real();
    if (N > 1) {
        krylov_logger().attr("debug")(
          "Lanczos N=%d, gap=%.3e, DeltaE0=%.3e, _result_krylov[-1]=%.3e",
          N,
          Es_at(N - 1, 1).real() - E0,
          Es_at(N - 2, 0).real() - E0,
          _result_krylov.back().real());
    } else {
        krylov_logger().attr("debug")("Lanczos N=%d, first alpha=%.3e, beta=%.3e",
                                      N,
                                      h_krylov(0, 0).real(),
                                      h_krylov(0, 1).real());
    }
    if (E_shift.has_value()) {
        E0 -= *E_shift;
    }
    if (N == 1) {
        return { E0, psi0, N };
    }
    return { E0, _calc_result_full(N), N };
}

int64
LanczosGroundState::_build_krylov()
{
    auto w = psi0;
    auto beta = abs_number(norm(VectorLikeCPtr(w)));
    if (beta < _cutoff) {
        throw std::invalid_argument(std::format("Norm of self.psi0 too small: {}", beta));
    }
    psi0 = scaled_num(w, 1.0 / beta);
    if (!_psi0_norm.has_value()) {
        // this is only needed for normalization in LanczosEvolution
        _psi0_norm = beta;
    }
    int64 k = 0;
    int64 performed = 0;
    for (; k < N_max; ++k) {
        w = scalar_multiply(as_scalar(*w, 1.0 / beta), w);
        _to_cache(w);
        w = H->matvec(w);
        auto alpha = inner(w, VectorLikeCPtr(_cache.back())).real().as_float64();
        set_h_krylov(k, k, alpha);
        _calc_result_krylov(k);
        w = _cache.back()->axpy(as_scalar(*w, -alpha), w);
        if (reortho) {
            for (std::size_t i = 0; i + 1 < _cache.size(); ++i) {
                auto ov = inner(VectorLikeCPtr(_cache[i]), w);
                w = _cache[i]->axpy(-ov, w);
            }
        } else if (k > 0) {
            w = _cache[_cache.size() - 2]->axpy(as_scalar(*w, -beta), w);
        }
        beta = abs_number(norm(VectorLikeCPtr(w)));
        set_h_krylov(k, k + 1, beta);
        set_h_krylov(k + 1, k, beta); // needed for the next step and convergence criteria
        performed = k + 1;
        if (std::abs(beta) < _cutoff || (k + 1 >= N_min && _converged(k))) {
            break;
        }
    }
    return performed;
}

bool
LanczosGroundState::_converged(int64 k)
{
    auto v0k = _result_krylov[static_cast<std::size_t>(k)];
    auto RitzRes = std::abs(v0k) * std::abs(h_krylov(k, k + 1));
    auto gap = std::max(Es_at(k, 1).real() - Es_at(k, 0).real(), min_gap);
    auto P_err = (RitzRes / gap) * (RitzRes / gap);
    auto Delta_E0 = Es_at(k - 1, 0).real() - Es_at(k, 0).real();
    return P_err < P_tol && Delta_E0 < E_tol;
}

VectorLike::Ptr
LanczosGroundState::_rebuild_krylov_for_result_full(VectorLike::Ptr psif, int64 N_max_rebuild)
{
    auto const& vf = _result_krylov;
    auto w = psi0;
    float64 beta = 0.;
    for (int64 k = 0; k < N_max_rebuild; ++k) {
        _to_cache(w);
        w = H->matvec(w);
        auto alpha = h_krylov(k, k).real();
        w = _cache.back()->axpy(as_scalar(*w, -alpha), w);
        if (reortho) {
            for (std::size_t i = 0; i + 1 < _cache.size(); ++i) {
                auto ov = inner(VectorLikeCPtr(_cache[i]), w);
                w = _cache[i]->axpy(-ov, w);
            }
        } else if (k > 0) {
            w = _cache[_cache.size() - 2]->axpy(as_scalar(*w, -beta), w);
        }
        beta = h_krylov(k, k + 1).real(); // = norm(w)
        w = scalar_multiply(as_scalar(*w, 1.0 / beta), w);
        psif = add_vec(std::move(psif), scaled_num(w, vf[static_cast<std::size_t>(k + 1)]));
    }
    return psif;
}

void
LanczosGroundState::_calc_result_krylov(int64 k)
{
    if (k == 0) {
        set_Es(0, 0, h_krylov(0, 0));
        _result_krylov = { 1. };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto np = numpy_mod();
    auto h = submatrix_numpy(_h_krylov, h_stride(), n, /*as_real=*/true);
    py::tuple ev = np.attr("linalg").attr("eigh")(h);
    auto E_kr = py_array_to_complex(ev[0]);
    auto v_kr = py_array_to_complex(ev[1]);
    for (int64 j = 0; j < n; ++j) {
        set_Es(k, j, E_kr[static_cast<std::size_t>(j)]);
    }
    _result_krylov.resize(static_cast<std::size_t>(n));
    _result_krylov_rows = n;
    _result_krylov_cols = 1;
    for (int64 row = 0; row < n; ++row) {
        _result_krylov[static_cast<std::size_t>(row)] = v_kr[idx2(row, 0, n)];
    }
}

LanczosEvolution::LanczosEvolution(LinearOperator::Ptr H_,
                                   VectorLike::Ptr psi0_,
                                   py::object options_)
  : LanczosGroundState(std::move(H_), std::move(psi0_), std::move(options_))
{
    _result_norm = 1.0;
    delta = std::nullopt;
}

std::tuple<VectorLike::Ptr, int64>
LanczosEvolution::run(complex128 delta_, std::optional<bool> normalize)
{
    delta = delta_;
    _reset_krylov_state();
    auto N = _build_krylov();
    if (N > 1) {
        krylov_logger().attr("debug")(
          "Lanczos N=%d, |result_krylov[-1]|=%.3e", N, std::abs(_result_krylov.back()));
    } else {
        krylov_logger().attr("debug")("Lanczos N=%d, first alpha=%.3e, beta=%.3e",
                                      N,
                                      h_krylov(0, 0).real(),
                                      h_krylov(0, 1).real());
    }
    VectorLike::Ptr result_full;
    if (N == 1) {
        result_full = scaled_num(psi0, _result_krylov[0]); // _result_krylov[0] is only a phase
    } else {
        result_full = _calc_result_full(N);
    }
    // result_full is normalized at this point
    bool do_normalize = normalize.value_or(delta_.real() == 0.0);
    if (do_normalize) {
        return { std::move(result_full), N };
    }
    auto scale = (_psi0_norm.value_or(1.0)) * _result_norm;
    return { scaled_num(result_full, scale), N };
}

void
LanczosEvolution::_calc_result_krylov(int64 k)
{
    // self._result_krylov should be a normalized vector.
    auto dlt = *delta;
    if (k == 0) {
        auto E = h_krylov(0, 0);
        auto exp_dE = std::exp(dlt * E);
        _result_norm = std::abs(exp_dE);
        _result_krylov = { exp_dE / _result_norm };
        _result_krylov_rows = 1;
        _result_krylov_cols = 1;
        return;
    }
    auto n = k + 1;
    auto np = numpy_mod();
    auto h = submatrix_numpy(_h_krylov, h_stride(), n, /*as_real=*/true);
    py::tuple ev = np.attr("linalg").attr("eigh")(h);
    py::object E_kr = ev[0];
    py::object v_kr = ev[1];
    auto exp_dH_e0 = np.attr("dot")(
      v_kr, np.attr("exp")(E_kr * py::cast(dlt)) * np.attr("conj")(v_kr[py::int_(0)]));
    _result_norm = np.attr("linalg").attr("norm")(exp_dH_e0).cast<float64>();
    _result_krylov = py_array_to_complex(exp_dH_e0 / py::cast(_result_norm));
    _result_krylov_rows = n;
    _result_krylov_cols = 1;
}

bool
LanczosEvolution::_converged(int64 k)
{
    return std::abs(_result_krylov[static_cast<std::size_t>(k)]) < P_tol;
}

std::tuple<float64, VectorLike::Ptr, int64>
lanczos(LinearOperator::Ptr H, VectorLike::Ptr psi, py::object options)
{
    return LanczosGroundState(std::move(H), std::move(psi), std::move(options)).run();
}

} // namespace cyten
