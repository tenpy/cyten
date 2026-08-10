#include <cyten/backends/tensor_backend.h>
#include <cyten/tools.h>

#include <format>
#include <sstream>
#include <stdexcept>

namespace cyten {

namespace {

py::array
combine_constraints_py(py::array good1, py::array good2, char const* warn)
{
    auto misc = py::module_::import("cyten.tools.misc");
    return misc.attr("combine_constraints")(good1, good2, warn).cast<py::array>();
}

std::string
backend_type_name(TensorBackend const& self)
{
    try {
        py::object py_self = py::cast(const_cast<TensorBackend*>(&self));
        return py::str(py::type::of(py_self).attr("__name__")).cast<std::string>();
    } catch (py::cast_error const&) {
        return "TensorBackend";
    } catch (py::error_already_set const&) {
        return "TensorBackend";
    }
}

bool
legs_equal(std::vector<Leg::Ptr> const& a, std::vector<Leg::Ptr> const& b)
{
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] == b[i])
            continue;
        if (!a[i] || !b[i] || !(*a[i] == *b[i]))
            return false;
    }
    return true;
}

} // namespace

TensorBackend::TensorBackend(std::shared_ptr<BlockBackend> block_backend_)
  : DataCls(py::none())
  , block_backend(std::move(block_backend_))
{
}

std::string
TensorBackend::__repr__() const
{
    std::ostringstream oss;
    oss << backend_type_name(*this) << '(';
    try {
        oss << py::repr(py::cast(block_backend)).cast<std::string>();
    } catch (...) {
        oss << "BlockBackend";
    }
    oss << ')';
    return oss.str();
}

std::string
TensorBackend::__str__() const
{
    return __repr__();
}

BlockBackend::Scalar
TensorBackend::item(py::object a)
{
    return data_item(a.attr("data").cast<DataPtr>());
}

void
TensorBackend::test_tensor_sanity(py::object a, bool /*is_diagonal*/)
{
    // --- hints from Python TensorBackend.test_tensor_sanity ---
    // subclasses will typically call super().test_tensor_sanity(a)
    // ---
    // subclasses will typically call super().test_tensor_sanity(a)
    py::object data = a.attr("data");
    if (!DataCls.is_none() && !py::isinstance(data, DataCls)) {
        throw std::runtime_error(std::format("expected data of type {}, got {}",
                                             py::str(DataCls).cast<std::string>(),
                                             py::str(py::type::of(data)).cast<std::string>()));
    }
}

void
TensorBackend::test_mask_sanity(py::object a)
{
    // --- hints from Python TensorBackend.test_mask_sanity ---
    // subclasses will typically call super().test_mask_sanity(a)
    // ---
    // subclasses will typically call super().test_mask_sanity(a)
    py::object data = a.attr("data");
    if (!DataCls.is_none() && !py::isinstance(data, DataCls)) {
        throw std::runtime_error(std::format("expected data of type {}, got {}",
                                             py::str(DataCls).cast<std::string>(),
                                             py::str(py::type::of(data)).cast<std::string>()));
    }
}

LegPipe::Ptr
TensorBackend::make_pipe(std::vector<Leg::Ptr> legs, bool is_dual, LegPipe::Ptr pipe)
{
    if (pipe) {
        assert(pipe->combine_cstyle == !is_dual);
        assert(pipe->is_dual == is_dual);
        assert(legs_equal(pipe->legs, legs));
        return pipe;
    }
    return std::make_shared<LegPipe>(std::move(legs), is_dual, /*combine_cstyle=*/!is_dual);
}

std::tuple<py::array, float64, float64>
TensorBackend::_truncate_singular_values_selection(py::array S,
                                                   py::object qdims,
                                                   std::optional<int64> chi_max,
                                                   int64 chi_min,
                                                   float64 degeneracy_tol,
                                                   float64 trunc_cut,
                                                   std::optional<float64> svd_min,
                                                   bool minimize_error)
{
    // --- hints from Python TensorBackend._truncate_singular_values_selection ---
    // qdims = qdims[piv]  # not needed again.
    // this is equivalent to
    // ``(S[cut] - S[cut-1])/S[cut-1] < exp(deg_tol) - 1 = deg_tol + O(deg_tol^2)``
    // keep only values S[i] >= svd_min
    // smallest cut for which good[cut] is True
    // largest cut for which good[cut] is True
    // ---
    // contributions ``err[i] = d[i] * S[i] ** 2`` to the error, if S[i] would be truncated.
    py::module_ np = py::module_::import("numpy");
    py::object S_obj = S;
    py::object marginal_errs;
    if (qdims.is_none()) {
        marginal_errs = S_obj.attr("__pow__")(2);
    } else {
        marginal_errs = qdims.attr("__mul__")(S_obj.attr("__pow__")(2));
    }

    // sort *ascending* by marginal errors (smallest first, should be truncated first)
    py::array piv = np.attr("argsort")(marginal_errs).cast<py::array>();
    S_obj = S_obj.attr("__getitem__")(piv);
    marginal_errs = marginal_errs.attr("__getitem__")(piv);

    // take safe logarithm, clipping small values to log(1e-100).
    // this is only used for degeneracy tol.
    py::object ones = np.attr("ones")(py::len(S_obj));
    py::object clipped = np.attr("choose")(S_obj.attr("__le__")(1.0e-100),
                                           py::make_tuple(S_obj, ones.attr("__mul__")(1.0e-100)));
    py::object logS = np.attr("log")(clipped);

    // goal: find an index 'cut' such that we keep piv[cut:], i.e. cut between `cut-1` and `cut`.
    // build an array good, where ``good[cut] = (is `cut` an allowed choice)``.
    // we then choose the smallest good cut, i.e. we keep as many singular values as possible
    py::ssize_t n = py::len(S_obj);
    py::array good = np.attr("ones")(n, py::arg("dtype") = np.attr("bool_")).cast<py::array>();

    if (chi_max.has_value() && *chi_max < n) {
        // keep at most chi_max values
        py::array good2 =
          np.attr("zeros")(n, py::arg("dtype") = np.attr("bool_")).cast<py::array>();
        good2.attr("__setitem__")(py::slice(-*chi_max, std::nullopt, std::nullopt), true);
        good = combine_constraints_py(good, good2, "chi_max");
    }

    if (chi_min > 1) {
        // keep at least chi_min values
        py::array good2 =
          np.attr("ones")(n, py::arg("dtype") = np.attr("bool_")).cast<py::array>();
        good2.attr("__setitem__")(py::slice(-chi_min + 1, std::nullopt, std::nullopt), false);
        good = combine_constraints_py(good, good2, "chi_min");
    }

    if (degeneracy_tol > 0) {
        // don't cut between values (cut-1, cut) with ``log(S[cut]/S[cut-1]) < deg_tol``
        py::array good2 = np.attr("empty")(n, np.attr("bool_")).cast<py::array>();
        good2.attr("__setitem__")(0, true);
        py::object dlog =
          logS.attr("__getitem__")(py::slice(1, std::nullopt, std::nullopt))
            .attr("__sub__")(logS.attr("__getitem__")(py::slice(std::nullopt, -1, std::nullopt)));
        good2.attr("__setitem__")(py::slice(1, std::nullopt, std::nullopt),
                                  np.attr("greater_equal")(dlog, degeneracy_tol));
        good = combine_constraints_py(good, good2, "degeneracy_tol");
    }

    if (svd_min.has_value()) {
        py::array good2 = np.attr("greater_equal")(S_obj, *svd_min).cast<py::array>();
        good = combine_constraints_py(good, good2, "svd_min");
    }

    {
        py::array good2 =
          np.attr("cumsum")(marginal_errs).attr("__gt__")(trunc_cut * trunc_cut).cast<py::array>();
        good = combine_constraints_py(good, good2, "trunc_cut");
    }

    py::array nonzero =
      np.attr("nonzero")(good).cast<py::tuple>().attr("__getitem__")(0).cast<py::array>();
    int64 cut;
    if (minimize_error) {
        cut = nonzero.attr("__getitem__")(0).cast<int64>(); // smallest cut
    } else {
        cut = nonzero.attr("__getitem__")(-1).cast<int64>(); // largest cut
    }
    float64 err =
      np.attr("sum")(marginal_errs.attr("__getitem__")(py::slice(std::nullopt, cut, std::nullopt)))
        .cast<float64>();
    float64 new_norm =
      np.attr("sum")(marginal_errs.attr("__getitem__")(py::slice(cut, std::nullopt, std::nullopt)))
        .cast<float64>();
    // build mask in the original order, before sorting
    py::array mask = np.attr("zeros")(n, py::arg("dtype") = np.attr("bool_")).cast<py::array>();
    np.attr("put")(
      mask, piv.attr("__getitem__")(py::slice(cut, std::nullopt, std::nullopt)), true);
    return { mask, err, new_norm };
}

bool
TensorBackend::is_real(py::object a)
{
    // --- hints from Python TensorBackend.is_real ---
    // FusionTree backend might implement this differently.
    // ---
    // FusionTree backend might implement this differently.
    return a.attr("dtype").attr("is_real").cast<bool>();
}

void
TensorBackend::save_hdf5(py::object hdf5_saver, py::object /*h5gr*/, std::string subpath)
{
    hdf5_saver.attr("save")(block_backend, subpath + "block_backend");
}

TensorBackend::Ptr
TensorBackend::from_hdf5(py::object /*hdf5_loader*/, py::object /*h5gr*/, std::string /*subpath*/)
{
    // Concrete backends construct the appropriate subclass; base cannot be instantiated.
    throw NotImplemented("TensorBackend::from_hdf5");
}

std::vector<py::object>
conventional_leg_order(TensorProduct::Ptr codomain, TensorProduct::Ptr domain)
{
    std::vector<py::object> out;
    out.reserve(codomain->factors.size() + domain->factors.size());
    for (auto const& f : codomain->factors)
        out.push_back(f);
    for (auto it = domain->factors.rbegin(); it != domain->factors.rend(); ++it)
        out.push_back(*it);
    return out;
}

std::vector<py::object>
conventional_leg_order(py::object tensor_or_codomain, py::object domain)
{
    TensorProduct::Ptr codomain_ptr;
    TensorProduct::Ptr domain_ptr;
    if (domain.is_none()) {
        codomain_ptr = tensor_or_codomain.attr("codomain").cast<TensorProduct::Ptr>();
        domain_ptr = tensor_or_codomain.attr("domain").cast<TensorProduct::Ptr>();
    } else {
        codomain_ptr = tensor_or_codomain.cast<TensorProduct::Ptr>();
        domain_ptr = domain.cast<TensorProduct::Ptr>();
    }
    return conventional_leg_order(codomain_ptr, domain_ptr);
}

TensorBackend::Ptr
get_same_backend(const std::vector<py::object>& objs, std::string error_msg)
{
    if (objs.empty())
        throw std::invalid_argument("Need at least one tensor");
    TensorBackend::Ptr backend = objs[0].attr("backend").cast<TensorBackend::Ptr>();
    for (std::size_t i = 1; i < objs.size(); ++i) {
        TensorBackend::Ptr other = objs[i].attr("backend").cast<TensorBackend::Ptr>();
        if (other.get() != backend.get())
            throw std::invalid_argument(std::move(error_msg));
    }
    return backend;
}

} // namespace cyten
