#include <cyten/tensors/mask.h>

#include <cyten/backends/abelian.h>
#include <cyten/backends/backend_factory.h>
#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/tools.h>
#include <cyten/warn.h>

#include <algorithm>
#include <cassert>
#include <format>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

namespace {

bool
is_mask_obj(py::handle obj)
{
    if (py::isinstance<Mask>(obj)) {
        return true;
    }
    try {
        return py::isinstance(obj, py::module_::import("cyten.tensors._tensors").attr("Mask"));
    } catch (py::error_already_set const&) {
        return false;
    }
}

Space::Ptr
as_space(py::object obj)
{
    return obj.cast<Space::Ptr>();
}

ElementarySpace::Ptr
as_elementary_space(py::object obj)
{
    if (py::isinstance<LegPipe>(obj)) {
        throw std::invalid_argument("Mask is not defined on LegPipes.");
    }
    if (!py::isinstance<ElementarySpace>(obj)) {
        throw std::invalid_argument("Expected ElementarySpace.");
    }
    return obj.cast<ElementarySpace::Ptr>();
}

BlockBackend::LegCPtr
as_leg_cptr(Space::Ptr const& space)
{
    auto es = std::dynamic_pointer_cast<ElementarySpace>(space);
    if (!es) {
        throw std::invalid_argument("Expected ElementarySpace for Mask leg.");
    }
    return std::static_pointer_cast<Leg const>(es);
}

/// Adapt a numpy-oriented bool function so backends can call it on Block objects.
py::function
adapt_block_bool_unary(py::function func, std::shared_ptr<BlockBackend> bb)
{
    return py::cpp_function([func, bb](py::object block_obj) {
        auto block = block_obj.cast<BlockBackend::BlockPtr>();
        auto arr = bb->to_numpy(block, py::module_::import("builtins").attr("bool"));
        auto out = func(arr);
        return bb->as_block(out, Dtype::Bool, block->device());
    });
}

py::function
adapt_block_bool_binary(py::function func, std::shared_ptr<BlockBackend> bb)
{
    return py::cpp_function([func, bb](py::object a_obj, py::object b_obj) {
        // ``b`` may be a bool scalar (unary-via-binary path) or a Block.
        if (py::isinstance<py::bool_>(b_obj) ||
            py::isinstance(b_obj, py::module_::import("numpy").attr("bool_"))) {
            auto a = a_obj.cast<BlockBackend::BlockPtr>();
            auto arr = bb->to_numpy(a, py::module_::import("builtins").attr("bool"));
            auto out = func(arr, b_obj);
            return bb->as_block(out, Dtype::Bool, a->device());
        }
        auto a = a_obj.cast<BlockBackend::BlockPtr>();
        auto b = b_obj.cast<BlockBackend::BlockPtr>();
        auto arr_a = bb->to_numpy(a, py::module_::import("builtins").attr("bool"));
        auto arr_b = bb->to_numpy(b, py::module_::import("builtins").attr("bool"));
        auto out = func(arr_a, arr_b);
        return bb->as_block(out, Dtype::Bool, a->device());
    });
}

int64
space_dim(Space const& space)
{
    return static_cast<int64>(space.Space::dim);
}

int64
sum_multiplicities(Space const& space)
{
    return std::accumulate(space.multiplicities.begin(), space.multiplicities.end(), int64{ 0 });
}

bool
basis_perm_trivial(ElementarySpace const& leg)
{
    if (!leg.has_custom_basis_perm()) {
        return true;
    }
    auto np = py::module_::import("numpy");
    auto perm = py::cast(leg.basis_perm());
    return py::bool_(np.attr("all")(perm.attr("__eq__")(np.attr("arange")(py::len(perm)))));
}

TensorBackend::Ptr
resolve_backend(TensorBackend::Ptr backend, Space::Ptr const& space)
{
    if (!backend) {
        return get_backend(py::cast(space->symmetry)).cast<TensorBackend::Ptr>();
    }
    return backend;
}

} // namespace

std::vector<Dtype> Mask::_forbidden_dtypes = {
    Dtype::Float32,
    Dtype::Float64,
    Dtype::Complex64,
    Dtype::Complex128,
};

Mask::Mask(TensorBackend::DataPtr data_in,
           py::object space_in_obj,
           py::object space_out_obj,
           std::optional<bool> is_projection_opt,
           TensorBackend::Ptr backend_in,
           py::object labels_obj)
  : Tensor(
      [&]() -> py::object {
          as_elementary_space(space_out_obj); // validate early
          return py::make_tuple(space_out_obj);
      }(),
      py::make_tuple(space_in_obj),
      std::move(backend_in),
      labels_obj,
      Dtype::Bool,
      "")
  , data(std::move(data_in))
{
    auto space_in = as_elementary_space(space_in_obj);
    auto space_out = as_elementary_space(space_out_obj);

    bool proj = false;
    if (!is_projection_opt.has_value()) {
        if (space_dim(*space_in) == space_dim(*space_out)) {
            throw std::invalid_argument("Need to specify is_projection for equal spaces.");
        }
        proj = space_dim(*space_in) > space_dim(*space_out);
    } else {
        proj = *is_projection_opt;
        if (proj) {
            assert(space_dim(*space_in) >= space_dim(*space_out));
        } else {
            assert(space_dim(*space_in) <= space_dim(*space_out));
        }
    }
    is_projection = proj;

    if (is_projection) {
        assert(space_out->is_subspace_of(*space_in));
    } else {
        assert(space_in->is_subspace_of(*space_out));
    }
    assert(space_out->is_dual == space_in->is_dual);

    dtype = Dtype::Bool;
    device = backend->get_device_from_data(data);
}

Mask::Mask(TensorBackend::DataPtr data_in,
           Space::Ptr space_in,
           Space::Ptr space_out,
           bool is_projection_in,
           TensorBackend::Ptr backend_in,
           Symmetry::Ptr symmetry_in,
           LegLabels labels_in,
           std::string device_in)
  : Tensor(std::make_shared<TensorProduct>(std::vector<py::object>{ py::cast(space_out) }),
           std::make_shared<TensorProduct>(std::vector<py::object>{ py::cast(space_in) }),
           std::move(backend_in),
           std::move(symmetry_in),
           std::move(labels_in),
           Dtype::Bool,
           std::move(device_in))
  , is_projection(is_projection_in)
  , data(std::move(data_in))
{
    if (py::isinstance<LegPipe>(py::cast(space_in)) ||
        py::isinstance<LegPipe>(py::cast(space_out))) {
        throw std::invalid_argument("Mask is not defined on LegPipes.");
    }
    if (!std::dynamic_pointer_cast<ElementarySpace>(space_in) ||
        !std::dynamic_pointer_cast<ElementarySpace>(space_out)) {
        throw std::invalid_argument("Expected ElementarySpace.");
    }
    if (is_projection) {
        assert(space_dim(*space_in) >= space_dim(*space_out));
        assert(space_out->is_subspace_of(*space_in));
    } else {
        assert(space_dim(*space_in) <= space_dim(*space_out));
        assert(space_in->is_subspace_of(*space_out));
    }
    assert(std::dynamic_pointer_cast<ElementarySpace>(space_out)->is_dual ==
           std::dynamic_pointer_cast<ElementarySpace>(space_in)->is_dual);
}

std::vector<Dtype> const&
Mask::forbidden_dtypes() const
{
    return _forbidden_dtypes;
}

std::string
Mask::ascii_diagram_type_name() const
{
    return "Mask";
}

std::string
Mask::class_name() const
{
    return "Mask";
}

py::object
Mask::as_py_object()
{
    return py::cast(std::static_pointer_cast<Mask>(shared_from_this()));
}

py::object
Mask::as_py_object() const
{
    return const_cast<Mask*>(this)->as_py_object();
}

ElementarySpace::Ptr
Mask::large_leg() const
{
    if (is_projection) {
        return domain->factors[0].cast<ElementarySpace::Ptr>();
    }
    return codomain->factors[0].cast<ElementarySpace::Ptr>();
}

ElementarySpace::Ptr
Mask::small_leg() const
{
    if (is_projection) {
        return codomain->factors[0].cast<ElementarySpace::Ptr>();
    }
    return domain->factors[0].cast<ElementarySpace::Ptr>();
}

void
Mask::test_sanity() const
{
    // --- hints from Python Mask.test_sanity ---
    // check consistency of the basis perm of the small leg.
    // this is consistent.
    // check if ranks is sorted
    // ---
    Tensor::test_sanity();
    backend->test_mask_sanity(as_py_object());
    assert(codomain->num_factors == 1 && domain->num_factors == 1);
    assert(py::isinstance<ElementarySpace>(codomain->factors[0]));
    assert(py::isinstance<ElementarySpace>(domain->factors[0]));
    auto large = large_leg();
    auto small = small_leg();
    assert(large->is_dual == small->is_dual);
    assert(small->is_subspace_of(*large));
    assert(dtype == Dtype::Bool);
    assert(device == backend->get_device_from_data(data));

    // check consistency of the basis perm of the small leg.
    if (!large->has_custom_basis_perm()) {
        if (!small->has_custom_basis_perm()) {
            // consistent
        } else {
            auto np = py::module_::import("numpy");
            auto expected = np.attr("arange")(space_dim(*small));
            auto actual = np.attr("asarray")(py::cast(small->basis_perm()));
            if (!np.attr("array_equal")(actual, expected).cast<bool>()) {
                throw std::logic_error(
                  "Mask.test_sanity: small_leg.basis_perm inconsistent with trivial large_leg");
            }
        }
    } else {
        auto np = py::module_::import("numpy");
        auto mask_in_internal_basis = backend->block_backend->to_numpy(
          backend->mask_to_block(as_py_object()), py::module_::import("builtins").attr("bool"));
        // Use Leg Python properties (perm_to_numpy) so empty perms stay integer dtype.
        // np.asarray([]) defaults to float64 and breaks advanced indexing.
        auto large_py = py::cast(large);
        auto small_py = py::cast(small);
        auto pi_1 = large_py.attr("basis_perm");
        auto pi_2_inv = small_py.attr("inverse_basis_perm");
        auto ranks =
          pi_1.attr("__getitem__")(mask_in_internal_basis).attr("__getitem__")(pi_2_inv);
        // check if ranks is sorted (strictly increasing)
        if (!np.attr("all")(np.attr("diff")(ranks).attr("__gt__")(0)).cast<bool>()) {
            throw std::logic_error("Mask.test_sanity: kept basis ranks are not sorted");
        }
    }
}

Mask::Ptr
Mask::from_eye(py::object leg,
               bool is_projection_flag,
               TensorBackend::Ptr backend,
               py::object labels,
               std::optional<std::string> device)
{
    auto diag = DiagonalTensor::from_eye(leg, backend, labels, Dtype::Bool, device);
    auto res = from_DiagonalTensor(py::cast(diag));
    if (!is_projection_flag) {
        return std::static_pointer_cast<Mask>(res->dagger());
    }
    return res;
}

Mask::Ptr
Mask::from_block_mask(py::object block_mask,
                      py::object large_leg_obj,
                      TensorBackend::Ptr backend,
                      py::object labels,
                      std::optional<std::string> device)
{
    auto large_leg = as_space(large_leg_obj);
    if (!large_leg->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}",
                      large_leg->symmetry->repr()));
    }
    backend = resolve_backend(std::move(backend), large_leg);
    auto block = backend->block_backend->as_block(block_mask, Dtype::Bool, device);
    block =
      backend->block_backend->apply_basis_perm(block, { as_leg_cptr(large_leg) }, /*inv=*/false);
    auto [data_out, small_leg] = backend->mask_from_block(block, large_leg);
    return std::make_shared<Mask>(
      data_out, large_leg_obj, py::cast(small_leg), true, backend, labels);
}

Mask::Ptr
Mask::from_DiagonalTensor(py::object diag_obj)
{
    DiagonalTensor::Ptr diag;
    if (py::isinstance<DiagonalTensor>(diag_obj)) {
        diag = diag_obj.cast<DiagonalTensor::Ptr>();
    } else {
        // Python DiagonalTensor until monkey-patch — go via attributes
        assert(diag_obj.attr("dtype").cast<Dtype>() == Dtype::Bool);
        auto backend = diag_obj.attr("backend").cast<TensorBackend::Ptr>();
        auto [data_out, small_leg] = backend->diagonal_to_mask(diag_obj);
        return std::make_shared<Mask>(data_out,
                                      diag_obj.attr("domain").attr("factors")[py::int_(0)],
                                      py::cast(small_leg),
                                      true,
                                      backend,
                                      diag_obj.attr("labels"));
    }
    assert(diag->dtype == Dtype::Bool);
    auto [data_out, small_leg] = diag->backend->diagonal_to_mask(py::cast(diag));
    return std::make_shared<Mask>(data_out,
                                  diag->domain->factors[0],
                                  py::cast(small_leg),
                                  true,
                                  diag->backend,
                                  py::cast(diag->labels()));
}

Mask::Ptr
Mask::from_indices(py::object indices,
                   py::object large_leg_obj,
                   TensorBackend::Ptr backend,
                   py::object labels,
                   std::optional<std::string> device)
{
    auto np = py::module_::import("numpy");
    auto large_leg = as_space(large_leg_obj);
    auto block_mask = np.attr("zeros")(space_dim(*large_leg), np.attr("bool_"));
    block_mask.attr("__setitem__")(indices, true);
    return from_block_mask(block_mask, large_leg_obj, std::move(backend), labels, device);
}

Mask::Ptr
Mask::from_random(py::object large_leg_obj,
                  py::object small_leg_obj,
                  TensorBackend::Ptr backend,
                  float64 p_keep,
                  int64 min_keep,
                  py::object labels,
                  std::optional<std::string> device,
                  py::object np_random)
{
    // --- hints from Python Mask.from_random ---
    // diagonal entries are uniform in [-1, 1].
    // explicitly constructing the small_leg with exactly min_keep sectors kept is
    // quite annoying bc of basis_perm. Instead we increase p_keep until we get there.
    // first, try a heuristic
    // step halfway towards 100%
    // ---
    auto large_leg = as_elementary_space(large_leg_obj);
    backend = resolve_backend(std::move(backend), large_leg);

    if (np_random.is_none()) {
        np_random = py::module_::import("numpy").attr("random").attr("default_rng")();
    }

    if (small_leg_obj.is_none()) {
        assert(0. <= p_keep && p_keep <= 1.);
        auto diag = DiagonalTensor::from_random_uniform(
          large_leg_obj, backend, labels, Dtype::Float32, device);
        float64 cutoff = 2. * p_keep - 1.; // diagonal entries are uniform in [-1, 1].
        auto res = from_DiagonalTensor(py::cast(diag).attr("__lt__")(cutoff));

        if (sum_multiplicities(*res->small_leg()) >= min_keep) {
            return res;
        }

        int64 large_leg_sector_num = sum_multiplicities(*large_leg);
        assert(min_keep <= large_leg_sector_num); // min_keep can not be fulfilled
        if (min_keep == large_leg_sector_num) {
            return from_eye(large_leg_obj, /*is_projection=*/true, backend, labels, device);
        }
        // explicitly constructing the small_leg with exactly min_keep sectors kept is
        // quite annoying bc of basis_perm. Instead we increase p_keep until we get there.
        // first, try a heuristic
        auto np = py::module_::import("numpy");
        p_keep =
          py::float_(np.attr("ceil")(1.05 * min_keep / large_leg_sector_num)).cast<float64>();
        res = from_DiagonalTensor(py::cast(diag).attr("__lt__")(2. * p_keep - 1.));
        for (int i = 0; i < 20; ++i) {
            if (sum_multiplicities(*res->small_leg()) >= min_keep) {
                return res;
            }
            p_keep = 0.5 * (p_keep + 1.); // step halfway towards 100%
            res = from_DiagonalTensor(py::cast(diag).attr("__lt__")(2. * p_keep - 1.));
        }
        throw std::runtime_error("Could not fulfill min_keep");
    }

    auto small_leg = as_elementary_space(small_leg_obj);
    if (!small_leg->is_subspace_of(*large_leg)) {
        throw std::invalid_argument("small_leg must be a subspace of the large leg.");
    }

    if ((!basis_perm_trivial(*large_leg)) || (!basis_perm_trivial(*small_leg))) {
        throw NotImplemented(
          "Generating random Masks with non-trivial, fixed basis_perm is hard and hopefully never "
          "needed.");
    }

    auto small_leg_cap = small_leg;
    auto np_random_cap = np_random;
    auto np = py::module_::import("numpy");
    py::function func =
      py::cpp_function([small_leg_cap, np_random_cap, np](py::object shape, py::object coupled) {
          int64 num_keep = small_leg_cap->sector_multiplicity(coupled.cast<Sector>());
          auto block = np.attr("zeros")(shape, np.attr("bool_"));
          auto which = np_random_cap.attr("choice")(
            shape[py::int_(0)], py::arg("size") = num_keep, py::arg("replace") = false);
          block.attr("__setitem__")(which, true);
          return block;
      });

    auto diag = DiagonalTensor::from_sector_block_func(
      func, large_leg_obj, backend, labels, py::none(), Dtype::Bool, device);
    auto res = from_DiagonalTensor(py::cast(diag));
    assert(static_cast<Space const&>(*res->small_leg()) == static_cast<Space const&>(*small_leg));
    return res;
}

Mask::Ptr
Mask::from_zero(py::object large_leg_obj,
                TensorBackend::Ptr backend,
                py::object labels,
                std::optional<std::string> device)
{
    auto large_leg = as_space(large_leg_obj);
    backend = resolve_backend(std::move(backend), large_leg);
    auto device_s = backend->block_backend->as_device(device);
    auto data_out = backend->zero_mask_data(large_leg, device_s);
    bool is_dual = false;
    if (auto es = std::dynamic_pointer_cast<ElementarySpace>(large_leg)) {
        is_dual = es->is_dual;
    }
    auto small_leg = ElementarySpace::from_null_space(large_leg->symmetry, is_dual);
    return std::make_shared<Mask>(
      data_out, large_leg_obj, py::cast(small_leg), true, backend, labels);
}

Tensor::Ptr
Mask::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    throw std::invalid_argument(
      "Mask requires Dtype.bool; use as_DiagonalTensor() or as_SymmetricTensor() "
      "for conversion to other tensor classes");
}

DiagonalTensor::Ptr
Mask::as_DiagonalTensor(Dtype out_dtype)
{
    return std::make_shared<DiagonalTensor>(backend->mask_to_diagonal(as_py_object(), out_dtype),
                                            py::cast(large_leg()),
                                            backend,
                                            py::cast(labels()));
}

py::object
Mask::as_SymmetricTensor(bool guarantee_copy, std::optional<std::string> warning)
{
    // --- hints from Python Mask.as_SymmetricTensor ---
    // OPTIMIZE how hard is it to deal with inclusions in the backend?
    // ---
    return as_SymmetricTensor(guarantee_copy, std::move(warning), Dtype::Complex128);
}

py::object
Mask::as_SymmetricTensor(bool /*guarantee_copy*/,
                         std::optional<std::string> warning,
                         Dtype out_dtype)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    if (!is_projection) {
        // OPTIMIZE how hard is it to deal with inclusions in the backend?
        auto proj = std::static_pointer_cast<Mask>(dagger());
        auto sym = proj->as_SymmetricTensor(false, std::nullopt, out_dtype);
        return py::module_::import("cyten.tensors._tensors").attr("dagger")(sym);
    }
    auto new_data = backend->full_data_from_mask(as_py_object(), out_dtype);
    return py::cast(
      std::make_shared<SymmetricTensor>(new_data, codomain, domain, backend, symmetry, labels()));
}

py::object
Mask::_binary_operand(py::object other,
                      py::function func,
                      std::string const& operand,
                      bool return_NotImplemented)
{
    // --- hints from Python Mask._binary_operand ---
    // deal with non-Mask types
    // remaining case: other is Mask
    // OPTIMIZE how hard is it to deal with inclusions in the backend?
    // ---
    // deal with non-Mask types
    if (py::isinstance<py::bool_>(other)) {
        bool other_b = other.cast<bool>();
        return py::cast(_unary_operand(py::cpp_function(
          [func, other_b](py::object block) { return func(block, py::bool_(other_b)); })));
    }
    if (is_mask_obj(other)) {
        // remaining case: other is Mask
    } else if (return_NotImplemented &&
               !(py::isinstance<Tensor>(other) ||
                 py::isinstance(other, py::module_::import("numbers").attr("Number")))) {
        return py::cast(Py_NotImplemented);
    } else {
        throw std::invalid_argument(std::format("Invalid types for operand \"{}\": Mask and {}",
                                                operand,
                                                std::string(py::str(py::type::of(other)))));
    }

    bool other_is_projection = other.attr("is_projection").cast<bool>();
    if (is_projection != other_is_projection) {
        throw std::invalid_argument("Mismatching is_projection.");
    }
    if (!is_projection) {
        // OPTIMIZE how hard is it to deal with inclusions in the backend?
        // dagger is a property (like Python), not a callable method.
        auto self_proj = std::static_pointer_cast<Mask>(dagger());
        auto other_proj = other.attr("dagger");
        auto res_projection =
          self_proj->_binary_operand(other_proj, func, operand, return_NotImplemented);
        return res_projection.attr("dagger");
    }

    auto same = get_same_backend({ as_py_object(), other });
    if (!as_py_object().attr("domain").equal(other.attr("domain"))) {
        throw std::invalid_argument("Incompatible domain.");
    }
    auto adapted = adapt_block_bool_binary(func, same->block_backend);
    auto [data_out, small] = same->mask_binary_operand(as_py_object(), other, adapted);
    auto labs = _get_matching_labels(labels(), other.attr("labels").cast<LegLabels>());
    return py::cast(std::make_shared<Mask>(
      data_out, py::cast(large_leg()), py::cast(small), is_projection, same, py::cast(labs)));
}

Mask::Ptr
Mask::_unary_operand(py::function func)
{
    // --- hints from Python Mask._unary_operand ---
    // operate on the respective projection
    // OPTIMIZE: how hard is it to deal with inclusion Masks in the backends?
    // ---
    // operate on the respective projection
    if (!is_projection) {
        // OPTIMIZE: how hard is it to deal with inclusion Masks in the backends?
        auto proj = std::static_pointer_cast<Mask>(dagger());
        return std::static_pointer_cast<Mask>(proj->_unary_operand(func)->dagger());
    }

    auto [data_out, small] = backend->mask_unary_operand(
      as_py_object(), adapt_block_bool_unary(func, backend->block_backend));
    return std::make_shared<Mask>(
      data_out, py::cast(large_leg()), py::cast(small), true, backend, py::cast(labels()));
}

Tensor::Ptr
Mask::copy(bool deep, std::optional<std::string> device_opt, std::optional<Dtype> dtype_opt)
{
    if (dtype_opt.has_value() && *dtype_opt != dtype) {
        // Python: as_dtype then maybe move — Mask.as_dtype only allows bool
        return as_dtype(*dtype_opt);
    }
    TensorBackend::DataPtr new_data;
    if (deep) {
        std::optional<std::string> device_arg = device_opt;
        new_data = backend->copy_data(as_py_object(), device_arg);
    } else if (device_opt.has_value()) {
        new_data = backend->move_to_device(as_py_object(), *device_opt);
    } else {
        new_data = data;
    }
    auto space_in = is_projection ? large_leg() : small_leg();
    auto space_out = is_projection ? small_leg() : large_leg();
    // domain is space_in, codomain is space_out
    space_in = domain->factors[0].cast<ElementarySpace::Ptr>();
    space_out = codomain->factors[0].cast<ElementarySpace::Ptr>();
    return std::make_shared<Mask>(new_data,
                                  py::cast(space_in),
                                  py::cast(space_out),
                                  is_projection,
                                  backend,
                                  py::cast(labels()));
}

Tensor::Ptr
Mask::dagger() const
{
    auto labs = labels();
    LegLabels dual_rev;
    dual_rev.reserve(labs.size());
    for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
        dual_rev.push_back(_dual_leg_label(*it));
    }
    auto new_data = backend->mask_dagger(as_py_object());
    return std::make_shared<Mask>(new_data,
                                  codomain->factors[0].cast<Space::Ptr>(),
                                  domain->factors[0].cast<Space::Ptr>(),
                                  !is_projection,
                                  backend,
                                  symmetry,
                                  std::move(dual_rev),
                                  device);
}

BlockBackend::Scalar
Mask::_get_item(std::vector<int64> const& idx)
{
    return backend->get_element_mask(as_py_object(), idx);
}

Mask::Ptr
Mask::logical_not()
{
    return orthogonal_complement();
}

void
Mask::move_to_device(std::string device_in)
{
    data = backend->move_to_device(as_py_object(), device_in);
    device = backend->block_backend->as_device(device_in);
}

Mask::Ptr
Mask::orthogonal_complement()
{
    return _unary_operand(py::module_::import("operator").attr("invert"));
}

bool
Mask::all() const
{
    // --- hints from Python Mask.all ---
    // assuming subspace, it is enough to check that the total sector number is the same.
    // ---
    // assuming subspace, it is enough to check that the total sector number is the same.
    return sum_multiplicities(*small_leg()) == sum_multiplicities(*large_leg());
}

bool
Mask::any() const
{
    return space_dim(*small_leg()) > 0;
}

BlockBackend::BlockPtr
Mask::as_block_mask()
{
    auto res = backend->mask_to_block(as_py_object());
    return backend->block_backend->apply_basis_perm(
      res, { as_leg_cptr(large_leg()) }, /*inv=*/true);
}

py::array
Mask::as_numpy_mask()
{
    return backend->block_backend->to_numpy(as_block_mask(),
                                            py::module_::import("builtins").attr("bool"));
}

Tensor::Ptr
Mask::to_backend(TensorBackend::Ptr new_backend,
                 std::optional<Dtype> dtype_opt,
                 std::optional<std::string> device_opt)
{
    // --- hints from Python Mask.to_backend ---
    // similar to DiagonalTensor, we can just go via dense mask, with some exceptions.
    // these exceptions only occurr for FusionTreeBackend -> FusionTreeBackend, and that allows
    // a simple implementation directly
    // mask_from_block assumes projection mask -> swap block_inds for inclusion
    // ---
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }

    if (dtype_opt.has_value() && *dtype_opt != Dtype::Bool) {
        throw std::invalid_argument("Mask requires Dtype.bool");
    }

    // similar to DiagonalTensor, we can just go via dense mask, with some exceptions.
    // these exceptions only occurr for FusionTreeBackend -> FusionTreeBackend, and that allows
    // a simple implementation directly

    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });
    TensorBackend::DataPtr new_data;
    if (std::dynamic_pointer_cast<FusionTreeBackend>(backend) &&
        std::dynamic_pointer_cast<FusionTreeBackend>(new_backend)) {
        new_data =
          backend->to_block_backend(data, new_backend->block_backend, Dtype::Bool, device_s);
    } else {
        auto old_mask = backend->mask_to_block(as_py_object());
        auto new_mask =
          new_backend->block_backend->as_block(py::cast(old_mask), Dtype::Bool, device_s);
        auto [data_out, unused_small] = new_backend->mask_from_block(new_mask, large_leg());
        (void)unused_small;
        new_data = std::move(data_out);
        if (std::dynamic_pointer_cast<AbelianBackend>(new_backend) && !is_projection) {
            // mask_from_block assumes projection mask -> swap block_inds for inclusion
            auto abd = std::dynamic_pointer_cast<AbelianBackendData>(new_data);
            assert(abd);
            int64 cols[] = { 1, 0 };
            abd->block_inds = abd->block_inds.take_columns_i64(cols);
        }
    }
    return std::make_shared<Mask>(new_data,
                                  domain->factors[0].cast<Space::Ptr>(),
                                  codomain->factors[0].cast<Space::Ptr>(),
                                  is_projection,
                                  new_backend,
                                  symmetry,
                                  labels(),
                                  device_s);
}

BlockBackend::BlockPtr
Mask::to_dense_block(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
                     std::optional<Dtype> dtype_opt,
                     bool understood_braiding)
{
    // --- hints from Python Mask.to_dense_block ---
    // for Mask, defining via numpy is actually easier, to use numpy indexing
    // ---
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    if (!symmetry->has_trivial_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of to_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    // for Mask, defining via numpy is actually easier, to use numpy indexing
    py::object numpy_dtype = py::none();
    if (dtype_opt.has_value()) {
        numpy_dtype = dtype::to_numpy_dtype(*dtype_opt);
    }
    auto as_numpy = to_numpy(leg_order, numpy_dtype, understood_braiding);
    return backend->block_backend->as_block(as_numpy, dtype_opt, std::nullopt);
}

py::array
Mask::to_numpy(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
               py::object numpy_dtype,
               bool understood_braiding)
{
    // --- hints from Python Mask.to_numpy ---
    // sets the appropriate dtype. e.g. sets ``True`` for bool.
    // ---
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    if (!symmetry->has_trivial_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of to_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    assert(symmetry->can_be_dropped());
    auto np = py::module_::import("numpy");
    auto mask = as_numpy_mask();
    // Use Python shape property (int dims) — np.zeros rejects float dims.
    // Match Python: ``numpy_dtype or bool`` (the type, not the value False).
    auto res = np.attr("zeros")(
      as_py_object().attr("shape"),
      numpy_dtype.is_none() ? py::module_::import("builtins").attr("bool") : numpy_dtype);
    // shape is [m, n] for Mask
    assert(shape.size() == 2);
    auto m = static_cast<int64>(shape[0]);
    auto n = static_cast<int64>(shape[1]);
    if (is_projection) {
        res.attr("__setitem__")(py::make_tuple(np.attr("arange")(m), mask), 1);
    } else {
        res.attr("__setitem__")(py::make_tuple(mask, np.attr("arange")(n)), 1);
    }
    if (leg_order.has_value()) {
        auto idcs = as_py_object().attr("get_leg_idcs")(py::cast(*leg_order));
        res = np.attr("transpose")(res, idcs);
    }
    return res;
}

void
Mask::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    /// Export Mask to hdf5 such that it can be re-imported with from_hdf5
    hdf5_saver.attr("save")(py::cast(domain), subpath + "domain");
    hdf5_saver.attr("save")(py::cast(codomain), subpath + "codomain");
    hdf5_saver.attr("save")(py::cast(backend), subpath + "backend");
    hdf5_saver.attr("save")(py::cast(data), subpath + "data");
    hdf5_saver.attr("save")(py::cast(symmetry), subpath + "symmetry");
    h5gr.attr("attrs")["dtype"] = dtype::repr(dtype);
    h5gr.attr("attrs")["num_legs"] = num_legs;
    h5gr.attr("attrs")["shape"] = py::module_::import("numpy").attr("array")(
      py::cast(shape), py::module_::import("numpy").attr("intp"));
    h5gr.attr("attrs")["is_projection"] = is_projection;
    if (std::ranges::all_of(_labels, [](LegLabel const& l) { return !l; })) {
        h5gr.attr("attrs")["labels"] = py::list();
    } else {
        h5gr.attr("attrs")["labels"] = py::cast(_labels);
    }
}

Mask::Ptr
Mask::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    /// Import Mask from hdf5
    auto domain_tp = hdf5_loader.attr("load")(subpath + "domain").cast<TensorProduct::Ptr>();
    auto codomain_tp = hdf5_loader.attr("load")(subpath + "codomain").cast<TensorProduct::Ptr>();
    auto symmetry_in = hdf5_loader.attr("load")(subpath + "symmetry").cast<Symmetry::Ptr>();
    auto backend_in = hdf5_loader.attr("load")(subpath + "backend").cast<TensorBackend::Ptr>();
    auto data_in = hdf5_loader.attr("load")(subpath + "data").cast<TensorBackend::DataPtr>();
    (void)hdf5_loader.attr("get_attr")(h5gr, "dtype");
    (void)hdf5_loader.attr("get_attr")(h5gr, "num_legs");
    auto shape_in = hdf5_loader.attr("get_attr")(h5gr, "shape").cast<std::vector<float64>>();

    bool proj = true;
    try {
        proj = hdf5_loader.attr("get_attr")(h5gr, "is_projection").cast<bool>();
    } catch (py::error_already_set&) {
        auto space_in = domain_tp->factors[0].cast<Space::Ptr>();
        auto space_out = codomain_tp->factors[0].cast<Space::Ptr>();
        proj = space_dim(*space_in) >= space_dim(*space_out);
    }

    LegLabels labels_in(2, std::nullopt);
    try {
        labels_in = hdf5_loader.attr("get_attr")(h5gr, "labels").cast<LegLabels>();
        // Match Python save: all-None labels are stored as [].
        if (labels_in.empty()) {
            labels_in.assign(2, std::nullopt);
        }
    } catch (py::error_already_set&) {
        // older saves may omit labels
    }

    auto device_in = backend_in->get_device_from_data(data_in);
    auto obj = std::make_shared<Mask>(data_in,
                                      domain_tp->factors[0].cast<Space::Ptr>(),
                                      codomain_tp->factors[0].cast<Space::Ptr>(),
                                      proj,
                                      backend_in,
                                      symmetry_in,
                                      std::move(labels_in),
                                      device_in);
    obj->shape = std::move(shape_in);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
