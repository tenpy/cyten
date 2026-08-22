#include <cyten/tensors/mask.h>

#include <cyten/backends/abelian.h>
#include <cyten/backends/backend_factory.h>
#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/tools.h>
#include <cyten/tools/warn.h>

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

ElementarySpace::Ptr
as_elementary_space(Space::Ptr obj)
{
    if (std::dynamic_pointer_cast<LegPipe>(obj)) {
        throw std::invalid_argument("Mask is not defined on LegPipes.");
    }
    auto es = std::dynamic_pointer_cast<ElementarySpace>(obj);
    if (!es) {
        throw std::invalid_argument("Expected ElementarySpace.");
    }
    return es;
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
BlockUnaryFn
adapt_block_bool_unary(py::function func, std::shared_ptr<BlockBackend> bb)
{
    return [func, bb](BlockBackend::BlockPtr const& block) {
        auto arr = bb->to_numpy(block, py::module_::import("builtins").attr("bool"));
        auto out = func(arr);
        return bb->as_block(out, Dtype::Bool, block->device());
    };
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
        return get_backend(space->symmetry);
    }
    return backend;
}

Mask::Ptr
make_mask(TensorBackend::DataPtr data,
          Space::Ptr space_in,
          Space::Ptr space_out,
          bool is_projection,
          TensorBackend::Ptr backend,
          std::optional<LegLabels> labels)
{
    as_elementary_space(space_in);
    as_elementary_space(space_out);
    auto [codomain, domain, backend_tp, symmetry] = Tensor::_init_parse_args(
      std::make_shared<TensorProduct>(
        std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_out) }),
      std::make_shared<TensorProduct>(
        std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_in) }),
      std::move(backend));
    auto labs = Tensor::_init_parse_labels(std::move(labels), codomain, domain);
    auto device_s = backend_tp->get_device_from_data(data);
    return std::make_shared<Mask>(std::move(data),
                                  std::move(space_in),
                                  std::move(space_out),
                                  is_projection,
                                  std::move(backend_tp),
                                  std::move(symmetry),
                                  std::move(labs),
                                  std::move(device_s));
}

} // namespace

/// @cond
std::vector<Dtype> Mask::_forbidden_dtypes = {
    Dtype::Float32,
    Dtype::Float64,
    Dtype::Complex64,
    Dtype::Complex128,
};
/// @endcond

Mask::Mask(TensorBackend::DataPtr data_in,
           Space::Ptr space_in,
           Space::Ptr space_out,
           bool is_projection_in,
           TensorBackend::Ptr backend_in,
           Symmetry::Ptr symmetry_in,
           LegLabels labels_in,
           std::string device_in)
  : Tensor(std::make_shared<TensorProduct>(
             std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_out) }),
           std::make_shared<TensorProduct>(
             std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(space_in) }),
           std::move(backend_in),
           std::move(symmetry_in),
           std::move(labels_in),
           Dtype::Bool,
           std::move(device_in))
  , is_projection(is_projection_in)
  , data(std::move(data_in))
{
    assert(backend->is_correct_data_type(data));
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

ElementarySpace::Ptr
Mask::large_leg() const
{
    if (is_projection) {
        return std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]);
    }
    return std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]);
}

ElementarySpace::Ptr
Mask::small_leg() const
{
    if (is_projection) {
        return std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]);
    }
    return std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]);
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
    backend->test_mask_sanity(std::static_pointer_cast<Mask const>(shared_from_this()));
    assert(codomain->num_factors == 1 && domain->num_factors == 1);
    assert(std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]));
    assert(std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]));
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
          backend->mask_to_block(std::static_pointer_cast<Mask const>(shared_from_this())),
          py::module_::import("builtins").attr("bool"));
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
Mask::from_eye(Space::Ptr leg,
               bool is_projection_flag,
               TensorBackend::Ptr backend,
               std::optional<LegLabels> labels,
               std::optional<std::string> device)
{
    auto diag = DiagonalTensor::from_eye(std::move(leg), backend, labels, Dtype::Bool, device);
    auto res = from_DiagonalTensor(diag);
    if (!is_projection_flag) {
        return std::static_pointer_cast<Mask>(res->dagger());
    }
    return res;
}

Mask::Ptr
Mask::from_block_mask(BlockBackend::BlockPtr block_mask,
                      Space::Ptr large_leg,
                      TensorBackend::Ptr backend,
                      std::optional<LegLabels> labels,
                      std::optional<std::string> device)
{
    if (!large_leg->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}",
                      large_leg->symmetry->repr()));
    }
    backend = resolve_backend(std::move(backend), large_leg);
    auto block = backend->block_backend->as_block(py::cast(block_mask), Dtype::Bool, device);
    block =
      backend->block_backend->apply_basis_perm(block, { as_leg_cptr(large_leg) }, /*inv=*/false);
    auto [data_out, small_leg] = backend->mask_from_block(block, large_leg);
    return make_mask(data_out, large_leg, small_leg, true, backend, std::move(labels));
}

Mask::Ptr
Mask::from_DiagonalTensor(DiagonalTensorCPtr diag)
{
    assert(diag);
    assert(diag->dtype == Dtype::Bool);
    auto [data_out, small_leg] = diag->backend->diagonal_to_mask(diag);
    return std::make_shared<Mask>(data_out,
                                  as_space(diag->domain->factors[0]),
                                  small_leg,
                                  true,
                                  diag->backend,
                                  diag->symmetry,
                                  diag->labels(),
                                  diag->device);
}

Mask::Ptr
Mask::from_indices(py::object indices,
                   Space::Ptr large_leg,
                   TensorBackend::Ptr backend,
                   std::optional<LegLabels> labels,
                   std::optional<std::string> device)
{
    auto np = py::module_::import("numpy");
    auto block_mask = np.attr("zeros")(space_dim(*large_leg), np.attr("bool_"));
    block_mask.attr("__setitem__")(indices, true);
    backend = resolve_backend(std::move(backend), large_leg);
    auto block = backend->block_backend->as_block(block_mask, Dtype::Bool, device);
    return from_block_mask(
      block, std::move(large_leg), std::move(backend), std::move(labels), device);
}

Mask::Ptr
Mask::from_random(Space::Ptr large_leg_in,
                  Space::Ptr small_leg_in,
                  TensorBackend::Ptr backend,
                  float64 p_keep,
                  int64 min_keep,
                  std::optional<LegLabels> labels,
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
    auto large_leg = as_elementary_space(std::move(large_leg_in));
    backend = resolve_backend(std::move(backend), large_leg);

    if (np_random.is_none()) {
        np_random = py::module_::import("numpy").attr("random").attr("default_rng")();
    }

    if (!small_leg_in) {
        assert(0. <= p_keep && p_keep <= 1.);
        auto diag =
          DiagonalTensor::from_random_uniform(large_leg, backend, labels, Dtype::Float32, device);
        float64 cutoff = 2. * p_keep - 1.; // diagonal entries are uniform in [-1, 1].
        auto res =
          from_DiagonalTensor(py::cast(diag).attr("__lt__")(cutoff).cast<DiagonalTensorCPtr>());

        if (sum_multiplicities(*res->small_leg()) >= min_keep) {
            return res;
        }

        int64 large_leg_sector_num = sum_multiplicities(*large_leg);
        assert(min_keep <= large_leg_sector_num); // min_keep can not be fulfilled
        if (min_keep == large_leg_sector_num) {
            return from_eye(large_leg, /*is_projection=*/true, backend, labels, device);
        }
        // explicitly constructing the small_leg with exactly min_keep sectors kept is
        // quite annoying bc of basis_perm. Instead we increase p_keep until we get there.
        // first, try a heuristic
        auto np = py::module_::import("numpy");
        p_keep =
          py::float_(np.attr("ceil")(1.05 * min_keep / large_leg_sector_num)).cast<float64>();
        res = from_DiagonalTensor(
          py::cast(diag).attr("__lt__")(2. * p_keep - 1.).cast<DiagonalTensorCPtr>());
        for (int i = 0; i < 20; ++i) {
            if (sum_multiplicities(*res->small_leg()) >= min_keep) {
                return res;
            }
            p_keep = 0.5 * (p_keep + 1.); // step halfway towards 100%
            res = from_DiagonalTensor(
              py::cast(diag).attr("__lt__")(2. * p_keep - 1.).cast<DiagonalTensorCPtr>());
        }
        throw std::runtime_error("Could not fulfill min_keep");
    }

    auto small_leg = as_elementary_space(std::move(small_leg_in));
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
    auto bb = backend->block_backend;
    SectorBlockFactoryFn func = [small_leg_cap, np_random_cap, np, bb, device](
                                  std::vector<int64> const& shape, Sector const& coupled) {
        int64 num_keep = small_leg_cap->sector_multiplicity(coupled);
        py::object block = np.attr("zeros")(py::cast(shape), np.attr("bool_"));
        auto which = np_random_cap.attr("choice")(
          shape[0], py::arg("size") = num_keep, py::arg("replace") = false);
        block.attr("__setitem__")(which, true);
        return bb->as_block(block, Dtype::Bool, device);
    };

    auto diag = DiagonalTensor::from_sector_block_func(
      std::move(func), large_leg, backend, labels, Dtype::Bool, device);
    auto res = from_DiagonalTensor(diag);
    assert(static_cast<Space const&>(*res->small_leg()) == static_cast<Space const&>(*small_leg));
    return res;
}

Mask::Ptr
Mask::from_zero(Space::Ptr large_leg,
                TensorBackend::Ptr backend,
                std::optional<LegLabels> labels,
                std::optional<std::string> device)
{
    backend = resolve_backend(std::move(backend), large_leg);
    auto device_s = backend->block_backend->as_device(device);
    auto data_out = backend->zero_mask_data(large_leg, device_s);
    bool is_dual = false;
    if (auto es = std::dynamic_pointer_cast<ElementarySpace>(large_leg)) {
        is_dual = es->is_dual;
    }
    auto small_leg = ElementarySpace::from_null_space(large_leg->symmetry, is_dual);
    return make_mask(data_out, std::move(large_leg), small_leg, true, backend, std::move(labels));
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
    return std::make_shared<DiagonalTensor>(
      backend->mask_to_diagonal(std::static_pointer_cast<Mask const>(shared_from_this()),
                                out_dtype),
      large_leg(),
      backend,
      symmetry,
      labels());
}

SymmetricTensorPtr
Mask::as_SymmetricTensor(bool guarantee_copy, std::optional<std::string> warning)
{
    // --- hints from Python Mask.as_SymmetricTensor ---
    // OPTIMIZE how hard is it to deal with inclusions in the backend?
    // ---
    return as_SymmetricTensor(guarantee_copy, std::move(warning), Dtype::Complex128);
}

SymmetricTensorPtr
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
        return py::module_::import("cyten.tensors._tensors")
          .attr("dagger")(py::cast(sym))
          .cast<SymmetricTensorPtr>();
    }
    auto new_data = backend->full_data_from_mask(
      std::static_pointer_cast<Mask const>(shared_from_this()), out_dtype);
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels());
}

Mask::Ptr
Mask::_binary_operand(bool other, BlockBinaryFn func, std::string const& /*operand*/)
{
    auto bb = backend->block_backend;
    auto other_block = std::const_pointer_cast<BlockBackend::Block>(bb->as_scalar(other)._block());
    return _unary_operand([func, other_block](BlockBackend::BlockPtr const& block) {
        return func(block, other_block);
    });
}

Mask::Ptr
Mask::_binary_operand(MaskCPtr other, BlockBinaryFn func, std::string const& operand)
{
    // --- hints from Python Mask._binary_operand ---
    // remaining case: other is Mask
    // OPTIMIZE how hard is it to deal with inclusions in the backend?
    // ---
    if (is_projection != other->is_projection) {
        throw std::invalid_argument("Mismatching is_projection.");
    }
    if (!is_projection) {
        // OPTIMIZE how hard is it to deal with inclusions in the backend?
        auto self_proj = std::static_pointer_cast<Mask>(dagger());
        auto other_proj = std::dynamic_pointer_cast<Mask>(other->dagger());
        auto res_projection = self_proj->_binary_operand(other_proj, func, operand);
        return std::static_pointer_cast<Mask>(res_projection->dagger());
    }

    auto same = get_same_backend(std::vector<TensorCPtr>{
      std::static_pointer_cast<Tensor const>(shared_from_this()), other });
    if (!(static_cast<Space const&>(*domain) == static_cast<Space const&>(*other->domain))) {
        throw std::invalid_argument("Incompatible domain.");
    }
    auto [data_out, small] = same->mask_binary_operand(
      std::static_pointer_cast<Mask const>(shared_from_this()), other, std::move(func));
    auto labs = _get_matching_labels(labels(), other->labels());
    return make_mask(data_out, large_leg(), small, is_projection, same, labs);
}

Mask::Ptr
Mask::_unary_operand(BlockUnaryFn func)
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

    auto [data_out, small] =
      backend->mask_unary_operand(std::static_pointer_cast<Mask const>(shared_from_this()), func);
    return make_mask(data_out, large_leg(), small, true, backend, labels());
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
        new_data = backend->copy_data(shared_from_this(), device_arg);
    } else if (device_opt.has_value()) {
        new_data = backend->move_to_device(shared_from_this(), *device_opt);
    } else {
        new_data = data;
    }
    auto space_in = is_projection ? large_leg() : small_leg();
    auto space_out = is_projection ? small_leg() : large_leg();
    // domain is space_in, codomain is space_out
    space_in = std::dynamic_pointer_cast<ElementarySpace>(domain->factors[0]);
    space_out = std::dynamic_pointer_cast<ElementarySpace>(codomain->factors[0]);
    return std::make_shared<Mask>(new_data,
                                  space_in,
                                  space_out,
                                  is_projection,
                                  backend,
                                  symmetry,
                                  labels(),
                                  backend->get_device_from_data(new_data));
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
    auto new_data = backend->mask_dagger(std::static_pointer_cast<Mask const>(shared_from_this()));
    return std::make_shared<Mask>(new_data,
                                  as_space(codomain->factors[0]),
                                  as_space(domain->factors[0]),
                                  !is_projection,
                                  backend,
                                  symmetry,
                                  std::move(dual_rev),
                                  device);
}

BlockBackend::Scalar
Mask::_get_item(std::vector<int64> const& idx)
{
    return backend->get_element_mask(std::static_pointer_cast<Mask const>(shared_from_this()),
                                     idx);
}

Mask::Ptr
Mask::logical_not()
{
    return orthogonal_complement();
}

void
Mask::move_to_device(std::string device_in)
{
    data = backend->move_to_device(shared_from_this(), device_in);
    device = backend->block_backend->as_device(device_in);
}

Mask::Ptr
Mask::orthogonal_complement()
{
    return _unary_operand(adapt_block_bool_unary(py::module_::import("operator").attr("invert"),
                                                 backend->block_backend));
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
    auto res = backend->mask_to_block(std::static_pointer_cast<Mask const>(shared_from_this()));
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
        auto old_mask =
          backend->mask_to_block(std::static_pointer_cast<Mask const>(shared_from_this()));
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
                                  as_space(domain->factors[0]),
                                  as_space(codomain->factors[0]),
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
      py::make_tuple(static_cast<int64>(shape[0]), static_cast<int64>(shape[1])),
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
        auto idcs = get_leg_idcs(*leg_order);
        res = np.attr("transpose")(res, py::cast(idcs));
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
        auto space_in = as_space(domain_tp->factors[0]);
        auto space_out = as_space(codomain_tp->factors[0]);
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
                                      as_space(domain_tp->factors[0]),
                                      as_space(codomain_tp->factors[0]),
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
