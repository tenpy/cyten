#include <cyten/tensors/charged_tensor.h>

#include <cyten/backends/backend_factory.h>
#include <cyten/backends/no_symmetry.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/tensors/hidden_leg_tensor.h>
#include <cyten/tools.h>
#include <cyten/tools/warn.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

namespace {

py::object
tensors_mod()
{
    return py::module_::import("cyten.tensors._tensors");
}

} // namespace

ChargedTensor::ChargedTensor(SymmetricTensor::Ptr inv, BlockBackend::BlockPtr charged_state_in)
  : Tensor(
      inv->codomain,
      std::make_shared<TensorProduct>(
        std::vector<Leg::Ptr>(inv->domain->factors.begin() + 1, inv->domain->factors.end()),
        inv->symmetry),
      inv->backend,
      inv->symmetry,
      [&]() {
          auto labs = inv->labels();
          if (labs.empty()) {
              throw std::invalid_argument(
                "ChargedTensor invariant_part must have a charge-leg label");
          }
          return LegLabels(labs.begin(), labs.end() - 1);
      }(),
      inv->dtype,
      inv->device)
  , invariant_part(std::move(inv))
  , charged_state(std::move(charged_state_in))
  // Match Python: keep the domain factor as-is (ElementarySpace or LegPipe).
  , charge_leg(invariant_part->domain->factors[0])
{
    if (invariant_part->domain->num_factors <= 0) {
        throw std::invalid_argument(
          "ChargedTensor invariant_part must have a charge leg in the domain");
    }
    auto labs = invariant_part->labels();
    if (labs.empty() || !labs.back() || *labs.back() != _CHARGE_LEG_LABEL) {
        throw std::invalid_argument(
          std::format("ChargedTensor invariant_part last label must be '{}'", _CHARGE_LEG_LABEL));
    }
    if (!supports_symmetry(invariant_part->symmetry)) {
        throw SymmetryError(std::format("ChargedTensor is not well-defined for symmetry {}.",
                                        invariant_part->symmetry->repr()));
    }
    if (!charged_state) {
        throw std::invalid_argument(
          "ChargedTensor requires a charged_state. Use HiddenLegTensor to hide legs without a "
          "state.");
    }
    if (!invariant_part->symmetry->can_be_dropped()) {
        throw SymmetryError(std::format("charged_state can not be specified for symmetry {}",
                                        invariant_part->symmetry->repr()));
    }
    charged_state = invariant_part->backend->block_backend->as_block(
      py::cast(charged_state), invariant_part->dtype, invariant_part->device);
    invariant_part->allow_charge_leg_label = true;
    reject_exclamation_in_labels(labels(), "ChargedTensor");
}

void
ChargedTensor::test_sanity() const
{
    Tensor::test_sanity();
    auto inv_labs = invariant_part->labels();
    assert(labels() == LegLabels(inv_labs.begin(), inv_labs.end() - 1));
    invariant_part->allow_charge_leg_label = true;
    invariant_part->test_sanity();
    assert(invariant_part->device == device);
    assert(charged_state);
    backend->block_backend->test_block_sanity(
      charged_state,
      std::vector<int64>{ static_cast<int64>(charge_leg->dim) },
      std::nullopt,
      device);
    reject_exclamation_in_labels(labels(), "ChargedTensor");
}

std::string
ChargedTensor::ascii_diagram_type_name() const
{
    return "Chrg";
}

std::string
ChargedTensor::class_name() const
{
    return "ChargedTensor";
}

bool
ChargedTensor::supports_symmetry(Symmetry::Ptr const& symmetry)
{
    // charged_state is always required and needs a droppable symmetry.
    return symmetry->can_be_dropped() && symmetry->has_symmetric_braid();
}

std::tuple<TensorProduct::Ptr, Space::Ptr>
ChargedTensor::_parse_inv_domain(TensorProduct::Ptr domain,
                                 std::variant<ElementarySpace::Ptr, Sector> charge)
{
    if (!domain) {
        throw std::invalid_argument("domain must be parsed before constructing ChargedTensor");
    }
    Space::Ptr charge_leg_sp;
    if (auto const* es = std::get_if<ElementarySpace::Ptr>(&charge)) {
        charge_leg_sp = *es;
    } else {
        Sector const& sec = std::get<Sector>(charge);
        charge_leg_sp =
          std::make_shared<ElementarySpace>(domain->symmetry, SectorArray::from_sector(sec));
    }
    return { domain->left_multiply(std::dynamic_pointer_cast<Leg>(charge_leg_sp)), charge_leg_sp };
}

std::tuple<LegLabels, LegLabels>
ChargedTensor::_parse_inv_labels(std::optional<LegLabels> labels,
                                 TensorProduct::Ptr const& codomain,
                                 TensorProduct::Ptr const& domain)
{
    auto labs = _init_parse_labels(std::move(labels), codomain, domain);
    auto inv_labels = labs;
    inv_labels.emplace_back(std::string(_CHARGE_LEG_LABEL));
    return { labs, inv_labels };
}

ChargedTensor::Ptr
ChargedTensor::from_block_func(BlockFactoryFn func,
                               std::variant<ElementarySpace::Ptr, Sector> charge,
                               TensorProduct::Ptr codomain,
                               TensorProduct::Ptr domain,
                               BlockBackend::BlockPtr charged_state,
                               TensorBackend::Ptr backend,
                               std::optional<LegLabels> labels,
                               std::optional<Dtype> dtype,
                               std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    (void)symmetry;
    std::string device_s;
    if (!device.has_value()) {
        if (!charged_state) {
            device_s = backend_tp->block_backend->default_device;
        } else {
            device_s = backend_tp->block_backend->get_device(charged_state);
        }
    } else {
        device_s = *device;
    }
    auto [inv_domain, charge_leg_sp] = _parse_inv_domain(domain_tp, std::move(charge));
    (void)charge_leg_sp;
    auto [labs, inv_labels] = _parse_inv_labels(std::move(labels), codomain_tp, domain_tp);
    (void)labs;
    auto data = backend_tp->from_sector_block_func(
      [func = std::move(func)](std::vector<int64> const& shape, Sector const& /*coupled*/) {
          return func(shape);
      },
      codomain_tp,
      inv_domain);
    auto inv = std::make_shared<SymmetricTensor>(data,
                                                 codomain_tp,
                                                 inv_domain,
                                                 backend_tp,
                                                 symmetry,
                                                 std::move(inv_labels));
    inv->allow_charge_leg_label = true;
    inv->test_sanity();
    return std::make_shared<ChargedTensor>(inv, charged_state);
}

ChargedTensor::Ptr
ChargedTensor::from_dense_block(BlockBackend::BlockPtr block,
                                TensorProduct::Ptr codomain,
                                TensorProduct::Ptr domain,
                                std::optional<std::variant<ElementarySpace::Ptr, Sector>> charge,
                                TensorBackend::Ptr backend,
                                std::optional<LegLabels> labels,
                                std::optional<Dtype> dtype,
                                std::optional<std::string> device,
                                float64 tol,
                                bool understood_braiding)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    auto [labs, inv_labels] = _parse_inv_labels(std::move(labels), codomain_tp, domain_tp);
    (void)labs;
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    auto block_ptr = backend_tp->block_backend->as_block(py::cast(block), dtype, device);
    if (!charge.has_value()) {
        throw NotImplemented("ChargedTensor::from_dense_block with charge=None");
    }
    auto [inv_domain, charge_leg_sp] = _parse_inv_domain(domain_tp, *charge);
    if (charge_leg_sp->Space::dim != 1.) {
        throw NotImplemented("ChargedTensor::from_dense_block with charge_leg.dim != 1");
    }
    auto inv_part =
      SymmetricTensor::from_dense_block(backend_tp->block_backend->add_axis(block_ptr, -1),
                                        codomain_tp,
                                        inv_domain,
                                        backend_tp,
                                        inv_labels,
                                        std::nullopt,
                                        std::nullopt,
                                        tol,
                                        understood_braiding);
    auto cs = backend_tp->block_backend->as_block(
      py::cast(std::vector<int64>{ 1 }), inv_part->dtype, inv_part->device);
    return std::make_shared<ChargedTensor>(inv_part, cs);
}

ChargedTensor::Ptr
ChargedTensor::from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                              Leg::Ptr space,
                                              Sector sector,
                                              TensorBackend::Ptr backend,
                                              std::optional<std::string> label,
                                              std::optional<std::string> device)
{
    if (!space) {
        throw std::invalid_argument("space must be specified");
    }
    if (!backend) {
        backend = get_backend(space->symmetry);
    }
    if (space->symmetry->qdim(sector) > 1.) {
        throw NotImplemented("from_dense_block_single_sector does not support higher-dim sectors");
    }
    auto charge_leg =
      std::make_shared<ElementarySpace>(space->symmetry, SectorArray::from_sector(sector));
    auto space_as_space = as_space(space);
    auto data =
      backend->inv_part_from_dense_block_single_sector(vector, space_as_space, charge_leg);
    auto codomain =
      std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ space }, space->symmetry);
    auto inv_domain =
      std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ charge_leg }, space->symmetry);
    LegLabels inv_labels{ label, std::string(_CHARGE_LEG_LABEL) };
    auto inv_part = std::make_shared<SymmetricTensor>(
      data, codomain, inv_domain, backend, space->symmetry, std::move(inv_labels));
    auto charged_state =
      backend->block_backend->as_block(py::cast(std::vector<int64>{ 1 }), inv_part->dtype, device);
    return std::make_shared<ChargedTensor>(inv_part, charged_state);
}

std::variant<ChargedTensor::Ptr, BlockBackend::Scalar>
ChargedTensor::from_invariant_part(SymmetricTensor::Ptr inv, BlockBackend::BlockPtr charged_state)
{
    // --- hints from Python ChargedTensor.from_invariant_part ---
    // OPTIMIZE ?
    // ---
    if (!inv) {
        throw std::invalid_argument("invariant_part must be specified");
    }
    if (!charged_state) {
        throw std::invalid_argument(
          "ChargedTensor.from_invariant_part requires a charged_state. "
          "Use HiddenLegTensor to hide legs without a state.");
    }
    if (inv->num_legs == 1) {
        // OPTIMIZE ?
        auto inv_block =
          inv->to_dense_block(std::nullopt, std::nullopt, /*understood_braiding=*/true);
        auto state =
          inv->backend->block_backend->as_block(py::cast(charged_state), inv->dtype, inv->device);
        return inv->backend->block_backend->inner(inv_block, state, /*do_dagger=*/false);
    }
    inv->allow_charge_leg_label = true;
    return std::make_shared<ChargedTensor>(std::move(inv), std::move(charged_state));
}

std::variant<ChargedTensor::Ptr, BlockBackend::Scalar>
ChargedTensor::from_two_charge_legs(SymmetricTensor::Ptr invariant_part,
                                    BlockBackend::BlockPtr state1,
                                    BlockBackend::BlockPtr state2)
{
    // Uses combine_legs free function — keep via Python helper when needed.
    auto inv_obj = py::cast(invariant_part);
    auto labs = invariant_part->labels();
    if (labs.size() < 2) {
        throw std::invalid_argument("from_two_charge_legs requires at least two labels");
    }
    if (!labs[labs.size() - 1] || !labs[labs.size() - 1]->starts_with(_CHARGE_LEG_LABEL) ||
        !labs[labs.size() - 2] || !labs[labs.size() - 2]->starts_with(_CHARGE_LEG_LABEL)) {
        throw std::invalid_argument(
          std::format("from_two_charge_legs requires the last two labels to start with '{}'",
                      _CHARGE_LEG_LABEL));
    }
    if (!state1 || !state2) {
        throw std::invalid_argument(
          "from_two_charge_legs requires both charged states. "
          "Use HiddenLegTensor to hide legs without a state.");
    }
    invariant_part->allow_charge_leg_label = true;
    auto inv_part = tensors_mod().attr("combine_legs")(inv_obj, py::make_tuple(-2, -1));
    inv_part.attr("set_label")(-1, _CHARGE_LEG_LABEL);
    auto pipe = inv_part.attr("domain").attr("__getitem__")(0).cast<LegPipe::Ptr>();
    auto state = invariant_part->backend->state_tensor_product(state1, state2, pipe)
                   .cast<BlockBackend::BlockPtr>();
    return from_invariant_part(inv_part.cast<SymmetricTensor::Ptr>(), state);
}

ChargedTensor::Ptr
ChargedTensor::from_zero(TensorProduct::Ptr codomain,
                         TensorProduct::Ptr domain,
                         std::variant<ElementarySpace::Ptr, Sector> charge,
                         BlockBackend::BlockPtr charged_state,
                         TensorBackend::Ptr backend,
                         std::optional<LegLabels> labels,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    (void)symmetry;
    std::string device_s;
    if (!device.has_value()) {
        if (!charged_state) {
            throw std::invalid_argument("ChargedTensor.from_zero requires a charged_state");
        }
        device_s = backend_tp->block_backend->get_device(charged_state);
    } else {
        device_s = *device;
    }
    auto [inv_domain, charge_leg_sp] = _parse_inv_domain(domain_tp, std::move(charge));
    (void)charge_leg_sp;
    auto [labs, inv_labels] = _parse_inv_labels(std::move(labels), codomain_tp, domain_tp);
    (void)labs;
    auto inv_part =
      SymmetricTensor::from_zero(codomain_tp, inv_domain, backend_tp, inv_labels, dtype, device_s);
    inv_part->allow_charge_leg_label = true;
    return std::make_shared<ChargedTensor>(inv_part, charged_state);
}

Tensor::Ptr
ChargedTensor::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(invariant_part->as_dtype(new_dtype));
    assert(inv);
    return std::make_shared<ChargedTensor>(inv, charged_state);
}

SymmetricTensorPtr
ChargedTensor::as_SymmetricTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    // LegPipe charge legs (from combine_legs) expose Space APIs via as_Space().
    Space::Ptr charge_space = as_space(charge_leg);
    if (charge_space->sector_decomposition.size() != 1 ||
        charge_space->sector_decomposition[0] != symmetry->trivial_sector) {
        throw SymmetryError("Not a symmetric tensor");
    }
    if (charge_leg->dim == 1.) {
        auto res = tensors_mod().attr("squeeze_legs")(py::cast(invariant_part), -1);
        auto scale = backend->block_backend->item(charged_state);
        res = res.attr("__mul__")(py::cast(scale));
        return res.cast<SymmetricTensorPtr>();
    }
    // charge_leg.dual (Python wrote charged_state.dual — treat as charge_leg.dual)
    auto dual = charge_leg->dual_leg();
    auto state_codomain = std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ dual });
    auto state = SymmetricTensor::from_dense_block(
      charged_state,
      state_codomain,
      nullptr,
      backend,
      LegLabels{ _dual_leg_label(std::string(_CHARGE_LEG_LABEL)) },
      dtype,
      std::nullopt,
      1e-6,
      /*understood_braiding=*/true);
    state->allow_charge_leg_label = true;
    auto res = tensors_mod().attr("tdot")(py::cast(state), py::cast(invariant_part), 0, -1);
    return tensors_mod()
      .attr("bend_legs")(res, py::arg("num_codomain_legs") = num_codomain_legs())
      .cast<SymmetricTensorPtr>();
}

Tensor::Ptr
ChargedTensor::copy(bool deep,
                    std::optional<std::string> device_opt,
                    std::optional<Dtype> dtype_opt)
{
    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(
      invariant_part->copy(deep, device_opt, dtype_opt));
    assert(inv);
    inv->allow_charge_leg_label = true;
    BlockBackend::BlockPtr cs = charged_state;
    if ((device_opt.has_value() && !deep) || (dtype_opt.has_value() && *dtype_opt != dtype)) {
        cs = backend->block_backend->as_block(py::cast(cs), dtype_opt, device_opt);
    }
    if (deep) {
        cs = backend->block_backend->copy_block(cs, device_opt);
    }
    return std::make_shared<ChargedTensor>(inv, cs);
}

Tensor::Ptr
ChargedTensor::dagger() const
{
    // Match free-function dagger(ChargedTensor); dagger the invariant part in C++.
    auto labs = invariant_part->labels();
    LegLabels dual_rev;
    dual_rev.reserve(labs.size());
    for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
        dual_rev.push_back(_dual_leg_label(*it));
    }
    auto inv_data = backend->dagger(invariant_part);
    auto inv_sym = std::make_shared<SymmetricTensor>(inv_data,
                                                     invariant_part->domain,
                                                     invariant_part->codomain,
                                                     backend,
                                                     symmetry,
                                                     std::move(dual_rev));
    auto inv_part = py::cast(inv_sym);
    inv_part.attr("set_label")(0, _CHARGE_LEG_LABEL);
    inv_part = tensors_mod().attr("move_leg")(
      inv_part, 0, py::arg("domain_pos") = 0, py::arg("bend_right") = true);
    BlockBackend::BlockPtr cs = backend->block_backend->conj(charged_state);
    return std::get<ChargedTensor::Ptr>(
      from_invariant_part(inv_part.cast<SymmetricTensor::Ptr>(), cs));
}

BlockBackend::Scalar
ChargedTensor::_get_item(std::vector<int64> const& idx)
{
    // --- hints from Python ChargedTensor._get_item ---
    // should do sth smarter...
    // ---
    if (charged_state->shape()[0] > 10) {
        throw NotImplemented("ChargedTensor::_get_item for large charged_state");
    }
    auto bb = backend->block_backend;
    auto acc = bb->as_scalar(dtype::zero_scalar(dtype), dtype);
    int64 n_charge = charged_state->shape()[0];
    for (int64 n = 0; n < n_charge; ++n) {
        auto a = charged_state->get_item(n);
        std::vector<int64> inv_idx = idx;
        inv_idx.push_back(n);
        auto term = a * invariant_part->_get_item(inv_idx);
        acc = acc + term;
    }
    return acc;
}

void
ChargedTensor::move_to_device(std::string device_in)
{
    invariant_part->move_to_device(device_in);
    device = invariant_part->device;
    charged_state =
      backend->block_backend->as_block(py::cast(charged_state), std::nullopt, device);
}

std::vector<std::string>
ChargedTensor::_repr_header_lines(std::string const& indent, bool use_symm_str) const
{
    auto linewidth = get_config().print_linewidth;
    auto lines = Tensor::_repr_header_lines(indent, use_symm_str);
    Space::Ptr charge_space = as_space(charge_leg);
    lines.push_back(
      std::format("{}* Charge Leg: dim={} sectors={}",
                  indent,
                  std::round(charge_leg->dim * 1000.) / 1000.,
                  py::str(py::cast(charge_space->sector_decomposition)).cast<std::string>()));
    std::string start = indent + "* Charged State: ";
    auto state_lines = backend->block_backend->_block_repr_lines(
      charged_state, indent + "  ", linewidth - static_cast<int64>(start.size()), 1);
    lines.push_back(start + state_lines[0]);
    return lines;
}

LabelledLegs&
ChargedTensor::set_label(int64 pos, LegLabel label)
{
    pos = to_valid_idx(pos, num_legs);
    if (label_contains_exclamation(label)) {
        throw std::invalid_argument(
          "ChargedTensor public labels must not contain '!'. Use HiddenLegTensor to hide legs.");
    }
    invariant_part->set_label(pos, label);
    return LabelledLegs::set_label(pos, label);
}

Tensor&
ChargedTensor::set_labels(LegLabels labels_in)
{
    reject_exclamation_in_labels(labels_in, "ChargedTensor");
    Tensor::set_labels(labels_in);
    auto inv_labs = labels();
    inv_labs.emplace_back(std::string(_CHARGE_LEG_LABEL));
    invariant_part->allow_charge_leg_label = true;
    invariant_part->set_labels(std::move(inv_labs));
    return *this;
}

Tensor::Ptr
ChargedTensor::to_backend(TensorBackend::Ptr new_backend,
                          std::optional<Dtype> dtype_opt,
                          std::optional<std::string> device_opt)
{
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }
    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });

    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(
      invariant_part->to_backend(new_backend, dtype_opt, device_s));
    assert(inv);
    BlockBackend::BlockPtr cs;
    if (charged_state) {
        cs = new_backend->block_backend->as_block(py::cast(charged_state), dtype_opt, device_s);
    }
    return std::make_shared<ChargedTensor>(inv, cs);
}

BlockBackend::BlockPtr
ChargedTensor::to_dense_block(
  std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
  std::optional<Dtype> dtype_opt,
  bool understood_braiding)
{
    auto inv_block = invariant_part->to_dense_block(std::nullopt, dtype_opt, understood_braiding);
    auto block = backend->block_backend->tdot(inv_block, charged_state, { -1 }, { 0 });
    if (dtype_opt.has_value()) {
        block = backend->block_backend->to_dtype(block, *dtype_opt);
    }
    if (leg_order.has_value()) {
        block = backend->block_backend->permute_axes(block, get_leg_idcs(*leg_order));
    }
    return block;
}

BlockBackend::BlockPtr
ChargedTensor::to_dense_block_single_sector()
{
    if (num_legs > 1) {
        throw std::invalid_argument("Expected a single leg");
    }
    Space::Ptr charge_space = as_space(charge_leg);
    if (charge_space->num_sectors != 1 || charge_space->multiplicities[0] != 1) {
        throw std::invalid_argument("Not a single sector.");
    }
    auto sector_dims = charge_space->sector_dims;
    if (sector_dims.has_value() && (*sector_dims)[0] > 1) {
        throw NotImplemented("to_dense_block_single_sector does not support higher-dim sectors");
    }
    auto block = backend->inv_part_to_dense_block_single_sector(invariant_part);
    return backend->block_backend->item(charged_state) * *block;
}

void
ChargedTensor::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    hdf5_saver.attr("save")(py::cast(invariant_part), subpath + "invariant_part");
    hdf5_saver.attr("save")(py::cast(charged_state), subpath + "charged_state");
    h5gr.attr("attrs")["has_charged_state"] = true;
    h5gr.attr("attrs")["dtype"] = dtype::repr(dtype);
    h5gr.attr("attrs")["num_legs"] = num_legs;
    h5gr.attr("attrs")["shape"] = py::module_::import("numpy").attr("array")(
      py::cast(shape), py::module_::import("numpy").attr("intp"));
    if (std::ranges::all_of(_labels, [](LegLabel const& l) { return !l; })) {
        h5gr.attr("attrs")["labels"] = py::list();
    } else {
        h5gr.attr("attrs")["labels"] = py::cast(_labels);
    }
}

ChargedTensor::Ptr
ChargedTensor::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto inv = hdf5_loader.attr("load")(subpath + "invariant_part").cast<SymmetricTensor::Ptr>();
    inv->allow_charge_leg_label = true;
    bool has_cs = true;
    try {
        has_cs = hdf5_loader.attr("get_attr")(h5gr, "has_charged_state").cast<bool>();
    } catch (py::error_already_set&) {
        has_cs = true;
    }
    if (!has_cs) {
        throw std::invalid_argument(
          "HDF5 ChargedTensor without charged_state is no longer supported. "
          "Re-save or convert to HiddenLegTensor.");
    }
    auto cs = hdf5_loader.attr("load")(subpath + "charged_state").cast<BlockBackend::BlockPtr>();
    auto obj = std::make_shared<ChargedTensor>(inv, cs);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
