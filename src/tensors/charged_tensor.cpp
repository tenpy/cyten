#include <cyten/tensors/charged_tensor.h>

#include <cyten/backends/backend_factory.h>
#include <cyten/backends/no_symmetry.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/tools.h>
#include <cyten/warn.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

namespace {

SymmetricTensor::Ptr
as_symmetric_tensor(py::object obj)
{
    if (py::isinstance<SymmetricTensor>(obj)) {
        return obj.cast<SymmetricTensor::Ptr>();
    }
    // Coerce Python SymmetricTensor (e.g. from free functions) into C++.
    auto backend = obj.attr("backend").cast<TensorBackend::Ptr>();
    TensorBackend::DataPtr data;
    py::object data_obj = obj.attr("data");
    try {
        data = data_obj.cast<TensorBackend::DataPtr>();
    } catch (py::cast_error const&) {
        if (std::dynamic_pointer_cast<NoSymmetryBackend>(backend)) {
            data = NoSymmetryBackend::wrap(data_obj.cast<BlockBackend::BlockPtr>());
        } else {
            throw;
        }
    }
    return std::make_shared<SymmetricTensor>(data,
                                             obj.attr("codomain").cast<TensorProduct::Ptr>(),
                                             obj.attr("domain").cast<TensorProduct::Ptr>(),
                                             backend,
                                             obj.attr("symmetry").cast<Symmetry::Ptr>(),
                                             obj.attr("labels").cast<LegLabels>());
}

py::object
tensors_mod()
{
    return py::module_::import("cyten.tensors._tensors");
}

} // namespace

ChargedTensor::ChargedTensor(py::object invariant_part_obj, py::object charged_state_obj)
  : ChargedTensor(
      as_symmetric_tensor(invariant_part_obj),
      [&]() -> BlockBackend::BlockPtr {
          if (charged_state_obj.is_none()) {
              return nullptr;
          }
          try {
              return charged_state_obj.cast<BlockBackend::BlockPtr>();
          } catch (py::cast_error const&) {
              auto inv = as_symmetric_tensor(invariant_part_obj);
              return inv->backend->block_backend->as_block(
                charged_state_obj, inv->dtype, inv->device);
          }
      }())
{
}

ChargedTensor::ChargedTensor(SymmetricTensor::Ptr inv, BlockBackend::BlockPtr charged_state_in)
  : Tensor(
      inv->codomain,
      std::make_shared<TensorProduct>(
        std::vector<py::object>(inv->domain->factors.begin() + 1, inv->domain->factors.end()),
        inv->symmetry),
      inv->backend,
      inv->symmetry,
      [&]() {
          auto labs = inv->labels();
          assert(!labs.empty());
          return LegLabels(labs.begin(), labs.end() - 1);
      }(),
      inv->dtype,
      inv->device)
  , invariant_part(std::move(inv))
  , charged_state(std::move(charged_state_in))
  // Match Python: keep the domain factor as-is (ElementarySpace or LegPipe).
  , charge_leg(invariant_part->domain->factors[0])
{
    assert(invariant_part->domain->num_factors > 0);
    auto labs = invariant_part->labels();
    assert(!labs.empty() && labs.back() && *labs.back() == _CHARGE_LEG_LABEL);
    if (!supports_symmetry(invariant_part->symmetry)) {
        throw SymmetryError(std::format(
          "ChargedTensor is not well-defined for symmetry {}.", invariant_part->symmetry->repr()));
    }
    if (charged_state) {
        if (!invariant_part->symmetry->can_be_dropped()) {
            throw SymmetryError(std::format(
              "charged_state can not be specified for symmetry {}", invariant_part->symmetry->repr()));
        }
        charged_state = invariant_part->backend->block_backend->as_block(
          py::cast(charged_state), invariant_part->dtype, invariant_part->device);
    }
}

void
ChargedTensor::test_sanity() const
{
    Tensor::test_sanity();
    auto inv_labs = invariant_part->labels();
    assert(labels() == LegLabels(inv_labs.begin(), inv_labs.end() - 1));
    invariant_part->test_sanity();
    assert(invariant_part->device == device);
    if (charged_state) {
        backend->block_backend->test_block_sanity(
          charged_state,
          std::vector<int64>{ static_cast<int64>(charge_leg.attr("dim").cast<float64>()) },
          std::nullopt,
          device);
    }
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

py::object
ChargedTensor::as_py_object()
{
    return py::cast(std::static_pointer_cast<ChargedTensor>(shared_from_this()));
}

py::object
ChargedTensor::as_py_object() const
{
    return const_cast<ChargedTensor*>(this)->as_py_object();
}

bool
ChargedTensor::supports_symmetry(Symmetry::Ptr const& symmetry)
{
    return symmetry->has_symmetric_braid();
}

std::tuple<TensorProduct::Ptr, Space::Ptr>
ChargedTensor::_parse_inv_domain(TensorProduct::Ptr domain, py::object charge)
{
    assert(domain); // call _init_parse_args first?
    Space::Ptr charge_leg;
    if (py::isinstance<ElementarySpace>(charge)) {
        charge_leg = charge.cast<ElementarySpace::Ptr>();
    } else if (py::isinstance<Space>(charge) || py::isinstance<Leg>(charge)) {
        throw std::invalid_argument("Invalid type for charge. Expected ElementarySpace or sector");
    } else {
        Sector sec;
        if (py::isinstance<Sector>(charge)) {
            sec = charge.cast<Sector>();
        } else {
            sec = charge.cast<Sector>(); // pybind Sector caster accepts sequences
        }
        charge_leg = std::make_shared<ElementarySpace>(domain->symmetry, SectorArray::from_sector(sec));
    }
    return { domain->left_multiply(py::cast(charge_leg)), charge_leg };
}

std::tuple<LegLabels, LegLabels>
ChargedTensor::_parse_inv_labels(py::object labels,
                                 TensorProduct::Ptr const& codomain,
                                 TensorProduct::Ptr const& domain)
{
    auto labs = _init_parse_labels(labels, codomain, domain);
    auto inv_labels = labs;
    inv_labels.emplace_back(std::string(_CHARGE_LEG_LABEL));
    return { labs, inv_labels };
}

ChargedTensor::Ptr
ChargedTensor::from_block_func(py::function func,
                               py::object charge,
                               py::object codomain,
                               py::object domain,
                               py::object charged_state,
                               TensorBackend::Ptr backend,
                               py::object labels,
                               py::object func_kwargs,
                               std::optional<std::string> shape_kw,
                               std::optional<Dtype> dtype,
                               std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    (void)symmetry;
    std::string device_s;
    if (!device.has_value()) {
        if (charged_state.is_none()) {
            device_s = backend_tp->block_backend->default_device;
        } else {
            // may be raw array — as_block later; try get_device if already Block
            try {
                device_s = backend_tp->block_backend->get_device(charged_state.cast<BlockBackend::BlockPtr>());
            } catch (py::cast_error const&) {
                device_s = backend_tp->block_backend->get_device(
                  backend_tp->block_backend->as_block(charged_state, std::nullopt, std::nullopt));
            }
        }
    } else {
        device_s = *device;
    }
    auto [inv_domain, charge_leg] = _parse_inv_domain(domain_tp, charge);
    (void)charge_leg;
    auto inv = SymmetricTensor::from_block_func(func,
                                                py::cast(codomain_tp),
                                                py::cast(inv_domain),
                                                backend_tp,
                                                labels,
                                                func_kwargs,
                                                shape_kw,
                                                dtype,
                                                std::optional<std::string>{ device_s });
    return std::make_shared<ChargedTensor>(py::cast(inv), charged_state);
}

ChargedTensor::Ptr
ChargedTensor::from_dense_block(py::object block,
                                py::object codomain,
                                py::object domain,
                                py::object charge,
                                TensorBackend::Ptr backend,
                                py::object labels,
                                std::optional<Dtype> dtype,
                                std::optional<std::string> device,
                                float64 tol,
                                bool understood_braiding)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    auto [labs, inv_labels] = _parse_inv_labels(labels, codomain_tp, domain_tp);
    (void)labs;
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    auto block_ptr = backend_tp->block_backend->as_block(block, dtype, device);
    if (charge.is_none()) {
        throw NotImplemented("ChargedTensor::from_dense_block with charge=None");
    }
    auto [inv_domain, charge_leg] = _parse_inv_domain(domain_tp, charge);
    if (charge_leg->Space::dim != 1.) {
        throw NotImplemented("ChargedTensor::from_dense_block with charge_leg.dim != 1");
    }
    auto inv_part = SymmetricTensor::from_dense_block(
      py::cast(backend_tp->block_backend->add_axis(block_ptr, -1)),
      py::cast(codomain_tp),
      py::cast(inv_domain),
      backend_tp,
      py::cast(inv_labels),
      std::nullopt,
      std::nullopt,
      tol,
      understood_braiding);
    return std::make_shared<ChargedTensor>(py::cast(inv_part), py::cast(std::vector<int64>{ 1 }));
}

ChargedTensor::Ptr
ChargedTensor::from_dense_block_single_sector(py::object /*vector*/,
                                              py::object /*space*/,
                                              Sector /*sector*/,
                                              TensorBackend::Ptr /*backend*/,
                                              std::optional<std::string> /*label*/,
                                              std::optional<std::string> /*device*/)
{
    // --- hints from Python ChargedTensor.from_dense_block_single_sector ---
    // how to handle multi-dim sectors? which dummy leg state to give?
    // ---
    throw NotImplemented("ChargedTensor::from_dense_block_single_sector");
}

py::object
ChargedTensor::from_invariant_part(py::object invariant_part_obj, py::object charged_state)
{
    // --- hints from Python ChargedTensor.from_invariant_part ---
    // OPTIMIZE ?
    // ---
    auto inv = as_symmetric_tensor(invariant_part_obj);
    if (inv->num_legs == 1) {
        if (charged_state.is_none()) {
            throw std::invalid_argument(
              "Can not instantiate ChargedTensor with no legs and unspecified charged_states.");
        }
        // OPTIMIZE ?
        auto inv_block = inv->to_dense_block(std::nullopt, std::nullopt, /*understood_braiding=*/true);
        auto state = inv->backend->block_backend->as_block(charged_state, inv->dtype, inv->device);
        return py::cast(inv->backend->block_backend->inner(inv_block, state, /*do_dagger=*/false));
    }
    return py::cast(std::make_shared<ChargedTensor>(inv, [&]() -> BlockBackend::BlockPtr {
        if (charged_state.is_none()) {
            return nullptr;
        }
        try {
            return charged_state.cast<BlockBackend::BlockPtr>();
        } catch (py::cast_error const&) {
            return inv->backend->block_backend->as_block(charged_state, inv->dtype, inv->device);
        }
    }()));
}

py::object
ChargedTensor::from_two_charge_legs(py::object invariant_part_obj,
                                    py::object state1,
                                    py::object state2)
{
    // Uses combine_legs free function — keep via Python helper when needed.
    auto inv_obj = invariant_part_obj;
    auto labs = inv_obj.attr("labels");
    assert(std::string(py::str(labs[py::int_(-1)])).starts_with(_CHARGE_LEG_LABEL));
    assert(std::string(py::str(labs[py::int_(-2)])).starts_with(_CHARGE_LEG_LABEL));
    auto inv_part = tensors_mod().attr("combine_legs")(inv_obj, py::make_tuple(-2, -1));
    inv_part.attr("set_label")(-1, _CHARGE_LEG_LABEL);
    py::object state;
    if (state1.is_none() && state2.is_none()) {
        state = py::none();
    } else if (state1.is_none() || state2.is_none()) {
        throw std::invalid_argument("Must specify either both or none of the states");
    } else {
        auto backend = inv_obj.attr("backend").cast<TensorBackend::Ptr>();
        auto pipe = inv_part.attr("domain").attr("__getitem__")(0).cast<LegPipe::Ptr>();
        state = backend->state_tensor_product(state1.cast<BlockBackend::BlockPtr>(),
                                              state2.cast<BlockBackend::BlockPtr>(),
                                              pipe);
    }
    return from_invariant_part(inv_part, state);
}

ChargedTensor::Ptr
ChargedTensor::from_zero(py::object codomain,
                         py::object domain,
                         py::object charge,
                         py::object charged_state,
                         TensorBackend::Ptr backend,
                         py::object labels,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    (void)symmetry;
    std::string device_s;
    if (!device.has_value()) {
        if (charged_state.is_none()) {
            device_s = backend_tp->block_backend->default_device;
        } else {
            try {
                device_s = backend_tp->block_backend->get_device(charged_state.cast<BlockBackend::BlockPtr>());
            } catch (py::cast_error const&) {
                device_s = backend_tp->block_backend->get_device(
                  backend_tp->block_backend->as_block(charged_state, std::nullopt, std::nullopt));
            }
        }
    } else {
        device_s = *device;
    }
    auto [inv_domain, charge_leg] = _parse_inv_domain(domain_tp, charge);
    (void)charge_leg;
    auto [labs, inv_labels] = _parse_inv_labels(labels, codomain_tp, domain_tp);
    (void)labs;
    auto inv_part = SymmetricTensor::from_zero(py::cast(codomain_tp),
                                               py::cast(inv_domain),
                                               backend_tp,
                                               py::cast(inv_labels),
                                               dtype,
                                               device_s);
    return std::make_shared<ChargedTensor>(py::cast(inv_part), charged_state);
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

py::object
ChargedTensor::as_SymmetricTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    // LegPipe charge legs (from combine_legs) expose Space APIs via as_Space().
    Space::Ptr charge_space =
      py::isinstance<Space>(charge_leg) ? charge_leg.cast<Space::Ptr>()
                                        : charge_leg.attr("as_Space")().cast<Space::Ptr>();
    if (charge_space->sector_decomposition.size() != 1 ||
        charge_space->sector_decomposition[0] != symmetry->trivial_sector) {
        throw SymmetryError("Not a symmetric tensor");
    }
    if (charge_leg.attr("dim").cast<float64>() == 1.) {
        auto res = tensors_mod().attr("squeeze_legs")(py::cast(invariant_part), -1);
        if (charged_state) {
            auto scale = backend->block_backend->item(charged_state);
            res = res.attr("__mul__")(py::cast(scale));
        }
        return res;
    }
    if (!charged_state) {
        throw std::invalid_argument("Can not convert to SymmetricTensor. charged_state is not defined.");
    }
    // charge_leg.dual (Python wrote charged_state.dual — treat as charge_leg.dual)
    auto state = SymmetricTensor::from_dense_block(
      py::cast(charged_state),
      py::make_tuple(charge_leg.attr("dual")),
      py::none(),
      backend,
      py::make_tuple(_dual_leg_label(std::string(_CHARGE_LEG_LABEL))),
      dtype,
      std::nullopt,
      1e-6,
      /*understood_braiding=*/true);
    auto res = tensors_mod().attr("tdot")(py::cast(state), py::cast(invariant_part), 0, -1);
    return tensors_mod().attr("bend_legs")(res, py::arg("num_codomain_legs") = num_codomain_legs());
}

Tensor::Ptr
ChargedTensor::copy(bool deep, std::optional<std::string> device_opt, std::optional<Dtype> dtype_opt)
{
    auto inv = std::dynamic_pointer_cast<SymmetricTensor>(
      invariant_part->copy(deep, device_opt, dtype_opt));
    assert(inv);
    BlockBackend::BlockPtr cs = charged_state;
    if (cs) {
        if ((device_opt.has_value() && !deep) || (dtype_opt.has_value() && *dtype_opt != dtype)) {
            cs = backend->block_backend->as_block(py::cast(cs), dtype_opt, device_opt);
        }
        if (deep) {
            cs = backend->block_backend->copy_block(cs, device_opt);
        }
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
    auto inv_data = backend->dagger(py::cast(invariant_part));
    auto inv_sym = std::make_shared<SymmetricTensor>(
      inv_data, invariant_part->domain, invariant_part->codomain, backend, symmetry, std::move(dual_rev));
    auto inv_part = py::cast(inv_sym);
    inv_part.attr("set_label")(0, _CHARGE_LEG_LABEL);
    inv_part = tensors_mod().attr("move_leg")(
      inv_part, 0, py::arg("domain_pos") = 0, py::arg("bend_right") = true);
    py::object cs = py::none();
    if (charged_state) {
        cs = py::cast(backend->block_backend->conj(charged_state));
    }
    return from_invariant_part(inv_part, cs).cast<ChargedTensor::Ptr>();
}

BlockBackend::Scalar
ChargedTensor::_get_item(std::vector<int64> const& idx)
{
    // --- hints from Python ChargedTensor._get_item ---
    // should do sth smarter...
    // ---
    if (!charged_state) {
        throw std::out_of_range("Can not index a ChargedTensor with unspecified charged_state.");
    }
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
    if (charged_state) {
        charged_state = backend->block_backend->as_block(py::cast(charged_state), std::nullopt, device);
    }
}

std::vector<std::string>
ChargedTensor::_repr_header_lines(std::string const& indent, bool use_symm_str) const
{
    auto linewidth = get_config().print_linewidth;
    auto lines = Tensor::_repr_header_lines(indent, use_symm_str);
    Space::Ptr charge_space =
      py::isinstance<Space>(charge_leg) ? charge_leg.cast<Space::Ptr>()
                                        : charge_leg.attr("as_Space")().cast<Space::Ptr>();
    lines.push_back(std::format("{}* Charge Leg: dim={} sectors={}",
                                indent,
                                std::round(charge_leg.attr("dim").cast<float64>() * 1000.) / 1000.,
                                py::str(py::cast(charge_space->sector_decomposition)).cast<std::string>()));
    std::string start = indent + "* Charged State: ";
    if (!charged_state) {
        lines.push_back(start + "unspecified");
    } else {
        auto state_lines = backend->block_backend->_block_repr_lines(
          charged_state, indent + "  ", linewidth - static_cast<int64>(start.size()), 1);
        lines.push_back(start + state_lines[0]);
    }
    return lines;
}

LabelledLegs&
ChargedTensor::set_label(int64 pos, LegLabel label)
{
    pos = to_valid_idx(pos, num_legs);
    invariant_part->set_label(pos, label);
    return LabelledLegs::set_label(pos, label);
}

Tensor&
ChargedTensor::set_labels(LegLabels labels_in)
{
    Tensor::set_labels(labels_in);
    auto inv_labs = labels();
    inv_labs.emplace_back(std::string(_CHARGE_LEG_LABEL));
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
    if (!charged_state) {
        throw std::invalid_argument("charged_state not specified.");
    }
    auto inv_block =
      invariant_part->to_dense_block(std::nullopt, dtype_opt, understood_braiding);
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
    if (!charged_state) {
        throw std::invalid_argument("Unspecified charged_state");
    }
    if (num_legs > 1) {
        throw std::invalid_argument("Expected a single leg");
    }
    Space::Ptr charge_space =
      py::isinstance<Space>(charge_leg) ? charge_leg.cast<Space::Ptr>()
                                        : charge_leg.attr("as_Space")().cast<Space::Ptr>();
    if (charge_space->num_sectors != 1 || charge_space->multiplicities[0] != 1) {
        throw std::invalid_argument("Not a single sector.");
    }
    auto sector_dims = charge_space->sector_dims;
    if (sector_dims.has_value() && (*sector_dims)[0] > 1) {
        throw NotImplemented(
          "to_dense_block_single_sector does not support higher-dim sectors");
    }
    auto block = backend->inv_part_to_dense_block_single_sector(py::cast(invariant_part));
    return backend->block_backend->item(charged_state) * *block;
}

void
ChargedTensor::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    hdf5_saver.attr("save")(py::cast(invariant_part), subpath + "invariant_part");
    if (charged_state) {
        hdf5_saver.attr("save")(py::cast(charged_state), subpath + "charged_state");
        h5gr.attr("attrs")["has_charged_state"] = true;
    } else {
        h5gr.attr("attrs")["has_charged_state"] = false;
    }
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
    BlockBackend::BlockPtr cs;
    bool has_cs = false;
    try {
        has_cs = hdf5_loader.attr("get_attr")(h5gr, "has_charged_state").cast<bool>();
    } catch (py::error_already_set&) {
        has_cs = false;
    }
    if (has_cs) {
        cs = hdf5_loader.attr("load")(subpath + "charged_state").cast<BlockBackend::BlockPtr>();
    }
    auto obj = std::make_shared<ChargedTensor>(inv, cs);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
