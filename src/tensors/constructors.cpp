#include <cyten/tensors/constructors.h>

#include <cyten/backends/backend_factory.h>
#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tools.h>

#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

namespace {

std::vector<int64>
cumsum_with_leading_zero(std::vector<int64> const& mults)
{
    std::vector<int64> out;
    out.reserve(mults.size() + 1);
    out.push_back(0);
    int64 running = 0;
    for (auto m : mults) {
        running += m;
        out.push_back(running);
    }
    return out;
}

bool
legs_equal(Leg::Ptr const& a, Leg::Ptr const& b)
{
    return a && b && a->operator==(*b);
}

bool
products_equal(TensorProduct::Ptr const& a, TensorProduct::Ptr const& b)
{
    return a && b && a->operator==(*b);
}

std::vector<Leg::Ptr>
factors_slice(TensorProduct::Ptr const& tp, int64 start, int64 stop)
{
    auto const n = static_cast<int64>(tp->factors.size());
    if (start < 0) {
        start += n;
    }
    if (stop < 0) {
        stop += n;
    }
    std::vector<Leg::Ptr> out;
    for (int64 i = start; i < stop; ++i) {
        out.push_back(tp->factors[static_cast<std::size_t>(i)]);
    }
    return out;
}

bool
factor_slices_equal(std::vector<Leg::Ptr> const& a, std::vector<Leg::Ptr> const& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!legs_equal(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

} // namespace

TensorPtr
eye(Space::Ptr leg,
    TensorBackend::Ptr backend,
    std::optional<LegLabels> labels,
    Dtype dtype,
    std::optional<std::string> device,
    bool diagonal)
{
    auto res = DiagonalTensor::from_eye(
      std::move(leg), std::move(backend), std::move(labels), dtype, std::move(device));
    if (diagonal) {
        return res;
    }
    return res->as_SymmetricTensor();
}

TensorPtr
tensor(TensorCPtr obj,
       TensorProduct::Ptr codomain,
       TensorProduct::Ptr domain,
       TensorBackend::Ptr backend,
       std::optional<LegLabels> labels,
       std::optional<Dtype> dtype,
       std::optional<std::string> device)
{
    if (!products_equal(codomain, obj->codomain)) {
        throw std::invalid_argument("Mismatching codomain");
    }
    if (domain && !products_equal(domain, obj->domain)) {
        throw std::invalid_argument("Mismatching domain");
    }
    if (backend && (!obj->backend || !(*backend == *obj->backend))) {
        throw std::invalid_argument("Mismatching backend");
    }
    TensorPtr out;
    if (labels.has_value() && *labels != obj->labels()) {
        out = std::const_pointer_cast<Tensor>(obj)->copy();
        out->set_labels(*labels);
    } else {
        out = std::const_pointer_cast<Tensor>(obj);
    }
    if (dtype.has_value()) {
        throw std::invalid_argument("Mismatching dtype");
    }
    if (device.has_value()) {
        throw std::invalid_argument("Mismatching device");
    }
    return out->as_SymmetricTensor();
}

SymmetricTensorPtr
tensor(BlockBackend::BlockPtr obj,
       TensorProduct::Ptr codomain,
       TensorProduct::Ptr domain,
       TensorBackend::Ptr backend,
       std::optional<LegLabels> labels,
       std::optional<Dtype> dtype,
       std::optional<std::string> device,
       bool understood_braiding)
{
    return SymmetricTensor::from_dense_block(std::move(obj),
                                             std::move(codomain),
                                             std::move(domain),
                                             std::move(backend),
                                             std::move(labels),
                                             dtype,
                                             std::move(device),
                                             1e-6,
                                             understood_braiding);
}

TensorPtr
add_trivial_leg(TensorCPtr tens,
                std::optional<int64> legs_pos_opt,
                std::optional<int64> codomain_pos_opt,
                std::optional<int64> domain_pos_opt,
                LegLabel label,
                bool is_dual)
{
    // --- hints from Python add_trivial_leg ---
    // parse position to format:
    // - leg_pos: int,  0 <= leg_pos < res_num_legs
    // - add_to_domain: bool
    // - co_domain_pos: int, 0 <= co_domain_pos < num_[co]domain_legs
    // - is_dual: bool, if the leg in the [co]domain should be dual
    // domain[0] is the charge leg, so we need to add 1
    // ---
    int64 res_num_legs = tens->num_legs + 1;
    int64 legs_pos = 0;
    int64 co_domain_pos = 0;
    bool add_to_domain = false;

    if (legs_pos_opt.has_value()) {
        if (codomain_pos_opt.has_value() || domain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        legs_pos = to_valid_idx(*legs_pos_opt, res_num_legs);
        add_to_domain = legs_pos > tens->num_codomain_legs();
        if (add_to_domain) {
            co_domain_pos = res_num_legs - 1 - legs_pos;
        } else {
            co_domain_pos = legs_pos;
        }
    } else if (codomain_pos_opt.has_value()) {
        if (legs_pos_opt.has_value() || domain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        int64 res_codomain_legs = tens->num_codomain_legs() + 1;
        int64 codomain_pos = to_valid_idx(*codomain_pos_opt, res_codomain_legs);
        add_to_domain = false;
        co_domain_pos = codomain_pos;
        legs_pos = codomain_pos;
    } else if (domain_pos_opt.has_value()) {
        if (legs_pos_opt.has_value() || codomain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        int64 res_domain_legs = tens->num_domain_legs() + 1;
        int64 domain_pos = to_valid_idx(*domain_pos_opt, res_domain_legs);
        add_to_domain = true;
        co_domain_pos = domain_pos;
        legs_pos = res_num_legs - 1 - domain_pos;
    } else {
        add_to_domain = false;
        co_domain_pos = 0;
        legs_pos = 0;
    }

    if (std::dynamic_pointer_cast<DiagonalTensor const>(tens) ||
        std::dynamic_pointer_cast<Mask const>(tens) ||
        std::dynamic_pointer_cast<Identity const>(tens)) {
        std::string msg = "Converting to SymmetricTensor for add_trivial_leg. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tens = std::const_pointer_cast<Tensor>(tens)->as_SymmetricTensor(false, std::move(msg));
    }
    if (auto charged = std::dynamic_pointer_cast<ChargedTensor const>(tens)) {
        TensorPtr inv_part;
        if (add_to_domain) {
            // domain[0] is the charge leg, so we need to add 1
            inv_part = add_trivial_leg(charged->invariant_part,
                                       /*legs_pos=*/std::nullopt,
                                       /*codomain_pos=*/std::nullopt,
                                       /*domain_pos=*/co_domain_pos + 1,
                                       label,
                                       is_dual);
        } else {
            inv_part = add_trivial_leg(charged->invariant_part,
                                       /*legs_pos=*/std::nullopt,
                                       /*codomain_pos=*/co_domain_pos,
                                       /*domain_pos=*/std::nullopt,
                                       label,
                                       is_dual);
        }
        auto inv_sym = std::dynamic_pointer_cast<SymmetricTensor>(inv_part);
        if (!inv_sym) {
            throw std::runtime_error("add_trivial_leg expected SymmetricTensor invariant_part");
        }
        return std::make_shared<ChargedTensor>(std::move(inv_sym), charged->charged_state);
    }
    auto sym = std::dynamic_pointer_cast<SymmetricTensor const>(tens);
    if (!sym) {
        throw py::type_error("Invalid type for tens. Expected a Tensor subtype");
    }

    auto new_leg = ElementarySpace::from_trivial_sector(1, tens->symmetry, is_dual);
    TensorProduct::Ptr domain;
    TensorProduct::Ptr codomain;
    if (add_to_domain) {
        domain = tens->domain->insert_multiply(new_leg, co_domain_pos);
        codomain = tens->codomain;
    } else {
        domain = tens->domain;
        codomain = tens->codomain->insert_multiply(new_leg, co_domain_pos);
    }
    auto backend = tens->backend;
    auto data =
      backend->add_trivial_leg(tens, legs_pos, add_to_domain, co_domain_pos, codomain, domain);

    LegLabels labels = tens->labels();
    LegLabels new_labels;
    new_labels.reserve(labels.size() + 1);
    new_labels.insert(new_labels.end(), labels.begin(), labels.begin() + legs_pos);
    new_labels.push_back(label);
    new_labels.insert(new_labels.end(), labels.begin() + legs_pos, labels.end());

    return std::make_shared<SymmetricTensor>(std::move(data),
                                             std::move(codomain),
                                             std::move(domain),
                                             backend,
                                             tens->symmetry,
                                             new_labels);
}

TensorPtr
zero_like(TensorCPtr tensor)
{
    if (auto mask = std::dynamic_pointer_cast<Mask const>(tensor)) {
        return Mask::from_zero(mask->large_leg(), mask->backend, mask->labels(), mask->device);
    }
    if (auto diag = std::dynamic_pointer_cast<DiagonalTensor const>(tensor)) {
        return DiagonalTensor::from_zero(
          diag->leg(), diag->backend, diag->labels(), diag->dtype, diag->device);
    }
    if (auto charged = std::dynamic_pointer_cast<ChargedTensor const>(tensor)) {
        auto charge = std::dynamic_pointer_cast<ElementarySpace>(charged->charge_leg);
        if (!charge) {
            throw std::invalid_argument("zero_like: charge_leg must be an ElementarySpace");
        }
        return ChargedTensor::from_zero(charged->codomain,
                                        charged->domain,
                                        charge,
                                        charged->charged_state,
                                        charged->backend,
                                        charged->labels(),
                                        charged->dtype,
                                        charged->device);
    }
    if (auto sym = std::dynamic_pointer_cast<SymmetricTensor const>(tensor)) {
        return SymmetricTensor::from_zero(
          sym->codomain, sym->domain, sym->backend, sym->labels(), sym->dtype, sym->device);
    }
    throw py::type_error("Invalid type for tensor.");
}

TensorPtr
tensor_from_grid(std::vector<std::vector<TensorPtr>> grid,
                 std::optional<LegLabels> labels,
                 std::optional<Dtype> dtype_opt)
{
    // --- hints from Python tensor_from_grid ---
    // check input
    // only ElementarySpaces have direct_sum
    // find op from same column
    // find op from same row
    // for each sector in the direct sum, find which multiplicities come from which space
    // ---
    std::vector<TensorCPtr> ops;
    for (auto const& row : grid) {
        for (auto const& op : row) {
            if (op) {
                ops.push_back(op);
            }
        }
    }
    if (ops.empty()) {
        throw std::invalid_argument("grid must contain at least one tensor");
    }

    auto backend = get_same_backend(ops);
    std::string device = ops[0]->device;
    for (auto const& op : ops) {
        if (op->device != device) {
            throw std::invalid_argument("Incompatible devices.");
        }
    }

    Dtype dtype;
    if (dtype_opt.has_value()) {
        dtype = *dtype_opt;
    } else {
        std::vector<Dtype> dtypes;
        dtypes.reserve(ops.size());
        for (auto const& op : ops) {
            dtypes.push_back(op->dtype);
        }
        dtype = dtype::common(dtypes);
    }

    auto const& ref = ops[0];
    int64 n_cod = ref->num_codomain_legs();
    int64 n_dom = ref->num_domain_legs();
    auto ref_cod_tail =
      factors_slice(ref->codomain, 1, static_cast<int64>(ref->codomain->factors.size()));
    auto ref_dom_head = factors_slice(ref->domain, 0, -1);

    for (auto const& op : ops) {
        if (op->num_codomain_legs() != n_cod || op->num_domain_legs() != n_dom) {
            throw std::runtime_error("inconsistent number of legs in grid");
        }
        if (!factor_slices_equal(
              factors_slice(op->codomain, 1, static_cast<int64>(op->codomain->factors.size())),
              ref_cod_tail) ||
            !factor_slices_equal(factors_slice(op->domain, 0, -1), ref_dom_head)) {
            throw std::runtime_error("inconsistent legs in grid");
        }
        if (!std::dynamic_pointer_cast<ElementarySpace>((*op->codomain)[0]) ||
            !std::dynamic_pointer_cast<ElementarySpace>((*op->domain)[-1])) {
            throw std::runtime_error("stacking legs must be ElementarySpace");
        }
    }

    auto n_rows = static_cast<int64>(grid.size());
    int64 n_cols = n_rows > 0 ? static_cast<int64>(grid[0].size()) : 0;
    for (auto const& row : grid) {
        if (static_cast<int64>(row.size()) != n_cols) {
            throw std::invalid_argument("grid rows must have equal length");
        }
    }

    std::vector<TensorPtr> right_ops(static_cast<std::size_t>(n_cols));
    if (n_rows > 0) {
        right_ops = grid[0];
    }
    for (int64 i = 0; i < n_cols; ++i) {
        if (right_ops[static_cast<std::size_t>(i)]) {
            continue;
        }
        for (int64 r = 0; r < n_rows; ++r) {
            auto const& new_op = grid[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)];
            if (!new_op) {
                continue;
            }
            right_ops[static_cast<std::size_t>(i)] = new_op;
            break;
        }
    }
    for (auto const& op : right_ops) {
        if (!op) {
            throw std::invalid_argument("Must have at least one nonzero entry in each column.");
        }
    }
    std::vector<ElementarySpace::Ptr> right_spaces;
    right_spaces.reserve(right_ops.size());
    for (auto const& op : right_ops) {
        right_spaces.push_back(std::dynamic_pointer_cast<ElementarySpace>((*op->domain)[-1]));
    }

    std::vector<TensorPtr> left_ops(static_cast<std::size_t>(n_rows));
    for (int64 i = 0; i < n_rows; ++i) {
        left_ops[static_cast<std::size_t>(i)] = grid[static_cast<std::size_t>(i)][0];
    }
    for (int64 i = 0; i < n_rows; ++i) {
        if (left_ops[static_cast<std::size_t>(i)]) {
            continue;
        }
        for (int64 c = 0; c < n_cols; ++c) {
            auto const& new_op = grid[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            if (!new_op) {
                continue;
            }
            left_ops[static_cast<std::size_t>(i)] = new_op;
            break;
        }
    }
    for (auto const& op : left_ops) {
        if (!op) {
            throw std::invalid_argument("Must have at least one nonzero entry in each row.");
        }
    }
    std::vector<ElementarySpace::Ptr> left_spaces;
    left_spaces.reserve(left_ops.size());
    for (auto const& op : left_ops) {
        left_spaces.push_back(std::dynamic_pointer_cast<ElementarySpace>((*op->codomain)[0]));
    }

    std::vector<ElementarySpace::Ptr> left_rest(left_spaces.begin() + 1, left_spaces.end());
    std::vector<ElementarySpace::Ptr> right_rest(right_spaces.begin() + 1, right_spaces.end());
    auto left_space = left_spaces[0]->direct_sum(left_rest);
    auto right_space = right_spaces[0]->direct_sum(right_rest);

    std::vector<std::vector<int64>> left_mult_slices;
    std::vector<std::vector<int64>> right_mult_slices;
    if (auto dss = std::dynamic_pointer_cast<DirectSumSpace>(left_space)) {
        left_mult_slices = dss->mult_slices();
    } else {
        // Single summand (n_rows == 1): one slice covering the full multiplicity.
        for (auto const& sector : left_space->sector_decomposition) {
            auto idx = left_space->sector_decomposition_where(sector);
            int64 m = idx.has_value() ? left_space->multiplicities[static_cast<std::size_t>(*idx)]
                                      : int64{ 0 };
            left_mult_slices.push_back(cumsum_with_leading_zero(std::vector<int64>{ m }));
        }
    }
    if (auto dss = std::dynamic_pointer_cast<DirectSumSpace>(right_space)) {
        right_mult_slices = dss->mult_slices();
    } else {
        for (auto const& sector : right_space->sector_decomposition) {
            auto idx = right_space->sector_decomposition_where(sector);
            int64 m = idx.has_value() ? right_space->multiplicities[static_cast<std::size_t>(*idx)]
                                      : int64{ 0 };
            right_mult_slices.push_back(cumsum_with_leading_zero(std::vector<int64>{ m }));
        }
    }

    std::vector<Leg::Ptr> cod_legs;
    cod_legs.push_back(left_space);
    auto ref_cod_rest =
      factors_slice(ref->codomain, 1, static_cast<int64>(ref->codomain->factors.size()));
    cod_legs.insert(cod_legs.end(), ref_cod_rest.begin(), ref_cod_rest.end());
    std::vector<Leg::Ptr> dom_legs = factors_slice(ref->domain, 0, -1);
    dom_legs.push_back(right_space);

    auto codomain = std::make_shared<TensorProduct>(std::move(cod_legs));
    auto domain = std::make_shared<TensorProduct>(std::move(dom_legs));

    std::vector<std::vector<py::object>> py_grid(
      static_cast<std::size_t>(n_rows), std::vector<py::object>(static_cast<std::size_t>(n_cols)));
    for (int64 i = 0; i < n_rows; ++i) {
        for (int64 j = 0; j < n_cols; ++j) {
            auto const& t = grid[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            py_grid[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
              t ? py::cast(t) : py::none();
        }
    }

    auto data = backend->from_grid(std::move(py_grid),
                                   codomain,
                                   domain,
                                   std::move(left_mult_slices),
                                   std::move(right_mult_slices),
                                   dtype,
                                   device);
    LegLabels labs = labels.value_or(LegLabels{});
    if (!labels.has_value()) {
        labs = Tensor::_init_parse_labels(std::nullopt, codomain, domain);
    }
    return std::make_shared<SymmetricTensor>(std::move(data),
                                             std::move(codomain),
                                             std::move(domain),
                                             backend,
                                             ref->symmetry,
                                             std::move(labs));
}

namespace {

[[nodiscard]] int64
normalize_summand_index(DirectSumSpace const& space, int64 i)
{
    auto const n = static_cast<int64>(space.spaces.size());
    if (i < 0) {
        i += n;
    }
    if (i < 0 || i >= n) {
        throw std::invalid_argument(
          std::format("summand index {} out of range for DirectSumSpace with {} summands", i, n));
    }
    return i;
}

[[nodiscard]] TensorBackend::Ptr
resolve_backend_for_space(TensorBackend::Ptr backend, Space::Ptr const& space)
{
    if (!backend) {
        return get_backend(space->symmetry);
    }
    return backend;
}

} // namespace

MaskPtr
DirectSumSpace::projection_onto_summand(int64 i,
                                        std::shared_ptr<TensorBackend> backend,
                                        std::optional<LegLabels> labels,
                                        std::optional<std::string> device) const
{
    i = normalize_summand_index(*this, i);
    backend = resolve_backend_for_space(std::move(backend), shared_es());

    auto const slices = mult_slices();
    auto space_cap = shared_dss();
    auto i_cap = i;
    auto np = py::module_::import("numpy");
    auto bb = backend->block_backend;

    SectorBlockFactoryFn func = [space_cap, slices, i_cap, np, bb, device](
                                  std::vector<int64> const& shape, Sector const& coupled) {
        auto sector_idx = space_cap->sector_decomposition_where(coupled);
        if (!sector_idx.has_value()) {
            throw std::runtime_error(
              "DirectSumSpace::projection_onto_summand: sector missing from DirectSumSpace");
        }
        auto const& slc = slices[static_cast<std::size_t>(*sector_idx)];
        int64 const start = slc[static_cast<std::size_t>(i_cap)];
        int64 const stop = slc[static_cast<std::size_t>(i_cap) + 1];
        if (shape.empty() || shape[0] != slc.back()) {
            throw std::runtime_error(
              "DirectSumSpace::projection_onto_summand: unexpected diagonal block shape");
        }
        py::object block = np.attr("zeros")(py::cast(shape), np.attr("bool_"));
        if (stop > start) {
            block.attr("__setitem__")(py::slice(start, stop, 1), true);
        }
        return bb->as_block(block, Dtype::Bool, device);
    };

    auto diag = DiagonalTensor::from_sector_block_func(
      std::move(func), shared_es(), backend, labels, Dtype::Bool, device);
    return Mask::from_DiagonalTensor(diag);
}

MaskPtr
DirectSumSpace::inclusion_of_summand(int64 i,
                                     std::shared_ptr<TensorBackend> backend,
                                     std::optional<LegLabels> labels,
                                     std::optional<std::string> device) const
{
    auto proj =
      projection_onto_summand(i, std::move(backend), std::move(labels), std::move(device));
    auto incl = std::dynamic_pointer_cast<Mask>(proj->dagger());
    if (!incl) {
        throw std::runtime_error("Mask::dagger did not return a Mask");
    }
    return incl;
}

SymmetricTensorPtr
DirectSumSpace::unit_vector_of_summand(int64 i,
                                       std::shared_ptr<TensorBackend> backend,
                                       std::optional<LegLabels> labels,
                                       std::optional<Dtype> dtype,
                                       std::optional<std::string> device) const
{
    i = normalize_summand_index(*this, i);
    auto const& summand = spaces[static_cast<std::size_t>(i)];
    // Must be the one-dimensional trivial sector.
    if (summand->num_sectors != 1 || summand->multiplicities[0] != 1 ||
        !(summand->defining_sectors[0] == Space::symmetry->trivial_sector)) {
        throw std::invalid_argument(
          "DirectSumSpace::unit_vector_of_summand requires the summand to be the "
          "1-dimensional trivial sector");
    }

    auto incl = inclusion_of_summand(i, std::move(backend), std::nullopt, device);
    Dtype out_dtype = dtype.value_or(Dtype::Complex128);
    auto tens = incl->as_SymmetricTensor(/*guarantee_copy=*/false, std::nullopt, out_dtype);
    // Inclusion is trivial → fused; squeeze the trivial domain to get a pure vector on `this`.
    auto squeezed = squeeze_legs(tens, std::vector<LegRef>{ LegRef{ int64{ -1 } } });
    auto out = std::dynamic_pointer_cast<SymmetricTensor>(squeezed);
    if (!out) {
        throw std::runtime_error(
          "DirectSumSpace::unit_vector_of_summand: squeeze_legs did not return SymmetricTensor");
    }
    if (labels.has_value()) {
        if (labels->size() != 1) {
            throw std::invalid_argument(
              "DirectSumSpace::unit_vector_of_summand labels must have length 1 "
              "(the fused DirectSumSpace leg)");
        }
        out->set_labels(*labels);
    }
    return out;
}

} // namespace cyten
