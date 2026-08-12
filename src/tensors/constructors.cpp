#include <cyten/tensors/constructors.h>

#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tools.h>

#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

namespace {

py::object
tensors_mod()
{
    return py::module_::import("cyten.tensors._tensors");
}

bool
is_python_instance(py::object obj, char const* class_name)
{
    return py::isinstance(obj, tensors_mod().attr(class_name));
}

bool
is_any_tensor(py::object obj)
{
    return is_python_instance(obj, "Tensor") || py::isinstance<Tensor>(obj);
}

py::object
data_as_python(TensorBackend::DataPtr data, TensorBackend::Ptr const& /*backend*/)
{
    // C++ SymmetricTensor/Mask/DiagonalTensor ctors take DataPtr (including NoSymmetry BlockData).
    return py::cast(std::move(data));
}

py::object
make_python_symmetric_tensor(TensorBackend::DataPtr data,
                             py::object codomain,
                             py::object domain,
                             TensorBackend::Ptr backend,
                             py::object labels)
{
    return tensors_mod().attr("SymmetricTensor")(data_as_python(std::move(data), backend),
                                                 codomain,
                                                 domain,
                                                 py::arg("backend") = py::cast(backend),
                                                 py::arg("labels") = labels);
}

bool
py_eq(py::object a, py::object b)
{
    py::object eq = a.attr("__eq__")(b);
    if (eq.is(py::reinterpret_borrow<py::object>(Py_NotImplemented))) {
        return false;
    }
    return eq.cast<bool>();
}

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

} // namespace

py::object
eye(py::object leg,
    TensorBackend::Ptr backend,
    py::object labels,
    Dtype dtype,
    std::optional<std::string> device,
    bool diagonal)
{
    py::object res =
      tensors_mod()
        .attr("DiagonalTensor")
        .attr("from_eye")(py::arg("leg") = leg,
                          py::arg("backend") = backend ? py::cast(backend) : py::none(),
                          py::arg("labels") = labels,
                          py::arg("dtype") = dtype,
                          py::arg("device") = device.has_value() ? py::cast(*device) : py::none());
    if (diagonal) {
        return res;
    }
    return res.attr("as_SymmetricTensor")();
}

py::object
tensor(py::object obj,
       py::object codomain,
       py::object domain,
       TensorBackend::Ptr backend,
       py::object labels,
       std::optional<Dtype> dtype,
       std::optional<std::string> device,
       bool understood_braiding)
{
    if (is_any_tensor(obj)) {
        bool copied = false;
        if (!py_eq(codomain, obj.attr("codomain"))) {
            throw std::invalid_argument("Mismatching codomain");
        }
        if (!domain.is_none() && !py_eq(domain, obj.attr("domain"))) {
            throw std::invalid_argument("Mismatching domain");
        }
        if (backend && !py_eq(py::cast(backend), obj.attr("backend"))) {
            throw std::invalid_argument("Mismatching backend");
        }
        if (!labels.is_none() && !py_eq(labels, obj.attr("_labels"))) {
            if (!copied) {
                obj = obj.attr("copy")();
                copied = true;
            }
            obj.attr("labels") = labels;
        }
        if (dtype.has_value()) {
            throw std::invalid_argument("Mismatching dtype");
        }
        if (device.has_value()) {
            throw std::invalid_argument("Mismatching device");
        }
        return obj.attr("as_SymmetricTensor")();
    }
    return tensors_mod()
      .attr("SymmetricTensor")
      .attr("from_dense_block")(
        obj,
        codomain,
        domain,
        py::arg("backend") = backend ? py::cast(backend) : py::none(),
        py::arg("labels") = labels,
        py::arg("dtype") = dtype.has_value() ? py::cast(*dtype) : py::none(),
        py::arg("device") = device.has_value() ? py::cast(*device) : py::none(),
        py::arg("understood_braiding") = understood_braiding);
}

py::object
add_trivial_leg(py::object tens,
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
    int64 res_num_legs = tens.attr("num_legs").cast<int64>() + 1;
    // parse position to format:
    //  - leg_pos: int,  0 <= leg_pos < res_num_legs
    //  - add_to_domain: bool
    //  - co_domain_pos: int, 0 <= co_domain_pos < num_[co]domain_legs
    //  - is_dual: bool, if the leg in the [co]domain should be dual
    int64 legs_pos = 0;
    int64 co_domain_pos = 0;
    bool add_to_domain = false;

    if (legs_pos_opt.has_value()) {
        if (codomain_pos_opt.has_value() || domain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        legs_pos = to_valid_idx(*legs_pos_opt, res_num_legs);
        add_to_domain = legs_pos > tens.attr("num_codomain_legs").cast<int64>();
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
        int64 res_codomain_legs = tens.attr("num_codomain_legs").cast<int64>() + 1;
        int64 codomain_pos = to_valid_idx(*codomain_pos_opt, res_codomain_legs);
        add_to_domain = false;
        co_domain_pos = codomain_pos;
        legs_pos = codomain_pos;
    } else if (domain_pos_opt.has_value()) {
        if (legs_pos_opt.has_value() || codomain_pos_opt.has_value()) {
            throw std::invalid_argument(
              "legs_pos, codomain_pos, domain_pos are mutually exclusive");
        }
        int64 res_domain_legs = tens.attr("num_domain_legs").cast<int64>() + 1;
        int64 domain_pos = to_valid_idx(*domain_pos_opt, res_domain_legs);
        add_to_domain = true;
        co_domain_pos = domain_pos;
        legs_pos = res_num_legs - 1 - domain_pos;
    } else {
        add_to_domain = false;
        co_domain_pos = 0;
        legs_pos = 0;
    }

    if (is_python_instance(tens, "DiagonalTensor") || py::isinstance<DiagonalTensor>(tens) ||
        is_python_instance(tens, "Mask") || py::isinstance<Mask>(tens) ||
        is_python_instance(tens, "Identity") || py::isinstance<Identity>(tens)) {
        std::string msg = "Converting to SymmetricTensor for add_trivial_leg. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tens = tens.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }
    if (is_python_instance(tens, "ChargedTensor") || py::isinstance<ChargedTensor>(tens)) {
        py::object inv_part;
        if (add_to_domain) {
            // domain[0] is the charge leg, so we need to add 1
            inv_part = add_trivial_leg(tens.attr("invariant_part"),
                                       /*legs_pos=*/std::nullopt,
                                       /*codomain_pos=*/std::nullopt,
                                       /*domain_pos=*/co_domain_pos + 1,
                                       label,
                                       is_dual);
        } else {
            inv_part = add_trivial_leg(tens.attr("invariant_part"),
                                       /*legs_pos=*/std::nullopt,
                                       /*codomain_pos=*/co_domain_pos,
                                       /*domain_pos=*/std::nullopt,
                                       label,
                                       is_dual);
        }
        return tensors_mod().attr("ChargedTensor")(
          inv_part, py::arg("charged_state") = tens.attr("charged_state"));
    }
    if (!(is_python_instance(tens, "SymmetricTensor") || py::isinstance<SymmetricTensor>(tens))) {
        throw py::type_error("Invalid type for tens. Expected a Tensor subtype");
    }

    auto new_leg = ElementarySpace::from_trivial_sector(
      1, tens.attr("symmetry").cast<Symmetry::Ptr>(), is_dual);
    auto domain_tp = tens.attr("domain").cast<TensorProduct::Ptr>();
    auto codomain_tp = tens.attr("codomain").cast<TensorProduct::Ptr>();
    TensorProduct::Ptr domain;
    TensorProduct::Ptr codomain;
    if (add_to_domain) {
        domain = domain_tp->insert_multiply(py::cast(new_leg), co_domain_pos);
        codomain = codomain_tp;
    } else {
        domain = domain_tp;
        codomain = codomain_tp->insert_multiply(py::cast(new_leg), co_domain_pos);
    }
    auto backend = tens.attr("backend").cast<TensorBackend::Ptr>();
    auto data =
      backend->add_trivial_leg(tens, legs_pos, add_to_domain, co_domain_pos, codomain, domain);

    LegLabels labels = tens.attr("labels").cast<LegLabels>();
    LegLabels new_labels;
    new_labels.reserve(labels.size() + 1);
    new_labels.insert(new_labels.end(), labels.begin(), labels.begin() + legs_pos);
    new_labels.push_back(label);
    new_labels.insert(new_labels.end(), labels.begin() + legs_pos, labels.end());

    return make_python_symmetric_tensor(
      std::move(data), py::cast(codomain), py::cast(domain), backend, py::cast(new_labels));
}

py::object
zero_like(py::object tensor)
{
    if (is_python_instance(tensor, "Mask") || py::isinstance<Mask>(tensor)) {
        return tensors_mod().attr("Mask").attr("from_zero")(
          py::arg("large_leg") = tensor.attr("large_leg"),
          py::arg("backend") = tensor.attr("backend"),
          py::arg("labels") = tensor.attr("labels"),
          py::arg("device") = tensor.attr("device"));
    }
    if (is_python_instance(tensor, "DiagonalTensor") || py::isinstance<DiagonalTensor>(tensor) ||
        is_python_instance(tensor, "Identity") || py::isinstance<Identity>(tensor)) {
        return tensors_mod()
          .attr("DiagonalTensor")
          .attr("from_zero")(py::arg("leg") = tensor.attr("leg"),
                             py::arg("backend") = tensor.attr("backend"),
                             py::arg("labels") = tensor.attr("labels"),
                             py::arg("dtype") = tensor.attr("dtype"),
                             py::arg("device") = tensor.attr("device"));
    }
    if (is_python_instance(tensor, "SymmetricTensor") || py::isinstance<SymmetricTensor>(tensor)) {
        return tensors_mod()
          .attr("SymmetricTensor")
          .attr("from_zero")(py::arg("codomain") = tensor.attr("codomain"),
                             py::arg("domain") = tensor.attr("domain"),
                             py::arg("backend") = tensor.attr("backend"),
                             py::arg("labels") = tensor.attr("labels"),
                             py::arg("dtype") = tensor.attr("dtype"),
                             py::arg("device") = tensor.attr("device"));
    }
    if (is_python_instance(tensor, "ChargedTensor") || py::isinstance<ChargedTensor>(tensor)) {
        return tensors_mod()
          .attr("ChargedTensor")
          .attr("from_zero")(py::arg("codomain") = tensor.attr("codomain"),
                             py::arg("domain") = tensor.attr("domain"),
                             py::arg("charge") = tensor.attr("charge_leg"),
                             py::arg("charged_state") = tensor.attr("charged_state"),
                             py::arg("backend") = tensor.attr("backend"),
                             py::arg("labels") = tensor.attr("labels"),
                             py::arg("dtype") = tensor.attr("dtype"),
                             py::arg("device") = tensor.attr("device"));
    }
    throw py::type_error("Invalid type for tensor.");
}

py::object
tensor_from_grid(py::object grid_obj, py::object labels, std::optional<Dtype> dtype_opt)
{
    // --- hints from Python tensor_from_grid ---
    // check input
    // only ElementarySpaces have direct_sum
    // find op from same column
    // find op from same row
    // for each sector in the direct sum, find which multiplicities come from which space
    // ---
    py::list grid = py::reinterpret_borrow<py::list>(grid_obj);
    py::list op_list;
    for (auto row_h : grid) {
        for (auto op : py::reinterpret_borrow<py::iterable>(row_h)) {
            if (!op.is_none()) {
                op_list.append(op);
            }
        }
    }
    if (py::len(op_list) == 0) {
        throw std::invalid_argument("grid must contain at least one tensor");
    }

    std::vector<py::object> ops_vec;
    ops_vec.reserve(static_cast<std::size_t>(py::len(op_list)));
    for (auto op : op_list) {
        ops_vec.push_back(py::reinterpret_borrow<py::object>(op));
    }
    auto backend = get_same_backend(ops_vec);
    std::string device =
      tensors_mod().attr("get_same_device")(*py::tuple(op_list)).cast<std::string>();

    Dtype dtype;
    if (dtype_opt.has_value()) {
        dtype = *dtype_opt;
    } else {
        std::vector<Dtype> dtypes;
        dtypes.reserve(ops_vec.size());
        for (auto const& op : ops_vec) {
            dtypes.push_back(op.attr("dtype").cast<Dtype>());
        }
        dtype = dtype::common(dtypes);
    }

    py::object ref = op_list[0];
    int64 n_cod = ref.attr("num_codomain_legs").cast<int64>();
    int64 n_dom = ref.attr("num_domain_legs").cast<int64>();
    py::slice slice_cod_tail(std::optional<py::ssize_t>(1), std::nullopt, std::nullopt);
    py::slice slice_dom_head(std::nullopt, std::optional<py::ssize_t>(-1), std::nullopt);
    py::object ref_cod_tail = ref.attr("codomain").attr("__getitem__")(slice_cod_tail);
    py::object ref_dom_head = ref.attr("domain").attr("__getitem__")(slice_dom_head);

    // check input
    for (auto op : op_list) {
        if (op.attr("num_codomain_legs").cast<int64>() != n_cod ||
            op.attr("num_domain_legs").cast<int64>() != n_dom) {
            throw std::runtime_error("inconsistent number of legs in grid");
        }
        if (!py_eq(op.attr("codomain").attr("__getitem__")(slice_cod_tail), ref_cod_tail) ||
            !py_eq(op.attr("domain").attr("__getitem__")(slice_dom_head), ref_dom_head)) {
            throw std::runtime_error("inconsistent legs in grid");
        }
        // only ElementarySpaces have direct_sum
        if (!py::isinstance<ElementarySpace>(op.attr("codomain").attr("__getitem__")(0)) ||
            !py::isinstance<ElementarySpace>(op.attr("domain").attr("__getitem__")(-1))) {
            throw std::runtime_error("stacking legs must be ElementarySpace");
        }
    }

    py::ssize_t n_rows = py::len(grid);
    py::ssize_t n_cols = n_rows > 0 ? py::len(py::reinterpret_borrow<py::list>(grid[0])) : 0;

    std::vector<std::vector<py::object>> grid_vec(
      static_cast<std::size_t>(n_rows), std::vector<py::object>(static_cast<std::size_t>(n_cols)));
    for (py::ssize_t i = 0; i < n_rows; ++i) {
        py::list row = py::reinterpret_borrow<py::list>(grid[i]);
        if (py::len(row) != n_cols) {
            throw std::invalid_argument("grid rows must have equal length");
        }
        for (py::ssize_t j = 0; j < n_cols; ++j) {
            grid_vec[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
              py::reinterpret_borrow<py::object>(row[j]);
        }
    }

    std::vector<py::object> right_ops(static_cast<std::size_t>(n_cols));
    if (n_rows > 0) {
        right_ops = grid_vec[0];
    }
    for (py::ssize_t i = 0; i < n_cols; ++i) {
        if (!right_ops[static_cast<std::size_t>(i)].is_none()) {
            continue;
        }
        // find op from same column
        for (py::ssize_t r = 0; r < n_rows; ++r) {
            auto const& new_op =
              grid_vec[static_cast<std::size_t>(r)][static_cast<std::size_t>(i)];
            if (new_op.is_none()) {
                continue;
            }
            right_ops[static_cast<std::size_t>(i)] = new_op;
            break;
        }
    }
    for (auto const& op : right_ops) {
        if (op.is_none()) {
            throw std::invalid_argument("Must have at least one nonzero entry in each column.");
        }
    }
    std::vector<ElementarySpace::Ptr> right_spaces;
    right_spaces.reserve(right_ops.size());
    for (auto const& op : right_ops) {
        right_spaces.push_back(
          op.attr("domain").attr("__getitem__")(-1).cast<ElementarySpace::Ptr>());
    }

    std::vector<py::object> left_ops(static_cast<std::size_t>(n_rows));
    for (py::ssize_t i = 0; i < n_rows; ++i) {
        left_ops[static_cast<std::size_t>(i)] = grid_vec[static_cast<std::size_t>(i)][0];
    }
    for (py::ssize_t i = 0; i < n_rows; ++i) {
        if (!left_ops[static_cast<std::size_t>(i)].is_none()) {
            continue;
        }
        // find op from same row
        for (py::ssize_t c = 0; c < n_cols; ++c) {
            auto const& new_op =
              grid_vec[static_cast<std::size_t>(i)][static_cast<std::size_t>(c)];
            if (new_op.is_none()) {
                continue;
            }
            left_ops[static_cast<std::size_t>(i)] = new_op;
            break;
        }
    }
    for (auto const& op : left_ops) {
        if (op.is_none()) {
            throw std::invalid_argument("Must have at least one nonzero entry in each row.");
        }
    }
    std::vector<ElementarySpace::Ptr> left_spaces;
    left_spaces.reserve(left_ops.size());
    for (auto const& op : left_ops) {
        left_spaces.push_back(
          op.attr("codomain").attr("__getitem__")(0).cast<ElementarySpace::Ptr>());
    }

    std::vector<ElementarySpace::Ptr> left_rest(left_spaces.begin() + 1, left_spaces.end());
    std::vector<ElementarySpace::Ptr> right_rest(right_spaces.begin() + 1, right_spaces.end());
    auto left_space = left_spaces[0]->direct_sum(left_rest);
    auto right_space = right_spaces[0]->direct_sum(right_rest);

    // for each sector in the direct sum, find which multiplicities come from which space
    std::vector<std::vector<int64>> left_mult_slices;
    for (auto const& sector : left_space->sector_decomposition) {
        std::vector<int64> mults;
        mults.reserve(left_spaces.size());
        for (auto const& space : left_spaces) {
            auto idx = space->sector_decomposition_where(sector);
            mults.push_back(idx.has_value() ? space->multiplicities[static_cast<std::size_t>(*idx)]
                                            : int64{ 0 });
        }
        left_mult_slices.push_back(cumsum_with_leading_zero(mults));
    }
    std::vector<std::vector<int64>> right_mult_slices;
    for (auto const& sector : right_space->sector_decomposition) {
        std::vector<int64> mults;
        mults.reserve(right_spaces.size());
        for (auto const& space : right_spaces) {
            auto idx = space->sector_decomposition_where(sector);
            mults.push_back(idx.has_value() ? space->multiplicities[static_cast<std::size_t>(*idx)]
                                            : int64{ 0 });
        }
        right_mult_slices.push_back(cumsum_with_leading_zero(mults));
    }

    py::list cod_factors;
    cod_factors.append(py::cast(left_space));
    for (auto item : py::reinterpret_borrow<py::iterable>(
           ref.attr("codomain").attr("__getitem__")(slice_cod_tail))) {
        cod_factors.append(item);
    }
    py::list dom_factors;
    for (auto item : py::reinterpret_borrow<py::iterable>(
           ref.attr("domain").attr("__getitem__")(slice_dom_head))) {
        dom_factors.append(item);
    }
    dom_factors.append(py::cast(right_space));

    auto codomain = std::make_shared<TensorProduct>(cod_factors.cast<std::vector<py::object>>());
    auto domain = std::make_shared<TensorProduct>(dom_factors.cast<std::vector<py::object>>());

    auto data = backend->from_grid(std::move(grid_vec),
                                   codomain,
                                   domain,
                                   std::move(left_mult_slices),
                                   std::move(right_mult_slices),
                                   dtype,
                                   device);
    return make_python_symmetric_tensor(
      std::move(data), py::cast(codomain), py::cast(domain), backend, labels);
}

} // namespace cyten
