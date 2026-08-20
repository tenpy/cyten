#include <cyten/models/degrees_of_freedom.h>

#include <cyten/backends/backend_factory.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/symmetries/factors/fermion_number.h>
#include <cyten/symmetries/factors/fermion_parity.h>
#include <cyten/symmetries/factors/no_symmetry.h>
#include <cyten/symmetries/factors/su2.h>
#include <cyten/symmetries/factors/u1.h>
#include <cyten/symmetries/factors/zn.h>
#include <cyten/tensors/constructors.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/helpers.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tools.h>

#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cyten {

namespace {

int64
py_object_to_int64(py::handle obj)
{
    if (py::isinstance<py::int_>(obj)) {
        return obj.cast<int64>();
    }
    if (py::hasattr(obj, "item")) {
        return obj.attr("item")().cast<int64>();
    }
    return py::module_::import("builtins").attr("int")(obj).cast<int64>();
}

py::module_
numpy()
{
    return py::module_::import("numpy");
}

py::module_
tensors_mod()
{
    return py::module_::import("cyten.tensors");
}

bool
is_python_tensor(py::object const& obj, char const* class_name)
{
    try {
        return py::isinstance(obj, tensors_mod().attr(class_name));
    } catch (py::error_already_set&) {
        return false;
    }
}

bool
is_symmetric_tensor(py::object const& op)
{
    return py::isinstance<SymmetricTensor>(op) || is_python_tensor(op, "SymmetricTensor");
}

bool
is_diagonal_tensor(py::object const& op)
{
    return py::isinstance<DiagonalTensor>(op) || py::isinstance<Identity>(op) ||
           is_python_tensor(op, "DiagonalTensor") || is_python_tensor(op, "Identity");
}

SymmetricTensorPtr
as_symmetric_tensor(py::object op)
{
    if (py::isinstance<SymmetricTensor>(op)) {
        return op.cast<SymmetricTensorPtr>();
    }
    return op.cast<SymmetricTensorPtr>();
}

TensorProduct::Ptr
product_of_leg(Leg::Ptr leg)
{
    return std::make_shared<TensorProduct>(std::vector<Leg::Ptr>{ std::move(leg) });
}

TensorProduct::Ptr
product_of_legs(std::vector<Leg::Ptr> factors)
{
    return std::make_shared<TensorProduct>(std::move(factors));
}

LegLabels
pp_labels()
{
    return LegLabels{ "p", "p*" };
}

bool
leg_labels_are_p_pstar(LegLabels const& labels)
{
    return labels.size() == 2 && labels[0] == "p" && labels[1] == "p*";
}

bool
single_leg_matches(Leg::Ptr const& leg, TensorProduct::Ptr const& tp)
{
    return tp && tp->num_factors == 1 && tp->factors[0] && leg && tp->factors[0]->operator==(*leg);
}

SymmetricTensorPtr
compose_sym(SymmetricTensorPtr a, SymmetricTensorPtr b)
{
    auto res = compose(a, b);
    if (auto* t = std::get_if<TensorPtr>(&res)) {
        if (auto sym = std::dynamic_pointer_cast<SymmetricTensor>(*t)) {
            return sym;
        }
        throw std::runtime_error("compose of onsite operators did not yield SymmetricTensor");
    }
    throw std::runtime_error("compose of onsite operators yielded a scalar");
}

std::string
site_class_name(Site const& site)
{
    if (dynamic_cast<SpinDOF const*>(&site)) {
        return "SpinDOF";
    }
    if (dynamic_cast<ClockDOF const*>(&site)) {
        return "ClockDOF";
    }
    if (dynamic_cast<AnyonDOF const*>(&site)) {
        return "AnyonDOF";
    }
    if (dynamic_cast<FermionicDOF const*>(&site)) {
        return "FermionicDOF";
    }
    if (dynamic_cast<BosonicDOF const*>(&site)) {
        return "BosonicDOF";
    }
    if (dynamic_cast<OccupationDOF const*>(&site)) {
        return "OccupationDOF";
    }
    return "Site";
}

int64
leg_dim(ElementarySpace::Ptr const& leg)
{
    return static_cast<int64>(leg->Space::dim);
}

Symmetry::Ptr
leg_symmetry(ElementarySpace::Ptr const& leg)
{
    return leg->Space::symmetry;
}

bool
py_bool(py::object const& obj)
{
    return obj.cast<bool>();
}

bool
np_allclose(py::object const& a, py::object const& b)
{
    return py_bool(numpy().attr("allclose")(a, b));
}

bool
np_all(py::object const& a)
{
    return py_bool(numpy().attr("all")(a));
}

py::object
py_reduce(py::object func, py::object iterable)
{
    return py::module_::import("functools").attr("reduce")(func, iterable);
}

Symmetry::Ptr
symmetry_from_factor(SymmetryFactor::Ptr factor)
{
    return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ std::move(factor) });
}

Symmetry::Ptr
symmetry_from_factors(std::vector<SymmetryFactor::Ptr> factors)
{
    return std::make_shared<Symmetry>(std::move(factors));
}

int64
infer_clock_q(py::array const& X, py::array const& Z)
{
    auto np = numpy();
    auto const dim = X.attr("shape").attr("__getitem__")(0).cast<int64>();
    auto I = np.attr("eye")(dim);
    for (int64 q = 2; q <= dim; ++q) {
        auto Xq = np.attr("linalg").attr("matrix_power")(X, q);
        auto Zq = np.attr("linalg").attr("matrix_power")(Z, q);
        auto phase = np.attr("exp")(
          py::cast(std::complex<double>{ 0., 2. * std::numbers::pi / static_cast<double>(q) }));
        auto XZ = np.attr("matmul")(X, Z);
        auto ZX = np.attr("matmul")(Z, X);
        if (np_allclose(Xq, I) && np_allclose(Zq, I) &&
            np_allclose(XZ, py::object(phase) * py::object(ZX))) {
            return q;
        }
    }
    throw std::runtime_error("failed to infer clock q from operators");
}

void
add_onsite_operators_from_map(Site& site,
                              std::map<std::string, SymmetricTensorPtr> const& onsite_operators)
{
    for (auto const& [name, op] : onsite_operators) {
        site.add_onsite_operator(name, py::cast(op));
    }
}

} // namespace

py::object&
all_species_sentinel()
{
    static py::object& sentinel =
      leak_py_object(py::module_::import("builtins").attr("object")());
    return sentinel;
}

bool
is_all_species(py::object const& species)
{
    if (species.is_none()) {
        return true;
    }
    return species.is(all_species_sentinel());
}

py::array
as_immutable_array(py::object a, py::object dtype)
{
    auto np = numpy();
    py::object arr = dtype.is_none() ? np.attr("asarray")(a) : np.attr("asarray")(a, dtype);
    arr.attr("setflags")(py::arg("write") = false);
    return arr.cast<py::array>();
}

Site::Site(ElementarySpace::Ptr leg,
           std::map<std::string, int64> state_labels,
           std::map<std::string, SymmetricTensorPtr> onsite_operators,
           TensorBackend::Ptr backend,
           std::optional<std::string> default_device)
  : leg(std::move(leg))
  , state_labels(std::move(state_labels))
  , backend(backend ? std::move(backend) : get_backend(leg_symmetry(this->leg)))
  , default_device(default_device.value_or("cpu"))
{
    auto const sym = leg_symmetry(this->leg);
    Dtype id_dtype = sym->has_complex_topological_data ? Dtype::Complex128 : Dtype::Float64;
    add_onsite_operator("Id",
                        py::cast(Identity::from_eye(
                          this->leg, this->backend, pp_labels(), id_dtype, this->default_device)));
    add_onsite_operators_from_map(*this, onsite_operators);
}

void
Site::test_sanity()
{
    leg->test_sanity();

    if (!symmetry()->can_be_dropped()) {
        assert(state_labels.empty());
    }
    for (auto const& [label, idx] : state_labels) {
        assert(py::isinstance<py::str>(py::cast(label)));
        assert(0 <= idx && idx < static_cast<int64>(dim()));
    }

    for (auto const& [name, op] : onsite_operators) {
        (void)name;
        assert(single_leg_matches(leg, op->codomain));
        assert(single_leg_matches(leg, op->domain));
        assert(leg_labels_are_p_pstar(op->labels()));
        op->test_sanity();
    }
}

Symmetry::Ptr
Site::symmetry() const
{
    return leg_symmetry(leg);
}

float64
Site::dim() const
{
    return leg->Space::dim;
}

void
Site::add_onsite_operator(std::string const& name,
                          py::object op,
                          std::optional<bool> is_diagonal,
                          bool understood_braiding)
{
    if (onsite_operators.contains(name)) {
        throw std::invalid_argument(std::format("Operator with name={} already exists.", name));
    }
    if (name.find(' ') != std::string::npos) {
        throw std::invalid_argument("operator names are not allowed to feature whitespace");
    }

    SymmetricTensorPtr tensor_op;
    if (is_symmetric_tensor(op)) {
        tensor_op = as_symmetric_tensor(op);
        if (is_diagonal.has_value()) {
            assert(is_diagonal_tensor(py::cast(tensor_op)) == *is_diagonal);
        }
        assert(single_leg_matches(leg, tensor_op->codomain));
        assert(single_leg_matches(leg, tensor_op->domain));
        if (!leg_labels_are_p_pstar(tensor_op->labels())) {
            tensor_op =
              std::dynamic_pointer_cast<SymmetricTensor>(tensor_op->copy(/*deep=*/false));
            tensor_op->set_labels(pp_labels());
        }
    } else if (is_diagonal.has_value() && *is_diagonal) {
        py::object block = op.attr("copy")();
        auto block_ptr = backend->block_backend->as_block(block, std::nullopt, default_device);
        tensor_op = DiagonalTensor::from_dense_block(block_ptr,
                                                     leg,
                                                     backend,
                                                     pp_labels(),
                                                     std::nullopt,
                                                     1e-6,
                                                     default_device,
                                                     understood_braiding);
    } else {
        py::object block = op.attr("copy")();
        auto block_ptr = backend->block_backend->as_block(block, std::nullopt, default_device);
        auto tp = product_of_leg(leg);
        tensor_op = SymmetricTensor::from_dense_block(block_ptr,
                                                      tp,
                                                      tp,
                                                      backend,
                                                      pp_labels(),
                                                      std::nullopt,
                                                      default_device,
                                                      1e-6,
                                                      understood_braiding);
    }
    onsite_operators[name] = std::move(tensor_op);
}

bool
Site::valid_opname(std::string const& name) const
{
    return onsite_operators.contains(name);
}

SymmetricTensorPtr
Site::get_op(std::string const& name)
{
    auto pos = name.find(' ');
    std::string first = pos == std::string::npos ? name : name.substr(0, pos);
    if (!onsite_operators.contains(first)) {
        throw std::invalid_argument(
          std::format("{} doesn't have the operator '{}'", repr(), first));
    }
    SymmetricTensorPtr op = onsite_operators.at(first);
    if (pos == std::string::npos) {
        return op;
    }
    std::string rest = name.substr(pos + 1);
    while (!rest.empty()) {
        auto pos2 = rest.find(' ');
        std::string name2 = pos2 == std::string::npos ? rest : rest.substr(0, pos2);
        if (!onsite_operators.contains(name2)) {
            throw std::invalid_argument(
              std::format("{} doesn't have the operator '{}'", repr(), name2));
        }
        op = compose_sym(op, onsite_operators.at(name2));
        if (pos2 == std::string::npos) {
            break;
        }
        rest = rest.substr(pos2 + 1);
    }
    return op;
}

std::string
Site::multiply_op_names(std::vector<std::string> const& names) const
{
    if (names.empty()) {
        return "Id";
    }
    std::string result;
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i > 0) {
            result += ' ';
        }
        result += names[i];
    }
    return result;
}

SymmetricTensorPtr
Site::multiply_operators(std::vector<py::object> const& operators)
{
    if (operators.empty()) {
        return onsite_operators.at("Id");
    }
    SymmetricTensorPtr op;
    if (py::isinstance<py::str>(operators[0])) {
        op = get_op(operators[0].cast<std::string>());
    } else {
        op = as_symmetric_tensor(operators[0]);
    }
    for (std::size_t i = 1; i < operators.size(); ++i) {
        SymmetricTensorPtr next_op;
        if (py::isinstance<py::str>(operators[i])) {
            next_op = get_op(operators[i].cast<std::string>());
        } else {
            next_op = as_symmetric_tensor(operators[i]);
        }
        op = compose_sym(op, next_op);
    }
    return op;
}

SymmetricTensorPtr
Site::identity_tensor(ElementarySpace::Ptr w, bool overbraid)
{
    auto co_domain = product_of_legs(std::vector<Leg::Ptr>{ leg, std::move(w) });
    auto tensor = SymmetricTensor::from_eye(co_domain, backend, LegLabels{ "p", "w" });
    auto permuted = permute_legs(tensor,
                                 std::vector<LegRef>{ "w", "p" },
                                 std::vector<LegRef>{ "p*", "w*" },
                                 std::nullopt,
                                 !overbraid);
    permuted->relabel({ { "w", "wL" }, { "w*", "wR" } });
    return std::dynamic_pointer_cast<SymmetricTensor>(permuted);
}

int64
Site::state_index(py::object label) const
{
    if (py::isinstance<py::str>(label)) {
        auto key = label.cast<std::string>();
        auto it = state_labels.find(key);
        if (it == state_labels.end()) {
            throw py::key_error(std::format("Label not found: {}", key));
        }
        return it->second;
    }
    int64 res = label.cast<int64>();
    auto d = static_cast<int64>(dim());
    if (!(-d <= res && res < d)) {
        throw std::invalid_argument("Index out of bounds");
    }
    if (res < 0) {
        return res + d;
    }
    return res;
}

std::vector<int64>
Site::state_indices(std::vector<py::object> const& labels) const
{
    std::vector<int64> result;
    result.reserve(labels.size());
    for (auto const& label : labels) {
        result.push_back(state_index(label));
    }
    return result;
}

std::string
Site::repr() const
{
    return std::format(
      "<{}, dim={}, symmetry={}>", site_class_name(*this), dim(), symmetry()->repr());
}

SpinDOF::SpinDOF(ElementarySpace::Ptr leg,
                 py::array spin_vector,
                 std::map<std::string, int64> state_labels,
                 std::map<std::string, SymmetricTensorPtr> onsite_operators,
                 TensorBackend::Ptr backend,
                 std::optional<std::string> default_device)
  : Site(leg, state_labels, {}, backend, default_device)
{
    auto buf = spin_vector.request();
    assert(buf.ndim == 3);
    assert(buf.shape[0] == static_cast<py::ssize_t>(leg_dim(leg)));
    assert(buf.shape[1] == static_cast<py::ssize_t>(leg_dim(leg)));
    assert(buf.shape[2] == 3);
    this->spin_vector = as_immutable_array(spin_vector);
    add_onsite_operators_from_map(*this, onsite_operators);
}

void
SpinDOF::test_sanity()
{
    Site::test_sanity();
    auto np = numpy();
    py::array Sx = spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0));
    py::array Sy = spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1));
    py::array Sz = spin_vector.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 2));
    auto imag = py::cast(std::complex<double>{ 0., 1. });
    assert(
      np_allclose(np.attr("matmul")(Sx, Sy) - np.attr("matmul")(Sy, Sx), imag * py::object(Sz)));
    assert(
      np_allclose(np.attr("matmul")(Sy, Sz) - np.attr("matmul")(Sz, Sy), imag * py::object(Sx)));
    assert(
      np_allclose(np.attr("matmul")(Sz, Sx) - np.attr("matmul")(Sx, Sz), imag * py::object(Sy)));
}

py::array
SpinDOF::spin_vector_from_Sp(py::array Sz, py::array Sp)
{
    auto np = numpy();
    auto dim = Sz.attr("shape").attr("__getitem__")(0).cast<int64>();
    assert(Sz.attr("shape").attr("__getitem__")(0).cast<int64>() == dim);
    assert(Sz.attr("shape").attr("__getitem__")(1).cast<int64>() == dim);
    assert(Sp.attr("shape").attr("__getitem__")(0).cast<int64>() == dim);
    assert(Sp.attr("shape").attr("__getitem__")(1).cast<int64>() == dim);
    py::array Sm = np.attr("conj")(Sp.attr("T"));
    py::object Sp_obj = Sp;
    py::object Sm_obj = Sm;
    py::array Sx = (py::float_(0.5) * (Sp_obj + Sm_obj)).cast<py::array>();
    py::array Sy =
      (py::cast(std::complex<double>{ 0., 0.5 }) * (Sm_obj - Sp_obj)).cast<py::array>();
    return np.attr("stack")(py::make_tuple(Sx, Sy, Sz), py::arg("axis") = -1);
}

Symmetry::Ptr
SpinDOF::conservation_law_to_symmetry(std::optional<std::string> conserve)
{
    if (!conserve.has_value() || *conserve == "None" || *conserve == "none") {
        return symmetry_from_factor(std::make_shared<NoSymmetry>());
    }
    if (*conserve == "SU(2)" || *conserve == "SU2" || *conserve == "Stot") {
        return symmetry_from_factor(std::make_shared<SU2>("spin"));
    }
    if (*conserve == "Sz" || *conserve == "U(1)" || *conserve == "U1") {
        return symmetry_from_factor(std::make_shared<U1>("2*Sz"));
    }
    if (*conserve == "parity" || *conserve == "Sz_parity" || *conserve == "Z_2" ||
        *conserve == "Z2") {
        return symmetry_from_factor(std::make_shared<ZN>(2, "Sz_parity"));
    }
    throw std::invalid_argument(std::format("Invalid `conserve`: {}", *conserve));
}

ClockDOF::ClockDOF(ElementarySpace::Ptr leg,
                   py::array clock_operators,
                   std::map<std::string, int64> state_labels,
                   std::map<std::string, SymmetricTensorPtr> onsite_operators,
                   TensorBackend::Ptr backend,
                   std::optional<std::string> default_device)
  : Site(leg, state_labels, {}, backend, default_device)
{
    auto buf = clock_operators.request();
    assert(buf.ndim == 3);
    assert(buf.shape[0] == static_cast<py::ssize_t>(leg_dim(leg)));
    assert(buf.shape[1] == static_cast<py::ssize_t>(leg_dim(leg)));
    assert(buf.shape[2] == 2);
    this->clock_operators = as_immutable_array(clock_operators);
    add_onsite_operators_from_map(*this, onsite_operators);

    auto np = numpy();
    py::array Z = clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1));
    py::array Zhc = np.attr("conj")(Z.attr("T"));
    add_onsite_operator("Z", Z, true);
    add_onsite_operator("Zhc", Zhc, true);
    add_onsite_operator("Zphc", Z + Zhc, true);
}

void
ClockDOF::test_sanity()
{
    Site::test_sanity();
    auto np = numpy();
    py::array X = clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 0));
    py::array Z = clock_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), 1));
    int64 q = infer_clock_q(X, Z);
    py::array Xhc = np.attr("conj")(X.attr("T"));
    py::array Zhc = np.attr("conj")(Z.attr("T"));
    auto phase = np.attr("exp")(
      py::cast(std::complex<double>{ 0., 2. * std::numbers::pi / static_cast<double>(q) }));
    assert(np_allclose(np.attr("matmul")(X, Z), py::object(phase) * np.attr("matmul")(Z, X)));
    auto identity = np.attr("eye")(X.attr("shape").attr("__getitem__")(0));
    assert(np_allclose(np.attr("linalg").attr("matrix_power")(X, q), identity));
    assert(np_allclose(np.attr("linalg").attr("matrix_power")(Z, q), identity));
    assert(np_allclose(np.attr("matmul")(X, Xhc), identity));
    assert(np_allclose(np.attr("matmul")(Z, Zhc), identity));
}

Symmetry::Ptr
ClockDOF::conservation_law_to_symmetry(std::optional<std::string> conserve)
{
    return SpinDOF::conservation_law_to_symmetry(conserve);
}

AnyonDOF::AnyonDOF(ElementarySpace::Ptr leg,
                   std::vector<std::string> sector_names,
                   std::map<std::string, int64> state_labels,
                   std::map<std::string, SymmetricTensorPtr> onsite_operators,
                   TensorBackend::Ptr backend,
                   std::optional<std::string> default_device)
  : Site(leg, state_labels, {}, backend, default_device)
{
    if (sector_names.empty()) {
        sector_names.assign(static_cast<std::size_t>(leg->num_sectors), std::string{});
    }
    assert(sector_names.size() == static_cast<std::size_t>(leg->num_sectors));
    this->sector_names = sector_names;

    std::map<std::string, SymmetricTensorPtr> ops = onsite_operators;
    auto tp = product_of_leg(leg);
    for (std::size_t i = 0; i < leg->sector_decomposition.size(); ++i) {
        if (sector_names[i].empty()) {
            continue;
        }
        auto P_sec = SymmetricTensor::from_sector_projection(
          tp, leg->sector_decomposition[i], backend, pp_labels(), std::nullopt, default_device);
        ops[std::format("P_{}", sector_names[i])] = P_sec;
    }
    add_onsite_operators_from_map(*this, ops);
}

void
AnyonDOF::test_sanity()
{
    Site::test_sanity();
}

OccupationDOF::OccupationDOF(ElementarySpace::Ptr leg,
                             py::array creators,
                             py::array annihilators,
                             int64 anti_commute_sign,
                             std::vector<std::optional<std::string>> species_names,
                             std::map<std::string, int64> state_labels,
                             std::map<std::string, SymmetricTensorPtr> onsite_operators,
                             TensorBackend::Ptr backend,
                             std::optional<std::string> default_device)
  : Site(leg, state_labels, {}, backend, default_device)
{
    num_species = py_object_to_int64(creators.attr("shape").attr("__getitem__")(2));
    auto const d = leg_dim(leg);
    auto creators_buf = creators.request();
    auto annihilators_buf = annihilators.request();
    assert(creators_buf.ndim == 3);
    assert(annihilators_buf.ndim == 3);
    assert(creators_buf.shape[0] == static_cast<py::ssize_t>(d));
    assert(creators_buf.shape[1] == static_cast<py::ssize_t>(d));
    assert(creators_buf.shape[2] == static_cast<py::ssize_t>(num_species));
    assert(annihilators_buf.shape[0] == static_cast<py::ssize_t>(d));
    assert(annihilators_buf.shape[1] == static_cast<py::ssize_t>(d));
    assert(annihilators_buf.shape[2] == static_cast<py::ssize_t>(num_species));

    this->creators = as_immutable_array(creators);
    this->annihilators = as_immutable_array(annihilators);
    this->anti_commute_sign = anti_commute_sign;

    if (species_names.empty()) {
        species_names.assign(static_cast<std::size_t>(num_species), std::nullopt);
    } else {
        assert(static_cast<int64>(species_names.size()) == num_species);
    }
    this->species_names = species_names;
    species_name_to_idx.clear();
    for (std::size_t idx = 0; idx < species_names.size(); ++idx) {
        if (species_names[idx].has_value()) {
            species_name_to_idx[*species_names[idx]] = static_cast<int64>(idx);
        }
    }

    auto np = numpy();
    py::array n_ops =
      np.attr("diagonal")(np.attr("tensordot")(creators, annihilators, py::make_tuple(1, 0)),
                          py::arg("axis1") = 1,
                          py::arg("axis2") = 3);
    number_operators = as_immutable_array(n_ops);
    n_tot = as_immutable_array(np.attr("sum")(number_operators, py::arg("axis") = 2));

    add_onsite_operators_from_map(*this, onsite_operators);
}

void
OccupationDOF::test_sanity()
{
    Site::test_sanity();
    auto np = numpy();
    auto const d = leg_dim(leg);
    for (int64 k = 0; k < num_species; ++k) {
        py::array n_k =
          number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        py::array c_k = creators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        py::array a_k =
          annihilators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        assert(np_allclose(np.attr("matmul")(c_k, a_k), n_k));
        assert(np_allclose(np.attr("diag")(np.attr("diag")(n_k)), n_k));
        assert(np_allclose(np.attr("around")(n_k, 0), n_k));
        assert(np_all(np.attr("greater_equal")(n_k, 0)));

        py::array BBd = np.attr("matmul")(a_k, c_k);
        if (anti_commute_sign == 1) {
            py::object BBd_obj = np.attr("array")(BBd, py::arg("copy") = true);
            py::array mask = np.attr("isclose")(np.attr("diag")(BBd_obj), 0);
            py::object increment = np.attr("max")(BBd_obj) + py::int_(1);
            BBd_obj.attr("__setitem__")(py::make_tuple(mask, mask),
                                        py::object(np.attr("diag")(BBd_obj)[mask]) + increment);
            BBd = BBd_obj.cast<py::array>();
        }
        assert(
          np_allclose(BBd - py::cast(anti_commute_sign) * py::object(n_k), np.attr("eye")(d)));

        for (int64 j = 0; j < k; ++j) {
            py::array Bk =
              annihilators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
            py::array Bj =
              annihilators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), j));
            py::array Bdj =
              creators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), j));
            py::array Bdk =
              creators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
            assert(np_allclose(np.attr("matmul")(Bk, Bdj), np.attr("matmul")(Bdj, Bk)));
            assert(np_allclose(np.attr("matmul")(Bk, Bj), np.attr("matmul")(Bj, Bk)));
            assert(np_allclose(np.attr("matmul")(Bdk, Bdj), np.attr("matmul")(Bdj, Bdk)));
        }
    }
}

void
OccupationDOF::add_individual_occupation_ops()
{
    for (int64 k = 0; k < num_species; ++k) {
        py::array N_k =
          number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        add_onsite_operator(std::format("N{}", k), N_k, true);
    }
    if (num_species == 1) {
        add_onsite_operator("N", py::cast(onsite_operators.at("N0")));
    }
}

void
BosonicDOF::add_individual_occupation_ops()
{
    OccupationDOF::add_individual_occupation_ops();
    auto np = numpy();
    for (int64 k = 0; k < num_species; ++k) {
        py::array N_k =
          number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        add_onsite_operator(std::format("N{}N{}", k, k), np.attr("matmul")(N_k, N_k), true);
        py::array P_k = np.attr("diag")(py::float_(1.0) -
                                        py::float_(2.0) * np.attr("mod")(np.attr("diag")(N_k), 2));
        add_onsite_operator(std::format("P{}", k), P_k, true);
    }
    if (num_species == 1) {
        add_onsite_operator("NN", py::cast(onsite_operators.at("N0N0")));
        add_onsite_operator("P", py::cast(onsite_operators.at("P0")));
    }
}

void
OccupationDOF::add_total_occupation_ops()
{
    auto np = numpy();
    add_onsite_operator("Ntot", n_tot, true);
    add_onsite_operator("NtotNtot", np.attr("matmul")(n_tot, n_tot), true);
    py::array P_tot = np.attr("diag")(py::float_(1.0) -
                                      py::float_(2.0) * np.attr("mod")(np.attr("diag")(n_tot), 2));
    add_onsite_operator("Ptot", P_tot, true);
}

py::array
OccupationDOF::get_annihilator_numpy(py::object /*species*/, bool /*include_JW*/)
{
    throw std::logic_error("OccupationDOF::get_annihilator_numpy is abstract");
}

py::array
OccupationDOF::get_creator_numpy(py::object /*species*/, bool /*include_JW*/)
{
    throw std::logic_error("OccupationDOF::get_creator_numpy is abstract");
}

py::array
OccupationDOF::get_occupation_numpy(py::object species)
{
    auto np = numpy();
    py::list indices;
    if (is_all_species(species)) {
        for (int64 k = 0; k < num_species; ++k) {
            indices.append(k);
        }
    } else {
        for (auto item : to_iterable(species)) {
            indices.append(get_species_idx(py::reinterpret_borrow<py::object>(item)));
        }
    }
    return np.attr("sum")(
      number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), indices)),
      py::arg("axis") = 2);
}

int64
OccupationDOF::get_species_idx(py::object species) const
{
    if (py::isinstance<py::str>(species)) {
        species = py::cast(species_name_to_idx.at(species.cast<std::string>()));
    }
    if (species.is_none()) {
        if (num_species > 1) {
            throw std::invalid_argument("Need to specify the species");
        }
        species = py::int_(0);
    }
    return to_valid_idx(species.cast<int64>(), num_species);
}

BosonicDOF::BosonicDOF(ElementarySpace::Ptr leg,
                       py::array /*Nmax_param*/,
                       py::array creators,
                       py::array annihilators,
                       std::vector<std::optional<std::string>> species_names,
                       std::map<std::string, int64> state_labels,
                       std::map<std::string, SymmetricTensorPtr> onsite_operators,
                       TensorBackend::Ptr backend,
                       std::optional<std::string> default_device)
  : Site(leg, state_labels, {}, backend, default_device)
  , OccupationDOF(leg,
                  creators,
                  annihilators,
                  +1,
                  species_names,
                  state_labels,
                  {},
                  backend,
                  default_device)
{
    if (dynamic_cast<FermionicDOF*>(this)) {
        throw SymmetryError("FermionicDOF and BosonicDOF are incompatible.");
    }

    auto np = numpy();
    py::list Nmax_list;
    for (int64 k = 0; k < num_species; ++k) {
        py::array N_k =
          number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        double N_k_max_ = np.attr("max")(np.attr("diag")(N_k)).cast<double>();
        int64 N_k_max = static_cast<int64>(std::llround(N_k_max_));
        assert(std::abs(N_k_max - N_k_max_) < 1e-9);
        assert(leg_dim(leg) % (N_k_max + 1) == 0);
        Nmax_list.append(N_k_max);
    }
    Nmax = as_immutable_array(np.attr("asarray")(Nmax_list, py::dtype("int")));
    assert(py_object_to_int64(np.attr("min")(Nmax)) > 0);
    JW = as_immutable_array(np.attr("diag")(np.attr("ones")(leg_dim(leg))));

    add_onsite_operators_from_map(*this, onsite_operators);
}

void
BosonicDOF::test_sanity()
{
    OccupationDOF::test_sanity();
    auto np = numpy();
    for (int64 k = 0; k < num_species; ++k) {
        py::array N_k =
          number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        py::array N_k_rounded = np.attr("around")(N_k, 0);
        assert(np_allclose(N_k_rounded, N_k));
        assert(np_allclose(np.attr("diag")(np.attr("diag")(N_k)), N_k));
        assert(np.attr("min")(N_k_rounded).cast<double>() == 0.);
        assert(np.attr("max")(N_k_rounded).cast<double>() ==
               Nmax.attr("__getitem__")(k).cast<double>());
    }
}

py::array
BosonicDOF::get_annihilator_numpy(py::object species, bool /*include_JW*/)
{
    return annihilators.attr("__getitem__")(
      py::make_tuple(py::slice(), py::slice(), get_species_idx(species)));
}

py::array
BosonicDOF::get_creator_numpy(py::object species, bool /*include_JW*/)
{
    return creators.attr("__getitem__")(
      py::make_tuple(py::slice(), py::slice(), get_species_idx(species)));
}

Symmetry::Ptr
BosonicDOF::conservation_law_to_symmetry(py::object conserve)
{
    if (conserve.is_none()) {
        return symmetry_from_factor(std::make_shared<NoSymmetry>());
    }
    if (py::isinstance<py::str>(conserve)) {
        auto const c = conserve.cast<std::string>();
        if (c == "None" || c == "none") {
            return symmetry_from_factor(std::make_shared<NoSymmetry>());
        }
        if (c == "N" || c == "Ntot" || c == "N_tot" || c == "U(1)" || c == "U1") {
            return symmetry_from_factor(std::make_shared<U1>("total_occupation"));
        }
        if (c == "parity" || c == "P" || c == "Ptot" || c == "P_tot" || c == "Z_2" || c == "Z2") {
            return symmetry_from_factor(std::make_shared<ZN>(2, "total_occupation_parity"));
        }
        throw std::invalid_argument(std::format("Invalid `conserve`: {}", c));
    }
    if (is_iterable(conserve)) {
        std::vector<SymmetryFactor::Ptr> sym_factors;
        int64 num_no_sym = 0;
        for (py::ssize_t k = 0; k < py::len(conserve); ++k) {
            py::object conserve_k = conserve[py::int_(k)];
            std::string ck = py::str(conserve_k).cast<std::string>();
            if (ck == "N" || ck == "Nk" || ck == "N_k" || ck == "U(1)" || ck == "U1") {
                sym_factors.push_back(
                  std::make_shared<U1>(std::format("species{}_occupation", k)));
            } else if (ck == "parity" || ck == "P" || ck == "Pi" || ck == "P_i" || ck == "Z_2" ||
                       ck == "Z2") {
                sym_factors.push_back(
                  std::make_shared<ZN>(2, std::format("species{}_occupation_parity", k)));
            } else if (ck == "None" || ck == "none" || conserve_k.is_none()) {
                sym_factors.push_back(std::make_shared<NoSymmetry>());
                ++num_no_sym;
            } else {
                throw std::invalid_argument(std::format("Invalid entry in `conserve`: {}", ck));
            }
        }
        if (num_no_sym == py::len(conserve)) {
            return symmetry_from_factor(std::make_shared<NoSymmetry>());
        }
        return symmetry_from_factors(std::move(sym_factors));
    }
    throw std::invalid_argument(
      std::format("Invalid `conserve`: {}", py::str(conserve).cast<std::string>()));
}

std::pair<py::array, py::array>
BosonicDOF::creation_annihilation_op_from_single_Nmax(int64 Nmax, int64 dim)
{
    auto np = numpy();
    assert(Nmax > 0);
    assert(dim == Nmax + 1);
    py::array B = np.attr("zeros")(py::make_tuple(dim, dim), py::dtype("float64"));
    for (int64 n = 1; n < dim; ++n) {
        B.attr("__setitem__")(py::make_tuple(n - 1, n), np.attr("sqrt")(static_cast<double>(n)));
    }
    return { np.attr("transpose")(B), B };
}

std::pair<py::array, py::array>
BosonicDOF::creation_annihilation_ops_from_Nmax(py::array Nmax, int64 dim)
{
    auto np = numpy();
    py::array Nmax_ = np.attr("asarray")(Nmax, py::dtype("int"));
    assert(np.attr("allclose")(Nmax_, Nmax).cast<bool>());
    py::list creators_i;
    py::list annihilators_i;
    for (auto N : Nmax_) {
        auto nmax = py_object_to_int64(N);
        auto single = creation_annihilation_op_from_single_Nmax(nmax, nmax + 1);
        creators_i.append(single.first);
        annihilators_i.append(single.second);
    }
    py::list ids_i;
    for (auto N : Nmax_) {
        ids_i.append(np.attr("eye")(py_object_to_int64(N) + 1));
    }
    py::list creators;
    py::list annihilators;
    auto len = py::len(Nmax_);
    for (py::ssize_t i = 0; i < len; ++i) {
        py::list c_factors;
        py::list a_factors;
        for (py::ssize_t j = 0; j < len; ++j) {
            if (j < i) {
                c_factors.append(ids_i[j]);
                a_factors.append(ids_i[j]);
            } else if (j == i) {
                c_factors.append(creators_i[j]);
                a_factors.append(annihilators_i[j]);
            } else {
                c_factors.append(ids_i[j]);
                a_factors.append(ids_i[j]);
            }
        }
        creators.append(py_reduce(np.attr("kron"), c_factors));
        annihilators.append(py_reduce(np.attr("kron"), a_factors));
    }
    (void)dim;
    return { np.attr("stack")(creators, py::arg("axis") = 2),
             np.attr("stack")(annihilators, py::arg("axis") = 2) };
}

std::pair<py::array, py::array>
BosonicDOF::creation_annihilation_ops(int64 num_species, py::array Nmax, int64 dim)
{
    (void)num_species;
    return creation_annihilation_ops_from_Nmax(std::move(Nmax), dim);
}

FermionicDOF::FermionicDOF(ElementarySpace::Ptr leg,
                           py::array creators,
                           py::array annihilators,
                           std::vector<std::optional<std::string>> species_names,
                           std::map<std::string, int64> state_labels,
                           std::map<std::string, SymmetricTensorPtr> onsite_operators,
                           TensorBackend::Ptr backend,
                           std::optional<std::string> default_device)
  : Site(leg, state_labels, {}, backend, default_device)
  , OccupationDOF(leg,
                  creators,
                  annihilators,
                  -1,
                  species_names,
                  state_labels,
                  {},
                  backend,
                  default_device)
{
    int64 fermion_factor_count = 0;
    for (auto const& factor : leg_symmetry(leg)->factors) {
        if (dynamic_cast<FermionParity const*>(factor.get()) ||
            dynamic_cast<FermionNumber const*>(factor.get())) {
            ++fermion_factor_count;
        }
    }
    assert(fermion_factor_count == 1);

    if (dynamic_cast<BosonicDOF*>(this)) {
        throw SymmetryError("FermionicDOF and BosonicDOF are incompatible.");
    }

    auto np = numpy();
    auto d = leg_dim(leg);
    py::array n_diag = number_operators.attr("__getitem__")(
      py::make_tuple(np.attr("arange")(d), np.attr("arange")(d), py::slice()));
    n_diag.attr("__setitem__")(
      py::make_tuple(py::slice(), py::slice(py::int_(1), py::none(), py::none())),
      n_diag.attr("__getitem__")(
        py::make_tuple(py::slice(), py::slice(py::none(), py::int_(-1), py::none()))));
    n_diag.attr("__setitem__")(py::make_tuple(py::slice(), 0), 0);
    py::array n_before = np.attr("cumsum")(n_diag, py::arg("axis") = 1);
    py::array partial_JW = np.attr("zeros")(py::make_tuple(d, d, num_species));
    partial_JW.attr("__setitem__")(
      py::make_tuple(np.attr("arange")(d), np.attr("arange")(d), py::slice()),
      np.attr("power")(py::cast(-1), n_before));
    partial_JWs = as_immutable_array(partial_JW);
    JW =
      as_immutable_array(np.attr("diag")(np.attr("power")(py::cast(-1), np.attr("diag")(n_tot))));

    for (int64 k = 0; k < num_species; ++k) {
        py::array N_k =
          number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        double N_k_max_ = np.attr("max")(np.attr("diag")(N_k)).cast<double>();
        int64 N_k_max = static_cast<int64>(std::llround(N_k_max_));
        assert(std::abs(N_k_max - N_k_max_) < 1e-9);
        assert(N_k_max == 1);
    }

    add_onsite_operators_from_map(*this, onsite_operators);
}

void
FermionicDOF::test_sanity()
{
    OccupationDOF::test_sanity();
    auto np = numpy();
    for (int64 k = 0; k < num_species; ++k) {
        py::array c_k =
          annihilators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        py::array cd_k = creators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        py::array N_k =
          number_operators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), k));
        py::array CC = np.attr("matmul")(c_k, c_k);
        py::array CdCd = np.attr("matmul")(cd_k, cd_k);
        assert(np_allclose(CC, np.attr("zeros_like")(CC)));
        assert(np_allclose(CdCd, np.attr("zeros_like")(CdCd)));
        assert(np.attr("max")(N_k).cast<double>() <= 1.);
    }
}

py::array
FermionicDOF::get_annihilator_numpy(py::object species, bool include_JW)
{
    auto idx = get_species_idx(species);
    py::array res =
      annihilators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), idx));
    if (include_JW) {
        auto np = numpy();
        res = np.attr("matmul")(
          res, partial_JWs.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), idx)));
    }
    return res;
}

py::array
FermionicDOF::get_creator_numpy(py::object species, bool include_JW)
{
    auto idx = get_species_idx(species);
    py::array res = creators.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), idx));
    if (include_JW) {
        auto np = numpy();
        res = np.attr("matmul")(
          res, partial_JWs.attr("__getitem__")(py::make_tuple(py::slice(), py::slice(), idx)));
    }
    return res;
}

Symmetry::Ptr
FermionicDOF::conservation_law_to_symmetry(std::optional<std::string> conserve)
{
    if (!conserve.has_value()) {
        throw std::invalid_argument("Invalid `conserve`: null");
    }
    if (*conserve == "N" || *conserve == "Ntot" || *conserve == "N_tot") {
        return symmetry_from_factors({ std::make_shared<U1>("total_fermion_occupation"),
                                       std::make_shared<FermionParity>("total_fermion_parity") });
    }
    if (*conserve == "parity" || *conserve == "P" || *conserve == "Ptot" || *conserve == "P_tot") {
        return symmetry_from_factor(std::make_shared<FermionParity>("total_fermion_parity"));
    }
    throw std::invalid_argument(std::format("Invalid `conserve`: {}", *conserve));
}

std::pair<py::array, py::array>
FermionicDOF::creation_annihilation_ops(int64 num_species)
{
    auto np = numpy();
    py::array Nmax = np.attr("ones")(num_species, py::dtype("int"));
    int64 dim = num_species >= 0 ? (int64{ 1 } << num_species) : 1;
    return BosonicDOF::creation_annihilation_ops_from_Nmax(Nmax, dim);
}

} // namespace cyten
