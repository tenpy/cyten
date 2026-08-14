#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/helpers.h>
#include <cyten/tensors/symmetric_tensor.h>

#include <cyten/backends/backend_factory.h>
#include <cyten/tools.h>
#include <cyten/warn.h>

#include <cassert>
#include <format>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

namespace {

std::vector<BlockBackend::LegCPtr>
legs_from_py(std::vector<Leg::Ptr> const& objs)
{
    std::vector<BlockBackend::LegCPtr> out;
    out.reserve(objs.size());
    for (auto const& o : objs) {
        out.push_back(o);
    }
    return out;
}

py::dict
copy_dict(py::object obj)
{
    if (obj.is_none()) {
        return py::dict();
    }
    return py::dict(obj);
}

py::tuple
shape_as_tuple(std::vector<int64> const& shape)
{
    py::tuple t(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        t[i] = py::int_(shape[i]);
    }
    return t;
}

} // namespace

SymmetricTensor::SymmetricTensor(TensorBackend::DataPtr data_in,
                                 TensorProduct::Ptr codomain_in,
                                 TensorProduct::Ptr domain_in,
                                 TensorBackend::Ptr backend_in,
                                 Symmetry::Ptr symmetry_in,
                                 LegLabels labels_in,
                                 bool check_complex_dtype)
  : Tensor(std::move(codomain_in),
           std::move(domain_in),
           backend_in,
           std::move(symmetry_in),
           std::move(labels_in),
           backend_in->get_dtype_from_data(data_in),
           backend_in->get_device_from_data(data_in))
  , data(std::move(data_in))
{
    if (!backend->DataCls.is_none() && !std::dynamic_pointer_cast<NoSymmetryBackend>(backend)) {
        // NoSymmetry stores BlockData while DataCls is BlockCls (Python stores the Block).
        assert(py::isinstance(py::cast(data), backend->DataCls));
    }
    if (check_complex_dtype) {
        verify_dtype();
    }
}

void
SymmetricTensor::test_sanity() const
{
    Tensor::test_sanity();
    assert(dtype == backend->get_dtype_from_data(data));
    assert(device == backend->get_device_from_data(data));
    // Prefer C++ RTTI so _core.DiagonalTensor is recognized before monkey-patch.
    bool is_diagonal = dynamic_cast<DiagonalTensor const*>(this) != nullptr;
    if (!is_diagonal && Py_IsInitialized()) {
        try {
            is_diagonal =
              py::isinstance(py::cast(shared_from_this()),
                             py::module_::import("cyten.tensors._tensors").attr("DiagonalTensor"));
        } catch (py::error_already_set& e) {
            e.restore();
            PyErr_Clear();
        }
    }
    backend->test_tensor_sanity(shared_from_this(), is_diagonal);
    verify_dtype();
}

void
SymmetricTensor::verify_dtype() const
{
    if (symmetry->has_complex_topological_data && dtype::is_real(dtype)) {
        throw std::invalid_argument(
          std::format("SymmetricTensor with {} must have complex dtype", symmetry->repr()));
    }
}

std::string
SymmetricTensor::ascii_diagram_type_name() const
{
    return "Symm";
}

std::string
SymmetricTensor::class_name() const
{
    return "SymmetricTensor";
}

std::optional<Dtype>
SymmetricTensor::_parse_default_dtype(std::optional<Dtype> dtype, Symmetry::Ptr const& symmetry)
{
    if (symmetry->has_complex_topological_data) {
        if (!dtype.has_value()) {
            dtype = Dtype::Complex128;
        }
        if (dtype::is_real(*dtype)) {
            throw std::invalid_argument(
              std::format("SymmetricTensor with {} must have complex dtype", symmetry->repr()));
        }
    }
    return dtype;
}

SymmetricTensor::Ptr
SymmetricTensor::from_zero(TensorProduct::Ptr codomain,
                           TensorProduct::Ptr domain,
                           TensorBackend::Ptr backend,
                           std::optional<LegLabels> labels,
                           Dtype dtype,
                           std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    auto dt = _parse_default_dtype(dtype, symmetry);
    assert(dt.has_value());
    auto device_s = backend_tp->block_backend->as_device(device);
    auto data = backend_tp->zero_data(codomain_tp, domain_tp, *dt, device_s);
    return std::make_shared<SymmetricTensor>(
      data,
      codomain_tp,
      domain_tp,
      backend_tp,
      symmetry,
      _init_parse_labels(std::move(labels), codomain_tp, domain_tp));
}

SymmetricTensor::Ptr
SymmetricTensor::from_eye(TensorProduct::Ptr co_domain,
                          TensorBackend::Ptr backend,
                          std::optional<LegLabels> labels,
                          Dtype dtype,
                          std::optional<std::string> device)
{
    auto [co_domain_tp, unused_domain, backend_tp, symmetry] =
      _init_parse_args(co_domain, co_domain, std::move(backend));
    (void)unused_domain;
    auto dt = _parse_default_dtype(dtype, symmetry);
    assert(dt.has_value());
    auto labels_parsed =
      _init_parse_labels(std::move(labels), co_domain_tp, co_domain_tp, /*is_endomorphism=*/true);
    auto device_s = backend_tp->block_backend->as_device(device);
    auto data = backend_tp->eye_data(co_domain_tp, *dt, device_s);
    return std::make_shared<SymmetricTensor>(
      data, co_domain_tp, co_domain_tp, backend_tp, symmetry, std::move(labels_parsed));
}

SymmetricTensor::Ptr
SymmetricTensor::from_block_func(py::function func,
                                 TensorProduct::Ptr codomain,
                                 TensorProduct::Ptr domain,
                                 TensorBackend::Ptr backend,
                                 std::optional<LegLabels> labels,
                                 py::object func_kwargs,
                                 std::optional<std::string> shape_kw,
                                 std::optional<Dtype> dtype,
                                 std::optional<std::string> device)
{
    // --- hints from Python SymmetricTensor.from_block_func ---
    // wrap func to consider func_kwargs, shape_kw, dtype, device
    // use same backend function as from_sector_block_func, so we include the coupled arg
    // but just ignore it.
    // OPTIMIZE remove?
    // ---
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    dtype = _parse_default_dtype(dtype, symmetry);

    py::dict kwargs = copy_dict(func_kwargs);
    py::object shape_kw_obj = shape_kw.has_value() ? py::cast(*shape_kw) : py::none();
    std::optional<Dtype> dtype_cap = dtype;
    std::optional<std::string> device_cap = device;
    auto bb = backend_tp->block_backend;

    // wrap func to consider func_kwargs, shape_kw, dtype, device
    SectorBlockFactoryFn block_func =
      [func, kwargs, shape_kw_obj, dtype_cap, device_cap, bb](std::vector<int64> const& shape,
                                                             Sector const& /*coupled*/) {
          py::object block;
          auto shape_t = shape_as_tuple(shape);
          if (shape_kw_obj.is_none()) {
              block = func(shape_t, **kwargs);
          } else {
              py::dict call_kwargs = py::dict(kwargs);
              call_kwargs[shape_kw_obj] = shape_t;
              block = func(**call_kwargs);
          }
          return bb->as_block(block, dtype_cap, device_cap);
      };

    auto data = backend_tp->from_sector_block_func(block_func, codomain_tp, domain_tp);
    auto res = std::make_shared<SymmetricTensor>(
      data,
      codomain_tp,
      domain_tp,
      backend_tp,
      symmetry,
      _init_parse_labels(std::move(labels), codomain_tp, domain_tp));
    res->test_sanity(); // OPTIMIZE remove?
    return res;
}

SymmetricTensor::Ptr
SymmetricTensor::from_sector_block_func(py::function func,
                                        TensorProduct::Ptr codomain,
                                        TensorProduct::Ptr domain,
                                        TensorBackend::Ptr backend,
                                        std::optional<LegLabels> labels,
                                        py::object func_kwargs,
                                        std::optional<Dtype> dtype,
                                        std::optional<std::string> device)
{
    // --- hints from Python SymmetricTensor.from_sector_block_func ---
    // wrap func to consider func_kwargs and dtype
    // ---
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    dtype = _parse_default_dtype(dtype, symmetry);

    // wrap func to consider func_kwargs and dtype
    py::dict kwargs = copy_dict(func_kwargs);
    std::optional<Dtype> dtype_cap = dtype;
    std::optional<std::string> device_cap = device;
    auto bb = backend_tp->block_backend;

    SectorBlockFactoryFn block_func =
      [func, kwargs, dtype_cap, device_cap, bb](std::vector<int64> const& shape,
                                                Sector const& coupled) {
          py::object block = func(shape_as_tuple(shape), py::cast(coupled), **kwargs);
          return bb->as_block(block, dtype_cap, device_cap);
      };

    auto data = backend_tp->from_sector_block_func(block_func, codomain_tp, domain_tp);
    auto res = std::make_shared<SymmetricTensor>(
      data,
      codomain_tp,
      domain_tp,
      backend_tp,
      symmetry,
      _init_parse_labels(std::move(labels), codomain_tp, domain_tp));
    res->test_sanity();
    return res;
}

SymmetricTensor::Ptr
SymmetricTensor::from_dense_block(BlockBackend::BlockPtr block,
                                  TensorProduct::Ptr codomain,
                                  TensorProduct::Ptr domain,
                                  TensorBackend::Ptr backend,
                                  std::optional<LegLabels> labels,
                                  std::optional<Dtype> dtype,
                                  std::optional<std::string> device,
                                  float64 tol,
                                  bool understood_braiding)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    dtype = _parse_default_dtype(dtype, symmetry);
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    if (!symmetry->has_trivial_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of from_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    auto block_ptr = backend_tp->block_backend->as_block(py::cast(block), dtype, device);
    assert(static_cast<int64>(backend_tp->block_backend->get_shape(block_ptr).size()) ==
           codomain_tp->num_factors + domain_tp->num_factors);
    block_ptr = backend_tp->block_backend->apply_basis_perm(
      block_ptr, legs_from_py(conventional_leg_order(codomain_tp, domain_tp)));
    auto data = backend_tp->from_dense_block(block_ptr, codomain_tp, domain_tp, tol);
    return std::make_shared<SymmetricTensor>(
      data,
      codomain_tp,
      domain_tp,
      backend_tp,
      symmetry,
      _init_parse_labels(std::move(labels), codomain_tp, domain_tp));
}

SymmetricTensor::Ptr
SymmetricTensor::from_dense_block_trivial_sector(BlockBackend::BlockPtr vector,
                                                 Space::Ptr space,
                                                 TensorBackend::Ptr backend,
                                                 std::optional<std::string> device,
                                                 LegLabel /*label*/)
{
    if (!backend) {
        backend = get_backend(space->symmetry);
    }
    auto vec = backend->block_backend->as_block(py::cast(vector), std::nullopt, device);
    // Python checks ``space._basis_perm is not None`` then applies a perm; unfinished below.
    if (py::isinstance<Leg>(py::cast(space)) &&
        py::cast(space).cast<Leg::Ptr>()->has_custom_basis_perm()) {
        auto i = space->sector_decomposition_where(space->symmetry->trivial_sector);
        assert(i.has_value());
        (void)vec;
        (void)i;
    }
    throw NotImplemented("SymmetricTensor.from_dense_block_trivial_sector");
}

SymmetricTensor::Ptr
SymmetricTensor::from_random_normal(TensorProduct::Ptr codomain,
                                    TensorProduct::Ptr domain,
                                    TensorCPtr mean,
                                    float64 sigma,
                                    TensorBackend::Ptr backend,
                                    std::optional<LegLabels> labels,
                                    std::optional<Dtype> dtype,
                                    std::optional<std::string> device)
{
    assert(sigma > 0.0);
    Symmetry::Ptr symmetry;
    TensorProduct::Ptr codomain_tp;
    TensorProduct::Ptr domain_tp;
    TensorBackend::Ptr backend_tp;

    if (mean) {
        if (!codomain) {
            codomain = mean->codomain;
        } else {
            assert(mean->codomain->operator==(*codomain));
        }
        if (!domain) {
            domain = mean->domain;
        } else {
            assert(mean->domain->operator==(*domain));
        }
        if (!backend) {
            backend = mean->backend;
        } else {
            assert(mean->backend == backend);
        }
        auto [c, d, b, s] = _init_parse_args(std::move(codomain), std::move(domain), backend);
        codomain_tp = std::move(c);
        domain_tp = std::move(d);
        backend_tp = std::move(b);
        symmetry = std::move(s);
        if (!labels.has_value()) {
            labels = mean->labels();
        } else {
            assert(mean->labels() == _init_parse_labels(labels, codomain_tp, domain_tp));
        }
        if (!dtype.has_value()) {
            dtype = mean->dtype;
        } else {
            assert(mean->dtype == *dtype);
        }
        if (!device.has_value()) {
            device = mean->device;
        }
    } else {
        if (!codomain) {
            throw std::invalid_argument("Must specify the codomain if mean is not given.");
        }
        auto [c, d, b, s] =
          _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
        codomain_tp = std::move(c);
        domain_tp = std::move(d);
        backend_tp = std::move(b);
        symmetry = std::move(s);
        if (!device.has_value()) {
            device = backend_tp->block_backend->default_device;
        }
    }

    dtype = _parse_default_dtype(dtype, symmetry);
    assert(dtype.has_value());
    assert(device.has_value());

    auto data = backend_tp->from_random_normal(codomain_tp, domain_tp, sigma, *dtype, *device);
    auto with_zero_mean = std::make_shared<SymmetricTensor>(
      data,
      codomain_tp,
      domain_tp,
      backend_tp,
      symmetry,
      _init_parse_labels(std::move(labels), codomain_tp, domain_tp));

    if (mean) {
        auto one = backend_tp->block_backend->as_scalar(1.0);
        auto new_data = backend_tp->linear_combination(one, mean, one, with_zero_mean);
        return std::make_shared<SymmetricTensor>(
          new_data, codomain_tp, domain_tp, backend_tp, symmetry, with_zero_mean->labels());
    }
    return with_zero_mean;
}

SymmetricTensor::Ptr
SymmetricTensor::from_random_uniform(TensorProduct::Ptr codomain,
                                     TensorProduct::Ptr domain,
                                     TensorBackend::Ptr backend,
                                     std::optional<LegLabels> labels,
                                     Dtype dtype,
                                     std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    auto dt = _parse_default_dtype(dtype, symmetry);
    assert(dt.has_value());
    auto bb = backend_tp->block_backend;
    py::cpp_function func([bb](py::object shape, py::kwargs kwargs) {
        auto dims = shape.cast<std::vector<int64>>();
        auto d = kwargs["dtype"].cast<Dtype>();
        std::optional<std::string> dev;
        if (!kwargs["device"].is_none()) {
            dev = kwargs["device"].cast<std::string>();
        }
        return bb->random_uniform(dims, d, dev);
    });
    py::dict func_kwargs;
    func_kwargs["dtype"] = py::cast(*dt);
    if (device.has_value()) {
        func_kwargs["device"] = py::cast(*device);
    } else {
        func_kwargs["device"] = py::none();
    }
    return from_block_func(func,
                           codomain_tp,
                           domain_tp,
                           backend_tp,
                           std::move(labels),
                           func_kwargs,
                           std::nullopt,
                           dt,
                           device);
}

SymmetricTensor::Ptr
SymmetricTensor::from_sector_projection(TensorProduct::Ptr co_domain,
                                        Sector sector,
                                        TensorBackend::Ptr backend,
                                        std::optional<LegLabels> labels,
                                        std::optional<Dtype> dtype,
                                        std::optional<std::string> device)
{
    auto [co_domain_tp, unused_domain, backend_tp, unused_symm] =
      _init_parse_args(co_domain, co_domain, std::move(backend));
    (void)unused_domain;
    (void)unused_symm;
    assert(co_domain_tp->symmetry->is_valid_sector(sector));
    if (co_domain_tp->sector_multiplicity(sector) == 0) {
        warn("Sector does not appear. from_sector_projection yields zero");
    }
    dtype = _parse_default_dtype(dtype, co_domain_tp->symmetry);
    std::optional<Dtype> dtype_cap = dtype;
    std::optional<std::string> device_cap = device;
    Sector sector_copy = sector;
    auto bb = backend_tp->block_backend;

    SectorBlockFactoryFn func =
      [bb, dtype_cap, device_cap, sector_copy](std::vector<int64> const& shape,
                                               Sector const& coupled) {
          Dtype dt = dtype_cap.value_or(Dtype::Complex128);
          if (coupled == sector_copy) {
              std::vector<int64> half(shape.begin(),
                                      shape.begin() + static_cast<std::ptrdiff_t>(shape.size() / 2));
              return bb->eye_block(half, dt, device_cap);
          }
          return bb->zeros(shape, dt, device_cap);
      };

    auto data = backend_tp->from_sector_block_func(func, co_domain_tp, co_domain_tp);
    auto res = std::make_shared<SymmetricTensor>(
      data,
      co_domain_tp,
      co_domain_tp,
      backend_tp,
      co_domain_tp->symmetry,
      _init_parse_labels(std::move(labels), co_domain_tp, co_domain_tp));
    res->test_sanity();
    return res;
}

SymmetricTensor::Ptr
SymmetricTensor::from_tree_pairs(py::object trees_obj,
                                 TensorProduct::Ptr codomain,
                                 TensorProduct::Ptr domain,
                                 TensorBackend::Ptr backend,
                                 std::optional<LegLabels> labels,
                                 std::optional<Dtype> dtype,
                                 std::optional<std::string> device)
{
    py::dict trees = trees_obj.cast<py::dict>();
    if (py::len(trees) == 0) {
        if (!dtype.has_value()) {
            throw std::invalid_argument("dtype is required if trees is empty");
        }
        if (!device.has_value()) {
            throw std::invalid_argument("device is required if trees is empty");
        }
        return from_zero(std::move(codomain),
                         std::move(domain),
                         std::move(backend),
                         std::move(labels),
                         *dtype,
                         device);
    }
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(std::move(codomain), std::move(domain), std::move(backend));
    if (codomain_tp->has_pipes() || domain_tp->has_pipes()) {
        throw NotImplemented("from_tree_pairs does not support pipes (yet?)");
    }
    dtype = _parse_default_dtype(dtype, symmetry);
    std::string device_s;
    if (!device.has_value()) {
        auto some_block = backend_tp->block_backend->as_block(
          py::reinterpret_borrow<py::object>(trees.begin()->second));
        device_s = backend_tp->block_backend->get_device(some_block);
    } else {
        device_s = *device;
    }

    std::vector<std::uint8_t> X_are_dual;
    std::vector<std::uint8_t> Y_are_dual;
    X_are_dual.reserve(codomain_tp->factors.size());
    Y_are_dual.reserve(domain_tp->factors.size());
    for (auto const& leg : codomain_tp->factors) {
        X_are_dual.push_back(leg->is_dual ? 1 : 0);
    }
    for (auto const& leg : domain_tp->factors) {
        Y_are_dual.push_back(leg->is_dual ? 1 : 0);
    }

    std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> block_trees;
    // Match Python: convert BlockLikes in-place so callers can use block.to_numpy().
    for (auto item : trees) {
        auto key_obj = py::reinterpret_borrow<py::object>(item.first);
        auto key_tup = key_obj.cast<py::tuple>();
        FusionTree X = key_tup[0].cast<FusionTree>();
        FusionTree Y = key_tup[1].cast<FusionTree>();
        assert(X.coupled == Y.coupled);
        assert(X.are_dual == X_are_dual);
        assert(Y.are_dual == Y_are_dual);
        auto block = backend_tp->block_backend->as_block(
          py::reinterpret_borrow<py::object>(item.second), dtype, device_s);
        assert(backend_tp->block_backend->get_device(block) == device_s);
        trees[key_obj] = py::cast(block);
        block_trees.emplace(std::make_pair(std::move(X), std::move(Y)), std::move(block));
    }
    if (!dtype.has_value()) {
        std::vector<Dtype> dts;
        dts.reserve(block_trees.size());
        for (auto const& [_, b] : block_trees) {
            (void)_;
            dts.push_back(backend_tp->block_backend->get_dtype(b));
        }
        dtype = dtype::common(dts);
    }
    auto data = backend_tp->from_tree_pairs(block_trees, codomain_tp, domain_tp, *dtype, device_s);
    return std::make_shared<SymmetricTensor>(
      data,
      codomain_tp,
      domain_tp,
      backend_tp,
      symmetry,
      _init_parse_labels(std::move(labels), codomain_tp, domain_tp));
}

Tensor::Ptr
SymmetricTensor::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    auto new_data = backend->to_dtype(shared_from_this(), new_dtype);
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels());
}

SymmetricTensorPtr
SymmetricTensor::as_SymmetricTensor(bool guarantee_copy, std::optional<std::string> /*warning*/)
{
    if (guarantee_copy) {
        return std::static_pointer_cast<SymmetricTensor>(copy());
    }
    return std::static_pointer_cast<SymmetricTensor>(shared_from_this());
}

Tensor::Ptr
SymmetricTensor::copy(bool deep,
                      std::optional<std::string> device_opt,
                      std::optional<Dtype> dtype_opt)
{
    TensorBackend::DataPtr new_data;
    // Match Python: ``if dtype is not None and dtype != self.dtype:`` then
    // ``if device is not None or device != self.device:`` — the latter is always true when
    // ``device is None``, so the move branch is unreachable; effective behavior is as_dtype.
    if (dtype_opt.has_value() && *dtype_opt != dtype) {
        return as_dtype(*dtype_opt);
    }
    if (deep) {
        new_data = backend->copy_data(shared_from_this(), device_opt);
    } else if (device_opt.has_value()) {
        new_data = backend->move_to_device(shared_from_this(), *device_opt);
    } else {
        new_data = data;
    }
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels());
}

DiagonalTensorPtr
SymmetricTensor::diagonal(bool check_offdiagonal) const
{
    // Python passes check_offdiagonal as a kwarg name mismatch to from_tensor(tol=...);
    // Map True -> default tol, False -> None (skip check), matching intended semantics.
    std::optional<float64> tol =
      check_offdiagonal ? std::optional<float64>{ 1e-12 } : std::nullopt;
    return DiagonalTensor::from_tensor(
      std::static_pointer_cast<SymmetricTensor const>(shared_from_this()), tol);
}

BlockBackend::Scalar
SymmetricTensor::_get_item(std::vector<int64> const& idx)
{
    return backend->get_element(std::static_pointer_cast<SymmetricTensor>(shared_from_this()),
                                idx);
}

void
SymmetricTensor::move_to_device(std::string device_in)
{
    data = backend->move_to_device(shared_from_this(), device_in);
    device = backend->block_backend->as_device(device_in);
}

Tensor::Ptr
SymmetricTensor::to_backend(TensorBackend::Ptr new_backend,
                            std::optional<Dtype> dtype_opt,
                            std::optional<std::string> device_opt)
{
    // --- hints from Python SymmetricTensor.to_backend ---
    // Flatten the pipes, convert backends on flat leg basis, then recombine
    // This means we dont have to deal with the permutation induced by pipes in the AB backend
    // or with the special AbelianLegPipe type
    // OPTIMIZE do it directly if no abelian backend is involved?
    // ---
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }
    Dtype dt = dtype_opt.value_or(dtype);
    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });

    if (has_pipes()) {
        // Flatten the pipes, convert backends on flat leg basis, then recombine
        // This means we dont have to deal with the permutation induced by pipes in the AB backend
        // or with the special AbelianLegPipe type
        // OPTIMIZE do it directly if no abelian backend is involved?
        auto tensors_mod = py::module_::import("cyten.tensors._tensors");
        std::vector<py::object> combine;
        std::vector<bool> pipe_dualities;
        int64 flat_leg_counter = 0;
        for (auto const& leg : legs()) {
            if (auto pipe = std::dynamic_pointer_cast<LegPipe>(leg)) {
                auto num = pipe->num_legs;
                py::list group;
                for (int64 i = flat_leg_counter; i < flat_leg_counter + num; ++i) {
                    group.append(i);
                }
                combine.push_back(group);
                pipe_dualities.push_back(leg->is_dual);
                flat_leg_counter += num;
            } else {
                flat_leg_counter += 1;
            }
        }
        py::object flat = tensors_mod.attr("split_legs")(py::cast(shared_from_this()));
        py::object res_flat =
          flat.attr("to_backend")(py::cast(new_backend), py::cast(dt), py::cast(device_s));
        py::object res = tensors_mod.attr("combine_legs")(
          res_flat, *py::tuple(py::cast(combine)), py::arg("pipe_dualities") = pipe_dualities);
        // Do not cast res.attr("data") to DataPtr: NoSymmetry exposes the raw Block.
        return res.cast<SymmetricTensor::Ptr>();
    }

    TensorBackend::DataPtr new_data;
    if (std::dynamic_pointer_cast<NoSymmetryBackend>(new_backend)) {
        auto old_block = backend->to_dense_block(shared_from_this());
        auto new_block = new_backend->block_backend->as_block(py::cast(old_block), dt, device_s);
        new_data = NoSymmetryBackend::wrap(new_block);
    } else if (std::dynamic_pointer_cast<NoSymmetryBackend>(backend)) {
        auto old_block = NoSymmetryBackend::unwrap(data);
        auto new_block = new_backend->block_backend->as_block(py::cast(old_block), dt, device_s);
        new_data = new_backend->from_dense_block(new_block, codomain, domain, 0.);
    } else if (std::dynamic_pointer_cast<AbelianBackend>(new_backend)) {
        if (std::dynamic_pointer_cast<AbelianBackend>(backend)) {
            new_data = backend->to_block_backend(data, new_backend->block_backend, dt, device_s);
        } else if (std::dynamic_pointer_cast<FusionTreeBackend>(backend)) {
            new_data =
              _convert_FT_to_abelian(shared_from_this(),
                                     std::dynamic_pointer_cast<AbelianBackend>(new_backend),
                                     dt,
                                     device_s);
        } else {
            // --- hints from Python _convert_FT_to_abelian ---
            // fusion rule violated
            // no block for this coupled sector -> dont need to add a result block either
            // convert to new block_backend
            // sector combination violates fusion rules -> no contributions
            // move down by one tree-block
            // reset to the top
            // move to the right by one tree-block, for the next time we visit this block
            // ---
            throw std::runtime_error("Unexpected backend combination");
        }
    } else if (std::dynamic_pointer_cast<FusionTreeBackend>(new_backend)) {
        if (std::dynamic_pointer_cast<AbelianBackend>(backend)) {
            new_data =
              _convert_abelian_to_FT(shared_from_this(),
                                     std::dynamic_pointer_cast<FusionTreeBackend>(new_backend),
                                     dt,
                                     device_s);
        } else if (std::dynamic_pointer_cast<FusionTreeBackend>(backend)) {
            // --- hints from Python _convert_abelian_to_FT ---
            // Start with all allowed blocks initialized with zeros
            // OPTIMIZE create the blocks on-demand instead?
            // block is missing (zero) -> nothing to do
            // this can happen if c does not appear in the codomain at all -> no block
            // sector combination violates fusion rules -> no contributions
            // OPTIMIZE use that the data.block_inds are lexsorted for this lookup (also above)
            // cstyle combine in the codomain, Fstyle in the domain
            // move down by one tree-block
            // reset to the top
            // move to the right by one tree-block, for the next time we visit this block
            // ---
            new_data = backend->to_block_backend(data, new_backend->block_backend, dt, device_s);
        } else {
            throw std::runtime_error("Unexpected backend combination");
        }
    } else {
        throw std::invalid_argument(
          std::format("Unexpected backend type {}", typeid(*new_backend).name()));
    }

    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, new_backend, symmetry, labels());
}

BlockBackend::BlockPtr
SymmetricTensor::to_dense_block(
  std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
  std::optional<Dtype> dtype_opt,
  bool understood_braiding)
{
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
    auto block = backend->to_dense_block(shared_from_this());
    block = backend->block_backend->apply_basis_perm(
      block, legs_from_py(conventional_leg_order(shared_from_this())), /*inv=*/true);
    if (dtype_opt.has_value()) {
        block = backend->block_backend->to_dtype(block, *dtype_opt);
    }
    if (leg_order.has_value()) {
        auto idcs = get_leg_idcs(*leg_order);
        block = backend->block_backend->permute_axes(block, idcs);
    }
    return block;
}

BlockBackend::BlockPtr
SymmetricTensor::to_dense_block_trivial_sector() const
{
    // --- hints from Python SymmetricTensor.to_dense_block_trivial_sector ---
    // TODO assuming this for now to construct the perm. should we keep that?
    // ---
    assert(num_legs == 1);
    auto block = backend->to_dense_block_trivial_sector(shared_from_this());
    assert(num_codomain_legs() ==
           1); // TODO assuming this for now to construct the perm. should we keep that?
    auto space = as_space(codomain->factors[0]);
    auto leg = codomain->factors[0];
    if (leg->has_custom_basis_perm()) {
        auto i = space->sector_decomposition_where(symmetry->trivial_sector);
        assert(i.has_value());
        assert(space->slices.has_value());
        auto const& sl = (*space->slices)[static_cast<std::size_t>(*i)];
        auto basis_perm = leg->basis_perm();
        std::vector<int64> segment(basis_perm.begin() + sl[0], basis_perm.begin() + sl[1]);
        // Python: perm = np.argsort(leg.basis_perm[slice(*leg.slices[i])])
        std::vector<int64> order(segment.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&](int64 a, int64 b) {
            return segment[static_cast<std::size_t>(a)] < segment[static_cast<std::size_t>(b)];
        });
        block = backend->block_backend->apply_leg_permutations(
          block, { py::array_t<int64>(static_cast<py::ssize_t>(order.size()), order.data()) });
    }
    return block;
}

void
SymmetricTensor::save_hdf5(py::object hdf5_saver,
                           py::object h5gr,
                           std::string const& subpath) const
{
    /// Export SymmetricTensor to hdf5 such that it can be re-imported with from_hdf5
    hdf5_saver.attr("save")(py::cast(domain), subpath + "domain");
    hdf5_saver.attr("save")(py::cast(codomain), subpath + "codomain");
    hdf5_saver.attr("save")(py::cast(backend), subpath + "backend");
    hdf5_saver.attr("save")(py::cast(data), subpath + "data");
    hdf5_saver.attr("save")(py::cast(symmetry), subpath + "symmetry");
    hdf5_saver.attr("save")(dtype::to_numpy_dtype(dtype), subpath + "dtype");
    hdf5_saver.attr("save")(device, subpath + "device");
    h5gr.attr("attrs")["num_legs"] = num_legs;
    h5gr.attr("attrs")["shape"] = py::cast(shape);
    h5gr.attr("attrs")["cls"] = class_name();
    if (std::ranges::all_of(_labels, [](LegLabel const& l) { return !l; })) {
        h5gr.attr("attrs")["labels"] = py::list();
    } else {
        h5gr.attr("attrs")["labels"] = py::cast(_labels);
    }
}

SymmetricTensor::Ptr
SymmetricTensor::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    /// Import SymmetricTensor from hdf5
    auto domain = hdf5_loader.attr("load")(subpath + "domain").cast<TensorProduct::Ptr>();
    auto codomain = hdf5_loader.attr("load")(subpath + "codomain").cast<TensorProduct::Ptr>();
    auto symmetry = hdf5_loader.attr("load")(subpath + "symmetry").cast<Symmetry::Ptr>();
    auto backend = get_backend(py::cast(symmetry), py::cast("numpy")).cast<TensorBackend::Ptr>();
    auto data = hdf5_loader.attr("load")(subpath + "data").cast<TensorBackend::DataPtr>();
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    auto dt = dtype::from_numpy_dtype(hdf5_loader.attr("load")(subpath + "dtype"));
    (void)hdf5_loader.attr("get_attr")(h5gr, "num_legs");
    auto shape = hdf5_loader.attr("get_attr")(h5gr, "shape").cast<std::vector<float64>>();
    auto labels = hdf5_loader.attr("get_attr")(h5gr, "labels").cast<LegLabels>();
    // Match Python save: all-None labels are stored as []; expand for the Tensor ctor.
    int64 nlegs = codomain->num_factors + domain->num_factors;
    if (labels.empty() && nlegs > 0) {
        labels.assign(static_cast<std::size_t>(nlegs), std::nullopt);
    }

    auto obj = std::make_shared<SymmetricTensor>(
      data, codomain, domain, backend, symmetry, std::move(labels));
    // Constructor overwrites dtype/device from data; restore hdf5 values if needed
    obj->dtype = dt;
    obj->device = device;
    obj->shape = std::move(shape);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
