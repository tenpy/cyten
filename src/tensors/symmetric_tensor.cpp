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
legs_from_py(std::vector<py::object> const& objs)
{
    std::vector<BlockBackend::LegCPtr> out;
    out.reserve(objs.size());
    for (auto const& o : objs) {
        out.push_back(o.cast<Leg::Ptr>());
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

} // namespace

SymmetricTensor::SymmetricTensor(TensorBackend::DataPtr data_in,
                                 py::object codomain_obj,
                                 py::object domain_obj,
                                 TensorBackend::Ptr backend_in,
                                 py::object labels_obj)
  : Tensor(codomain_obj,
           domain_obj,
           std::move(backend_in),
           labels_obj,
           Dtype::Float64, // overwritten from data below
           "")
  , data(std::move(data_in))
{
    dtype = backend->get_dtype_from_data(data);
    device = backend->get_device_from_data(data);
    if (!backend->DataCls.is_none()) {
        assert(py::isinstance(py::cast(data), backend->DataCls));
    }
    verify_dtype();
}

SymmetricTensor::SymmetricTensor(TensorBackend::DataPtr data_in,
                                 TensorProduct::Ptr codomain_in,
                                 TensorProduct::Ptr domain_in,
                                 TensorBackend::Ptr backend_in,
                                 Symmetry::Ptr symmetry_in,
                                 LegLabels labels_in)
  : Tensor(std::move(codomain_in),
           std::move(domain_in),
           backend_in,
           std::move(symmetry_in),
           std::move(labels_in),
           backend_in->get_dtype_from_data(data_in),
           backend_in->get_device_from_data(data_in))
  , data(std::move(data_in))
{
    if (!backend->DataCls.is_none()) {
        assert(py::isinstance(py::cast(data), backend->DataCls));
    }
    verify_dtype();
}

void
SymmetricTensor::test_sanity() const
{
    Tensor::test_sanity();
    assert(dtype == backend->get_dtype_from_data(data));
    assert(device == backend->get_device_from_data(data));
    bool is_diagonal = false;
    if (Py_IsInitialized()) {
        try {
            is_diagonal = py::isinstance(
              as_py_object(), py::module_::import("cyten.tensors._tensors").attr("DiagonalTensor"));
        } catch (py::error_already_set& e) {
            e.restore();
            PyErr_Clear();
        }
    }
    backend->test_tensor_sanity(as_py_object(), is_diagonal);
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

py::object
SymmetricTensor::as_py_object()
{
    return py::cast(std::static_pointer_cast<SymmetricTensor>(shared_from_this()));
}

py::object
SymmetricTensor::as_py_object() const
{
    return const_cast<SymmetricTensor*>(this)->as_py_object();
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
SymmetricTensor::from_zero(py::object codomain,
                           py::object domain,
                           TensorBackend::Ptr backend,
                           py::object labels,
                           Dtype dtype,
                           std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    auto dt = _parse_default_dtype(dtype, symmetry);
    assert(dt.has_value());
    auto device_s = backend_tp->block_backend->as_device(device);
    auto data = backend_tp->zero_data(codomain_tp, domain_tp, *dt, device_s);
    return std::make_shared<SymmetricTensor>(
      data, codomain_tp, domain_tp, backend_tp, symmetry, _init_parse_labels(labels, codomain_tp, domain_tp));
}

SymmetricTensor::Ptr
SymmetricTensor::from_eye(py::object co_domain,
                          TensorBackend::Ptr backend,
                          py::object labels,
                          Dtype dtype,
                          std::optional<std::string> device)
{
    auto [co_domain_tp, unused_domain, backend_tp, symmetry] =
      _init_parse_args(co_domain, co_domain, std::move(backend));
    (void)unused_domain;
    auto dt = _parse_default_dtype(dtype, symmetry);
    assert(dt.has_value());
    auto labels_parsed =
      _init_parse_labels(labels, co_domain_tp, co_domain_tp, /*is_endomorphism=*/true);
    auto device_s = backend_tp->block_backend->as_device(device);
    auto data = backend_tp->eye_data(co_domain_tp, *dt, device_s);
    return std::make_shared<SymmetricTensor>(
      data, co_domain_tp, co_domain_tp, backend_tp, symmetry, std::move(labels_parsed));
}

SymmetricTensor::Ptr
SymmetricTensor::from_block_func(py::function func,
                                 py::object codomain,
                                 py::object domain,
                                 TensorBackend::Ptr backend,
                                 py::object labels,
                                 py::object func_kwargs,
                                 std::optional<std::string> shape_kw,
                                 std::optional<Dtype> dtype,
                                 std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    dtype = _parse_default_dtype(dtype, symmetry);

    py::dict kwargs = copy_dict(func_kwargs);
    py::object shape_kw_obj = shape_kw.has_value() ? py::cast(*shape_kw) : py::none();
    std::optional<Dtype> dtype_cap = dtype;
    std::optional<std::string> device_cap = device;
    auto bb = backend_tp->block_backend;

    // wrap func to consider func_kwargs, shape_kw, dtype, device
    py::cpp_function block_func(
      [func, kwargs, shape_kw_obj, dtype_cap, device_cap, bb](py::object shape,
                                                              py::object /*coupled*/) {
          // use same backend function as from_sector_block_func, so we include the coupled arg
          // but just ignore it.
          py::object block;
          if (shape_kw_obj.is_none()) {
              block = func(shape, **kwargs);
          } else {
              py::dict call_kwargs = py::dict(kwargs);
              call_kwargs[shape_kw_obj] = shape;
              block = func(**call_kwargs);
          }
          return bb->as_block(block, dtype_cap, device_cap);
      });

    auto data = backend_tp->from_sector_block_func(block_func, codomain_tp, domain_tp);
    auto res = std::make_shared<SymmetricTensor>(data,
                                                 codomain_tp,
                                                 domain_tp,
                                                 backend_tp,
                                                 symmetry,
                                                 _init_parse_labels(labels, codomain_tp, domain_tp));
    res->test_sanity(); // OPTIMIZE remove?
    return res;
}

SymmetricTensor::Ptr
SymmetricTensor::from_sector_block_func(py::function func,
                                        py::object codomain,
                                        py::object domain,
                                        TensorBackend::Ptr backend,
                                        py::object labels,
                                        py::object func_kwargs,
                                        std::optional<Dtype> dtype,
                                        std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    dtype = _parse_default_dtype(dtype, symmetry);

    // wrap func to consider func_kwargs and dtype
    py::dict kwargs = copy_dict(func_kwargs);
    std::optional<Dtype> dtype_cap = dtype;
    std::optional<std::string> device_cap = device;
    auto bb = backend_tp->block_backend;

    py::cpp_function block_func(
      [func, kwargs, dtype_cap, device_cap, bb](py::object shape, py::object coupled) {
          py::object block = func(shape, coupled, **kwargs);
          return bb->as_block(block, dtype_cap, device_cap);
      });

    auto data = backend_tp->from_sector_block_func(block_func, codomain_tp, domain_tp);
    auto res = std::make_shared<SymmetricTensor>(data,
                                                 codomain_tp,
                                                 domain_tp,
                                                 backend_tp,
                                                 symmetry,
                                                 _init_parse_labels(labels, codomain_tp, domain_tp));
    res->test_sanity();
    return res;
}

SymmetricTensor::Ptr
SymmetricTensor::from_dense_block(py::object block,
                                  py::object codomain,
                                  py::object domain,
                                  TensorBackend::Ptr backend,
                                  py::object labels,
                                  std::optional<Dtype> dtype,
                                  std::optional<std::string> device,
                                  float64 tol,
                                  bool understood_braiding)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    dtype = _parse_default_dtype(dtype, symmetry);
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    if (!symmetry->has_trivial_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of from_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    auto block_ptr = backend_tp->block_backend->as_block(block, dtype, device);
    assert(static_cast<int64>(backend_tp->block_backend->get_shape(block_ptr).size()) ==
           codomain_tp->num_factors + domain_tp->num_factors);
    block_ptr = backend_tp->block_backend->apply_basis_perm(
      block_ptr, legs_from_py(conventional_leg_order(codomain_tp, domain_tp)));
    auto data = backend_tp->from_dense_block(block_ptr, codomain_tp, domain_tp, tol);
    return std::make_shared<SymmetricTensor>(
      data, codomain_tp, domain_tp, backend_tp, symmetry, _init_parse_labels(labels, codomain_tp, domain_tp));
}

SymmetricTensor::Ptr
SymmetricTensor::from_dense_block_trivial_sector(py::object vector,
                                                 Space::Ptr space,
                                                 TensorBackend::Ptr backend,
                                                 std::optional<std::string> device,
                                                 LegLabel /*label*/)
{
    if (!backend) {
        backend = get_backend(py::cast(space->symmetry)).cast<TensorBackend::Ptr>();
    }
    auto vec = backend->block_backend->as_block(vector, std::nullopt, device);
    if (space->_basis_perm.has_value()) {
        auto i = space->sector_decomposition_where(space->symmetry->trivial_sector);
        assert(i.has_value());
        // Python: perm = rank_data(space.basis_perm[slice(*space.slices[i])])
        // then apply_leg_permutations — keep unfinished like the Python body.
        (void)vec;
        (void)i;
    }
    throw NotImplemented("SymmetricTensor.from_dense_block_trivial_sector");
}

SymmetricTensor::Ptr
SymmetricTensor::from_random_normal(py::object codomain,
                                    py::object domain,
                                    py::object mean,
                                    float64 sigma,
                                    TensorBackend::Ptr backend,
                                    py::object labels,
                                    std::optional<Dtype> dtype,
                                    std::optional<std::string> device)
{
    assert(sigma > 0.0);
    Symmetry::Ptr symmetry;
    TensorProduct::Ptr codomain_tp;
    TensorProduct::Ptr domain_tp;
    TensorBackend::Ptr backend_tp;

    if (!mean.is_none()) {
        if (codomain.is_none()) {
            codomain = mean.attr("codomain");
        } else {
            assert(mean.attr("codomain").equal(codomain));
        }
        if (domain.is_none()) {
            domain = mean.attr("domain");
        } else {
            assert(mean.attr("domain").equal(domain));
        }
        if (!backend) {
            backend = mean.attr("backend").cast<TensorBackend::Ptr>();
        } else {
            assert(mean.attr("backend").is(py::cast(backend)));
        }
        auto [c, d, b, s] = _init_parse_args(codomain, domain, backend);
        codomain_tp = std::move(c);
        domain_tp = std::move(d);
        backend_tp = std::move(b);
        symmetry = std::move(s);
        if (labels.is_none()) {
            labels = mean.attr("labels");
        } else {
            assert(mean.attr("labels").cast<LegLabels>() ==
                   _init_parse_labels(labels, codomain_tp, domain_tp));
        }
        if (!dtype.has_value()) {
            dtype = mean.attr("dtype").cast<Dtype>();
        } else {
            assert(mean.attr("dtype").cast<Dtype>() == *dtype);
        }
        if (!device.has_value()) {
            // Python writes ``device = mean.backend`` (likely meant mean.device); use mean.device.
            device = mean.attr("device").cast<std::string>();
        }
    } else {
        if (codomain.is_none()) {
            throw std::invalid_argument("Must specify the codomain if mean is not given.");
        }
        auto [c, d, b, s] = _init_parse_args(codomain, domain, std::move(backend));
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

    auto data =
      backend_tp->from_random_normal(codomain_tp, domain_tp, sigma, *dtype, *device);
    auto with_zero_mean = std::make_shared<SymmetricTensor>(
      data, codomain_tp, domain_tp, backend_tp, symmetry, _init_parse_labels(labels, codomain_tp, domain_tp));

    if (!mean.is_none()) {
        // mean + with_zero_mean
        auto one = backend_tp->block_backend->as_scalar(1.0);
        auto new_data = backend_tp->linear_combination(
          one, mean, one, with_zero_mean->as_py_object());
        return std::make_shared<SymmetricTensor>(
          new_data, codomain_tp, domain_tp, backend_tp, symmetry, with_zero_mean->labels());
    }
    return with_zero_mean;
}

SymmetricTensor::Ptr
SymmetricTensor::from_random_uniform(py::object codomain,
                                     py::object domain,
                                     TensorBackend::Ptr backend,
                                     py::object labels,
                                     Dtype dtype,
                                     std::optional<std::string> device)
{
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
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
                           py::cast(codomain_tp),
                           py::cast(domain_tp),
                           backend_tp,
                           labels,
                           func_kwargs,
                           std::nullopt,
                           dt,
                           device);
}

SymmetricTensor::Ptr
SymmetricTensor::from_sector_projection(py::object co_domain,
                                        Sector sector,
                                        TensorBackend::Ptr backend,
                                        py::object labels,
                                        std::optional<Dtype> dtype,
                                        std::optional<std::string> device)
{
    TensorProduct::Ptr co_domain_tp;
    if (py::isinstance<TensorProduct>(co_domain)) {
        co_domain_tp = co_domain.cast<TensorProduct::Ptr>();
    } else {
        std::vector<py::object> factors;
        for (auto item : co_domain) {
            factors.push_back(py::reinterpret_borrow<py::object>(item));
        }
        co_domain_tp = std::make_shared<TensorProduct>(std::move(factors));
    }
    assert(co_domain_tp->symmetry->is_valid_sector(sector));
    if (co_domain_tp->sector_multiplicity(sector) == 0) {
        warn("Sector does not appear. from_sector_projection yields zero");
    }
    if (!backend) {
        backend = get_backend(py::cast(co_domain_tp->symmetry)).cast<TensorBackend::Ptr>();
    }
    dtype = _parse_default_dtype(dtype, co_domain_tp->symmetry);
    std::optional<Dtype> dtype_cap = dtype;
    std::optional<std::string> device_cap = device;
    Sector sector_copy = sector;
    auto bb = backend->block_backend;

    py::cpp_function func([bb, dtype_cap, device_cap, sector_copy](py::object shape,
                                                                   py::object coupled) {
        Sector c = coupled.cast<Sector>();
        auto shape_vec = shape.cast<std::vector<int64>>();
        Dtype dt = dtype_cap.value_or(Dtype::Complex128);
        if (c == sector_copy) {
            std::vector<int64> half(shape_vec.begin(),
                                    shape_vec.begin() + static_cast<std::ptrdiff_t>(shape_vec.size() / 2));
            return bb->eye_block(half, dt, device_cap);
        }
        return bb->zeros(shape_vec, dt, device_cap);
    });

    auto data = backend->from_sector_block_func(func, co_domain_tp, co_domain_tp);
    auto res = std::make_shared<SymmetricTensor>(data,
                                                 co_domain_tp,
                                                 co_domain_tp,
                                                 backend,
                                                 co_domain_tp->symmetry,
                                                 _init_parse_labels(labels, co_domain_tp, co_domain_tp));
    res->test_sanity();
    return res;
}

SymmetricTensor::Ptr
SymmetricTensor::from_tree_pairs(
  std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
  py::object codomain,
  py::object domain,
  TensorBackend::Ptr backend,
  py::object labels,
  std::optional<Dtype> dtype,
  std::optional<std::string> device)
{
    if (trees.empty()) {
        if (!dtype.has_value()) {
            throw std::invalid_argument("dtype is required if trees is empty");
        }
        if (!device.has_value()) {
            throw std::invalid_argument("device is required if trees is empty");
        }
        return from_zero(codomain, domain, std::move(backend), labels, *dtype, device);
    }
    auto [codomain_tp, domain_tp, backend_tp, symmetry] =
      _init_parse_args(codomain, domain, std::move(backend));
    if (codomain_tp->has_pipes() || domain_tp->has_pipes()) {
        throw NotImplemented("from_tree_pairs does not support pipes (yet?)");
    }
    dtype = _parse_default_dtype(dtype, symmetry);
    std::string device_s;
    if (!device.has_value()) {
        auto some_block =
          backend_tp->block_backend->as_block(py::cast(trees.begin()->second));
        device_s = backend_tp->block_backend->get_device(some_block);
    } else {
        device_s = *device;
    }

    std::vector<std::uint8_t> X_are_dual;
    std::vector<std::uint8_t> Y_are_dual;
    X_are_dual.reserve(codomain_tp->factors.size());
    Y_are_dual.reserve(domain_tp->factors.size());
    for (auto const& leg : codomain_tp->factors) {
        X_are_dual.push_back(leg.attr("is_dual").cast<bool>() ? 1 : 0);
    }
    for (auto const& leg : domain_tp->factors) {
        Y_are_dual.push_back(leg.attr("is_dual").cast<bool>() ? 1 : 0);
    }

    for (auto& [key, block] : trees) {
        auto const& [X, Y] = key;
        assert(X.coupled == Y.coupled);
        assert(X.are_dual == X_are_dual);
        assert(Y.are_dual == Y_are_dual);
        block = backend_tp->block_backend->as_block(py::cast(block), dtype, device_s);
        assert(backend_tp->block_backend->get_device(block) == device_s);
    }
    if (!dtype.has_value()) {
        std::vector<Dtype> dts;
        dts.reserve(trees.size());
        for (auto const& [_, b] : trees) {
            (void)_;
            dts.push_back(backend_tp->block_backend->get_dtype(b));
        }
        dtype = dtype::common(dts);
    }
    auto data = backend_tp->from_tree_pairs(trees, codomain_tp, domain_tp, *dtype, device_s);
    return std::make_shared<SymmetricTensor>(
      data, codomain_tp, domain_tp, backend_tp, symmetry, _init_parse_labels(labels, codomain_tp, domain_tp));
}

Tensor::Ptr
SymmetricTensor::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    auto new_data = backend->to_dtype(as_py_object(), new_dtype);
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels());
}

py::object
SymmetricTensor::as_SymmetricTensor(bool guarantee_copy, std::optional<std::string> /*warning*/)
{
    if (guarantee_copy) {
        return py::cast(std::static_pointer_cast<SymmetricTensor>(copy()));
    }
    return as_py_object();
}

Tensor::Ptr
SymmetricTensor::copy(bool deep, std::optional<std::string> device_opt, std::optional<Dtype> dtype_opt)
{
    TensorBackend::DataPtr new_data;
    // Match Python: ``if dtype is not None and dtype != self.dtype:`` then
    // ``if device is not None or device != self.device:`` — the latter is always true when
    // ``device is None``, so the move branch is unreachable; effective behavior is as_dtype.
    if (dtype_opt.has_value() && *dtype_opt != dtype) {
        return as_dtype(*dtype_opt);
    }
    if (deep) {
        new_data = backend->copy_data(as_py_object(), device_opt);
    } else if (device_opt.has_value()) {
        new_data = backend->move_to_device(as_py_object(), *device_opt);
    } else {
        new_data = data;
    }
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels());
}

py::object
SymmetricTensor::diagonal(bool check_offdiagonal) const
{
    return py::module_::import("cyten.tensors._tensors")
      .attr("DiagonalTensor")
      .attr("from_tensor")(as_py_object(), py::arg("check_offdiagonal") = check_offdiagonal);
}

BlockBackend::Scalar
SymmetricTensor::_get_item(std::vector<int64> const& idx)
{
    return backend->get_element(as_py_object(), idx);
}

void
SymmetricTensor::move_to_device(std::string device_in)
{
    data = backend->move_to_device(as_py_object(), device_in);
    device = backend->block_backend->as_device(device_in);
}

Tensor::Ptr
SymmetricTensor::to_backend(TensorBackend::Ptr new_backend,
                            std::optional<Dtype> dtype_opt,
                            std::optional<std::string> device_opt)
{
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
            if (std::dynamic_pointer_cast<LegPipe>(leg)) {
                auto pipe = std::static_pointer_cast<LegPipe>(leg);
                py::list group;
                for (int64 i = flat_leg_counter; i < flat_leg_counter + pipe->num_legs; ++i) {
                    group.append(i);
                }
                combine.push_back(group);
                pipe_dualities.push_back(pipe->is_dual);
                flat_leg_counter += pipe->num_legs;
            } else {
                flat_leg_counter += 1;
            }
        }
        py::object flat = tensors_mod.attr("split_legs")(as_py_object());
        py::object res_flat =
          flat.attr("to_backend")(py::cast(new_backend), py::cast(dt), py::cast(device_s));
        py::object res = tensors_mod.attr("combine_legs")(
          res_flat, *py::tuple(py::cast(combine)), py::arg("pipe_dualities") = pipe_dualities);
        return std::make_shared<SymmetricTensor>(res.attr("data").cast<TensorBackend::DataPtr>(),
                                                 res.attr("codomain").cast<TensorProduct::Ptr>(),
                                                 res.attr("domain").cast<TensorProduct::Ptr>(),
                                                 res.attr("backend").cast<TensorBackend::Ptr>(),
                                                 res.attr("symmetry").cast<Symmetry::Ptr>(),
                                                 res.attr("labels").cast<LegLabels>());
    }

    TensorBackend::DataPtr new_data;
    if (std::dynamic_pointer_cast<NoSymmetryBackend>(new_backend)) {
        auto old_block = backend->to_dense_block(as_py_object());
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
            new_data = py::module_::import("cyten.tensors._tensors")
                         .attr("_convert_FT_to_abelian")(
                           as_py_object(), py::cast(new_backend), py::cast(dt), py::cast(device_s))
                         .cast<TensorBackend::DataPtr>();
        } else {
            throw std::runtime_error("Unexpected backend combination");
        }
    } else if (std::dynamic_pointer_cast<FusionTreeBackend>(new_backend)) {
        if (std::dynamic_pointer_cast<AbelianBackend>(backend)) {
            new_data = py::module_::import("cyten.tensors._tensors")
                         .attr("_convert_abelian_to_FT")(
                           as_py_object(), py::cast(new_backend), py::cast(dt), py::cast(device_s))
                         .cast<TensorBackend::DataPtr>();
        } else if (std::dynamic_pointer_cast<FusionTreeBackend>(backend)) {
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
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    if (!symmetry->has_trivial_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of to_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    auto block = backend->to_dense_block(as_py_object());
    block = backend->block_backend->apply_basis_perm(
      block, legs_from_py(conventional_leg_order(as_py_object())), /*inv=*/true);
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
    assert(num_legs == 1);
    auto block = backend->to_dense_block_trivial_sector(as_py_object());
    assert(num_codomain_legs() == 1); // TODO assuming this for now to construct the perm. should we keep that?
    auto leg = codomain->factors[0].cast<Space::Ptr>();
    if (leg->_basis_perm.has_value()) {
        auto i = leg->sector_decomposition_where(symmetry->trivial_sector);
        assert(i.has_value());
        assert(leg->slices.has_value());
        auto const& sl = (*leg->slices)[static_cast<std::size_t>(*i)];
        auto const& basis_perm = *leg->_basis_perm;
        std::vector<int64> segment(basis_perm.begin() + sl[0], basis_perm.begin() + sl[1]);
        // Python: perm = np.argsort(leg.basis_perm[slice(*leg.slices[i])])
        std::vector<int64> order(segment.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::sort(order, [&](int64 a, int64 b) { return segment[static_cast<std::size_t>(a)] <
                                                                segment[static_cast<std::size_t>(b)]; });
        block = backend->block_backend->apply_leg_permutations(
          block, { py::array_t<int64>(static_cast<py::ssize_t>(order.size()), order.data()) });
    }
    return block;
}

void
SymmetricTensor::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
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
    auto backend =
      get_backend(py::cast(symmetry), py::cast("numpy")).cast<TensorBackend::Ptr>();
    auto data = hdf5_loader.attr("load")(subpath + "data").cast<TensorBackend::DataPtr>();
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    auto dt = dtype::from_numpy_dtype(hdf5_loader.attr("load")(subpath + "dtype"));
    (void)hdf5_loader.attr("get_attr")(h5gr, "num_legs");
    auto shape = hdf5_loader.attr("get_attr")(h5gr, "shape").cast<std::vector<float64>>();
    auto labels = hdf5_loader.attr("get_attr")(h5gr, "labels").cast<LegLabels>();

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
