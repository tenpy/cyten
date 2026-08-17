#include <cyten/tensors/diagonal_tensor.h>

#include <cyten/backends/backend_factory.h>
#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/tools.h>
#include <cyten/warn.h>

#include <algorithm>
#include <cassert>
#include <format>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cyten {

namespace {

Space::Ptr
as_space_leg(Space::Ptr leg)
{
    if (!leg) {
        throw std::invalid_argument("Must specify the leg.");
    }
    if (std::dynamic_pointer_cast<LegPipe>(leg)) {
        throw std::invalid_argument("DiagonalTensor is not defined on LegPipes.");
    }
    return leg;
}

TensorProduct::Ptr
product_of_leg(Space::Ptr const& leg)
{
    auto sp = as_space_leg(leg);
    return std::make_shared<TensorProduct>(
      std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(sp) });
}

BlockBackend::LegCPtr
as_leg_cptr(Space::Ptr const& leg)
{
    return std::dynamic_pointer_cast<Leg const>(as_space_leg(leg));
}

} // namespace

std::vector<Dtype> DiagonalTensor::_forbidden_dtypes = {};

DiagonalTensor::DiagonalTensor(TensorBackend::DataPtr data_in,
                               Space::Ptr leg_in,
                               TensorBackend::Ptr backend_in,
                               Symmetry::Ptr symmetry_in,
                               LegLabels labels_in)
  : SymmetricTensor(std::move(data_in),
                    std::make_shared<TensorProduct>(
                      std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(leg_in) }),
                    std::make_shared<TensorProduct>(
                      std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(leg_in) }),
                    std::move(backend_in),
                    std::move(symmetry_in),
                    std::move(labels_in),
                    /*check_complex_dtype=*/false)
{
    if (std::dynamic_pointer_cast<LegPipe>(leg_in)) {
        throw std::invalid_argument("DiagonalTensor is not defined on LegPipes.");
    }
}

std::vector<Dtype> const&
DiagonalTensor::forbidden_dtypes() const
{
    return _forbidden_dtypes;
}

void
DiagonalTensor::test_sanity() const
{
    SymmetricTensor::test_sanity();
    assert(domain->operator==(*codomain));
    assert(domain->num_factors == 1);
}

void
DiagonalTensor::verify_dtype() const
{
    // --- hints from Python DiagonalTensor.verify_dtype ---
    // for diagonal tensors, we always allow real dtypes
    // ---
    // for diagonal tensors, we always allow real dtypes
}

std::string
DiagonalTensor::ascii_diagram_type_name() const
{
    return "Diag";
}

std::string
DiagonalTensor::class_name() const
{
    return "DiagonalTensor";
}

Space::Ptr
DiagonalTensor::leg() const
{
    return as_space(codomain->factors[0]);
}

DiagonalTensor::Ptr
DiagonalTensor::from_block_func(BlockFactoryFn func,
                                Space::Ptr leg,
                                TensorBackend::Ptr backend,
                                std::optional<LegLabels> labels,
                                std::optional<Dtype> dtype,
                                std::optional<std::string> device)
{
    // --- hints from Python DiagonalTensor.from_block_func ---
    // use same backend function as from_sector_block_func, so we include the coupled arg
    // but just ignore it.
    // ---
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)symmetry;
    (void)dtype;
    (void)device;

    SectorBlockFactoryFn block_func = [func](std::vector<int64> const& shape,
                                             Sector const& /*coupled*/) { return func(shape); };

    auto data = backend_tp->diagonal_from_sector_block_func(block_func, co_domain);
    auto res = std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
    res->test_sanity();
    return res;
}

DiagonalTensor::Ptr
DiagonalTensor::from_dense_block(BlockBackend::BlockPtr block,
                                 Space::Ptr leg,
                                 TensorBackend::Ptr backend,
                                 std::optional<LegLabels> labels,
                                 std::optional<Dtype> dtype,
                                 float64 tol,
                                 std::optional<std::string> device,
                                 bool understood_braiding)
{
    auto leg_sp = as_space_leg(std::move(leg));
    if (!leg_sp->symmetry->can_be_dropped()) {
        throw SymmetryError(
          std::format("Dense block representation is not supported for symmetry {}",
                      leg_sp->symmetry->repr()));
    }
    if (!leg_sp->symmetry->has_symmetric_braid() && !understood_braiding) {
        throw SymmetryError(
          "If the symmetry has non-trivial braids, dense block representations do not "
          "consistently reproduce the braiding statistics. Make sure you understand what "
          "that means (read the docstring of from_dense_block). Then you can disable "
          "this error by setting ``understood_braiding=True``.");
    }
    if (!backend) {
        backend = get_backend(leg_sp->symmetry);
    }
    auto block_ptr = backend->block_backend->as_block(py::cast(block), dtype, device);
    auto diag = backend->block_backend->get_diagonal(block_ptr, 1e-10);
    return from_diag_block(diag, leg_sp, backend, std::move(labels), dtype, device, tol);
}

DiagonalTensor::Ptr
DiagonalTensor::from_diag_block(BlockBackend::BlockPtr diag,
                                Space::Ptr leg,
                                TensorBackend::Ptr backend,
                                std::optional<LegLabels> labels,
                                std::optional<Dtype> dtype,
                                std::optional<std::string> device,
                                float64 tol)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)symmetry;
    auto diag_ptr = backend_tp->block_backend->as_block(py::cast(diag), dtype, device);
    diag_ptr = backend_tp->block_backend->apply_basis_perm(diag_ptr, { as_leg_cptr(leg_sp) });
    auto data = backend_tp->diagonal_from_block(diag_ptr, co_domain, tol);
    return std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
}

DiagonalTensor::Ptr
DiagonalTensor::from_eye(Space::Ptr leg,
                         TensorBackend::Ptr backend,
                         std::optional<LegLabels> labels,
                         Dtype dtype,
                         std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)co_domain;
    (void)unused_domain;
    (void)symmetry;
    auto bb = backend_tp->block_backend;
    BlockFactoryFn ones = [bb, dtype, device](std::vector<int64> const& shape) {
        return bb->ones_block(shape, dtype, device);
    };
    return from_block_func(
      std::move(ones), std::move(leg_sp), std::move(backend_tp), std::move(labels), dtype, device);
}

DiagonalTensor::Ptr
DiagonalTensor::from_random_normal(Space::Ptr leg,
                                   TensorCPtr mean,
                                   float64 sigma,
                                   TensorBackend::Ptr backend,
                                   std::optional<LegLabels> labels,
                                   Dtype dtype,
                                   std::optional<std::string> device)
{
    assert(dtype::is_complex(dtype));
    assert(sigma > 0.0);
    if (mean) {
        auto mean_diag = std::dynamic_pointer_cast<DiagonalTensor const>(mean);
        Space::Ptr mean_leg = mean_diag ? mean_diag->leg() : as_space(mean->codomain->factors[0]);
        if (!leg) {
            leg = mean_leg;
        } else {
            assert(*mean_leg == *leg);
        }
        if (!backend) {
            backend = mean->backend;
        } else {
            assert(mean->backend == backend);
        }
        if (!labels.has_value()) {
            labels = mean->labels();
        }
        if (!device.has_value()) {
            device = mean->device;
        }
    } else {
        if (!leg) {
            throw std::invalid_argument("Must specify the leg if mean is not given.");
        }
        auto tp = product_of_leg(leg);
        auto [co_domain, unused_domain, backend_tp, symmetry] =
          _init_parse_args(tp, tp, std::move(backend));
        (void)co_domain;
        (void)unused_domain;
        (void)symmetry;
        backend = std::move(backend_tp);
        if (!device.has_value()) {
            device = backend->block_backend->default_device;
        }
    }

    auto bb = backend->block_backend;
    BlockFactoryFn randn = [bb, dtype, sigma](std::vector<int64> const& shape) {
        return bb->random_normal(shape, dtype, sigma, std::nullopt);
    };
    auto with_zero_mean =
      from_block_func(std::move(randn), leg, backend, std::move(labels), dtype, device);

    if (mean) {
        auto one = backend->block_backend->as_scalar(1.0);
        auto new_data = backend->linear_combination(one, mean, one, with_zero_mean);
        return std::make_shared<DiagonalTensor>(new_data,
                                                with_zero_mean->leg(),
                                                backend,
                                                with_zero_mean->symmetry,
                                                with_zero_mean->labels());
    }
    return with_zero_mean;
}

DiagonalTensor::Ptr
DiagonalTensor::from_random_uniform(Space::Ptr leg,
                                    TensorBackend::Ptr backend,
                                    std::optional<LegLabels> labels,
                                    Dtype dtype,
                                    std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)co_domain;
    (void)unused_domain;
    (void)symmetry;
    auto bb = backend_tp->block_backend;
    BlockFactoryFn func = [bb, dtype, device](std::vector<int64> const& shape) {
        return bb->random_uniform(shape, dtype, device);
    };
    return from_block_func(
      std::move(func), std::move(leg_sp), std::move(backend_tp), std::move(labels), dtype, device);
}

DiagonalTensor::Ptr
DiagonalTensor::from_sector_block_func(SectorBlockFactoryFn func,
                                       Space::Ptr leg,
                                       TensorBackend::Ptr backend,
                                       std::optional<LegLabels> labels,
                                       std::optional<Dtype> dtype,
                                       std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, unused_symm] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)unused_symm;
    (void)dtype;
    (void)device;

    auto data = backend_tp->diagonal_from_sector_block_func(std::move(func), co_domain);
    auto res = std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
    res->test_sanity();
    return res;
}

DiagonalTensor::Ptr
DiagonalTensor::from_tensor(SymmetricTensorCPtr tens, std::optional<float64> tol)
{
    assert(tens->num_legs == 2);
    assert(tens->domain->operator==(*tens->codomain));
    auto data = tens->backend->diagonal_tensor_from_full_tensor(tens, tol);
    auto leg = as_space(tens->codomain->factors[0]);
    return std::make_shared<DiagonalTensor>(
      data, leg, tens->backend, tens->symmetry, tens->labels());
}

DiagonalTensor::Ptr
DiagonalTensor::from_zero(Space::Ptr leg,
                          TensorBackend::Ptr backend,
                          std::optional<LegLabels> labels,
                          Dtype dtype,
                          std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    (void)symmetry;
    auto device_s = backend_tp->block_backend->as_device(device);
    auto data = backend_tp->zero_diagonal_data(co_domain, dtype, device_s);
    return std::make_shared<DiagonalTensor>(
      data,
      leg_sp,
      backend_tp,
      co_domain->symmetry,
      _init_parse_labels(std::move(labels), co_domain, co_domain));
}

Tensor::Ptr
DiagonalTensor::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    auto new_data = backend->to_dtype(shared_from_this(), new_dtype);
    return std::make_shared<DiagonalTensor>(new_data, leg(), backend, symmetry, labels());
}

SymmetricTensorPtr
DiagonalTensor::as_SymmetricTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    auto new_data = backend->full_data_from_diagonal_tensor(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
    return std::make_shared<SymmetricTensor>(
      new_data, codomain, domain, backend, symmetry, labels());
}

DiagonalTensor::Ptr
DiagonalTensor::as_DiagonalTensor(bool guarantee_copy, std::optional<std::string> /*warning*/)
{
    if (guarantee_copy) {
        return std::static_pointer_cast<DiagonalTensor>(copy());
    }
    return std::static_pointer_cast<DiagonalTensor>(shared_from_this());
}

DiagonalTensor::Ptr
DiagonalTensor::_binary_operand(BlockBackend::Scalar other,
                                BlockBinaryFn func,
                                std::string const& /*operand*/,
                                bool right)
{
    auto other_block = std::const_pointer_cast<BlockBackend::Block>(other._block());
    TensorBackend::DataPtr new_data;
    if (right) {
        new_data = backend->diagonal_elementwise_unary(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          [func, other_block](BlockBackend::BlockPtr const& block) {
              return func(other_block, block);
          },
          /*maps_zero_to_zero=*/false);
    } else {
        new_data = backend->diagonal_elementwise_unary(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          [func, other_block](BlockBackend::BlockPtr const& block) {
              return func(block, other_block);
          },
          /*maps_zero_to_zero=*/false);
    }
    return std::make_shared<DiagonalTensor>(new_data, leg(), backend, symmetry, labels());
}

DiagonalTensor::Ptr
DiagonalTensor::_binary_operand(DiagonalTensorCPtr other,
                                BlockBinaryFn func,
                                std::string const& /*operand*/,
                                bool right)
{
    if (auto id = std::dynamic_pointer_cast<Identity const>(other)) {
        other = std::const_pointer_cast<Identity>(id)->as_DiagonalTensor();
    }
    if (!(*leg() == *other->leg())) {
        throw std::invalid_argument("Incompatible legs!");
    }
    auto same = get_same_backend(std::vector<TensorCPtr>{
      std::static_pointer_cast<Tensor const>(shared_from_this()), other });
    TensorBackend::DataPtr new_data;
    if (right) {
        new_data = same->diagonal_elementwise_binary(
          other,
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          func,
          /*partial_zero_is_zero=*/false);
    } else {
        new_data = same->diagonal_elementwise_binary(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
          other,
          func,
          /*partial_zero_is_zero=*/false);
    }
    auto out_labels = _get_matching_labels(labels(), other->labels());
    return std::make_shared<DiagonalTensor>(new_data, leg(), same, symmetry, out_labels);
}

Tensor::Ptr
DiagonalTensor::copy(bool deep,
                     std::optional<std::string> device_opt,
                     std::optional<Dtype> dtype_opt)
{
    TensorBackend::DataPtr new_data;
    // Match Python: dtype change effectively always takes as_dtype branch
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
    return std::make_shared<DiagonalTensor>(new_data, leg(), backend, symmetry, labels());
}

DiagonalTensorPtr
DiagonalTensor::diagonal(bool /*check_offdiagonal*/) const
{
    return std::const_pointer_cast<DiagonalTensor>(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
}

BlockBackend::BlockPtr
DiagonalTensor::diagonal_as_block(std::optional<Dtype> dtype_opt)
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    auto res = backend->diagonal_tensor_to_block(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
    res = backend->block_backend->apply_basis_perm(res, { codomain->factors[0] }, /*inv=*/true);
    if (dtype_opt.has_value()) {
        res = backend->block_backend->to_dtype(res, *dtype_opt);
    }
    return res;
}

py::array
DiagonalTensor::diagonal_as_numpy(py::object numpy_dtype)
{
    std::optional<Dtype> dt;
    if (!numpy_dtype.is_none()) {
        dt = dtype::from_numpy_dtype(numpy_dtype);
    }
    auto block = diagonal_as_block(dt);
    std::optional<py::object> np_dt =
      numpy_dtype.is_none() ? std::nullopt : std::optional<py::object>{ numpy_dtype };
    return py::reinterpret_borrow<py::array>(backend->block_backend->to_numpy(block, np_dt));
}

DiagonalTensor::Ptr
DiagonalTensor::elementwise_almost_equal(DiagonalTensorCPtr other, float64 rtol, float64 atol)
{
    // --- hints from Python DiagonalTensor.elementwise_almost_equal ---
    // no (Scalar + Block) operation defined, so requires explicit casting
    // ---
    if (auto id = std::dynamic_pointer_cast<Identity const>(other)) {
        other = std::const_pointer_cast<Identity>(id)->as_DiagonalTensor();
    }
    auto ones = from_eye(leg(), backend, labels(), dtype::to_real(dtype), device);
    auto self_m = std::static_pointer_cast<DiagonalTensor>(shared_from_this());
    BlockBinaryFn sub = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) - (*b);
    };
    BlockBinaryFn mul = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) * (*b);
    };
    BlockBinaryFn add = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) + (*b);
    };
    BlockBinaryFn le = [](BlockBackend::BlockPtr const& a, BlockBackend::BlockPtr const& b) {
        return (*a) <= (*b);
    };
    auto diff = self_m->_binary_operand(other, sub, "sub");
    auto left = diff->abs();
    auto bb = backend->block_backend;
    auto atol_ones = ones->_binary_operand(bb->as_scalar(atol), mul, "mul");
    auto rtol_abs = self_m->abs()->_binary_operand(bb->as_scalar(rtol), mul, "mul");
    auto right = atol_ones->_binary_operand(rtol_abs, add, "add");
    return left->_binary_operand(right, le, "le");
}

DiagonalTensor::Ptr
DiagonalTensor::_elementwise_binary(DiagonalTensorCPtr other,
                                    BlockBinaryFn func,
                                    bool partial_zero_is_zero)
{
    if (auto id = std::dynamic_pointer_cast<Identity const>(other)) {
        other = std::const_pointer_cast<Identity>(id)->as_DiagonalTensor();
    }
    if (!(*leg() == *other->leg())) {
        throw std::invalid_argument("Incompatible legs");
    }
    auto same = get_same_backend(std::vector<TensorCPtr>{
      std::static_pointer_cast<Tensor const>(shared_from_this()), other });
    auto data_out = same->diagonal_elementwise_binary(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      other,
      std::move(func),
      partial_zero_is_zero);
    auto labs = _get_matching_labels(labels(), other->labels());
    return std::make_shared<DiagonalTensor>(data_out, leg(), same, symmetry, labs);
}

DiagonalTensor::Ptr
DiagonalTensor::_elementwise_unary(BlockUnaryFn func, bool maps_zero_to_zero)
{
    auto data_out = backend->diagonal_elementwise_unary(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      std::move(func),
      maps_zero_to_zero);
    return std::make_shared<DiagonalTensor>(data_out, leg(), backend, symmetry, labels());
}

BlockBackend::Scalar
DiagonalTensor::_get_item(std::vector<int64> const& idx)
{
    assert(idx.size() == 2);
    int64 i1 = idx[0];
    int64 i2 = idx[1];
    if (i1 != i2) {
        return backend->block_backend->as_scalar(dtype::zero_scalar(dtype), dtype);
    }
    return backend->get_element_diagonal(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()), i1);
}

bool
DiagonalTensor::all() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("all is not defined for dtype {}", dtype::repr(dtype)));
    }
    return backend->diagonal_all(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
}

bool
DiagonalTensor::any() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("any is not defined for dtype {}", dtype::repr(dtype)));
    }
    return backend->diagonal_any(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
}

BlockBackend::Scalar
DiagonalTensor::max() const
{
    assert(dtype::is_real(dtype));
    auto bb = backend->block_backend;
    return backend->reduce_DiagonalTensor(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      [bb](BlockBackend::BlockPtr const& block) { return bb->max(block); },
      [](std::vector<BlockBackend::Scalar> const& xs) {
          return py::module_::import("builtins")
            .attr("max")(py::cast(xs))
            .cast<BlockBackend::Scalar>();
      });
}

BlockBackend::Scalar
DiagonalTensor::min() const
{
    assert(dtype::is_real(dtype));
    auto bb = backend->block_backend;
    return backend->reduce_DiagonalTensor(
      std::static_pointer_cast<DiagonalTensor const>(shared_from_this()),
      [bb](BlockBackend::BlockPtr const& block) { return bb->min(block); },
      [](std::vector<BlockBackend::Scalar> const& xs) {
          return py::module_::import("builtins")
            .attr("min")(py::cast(xs))
            .cast<BlockBackend::Scalar>();
      });
}

int64
DiagonalTensor::argmin(std::optional<Sector> s) const
{
    if (!dtype::is_real(dtype)) {
        throw std::invalid_argument(
          std::format("argmin is not defined for dtype {}", dtype::repr(dtype)));
    }
    auto public_diag = const_cast<DiagonalTensor*>(this)->diagonal_as_block();
    auto bb = backend->block_backend;
    if (!s) {
        auto idcs = bb->argmin(public_diag);
        assert(idcs.size() == 1);
        return idcs[0];
    }
    auto space = leg();
    auto sector_idx = space->sector_decomposition_where(*s);
    if (!sector_idx) {
        throw std::invalid_argument("Sector is not in the leg of the DiagonalTensor");
    }
    if (!space->slices) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    auto const& sl = (*space->slices)[static_cast<std::size_t>(*sector_idx)];
    int64 const start = sl[0];
    int64 const stop = sl[1];
    if (stop <= start) {
        throw std::invalid_argument("Cannot compute argmin of an empty sector");
    }
    auto es = std::dynamic_pointer_cast<ElementarySpace>(space);
    if (!es) {
        throw std::invalid_argument("DiagonalTensor.argmin requires an ElementarySpace leg");
    }
    auto const perm = es->basis_perm();
    std::vector<int64> pub_idcs;
    pub_idcs.reserve(static_cast<std::size_t>(stop - start));
    for (int64 i = start; i < stop; ++i) {
        pub_idcs.push_back(perm[static_cast<std::size_t>(i)]);
    }
    std::sort(pub_idcs.begin(), pub_idcs.end());
    auto gathered = (*public_diag)[{ pub_idcs }];
    auto local = bb->argmin(gathered);
    assert(local.size() == 1);
    return pub_idcs[static_cast<std::size_t>(local[0])];
}

DiagonalTensor::Ptr
DiagonalTensor::abs() const
{
    auto bb = backend->block_backend;
    return const_cast<DiagonalTensor*>(this)->_elementwise_unary(
      [bb](BlockBackend::BlockPtr const& block) { return bb->abs(block); },
      /*maps_zero_to_zero=*/true);
}

void
DiagonalTensor::move_to_device(std::string device_in)
{
    data = backend->move_to_device(shared_from_this(), device_in);
    device = backend->block_backend->as_device(device_in);
}

Tensor::Ptr
DiagonalTensor::to_backend(TensorBackend::Ptr new_backend,
                           std::optional<Dtype> dtype_opt,
                           std::optional<std::string> device_opt)
{
    // --- hints from Python DiagonalTensor.to_backend ---
    // In most cases, we can just go via a single block for the diagonal
    // exceptions:
    // - for non-abelian symmetries this is inefficient (needs to expand sectors into multiplets)
    // - for symmetries that can not be dropped, this is not possible
    // Both of these exceptions can only ocurr if both backends are FusionTreeBackend, which is
    // then also simple OPTIMIZE for abelian <-> fusion tree, this might be slightly inefficient.
    // I think the blocks should be the same already, so we could get away without first
    // concatenating all of them and them splitting them back up
    // ---
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }

    // In most cases, we can just go via a single block for the diagonal
    // exceptions:
    //   - for non-abelian symmetries this is inefficient (needs to expand sectors into multiplets)
    //   - for symmetries that can not be dropped, this is not possible
    // Both of these exceptions can only ocurr if both backends are FusionTreeBackend, which is
    // then also simple

    // OPTIMIZE
    //   for abelian <-> fusion tree, this might be slightly inefficient.
    //   I think the blocks should be the same already, so we could get away without first
    //   concatenating all of them and them splitting them back up

    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });
    TensorBackend::DataPtr new_data;
    if (std::dynamic_pointer_cast<FusionTreeBackend>(backend) &&
        std::dynamic_pointer_cast<FusionTreeBackend>(new_backend)) {
        new_data =
          backend->to_block_backend(data, new_backend->block_backend, dtype_opt, device_s);
    } else {
        auto old_diag = backend->diagonal_tensor_to_block(
          std::static_pointer_cast<DiagonalTensor const>(shared_from_this()));
        auto new_diag =
          new_backend->block_backend->as_block(py::cast(old_diag), dtype_opt, device_s);
        new_data = new_backend->diagonal_from_block(new_diag, codomain, 0.);
    }
    return std::make_shared<DiagonalTensor>(new_data, leg(), new_backend, symmetry, labels());
}

BlockBackend::BlockPtr
DiagonalTensor::to_dense_block(
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
    auto diag = diagonal_as_block(dtype_opt);
    auto res = backend->block_backend->block_from_diagonal(diag);
    if (leg_order.has_value()) {
        res = backend->block_backend->permute_axes(res, get_leg_idcs(*leg_order));
    }
    return res;
}

void
DiagonalTensor::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    /// Export DiagonalTensor to hdf5 such that it can be re-imported with from_hdf5
    SymmetricTensor::save_hdf5(hdf5_saver, h5gr, subpath);
}

DiagonalTensor::Ptr
DiagonalTensor::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    /// Import DiagonalTensor from hdf5
    auto domain = hdf5_loader.attr("load")(subpath + "domain").cast<TensorProduct::Ptr>();
    auto codomain = hdf5_loader.attr("load")(subpath + "codomain").cast<TensorProduct::Ptr>();
    auto symmetry = hdf5_loader.attr("load")(subpath + "symmetry").cast<Symmetry::Ptr>();
    auto backend = hdf5_loader.attr("load")(subpath + "backend").cast<TensorBackend::Ptr>();
    auto data = hdf5_loader.attr("load")(subpath + "data").cast<TensorBackend::DataPtr>();
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    auto dt = dtype::from_numpy_dtype(hdf5_loader.attr("load")(subpath + "dtype"));
    auto labels = hdf5_loader.attr("get_attr")(h5gr, "labels").cast<LegLabels>();
    int64 nlegs = codomain->num_factors + domain->num_factors;
    if (labels.empty() && nlegs > 0) {
        labels.assign(static_cast<std::size_t>(nlegs), std::nullopt);
    }

    auto obj = std::make_shared<DiagonalTensor>(
      data, as_space(codomain->factors[0]), backend, symmetry, std::move(labels));
    obj->dtype = dt;
    obj->device = std::move(device);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

void
Identity::unsupported_factory(char const* name)
{
    throw std::invalid_argument(std::format("{} is not supported for Identity", name));
}

Identity::Identity(Space::Ptr leg_in,
                   TensorBackend::Ptr backend_in,
                   Symmetry::Ptr symmetry_in,
                   LegLabels labels_in,
                   Dtype dtype_in,
                   std::string device_in)
  : DiagonalTensor(
      // Do not std::move(leg_in)/backend_in in this mem-initializer list: argument
      // evaluation order is unspecified, and eye_data still needs both.
      backend_in->eye_data(std::make_shared<TensorProduct>(
                             std::vector<Leg::Ptr>{ std::dynamic_pointer_cast<Leg>(leg_in) }),
                           dtype_in,
                           device_in),
      leg_in,
      backend_in,
      std::move(symmetry_in),
      std::move(labels_in))
{
}

void
Identity::test_sanity() const
{
    Tensor::test_sanity();
    verify_dtype();
}

std::string
Identity::class_name() const
{
    return "Identity";
}

Identity::Ptr
Identity::from_eye(Space::Ptr leg,
                   TensorBackend::Ptr backend,
                   std::optional<LegLabels> labels,
                   Dtype dtype,
                   std::optional<std::string> device)
{
    auto leg_sp = as_space_leg(std::move(leg));
    auto tp = product_of_leg(leg_sp);
    auto [co_domain, unused_domain, backend_tp, symmetry] =
      _init_parse_args(tp, tp, std::move(backend));
    (void)unused_domain;
    auto dt = _parse_default_dtype(dtype, symmetry);
    if (!dt.has_value()) {
        dt = Dtype::Float64;
    }
    std::string device_s =
      device.has_value() ? *device : backend_tp->block_backend->default_device;
    auto labs =
      _init_parse_labels(std::move(labels), co_domain, co_domain, /*is_endomorphism=*/true);
    return std::make_shared<Identity>(
      leg_sp, backend_tp, symmetry, std::move(labs), *dt, std::move(device_s));
}

Identity::Ptr
Identity::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto domain = hdf5_loader.attr("load")(subpath + "domain").cast<TensorProduct::Ptr>();
    (void)hdf5_loader.attr("load")(subpath + "codomain");
    auto symmetry = hdf5_loader.attr("load")(subpath + "symmetry").cast<Symmetry::Ptr>();
    auto backend = hdf5_loader.attr("load")(subpath + "backend").cast<TensorBackend::Ptr>();
    (void)hdf5_loader.attr("load")(subpath + "data");
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    auto dt = dtype::from_numpy_dtype(hdf5_loader.attr("load")(subpath + "dtype"));
    auto labels = hdf5_loader.attr("get_attr")(h5gr, "labels").cast<LegLabels>();
    if (labels.empty()) {
        labels.assign(2, std::nullopt);
    }

    auto obj = std::make_shared<Identity>(
      as_space(domain->factors[0]), backend, symmetry, std::move(labels), dt, std::move(device));
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

Tensor::Ptr
Identity::as_dtype(Dtype new_dtype)
{
    if (new_dtype == dtype) {
        return shared_from_this();
    }
    return std::make_shared<Identity>(leg(), backend, symmetry, labels(), new_dtype, device);
}

SymmetricTensorPtr
Identity::as_SymmetricTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    return SymmetricTensor::from_eye(codomain, backend, labels(), dtype, device);
}

DiagonalTensor::Ptr
Identity::as_DiagonalTensor(bool /*guarantee_copy*/, std::optional<std::string> warning)
{
    if (warning.has_value()) {
        warn(*warning);
    }
    return DiagonalTensor::from_eye(leg(), backend, labels(), dtype, device);
}

DiagonalTensor::Ptr
Identity::_binary_operand(BlockBackend::Scalar other,
                          BlockBinaryFn func,
                          std::string const& operand,
                          bool right)
{
    return as_DiagonalTensor()->_binary_operand(other, std::move(func), operand, right);
}

DiagonalTensor::Ptr
Identity::_binary_operand(DiagonalTensorCPtr other,
                          BlockBinaryFn func,
                          std::string const& operand,
                          bool right)
{
    return as_DiagonalTensor()->_binary_operand(std::move(other), std::move(func), operand, right);
}

Tensor::Ptr
Identity::copy(bool /*deep*/,
               std::optional<std::string> /*device_opt*/,
               std::optional<Dtype> dtype_opt)
{
    if (dtype_opt.has_value() && *dtype_opt != dtype) {
        // Python: ``return self.as_dtype(dtype, device=device)`` — as_dtype ignores device.
        return as_dtype(*dtype_opt);
    }
    return shared_from_this();
}

DiagonalTensorPtr
Identity::diagonal(bool /*check_offdiagonal*/) const
{
    return const_cast<Identity*>(this)->as_DiagonalTensor();
}

BlockBackend::BlockPtr
Identity::diagonal_as_block(std::optional<Dtype> dtype_opt)
{
    if (!symmetry->can_be_dropped()) {
        throw SymmetryError(std::format(
          "Dense block representation is not supported for symmetry {}", symmetry->repr()));
    }
    return backend->block_backend->ones_block(
      { static_cast<int64>(leg()->dim) }, dtype_opt.value_or(dtype), device);
}

py::array
Identity::diagonal_as_numpy(py::object numpy_dtype)
{
    if (numpy_dtype.is_none()) {
        numpy_dtype = dtype::to_numpy_dtype(dtype);
    }
    return py::module_::import("numpy").attr("ones")(static_cast<int64>(leg()->dim), numpy_dtype);
}

DiagonalTensor::Ptr
Identity::elementwise_almost_equal(DiagonalTensorCPtr other, float64 rtol, float64 atol)
{
    return as_DiagonalTensor()->elementwise_almost_equal(std::move(other), rtol, atol);
}

DiagonalTensor::Ptr
Identity::_elementwise_unary(BlockUnaryFn func, bool maps_zero_to_zero)
{
    return as_DiagonalTensor()->_elementwise_unary(std::move(func), maps_zero_to_zero);
}

DiagonalTensor::Ptr
Identity::_elementwise_binary(DiagonalTensorCPtr other,
                              BlockBinaryFn func,
                              bool partial_zero_is_zero)
{
    return as_DiagonalTensor()->_elementwise_binary(
      std::move(other), std::move(func), partial_zero_is_zero);
}

BlockBackend::Scalar
Identity::_get_item(std::vector<int64> const& idx)
{
    assert(idx.size() == 2);
    if (idx[0] != idx[1]) {
        return backend->block_backend->as_scalar(dtype::zero_scalar(dtype), dtype);
    }
    return backend->block_backend->as_scalar(dtype::one_scalar(dtype), dtype);
}

bool
Identity::all() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("all is not defined for dtype {}", dtype::repr(dtype)));
    }
    return true;
}

bool
Identity::any() const
{
    if (dtype != Dtype::Bool) {
        throw std::invalid_argument(
          std::format("any is not defined for dtype {}", dtype::repr(dtype)));
    }
    return leg()->dim > 0;
}

BlockBackend::Scalar
Identity::max() const
{
    assert(dtype::is_real(dtype));
    return backend->block_backend->as_scalar(dtype::one_scalar(dtype), dtype);
}

BlockBackend::Scalar
Identity::min() const
{
    assert(dtype::is_real(dtype));
    return backend->block_backend->as_scalar(dtype::one_scalar(dtype), dtype);
}

int64
Identity::argmin(std::optional<Sector> /*s*/) const
{
    throw std::invalid_argument("argmin is not supported for Identity");
}

DiagonalTensor::Ptr
Identity::abs() const
{
    return std::const_pointer_cast<Identity>(
      std::static_pointer_cast<Identity const>(shared_from_this()));
}

void
Identity::move_to_device(std::string device_in)
{
    device = backend->block_backend->as_device(device_in);
}

Tensor::Ptr
Identity::to_backend(TensorBackend::Ptr new_backend,
                     std::optional<Dtype> dtype_opt,
                     std::optional<std::string> device_opt)
{
    if (!new_backend->supports_symmetry(symmetry)) {
        throw SymmetryError("backend does not support symmetry");
    }
    Dtype dt = dtype_opt.value_or(dtype);
    auto device_s = new_backend->block_backend->as_device(
      device_opt.has_value() ? device_opt : std::optional<std::string>{ device });
    return std::make_shared<Identity>(leg(), new_backend, symmetry, labels(), dt, device_s);
}

BlockBackend::BlockPtr
Identity::to_dense_block(std::optional<std::vector<std::variant<int64, std::string>>> leg_order,
                         std::optional<Dtype> dtype_opt,
                         bool understood_braiding)
{
    return as_DiagonalTensor()->to_dense_block(leg_order, dtype_opt, understood_braiding);
}

} // namespace cyten
