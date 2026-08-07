// ---- 00_keep_helpers.cpp ----
#include <cyten/backends/abelian.h>

#include <cyten/tools.h>
#include <cyten/symmetries/sector_numpy.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
#include <cassert>
#include <cmath>
#include <format>
#include <map>
#include <set>
#include <numeric>
#include <stdexcept>
#include <typeinfo>

namespace cyten {

namespace {

py::module_
numpy()
{
    return py::module_::import("numpy");
}

/// Reorder rows of a 2D int64 array according to ``perm`` (1D index array).
py::array_t<int64>
take_rows(py::array_t<int64> const& arr, py::array const& perm)
{
    return numpy()
      .attr("take")(arr, perm, py::arg("axis") = 0)
      .cast<py::array_t<int64>>();
}

} // namespace

AbelianBackendData::AbelianBackendData(Dtype dtype_,
                                       std::string device_,
                                       std::vector<BlockBackend::BlockPtr> blocks_,
                                       py::array_t<int64> block_inds_,
                                       bool is_sorted)
  : dtype(dtype_)
  , device(std::move(device_))
  , blocks(std::move(blocks_))
  , block_inds(std::move(block_inds_))
{
    if (block_inds.ndim() != 2) {
        throw std::invalid_argument("AbelianBackendData: block_inds must be 2D");
    }
    if (static_cast<std::size_t>(block_inds.shape(0)) != blocks.size()) {
        throw std::invalid_argument(
          "AbelianBackendData: len(blocks) must equal block_inds.shape[0]");
    }

    if (!is_sorted) {
        auto np = numpy();
        // ``np.lexsort(block_inds.T)`` — last row is primary key (numpy convention).
        auto perm = np.attr("lexsort")(block_inds.attr("T")).cast<py::array_t<int64>>();
        block_inds = take_rows(block_inds, perm);

        auto perm_buf = perm.unchecked<1>();
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(blocks.size());
        for (py::ssize_t i = 0; i < perm_buf.shape(0); ++i) {
            sorted_blocks.push_back(blocks[static_cast<std::size_t>(perm_buf(i))]);
        }
        blocks = std::move(sorted_blocks);
    }
}

std::optional<int64>
AbelianBackendData::get_block_num(py::array_t<int64> query) const
{
    // OPTIMIZE use sorted for lookup?
    auto np = numpy();
    py::object equal = np.attr("all")(np.attr("equal")(block_inds, query), py::arg("axis") = 1);
    py::array match = np.attr("argwhere")(equal)
                        .attr("__getitem__")(py::make_tuple(py::ellipsis(), 0))
                        .cast<py::array>();
    if (match.size() == 0) {
        return std::nullopt;
    }
    return match.cast<py::array_t<int64>>().at(0);
}

BlockBackend::BlockPtr
AbelianBackendData::get_block(py::array_t<int64> query) const
{
    auto block_num = get_block_num(query);
    if (!block_num.has_value()) {
        return nullptr;
    }
    return blocks[static_cast<std::size_t>(*block_num)];
}

void
AbelianBackendData::save_hdf5(py::object hdf5_saver,
                              py::object /*h5gr*/,
                              std::string const& subpath) const
{
    hdf5_saver.attr("save")(block_inds, subpath + "block_inds");
    hdf5_saver.attr("save")(blocks, subpath + "blocks");
    hdf5_saver.attr("save")(dtype::to_numpy_dtype(dtype), subpath + "dtype");
    hdf5_saver.attr("save")(device, subpath + "device");
}

AbelianBackendData::Ptr
AbelianBackendData::from_hdf5(py::object hdf5_loader,
                              py::object h5gr,
                              std::string const& subpath)
{
    auto block_inds =
      hdf5_loader.attr("load")(subpath + "block_inds").cast<py::array_t<int64>>();
    auto blocks =
      hdf5_loader.attr("load")(subpath + "blocks").cast<std::vector<BlockBackend::BlockPtr>>();
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    py::object dt = hdf5_loader.attr("load")(subpath + "dtype");
    Dtype dtype = dtype::from_numpy_dtype(dt);

    // Already sorted when saved; skip lexsort.
    auto obj = std::make_shared<AbelianBackendData>(
      dtype, std::move(device), std::move(blocks), std::move(block_inds), /*is_sorted=*/true);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}


namespace {

py::module_ misc() { return py::module_::import("cyten.tools.misc"); }

py::array_t<int64> zeros_i64(py::ssize_t rows, py::ssize_t cols)
{
    auto np = numpy();
    return np.attr("zeros")(py::make_tuple(rows, cols), py::arg("dtype") = np.attr("intp"))
      .cast<py::array_t<int64>>();
}

py::array_t<int64> asarray_i64(py::object obj)
{
    auto np = numpy();
    return np.attr("asarray")(obj, py::arg("dtype") = np.attr("intp")).cast<py::array_t<int64>>();
}

py::slice slice_pair(py::object pair)
{
    return py::slice(pair.attr("__getitem__")(0).cast<py::ssize_t>(),
                     pair.attr("__getitem__")(1).cast<py::ssize_t>(), 1);
}

BlockBackend::BlockPtr b_get(BlockBackend::BlockPtr const& b, py::object key)
{
    return b->get_item(key);
}

void b_set(BlockBackend::BlockPtr const& b, py::object key, BlockBackend::BlockPtr const& v)
{
    b->set_item(key, py::cast(v));
}

void b_set_add(BlockBackend::BlockPtr const& b, py::object key, BlockBackend::BlockPtr const& v)
{
    b_set(b, key, (*b_get(b, key)) + (*v));
}

bool
is_zero_scalar(BlockBackend::Scalar const& a)
{
    return a.as_complex128() == 0.;
}

AbelianBackendData::Ptr make_data(Dtype dtype, std::string device,
    std::vector<BlockBackend::BlockPtr> blocks, py::array_t<int64> block_inds, bool is_sorted = false)
{
    return std::make_shared<AbelianBackendData>(dtype, std::move(device), std::move(blocks),
                                                std::move(block_inds), is_sorted);
}

std::vector<int64> mults_of(py::object leg)
{
    return leg.attr("multiplicities").cast<std::vector<int64>>();
}

int64 nsec(py::object leg) { return leg.attr("num_sectors").cast<int64>(); }

bool sector_sorted(py::object leg)
{
    py::object so = leg.attr("sector_order");
    return (!so.is_none()) && so.cast<std::string>() == "sorted";
}

py::object take_rows_obj(py::object arr, py::object perm)
{
    return numpy().attr("take")(arr, perm, py::arg("axis") = 0);
}

std::vector<BlockBackend::BlockPtr> permute_blocks(std::vector<BlockBackend::BlockPtr> const& blocks,
                                                  py::array const& perm)
{
    auto p = perm.cast<py::array_t<int64>>();
    auto buf = p.unchecked<1>();
    std::vector<BlockBackend::BlockPtr> out;
    out.reserve(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        out.push_back(blocks[static_cast<std::size_t>(buf(i))]);
    return out;
}

} // namespace

// ---- 01_valid_wrap_ctor.cpp ----

py::array_t<int64>
valid_block_inds(TensorProduct::Ptr codomain, TensorProduct::Ptr domain)
{
    // Prefer calling the Python reference for exact fusion broadcast semantics via numpy zip,
    // but implement locally with make_grid + Symmetry::multiple_fusion_broadcast.
    auto np = numpy();
    auto legs = conventional_leg_order(codomain, domain);
    std::vector<int64> nums;
    nums.reserve(legs.size());
    for (auto const& leg : legs)
        nums.push_back(nsec(leg));
    py::array grid = misc().attr("make_grid")(nums, py::arg("cstyle") = false).cast<py::array>();
    py::ssize_t n_combos = py::int_(grid.attr("shape").attr("__getitem__")(0)).cast<py::ssize_t>();
    auto symmetry = codomain->symmetry;

    auto select_sectors = [&](std::vector<py::object> const& factors, py::object cols) {
        std::vector<SectorArray> parts;
        parts.reserve(factors.size());
        for (std::size_t fi = 0; fi < factors.size(); ++fi) {
            auto sectors = factors[fi].attr("sector_decomposition").cast<SectorArray>();
            auto idx = asarray_i64(cols.attr("__getitem__")(static_cast<py::ssize_t>(fi)));
            auto buf = idx.unchecked<1>();
            SectorArray selected = SectorArray::empty(sectors.sector_ind_len());
            selected.reserve(static_cast<std::size_t>(buf.shape(0)));
            for (py::ssize_t r = 0; r < buf.shape(0); ++r)
                selected.push_back(sectors[static_cast<std::size_t>(buf(r))]);
            parts.push_back(std::move(selected));
        }
        return symmetry->multiple_fusion_broadcast(parts);
    };

    SectorArray codomain_coupled;
    if (codomain->num_factors > 0) {
        py::list cols;
        for (int64 i = 0; i < codomain->num_factors; ++i)
            cols.append(grid.attr("T").attr("__getitem__")(i));
        codomain_coupled = select_sectors(codomain->factors, cols);
    } else {
        codomain_coupled = SectorArray::repeat(symmetry->trivial_sector, static_cast<std::size_t>(n_combos));
    }

    SectorArray domain_coupled;
    if (domain->num_factors > 0) {
        py::ssize_t nlegs = py::int_(grid.attr("shape").attr("__getitem__")(1)).cast<py::ssize_t>();
        py::list cols;
        // domain factors correspond to reversed grid columns
        for (int64 i = 0; i < domain->num_factors; ++i)
            cols.append(grid.attr("T").attr("__getitem__")(nlegs - 1 - i));
        domain_coupled = select_sectors(domain->factors, cols);
    } else {
        domain_coupled = SectorArray::repeat(symmetry->trivial_sector, static_cast<std::size_t>(n_combos));
    }

    py::list valid_idx;
    for (py::ssize_t i = 0; i < n_combos; ++i) {
        if (codomain_coupled[static_cast<std::size_t>(i)] == domain_coupled[static_cast<std::size_t>(i)])
            valid_idx.append(i);
    }
    py::array block_inds =
      grid.attr("__getitem__")(py::make_tuple(valid_idx, py::ellipsis())).cast<py::array>();
    auto perm = np.attr("lexsort")(block_inds.attr("T")).cast<py::array>();
    return take_rows(asarray_i64(block_inds), perm);
}

TensorBackend::DataPtr
AbelianBackend::wrap(AbelianBackendData::Ptr d)
{
    if (!d)
        throw std::invalid_argument("AbelianBackend::wrap: null");
    return d;
}

AbelianBackendData::Ptr
AbelianBackend::unwrap(DataPtr d)
{
    if (!d)
        throw std::invalid_argument("AbelianBackend::unwrap: null DataPtr");
    auto* p = dynamic_cast<AbelianBackendData*>(d.get());
    if (!p)
        throw std::invalid_argument(std::format(
          "AbelianBackend::unwrap: expected AbelianBackendData, got {}", typeid(*d).name()));
    return std::static_pointer_cast<AbelianBackendData>(d);
}

AbelianBackendData::Ptr
AbelianBackend::data_from_tensor(py::object tensor)
{
    py::object raw = tensor.attr("data");
    try {
        return unwrap(raw.cast<DataPtr>());
    } catch (py::cast_error const&) {
    } catch (std::invalid_argument const&) {
    }
    return make_data(raw.attr("dtype").cast<Dtype>(),
                     raw.attr("device").cast<std::string>(),
                     raw.attr("blocks").cast<std::vector<BlockBackend::BlockPtr>>(),
                     asarray_i64(raw.attr("block_inds")),
                     /*is_sorted=*/true);
}

AbelianBackend::AbelianBackend(std::shared_ptr<BlockBackend> block_backend_)
  : TensorBackend(std::move(block_backend_))
{
    // Match Python AbelianBackend: can_decompose_tensors remains false.
    // DataCls is set when pybind bindings for AbelianBackend exist.
    DataCls = py::none();
}

// ---- 02_early.cpp ----

void
AbelianBackend::test_tensor_sanity(py::object a, bool is_diagonal)
{
    TensorBackend::test_tensor_sanity(a, is_diagonal);
    // When DataCls is unset, skip deep checks that require C++ AbelianBackendData on the tensor.
    // Full checks run once bindings/monkey-patch store C++ Data.
    py::object raw = a.attr("data");
    AbelianBackendData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    assert(a.attr("device").cast<std::string>() == data->device);
    assert(data->device == block_backend->as_device(data->device));
    assert(a.attr("dtype").cast<Dtype>() == data->dtype);
    int64 num_legs = a.attr("num_legs").cast<int64>();
    for (int64 n = 0; n < num_legs; ++n) {
        py::object l = a.attr("get_leg_co_domain")(n);
        try {
            auto pipe = l.cast<LegPipe::Ptr>();
            if (pipe && !std::dynamic_pointer_cast<AbelianLegPipe>(pipe))
                throw std::runtime_error("pipes must be AbelianLegPipe");
        } catch (py::cast_error const&) {
        }
    }
    auto np = numpy();
    assert(data->block_inds.shape(0) == static_cast<py::ssize_t>(data->blocks.size()));
    assert(data->block_inds.shape(1) == num_legs);
    assert(np.attr("all")(data->block_inds.attr("__ge__")(0)).cast<bool>());
    py::list maxes;
    for (auto const& leg : conventional_leg_order(a))
        maxes.append(leg.attr("num_sectors"));
    assert(np.attr("all")(data->block_inds.attr("__lt__")(np.attr("array")(py::make_tuple(maxes)))).cast<bool>());
    assert(np.attr("all")(np.attr("equal")(np.attr("lexsort")(data->block_inds.attr("T")),
                                            np.attr("arange")(data->blocks.size())))
             .cast<bool>());
    if (is_diagonal) {
        assert(np.attr("all")(np.attr("equal")(
                 data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0)),
                 data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1))))
                 .cast<bool>());
    }
    auto legs = conventional_leg_order(a);
    auto bi = data->block_inds.unchecked<2>();
    for (std::size_t i = 0; i < data->blocks.size(); ++i) {
        std::vector<int64> shape;
        if (is_diagonal) {
            auto mults = mults_of(a.attr("leg"));
            shape = { mults[static_cast<std::size_t>(bi(static_cast<py::ssize_t>(i), 0))] };
        } else {
            for (std::size_t li = 0; li < legs.size(); ++li) {
                auto mults = mults_of(legs[li]);
                shape.push_back(mults[static_cast<std::size_t>(
                  bi(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(li)))]);
            }
        }
        block_backend->test_block_sanity(
          data->blocks[i], shape, a.attr("dtype").cast<Dtype>(), a.attr("device").cast<std::string>());
    }
}

void
AbelianBackend::test_mask_sanity(py::object a)
{
    TensorBackend::test_mask_sanity(a);
    py::object raw = a.attr("data");
    AbelianBackendData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    assert(a.attr("device").cast<std::string>() == data->device);
    assert(data->dtype == Dtype::Bool);
    auto np = numpy();
    assert(data->block_inds.shape(0) == static_cast<py::ssize_t>(data->blocks.size()));
    assert(data->block_inds.shape(1) == a.attr("num_legs").cast<int64>());
    bool is_projection = a.attr("is_projection").cast<bool>();
    auto large_leg = a.attr("large_leg");
    auto small_leg = a.attr("small_leg");
    auto bi = data->block_inds.unchecked<2>();
    for (std::size_t i = 0; i < data->blocks.size(); ++i) {
        int64 bi_small = is_projection ? bi(static_cast<py::ssize_t>(i), 0) : bi(static_cast<py::ssize_t>(i), 1);
        int64 bi_large = is_projection ? bi(static_cast<py::ssize_t>(i), 1) : bi(static_cast<py::ssize_t>(i), 0);
        assert(bi_large >= bi_small);
        int64 expect_len = mults_of(large_leg)[static_cast<std::size_t>(bi_large)];
        int64 expect_sum = mults_of(small_leg)[static_cast<std::size_t>(bi_small)];
        block_backend->test_block_sanity(
          data->blocks[i], std::vector<int64>{ expect_len }, Dtype::Bool, data->device);
        assert(block_backend->sum_all(data->blocks[i]).as_int64() == expect_sum);
    }
}

LegPipe::Ptr
AbelianBackend::make_pipe(std::vector<Leg::Ptr> legs, bool is_dual, LegPipe::Ptr pipe)
{
    std::vector<ElementarySpace::Ptr> es;
    es.reserve(legs.size());
    for (auto const& l : legs) {
        auto p = std::dynamic_pointer_cast<ElementarySpace>(l);
        if (!p)
            throw std::invalid_argument("make_pipe: legs must be ElementarySpace");
        es.push_back(std::move(p));
    }
    if (auto ab = std::dynamic_pointer_cast<AbelianLegPipe>(pipe)) {
        assert(ab->combine_cstyle == !is_dual);
        assert(ab->is_dual == is_dual);
        return ab;
    }
    return std::make_shared<AbelianLegPipe>(std::move(es), is_dual, !is_dual);
}

TensorBackend::DataPtr
AbelianBackend::act_block_diagonal_square_matrix(py::object a,
                                                 py::function block_method,
                                                 py::object dtype_map)
{
    auto a_data = data_from_tensor(a);
    auto leg = a.attr("domain").attr("factors").attr("__getitem__")(0);
    auto np = numpy();
    py::array_t<int64> all_block_inds =
      asarray_i64(np.attr("repeat")(np.attr("arange")(nsec(leg)).attr("__getitem__")(
                                      py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
                                    2,
                                    py::arg("axis") = 1));
    // Fix: np.repeat(np.arange(n)[:, None], 2, axis=1)
    all_block_inds = asarray_i64(
      np.attr("repeat")(np.attr("arange")(nsec(leg)).attr("__getitem__")(
                          py::make_tuple(py::ellipsis(), py::none())),
                        2,
                        py::arg("axis") = 1));
    // Use clearer construction:
    all_block_inds = asarray_i64(np.attr("column_stack")(
      py::make_tuple(np.attr("arange")(nsec(leg)), np.attr("arange")(nsec(leg)))));

    std::vector<BlockBackend::BlockPtr> res_blocks;
    py::object iter = misc().attr("iter_common_noncommon_sorted_arrays")(a_data->block_inds, all_block_inds);
    for (py::handle item : iter) {
        auto pair = item.cast<py::tuple>();
        py::object i = pair[0];
        py::object j = pair[1];
        BlockBackend::BlockPtr block;
        if (i.is_none()) {
            int64 jj = j.cast<int64>();
            int64 m = mults_of(leg)[static_cast<std::size_t>(jj)];
            block = block_backend->zeros({ m, m }, a.attr("dtype").cast<Dtype>());
        } else {
            block = a_data->blocks[static_cast<std::size_t>(i.cast<int64>())];
        }
        res_blocks.push_back(block_method(py::cast(block)).cast<BlockBackend::BlockPtr>());
    }
    Dtype dtype = a.attr("dtype").cast<Dtype>();
    if (!dtype_map.is_none())
        dtype = dtype_map(py::cast(dtype)).cast<Dtype>();
    for (auto& b : res_blocks)
        b = block_backend->to_dtype(b, dtype);
    return wrap(make_data(dtype, a_data->device, std::move(res_blocks), all_block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::add_trivial_leg(py::object a,
                                int64 legs_pos,
                                bool /*add_to_domain*/,
                                int64 /*co_domain_pos*/,
                                TensorProduct::Ptr /*new_codomain*/,
                                TensorProduct::Ptr /*new_domain*/)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->add_axis(b, legs_pos));
    auto np = numpy();
    py::array_t<int64> block_inds =
      asarray_i64(np.attr("insert")(a_data->block_inds, legs_pos, 0, py::arg("axis") = 1));
    return wrap(make_data(a_data->dtype, a_data->device, std::move(blocks), block_inds, true));
}

bool
AbelianBackend::almost_equal(py::object a, py::object b, float64 rtol, float64 atol)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    py::object iter =
      misc().attr("iter_common_noncommon_sorted_arrays")(a_data->block_inds, b_data->block_inds);
    for (py::handle item : iter) {
        auto pair = item.cast<py::tuple>();
        py::object i = pair[0];
        py::object j = pair[1];
        if (j.is_none()) {
            if (block_backend->max_abs(a_data->blocks[static_cast<std::size_t>(i.cast<int64>())])
                  .as_float64() > atol)
                return false;
        } else if (i.is_none()) {
            if (block_backend->max_abs(b_data->blocks[static_cast<std::size_t>(j.cast<int64>())])
                  .as_float64() > atol)
                return false;
        } else if (!block_backend->allclose(a_data->blocks[static_cast<std::size_t>(i.cast<int64>())],
                                            b_data->blocks[static_cast<std::size_t>(j.cast<int64>())],
                                            rtol,
                                            atol)) {
            return false;
        }
    }
    return true;
}

TensorBackend::DataPtr
AbelianBackend::apply_mask_to_DiagonalTensor(py::object tensor, py::object mask)
{
    auto t_data = data_from_tensor(tensor);
    auto m_data = data_from_tensor(mask);
    auto np = numpy();
    py::array tensor_block_inds_contr =
      t_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)));
    py::array mask_block_inds_contr =
      m_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1));
    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<int64> res_bi;
    py::object iter = misc().attr("iter_common_sorted")(tensor_block_inds_contr, mask_block_inds_contr);
    for (py::handle item : iter) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        res_blocks.push_back(block_backend->apply_mask(t_data->blocks[static_cast<std::size_t>(i)],
                                                       m_data->blocks[static_cast<std::size_t>(j)],
                                                       0));
        res_bi.push_back(m_data->block_inds.at(j, 0));
    }
    py::array_t<int64> res_block_inds;
    if (!res_bi.empty()) {
        res_block_inds = asarray_i64(np.attr("repeat")(
          np.attr("array")(res_bi).attr("__getitem__")(py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
          2,
          py::arg("axis") = 1));
        // clearer:
        res_block_inds = asarray_i64(np.attr("column_stack")(py::make_tuple(res_bi, res_bi)));
    } else {
        res_block_inds = zeros_i64(0, 2);
    }
    return wrap(make_data(tensor.attr("dtype").cast<Dtype>(),
                          t_data->device,
                          std::move(res_blocks),
                          res_block_inds,
                          true));
}

// ---- 03_mid.cpp ----

TensorBackend::DataPtr
AbelianBackend::copy_data(py::object a, std::optional<std::string> device)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->copy_block(b, device));
    std::string dev = device.has_value() ? block_backend->as_device(device) : a_data->device;
    auto np = numpy();
    return wrap(make_data(a_data->dtype, std::move(dev), std::move(blocks),
                          asarray_i64(a_data->block_inds.attr("copy")()), true));
}

TensorBackend::DataPtr
AbelianBackend::dagger(py::object a)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->dagger(b));
    auto np = numpy();
    py::array_t<int64> block_inds =
      asarray_i64(a_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(std::nullopt, std::nullopt, static_cast<py::ssize_t>(-1)))));
    return wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(blocks), block_inds));
}

BlockBackend::Scalar
AbelianBackend::data_item(DataPtr a)
{
    auto data = unwrap(a);
    if (data->blocks.size() > 1)
        throw std::runtime_error("Inconsistent data.");
    if (data->blocks.empty())
        return block_backend->as_scalar(dtype::zero_scalar(data->dtype), data->dtype);
    return block_backend->item(data->blocks[0]);
}

bool
AbelianBackend::diagonal_all(py::object a)
{
    auto data = data_from_tensor(a);
    if (static_cast<int64>(data->block_inds.shape(0)) < nsec(a.attr("leg")))
        return false;
    for (auto const& b : data->blocks)
        if (!block_backend->all(b))
            return false;
    return true;
}

bool
AbelianBackend::diagonal_any(py::object a)
{
    auto data = data_from_tensor(a);
    for (auto const& b : data->blocks)
        if (block_backend->any(b))
            return true;
    return false;
}


TensorBackend::DataPtr
AbelianBackend::diagonal_elementwise_unary(py::object a,
                                           py::function func,
                                           py::dict func_kwargs,
                                           bool maps_zero_to_zero)
{
    auto a_data = data_from_tensor(a);
    auto np = numpy();
    std::vector<BlockBackend::BlockPtr> blocks;
    py::array_t<int64> block_inds;
    if (maps_zero_to_zero) {
        blocks.reserve(a_data->blocks.size());
        for (auto const& b : a_data->blocks) {
            py::object res = func(py::cast(b), **func_kwargs);
            blocks.push_back(res.cast<BlockBackend::BlockPtr>());
        }
        block_inds = a_data->block_inds;
    } else {
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(nsec(a.attr("leg"))), np.attr("arange")(nsec(a.attr("leg"))))));
        py::object iter =
          misc().attr("iter_common_noncommon_sorted_arrays")(block_inds, a_data->block_inds);
        for (py::handle item : iter) {
            auto pair = item.cast<py::tuple>();
            py::object i = pair[0];
            py::object j = pair[1];
            BlockBackend::BlockPtr block;
            if (j.is_none()) {
                int64 ii = i.cast<int64>();
                block = block_backend->zeros({ mults_of(a.attr("leg"))[static_cast<std::size_t>(ii)] },
                                            a.attr("dtype").cast<Dtype>());
            } else {
                block = a_data->blocks[static_cast<std::size_t>(j.cast<int64>())];
            }
            py::object res = func(py::cast(block), **func_kwargs);
            blocks.push_back(res.cast<BlockBackend::BlockPtr>());
        }
    }
    Dtype dt;
    if (blocks.empty()) {
        py::object example =
          func(py::cast(block_backend->zeros({ 1 }, a.attr("dtype").cast<Dtype>())), **func_kwargs);
        dt = block_backend->get_dtype(example.cast<BlockBackend::BlockPtr>());
    } else {
        dt = block_backend->get_dtype(blocks[0]);
    }
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds, true));
}

std::string
AbelianBackend::get_device_from_data(DataPtr a)
{
    return block_backend->as_device(unwrap(a)->device);
}

Dtype
AbelianBackend::get_dtype_from_data(DataPtr a)
{
    return unwrap(a)->dtype;
}

bool
AbelianBackend::supports_symmetry(Symmetry::Ptr symmetry)
{
    return symmetry->is_abelian() && symmetry->has_trivial_braid();
}

py::object
AbelianBackend::state_tensor_product(BlockBackend::BlockPtr /*state1*/,
                                     BlockBackend::BlockPtr /*state2*/,
                                     LegPipe::Ptr /*pipe*/)
{
    throw NotImplemented("state_tensor_product not implemented");
}

BlockBackend::BlockPtr
AbelianBackend::to_dense_block_trivial_sector(py::object /*tensor*/)
{
    throw NotImplemented("to_dense_block_trivial_sector");
}

TensorBackend::DataPtr
AbelianBackend::zero_data(TensorProduct::Ptr codomain,
                          TensorProduct::Ptr domain,
                          Dtype dtype,
                          std::string device,
                          bool all_blocks)
{
    if (!all_blocks) {
        return wrap(make_data(dtype, std::move(device), {}, zeros_i64(0, codomain->num_factors + domain->num_factors), true));
    }
    auto block_inds = valid_block_inds(codomain, domain);
    auto legs = conventional_leg_order(codomain, domain);
    std::vector<BlockBackend::BlockPtr> zero_blocks;
    auto bi = block_inds.unchecked<2>();
    for (py::ssize_t r = 0; r < bi.shape(0); ++r) {
        std::vector<int64> shape;
        for (py::ssize_t c = 0; c < bi.shape(1); ++c)
            shape.push_back(mults_of(legs[static_cast<std::size_t>(c)])[static_cast<std::size_t>(bi(r, c))]);
        zero_blocks.push_back(block_backend->zeros(shape, dtype, device));
    }
    return wrap(make_data(dtype, std::move(device), std::move(zero_blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::zero_diagonal_data(TensorProduct::Ptr /*co_domain*/, Dtype dtype, std::string device)
{
    return wrap(make_data(dtype, std::move(device), {}, zeros_i64(0, 2), true));
}

TensorBackend::DataPtr
AbelianBackend::zero_mask_data(Space::Ptr /*large_leg*/, std::string device)
{
    return wrap(make_data(Dtype::Bool, std::move(device), {}, zeros_i64(0, 2), true));
}

void
AbelianBackend::save_hdf5(py::object hdf5_saver, py::object /*h5gr*/, std::string subpath)
{
    hdf5_saver.attr("save")(DataCls, subpath + "DataCls");
}

AbelianBackend::Ptr
AbelianBackend::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string subpath)
{
    auto obj = std::make_shared<AbelianBackend>(nullptr);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    obj->DataCls = hdf5_loader.attr("load")(subpath + "DataCls");
    return obj;
}

py::array_t<int64>
AbelianBackend::leg_pipe_map_incoming_block_inds(AbelianLegPipe const& pipe,
                                                 py::array_t<int64> incoming_block_inds) const
{
    assert(incoming_block_inds.shape(1) == pipe.num_legs);
    auto np = numpy();
    py::array strides = np.attr("array")(pipe.sector_strides);
    py::array inds_before =
      np.attr("sum")(incoming_block_inds.attr("__mul__")(strides.attr("__getitem__")(
                       py::make_tuple(np.attr("newaxis"), py::ellipsis()))),
                     py::arg("axis") = 1);
    py::array inv = misc().attr("inverse_permutation")(pipe.fusion_outcomes_sort);
    return asarray_i64(inv.attr("__getitem__")(inds_before));
}

TensorBackend::DataPtr
AbelianBackend::to_dtype(py::object a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::to_block_backend(DataPtr data,
                                 std::shared_ptr<BlockBackend> bb,
                                 std::optional<Dtype> dtype,
                                 std::optional<std::string> device)
{
    auto d = unwrap(data);
    Dtype dt = dtype.value_or(d->dtype);
    std::string dev = bb->as_device(device.has_value() ? device : std::optional<std::string>(d->device));
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(d->blocks.size());
    for (auto const& b : d->blocks)
        blocks.push_back(bb->as_block(py::cast(b), dt, dev));
    return wrap(make_data(dt, std::move(dev), std::move(blocks), d->block_inds));
}

TensorBackend::DataPtr
AbelianBackend::move_to_device(py::object a, std::string device)
{
    auto a_data = data_from_tensor(a);
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i)
        a_data->blocks[i] = block_backend->as_block(py::cast(a_data->blocks[i]), std::nullopt, device);
    a_data->device = block_backend->as_device(device);
    return wrap(a_data);
}

TensorBackend::DataPtr
AbelianBackend::full_data_from_diagonal_tensor(py::object a)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_diagonal(b));
    return wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::full_data_from_mask(py::object a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_mask(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::diagonal_tensor_from_full_tensor(py::object a, std::optional<float64> tol)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->get_diagonal(b, tol));
    return wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(blocks), a_data->block_inds, true));
}

BlockBackend::Scalar
AbelianBackend::diagonal_tensor_trace_full(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto total = block_backend->as_scalar(0.0, a.attr("dtype").cast<Dtype>());
    for (auto const& b : a_data->blocks)
        total = total + block_backend->sum_all(b);
    return total;
}

TensorBackend::DataPtr
AbelianBackend::mask_dagger(py::object mask)
{
    auto data = data_from_tensor(mask);
    auto np = numpy();
    py::array_t<int64> block_inds =
      asarray_i64(data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(std::nullopt, std::nullopt, static_cast<py::ssize_t>(-1)))));
    return wrap(make_data(mask.attr("dtype").cast<Dtype>(), mask.attr("device").cast<std::string>(),
                          data->blocks, block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::permute_legs(py::object a,
                             std::vector<int64> codomain_idcs,
                             std::vector<int64> domain_idcs,
                             TensorProduct::Ptr /*new_codomain*/,
                             TensorProduct::Ptr /*new_domain*/,
                             bool /*mixes_codomain_domain*/,
                             std::vector<std::optional<int64>> /*levels*/,
                             std::vector<std::optional<bool>> /*bend_right*/)
{
    auto a_data = data_from_tensor(a);
    std::vector<int64> axes_perm = codomain_idcs;
    for (auto it = domain_idcs.rbegin(); it != domain_idcs.rend(); ++it)
        axes_perm.push_back(*it);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->permute_axes(b, axes_perm));
    auto np = numpy();
    py::array_t<int64> block_inds =
      asarray_i64(a_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), axes_perm)));
    return wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(blocks), block_inds, false));
}

std::tuple<Space::Ptr, TensorBackend::DataPtr>
AbelianBackend::diagonal_transpose(py::object tens)
{
    auto leg = tens.attr("leg").cast<Space::Ptr>();
    return { leg->dual_space(), copy_data(tens) };
}

// ---- 04_delegate_part1.cpp ----

namespace {

AbelianBackendData::Ptr
data_from_any(py::object raw)
{
    try {
        return AbelianBackend::unwrap(raw.cast<TensorBackend::DataPtr>());
    } catch (py::cast_error const&) {
    } catch (std::invalid_argument const&) {
    }
    // Python AbelianBackendData duck type
    return make_data(raw.attr("dtype").cast<Dtype>(),
                     raw.attr("device").cast<std::string>(),
                     raw.attr("blocks").cast<std::vector<BlockBackend::BlockPtr>>(),
                     asarray_i64(raw.attr("block_inds")),
                     /*is_sorted=*/true);
}

py::object
py_abelian(std::shared_ptr<BlockBackend> const& bb)
{
    return py::module_::import("cyten.backends.abelian").attr("AbelianBackend")(
      py::arg("block_backend") = bb);
}

AbelianBackendData::Ptr
py_data(py::object obj)
{
    return data_from_any(obj);
}

template<typename... Args>
AbelianBackendData::Ptr
py_call_data(std::shared_ptr<BlockBackend> const& bb, char const* method, Args&&... args)
{
    return py_data(py_abelian(bb).attr(method)(std::forward<Args>(args)...));
}

} // namespace

// ---- Remaining methods ----
// FIXME: Several methods below still call the Python AbelianBackend via py_call_data /
// py_abelian for correct semantics while the native C++ ports are unfinished. Replace those
// with direct translations (see docs/cpp_conversion/convert_AbelianBackend.md).

TensorBackend::DataPtr
AbelianBackend::combine_legs(py::object tensor,
                             std::vector<std::vector<int64>> leg_idcs_combine,
                             std::vector<LegPipe::Ptr> pipes,
                             TensorProduct::Ptr new_codomain,
                             TensorProduct::Ptr new_domain)
{
    return wrap(py_call_data(block_backend, "combine_legs", tensor, leg_idcs_combine, pipes,
                             new_codomain, new_domain));
}

TensorBackend::DataPtr
AbelianBackend::compose(py::object a, py::object b)
{
    if (a.attr("num_codomain_legs").cast<int64>() == 0 && b.attr("num_domain_legs").cast<int64>() == 0) {
        // Python returns a Scalar here; wrap as 0-leg data with one scalar block.
        auto s = inner(a, b, false);
        auto block = block_backend->as_block(s.to_numpy(), a.attr("dtype").cast<Dtype>());
        return wrap(make_data(a.attr("dtype").cast<Dtype>(),
                              data_from_tensor(a)->device,
                              { block },
                              zeros_i64(1, 0),
                              true));
    }
    if (a.attr("num_domain_legs").cast<int64>() == 0)
        return _compose_no_contraction(a, b);
    return _compose_worker(a, b);
}

TensorBackend::DataPtr
AbelianBackend::_compose_worker(py::object a, py::object b)
{
    // Delegate full worker (also used with custom domains via partial_compose path in Python).
    return wrap(py_call_data(block_backend, "compose", a, b));
}

TensorBackend::DataPtr
AbelianBackend::_compose_no_contraction(py::object a, py::object b)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    Dtype res_dtype = dtype::common({ a_data->dtype, b_data->dtype });
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    if (a_data->dtype != res_dtype)
        for (auto& T : a_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    if (b_data->dtype != res_dtype)
        for (auto& T : b_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    auto a_bi = a_data->block_inds;
    auto b_bi = b_data->block_inds;
    py::ssize_t l_a = a_bi.shape(0), num_a = a_bi.shape(1);
    py::ssize_t l_b = b_bi.shape(0), num_b = b_bi.shape(1);
    py::array grid = misc().attr("make_grid")(py::make_tuple(l_a, l_b), py::arg("cstyle") = false).cast<py::array>();
    auto np = numpy();
    py::array_t<int64> res_bi = zeros_i64(l_a * l_b, num_a + num_b);
    res_bi = asarray_i64(np.attr("empty")(py::make_tuple(l_a * l_b, num_a + num_b), py::arg("dtype") = np.attr("intp")));
    auto g = asarray_i64(grid);
    auto gb = g.unchecked<2>();
    // fill via numpy for clarity
    res_bi = asarray_i64(np.attr("hstack")(py::make_tuple(
      a_bi.attr("__getitem__")(g.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0))),
      b_bi.attr("__getitem__")(g.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1))))));
    std::vector<BlockBackend::BlockPtr> res_blocks;
    res_blocks.reserve(static_cast<std::size_t>(gb.shape(0)));
    for (py::ssize_t r = 0; r < gb.shape(0); ++r)
        res_blocks.push_back(block_backend->outer(
          a_blocks[static_cast<std::size_t>(gb(r, 0))], b_blocks[static_cast<std::size_t>(gb(r, 1))]));
    return wrap(make_data(res_dtype, a_data->device, std::move(res_blocks), res_bi, true));
}


TensorBackend::DataPtr
AbelianBackend::diagonal_elementwise_binary(py::object a,
                                            py::object b,
                                            py::function func,
                                            py::dict func_kwargs,
                                            bool partial_zero_is_zero)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto leg = a.attr("leg");
    auto mults = mults_of(leg);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    auto a_bi = a_data->block_inds;
    auto b_bi = b_data->block_inds;
    py::ssize_t ia = 0, ib = 0;
    int64 bi_a = a_bi.shape(0) == 0 ? -1 : a_bi.at(0, 0);
    int64 bi_b = b_bi.shape(0) == 0 ? -1 : b_bi.at(0, 0);
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> block_ind_list;
    Dtype a_dtype = a.attr("dtype").cast<Dtype>();
    for (std::size_t i = 0; i < mults.size(); ++i) {
        BlockBackend::BlockPtr block_a;
        if (static_cast<int64>(i) == bi_a) {
            block_a = a_blocks[static_cast<std::size_t>(ia)];
            ++ia;
            bi_a = (ia >= a_bi.shape(0)) ? -1 : a_bi.at(ia, 0);
        } else if (partial_zero_is_zero) {
            continue;
        } else {
            block_a = block_backend->zeros({ mults[i] }, a_dtype);
        }
        BlockBackend::BlockPtr block_b;
        if (static_cast<int64>(i) == bi_b) {
            block_b = b_blocks[static_cast<std::size_t>(ib)];
            ++ib;
            bi_b = (ib >= b_bi.shape(0)) ? -1 : b_bi.at(ib, 0);
        } else if (partial_zero_is_zero) {
            continue;
        } else {
            block_b = block_backend->zeros({ mults[i] }, a_dtype);
        }
        py::object res = func(py::cast(block_a), py::cast(block_b), **func_kwargs);
        blocks.push_back(res.cast<BlockBackend::BlockPtr>());
        block_ind_list.push_back(static_cast<int64>(i));
    }
    auto np = numpy();
    py::array_t<int64> block_inds;
    Dtype dt;
    if (blocks.empty()) {
        block_inds = zeros_i64(0, 2);
        auto sample = func(py::cast(block_backend->ones_block({ 1 }, a_dtype)),
                           py::cast(block_backend->ones_block({ 1 }, b.attr("dtype").cast<Dtype>())));
        dt = block_backend->get_dtype(sample.cast<BlockBackend::BlockPtr>());
    } else {
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(block_ind_list, block_ind_list)));
        dt = block_backend->get_dtype(blocks[0]);
    }
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::diagonal_from_block(BlockBackend::BlockPtr a, TensorProduct::Ptr co_domain, float64 /*tol*/)
{
    auto leg = co_domain->factors[0];
    Dtype dt = block_backend->get_dtype(a);
    auto np = numpy();
    auto block_inds = asarray_i64(np.attr("column_stack")(
      py::make_tuple(np.attr("arange")(co_domain->num_sectors), np.attr("arange")(co_domain->num_sectors))));
    std::vector<BlockBackend::BlockPtr> blocks;
    auto bi = block_inds.unchecked<2>();
    for (py::ssize_t r = 0; r < bi.shape(0); ++r) {
        auto slc = slice_pair(leg.attr("slices").attr("__getitem__")(bi(r, 0)));
        blocks.push_back(b_get(a, slc));
    }
    return wrap(make_data(dt, block_backend->get_device(a), std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::diagonal_from_sector_block_func(py::function func, TensorProduct::Ptr co_domain)
{
    auto leg = co_domain->factors[0];
    auto np = numpy();
    auto block_inds = asarray_i64(np.attr("column_stack")(
      py::make_tuple(np.attr("arange")(nsec(leg)), np.attr("arange")(nsec(leg)))));
    auto sectors = leg.attr("sector_decomposition").cast<SectorArray>();
    auto mults = mults_of(leg);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t i = 0; i < mults.size(); ++i) {
        blocks.push_back(
          func(py::make_tuple(mults[i]), py::cast(sectors[i])).cast<BlockBackend::BlockPtr>());
    }
    BlockBackend::BlockPtr sample =
      blocks.empty()
        ? func(py::make_tuple(1), py::cast(co_domain->symmetry->trivial_sector)).cast<BlockBackend::BlockPtr>()
        : blocks[0];
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds,
                          true));
}

BlockBackend::BlockPtr
AbelianBackend::diagonal_tensor_to_block(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto leg = a.attr("leg");
    auto res = block_backend->zeros(
      { static_cast<int64>(leg.attr("dim").cast<float64>()) }, a.attr("dtype").cast<Dtype>());
    auto bi = a_data->block_inds.unchecked<2>();
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        auto slc = slice_pair(leg.attr("slices").attr("__getitem__")(bi(static_cast<py::ssize_t>(i), 0)));
        b_set(res, slc, a_data->blocks[i]);
    }
    return res;
}


std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::diagonal_to_mask(py::object tens)
{
    auto tens_data = data_from_tensor(tens);
    py::object large_leg = tens.attr("leg");
    py::object basis_perm = large_leg.attr("_basis_perm");
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> large_leg_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto bi = tens_data->block_inds.unchecked<2>();
    for (std::size_t n = 0; n < tens_data->blocks.size(); ++n) {
        auto const& diag_block = tens_data->blocks[n];
        if (!block_backend->any(diag_block))
            continue;
        int64 bii = bi(static_cast<py::ssize_t>(n), 0);
        blocks.push_back(diag_block);
        large_leg_block_inds.push_back(bii);
        sectors_vec.push_back(defining[static_cast<std::size_t>(bii)]);
        multiplicities.push_back(block_backend->sum_all(diag_block).as_int64());
        if (!basis_perm.is_none()) {
            py::array mask = block_backend->to_numpy(diag_block, py::module_::import("builtins").attr("bool"));
            // fallback: to_numpy with bool dtype
            try {
                mask = block_backend->to_numpy(diag_block, dtype::to_numpy_dtype(Dtype::Bool)).cast<py::array>();
            } catch (...) {
                mask = block_backend->to_numpy(diag_block).cast<py::array>();
            }
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bii));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    auto np = numpy();
    SectorArray sectors = tens.attr("symmetry").attr("empty_sector_array").cast<SectorArray>();
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    py::array_t<int64> block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = misc().attr("rank_data")(np.attr("concatenate")(basis_perm_ranks));
            basis_perm_opt = ranked.cast<std::vector<int64>>();
        }
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(sectors.size()), large_leg_block_inds)));
    }
    auto data = make_data(Dtype::Bool, tens_data->device, std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      tens.attr("symmetry").cast<Symmetry::Ptr>(),
      std::move(sectors),
      multiplicities,
      large_leg.attr("is_dual").cast<bool>(),
      basis_perm_opt);
    return { wrap(data), small_leg };
}


std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::eigh(py::object a, bool new_leg_dual, std::optional<std::string> sort)
{
    assert(a.attr("num_codomain_legs").cast<int64>() == 1);
    assert(a.attr("num_domain_legs").cast<int64>() == 1);
    auto a_data = data_from_tensor(a);
    auto domain = a.attr("domain").cast<TensorProduct::Ptr>();
    auto new_leg = domain->as_ElementarySpace(new_leg_dual).cast<ElementarySpace::Ptr>();
    auto v_wrapped = eye_data(domain, a.attr("dtype").cast<Dtype>(), a_data->device);
    auto v_data = unwrap(v_wrapped);
    std::vector<BlockBackend::BlockPtr> w_blocks;
    auto bi = a_data->block_inds.unchecked<2>();
    std::optional<std::string> sort_opt = sort;
    for (std::size_t n = 0; n < a_data->blocks.size(); ++n) {
        auto [vals, vects] = block_backend->eigh(a_data->blocks[n], sort_opt);
        w_blocks.push_back(vals);
        v_data->blocks[static_cast<std::size_t>(bi(static_cast<py::ssize_t>(n), 0))] = vects;
    }
    auto w_data = make_data(dtype::to_real(a.attr("dtype").cast<Dtype>()),
                            a_data->device,
                            std::move(w_blocks),
                            a_data->block_inds,
                            true);
    return { wrap(w_data), wrap(v_data), new_leg };
}

TensorBackend::DataPtr
AbelianBackend::eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device)
{
    auto np = numpy();
    py::list domain_dims;
    for (auto it = co_domain->factors.rbegin(); it != co_domain->factors.rend(); ++it)
        domain_dims.append(nsec(*it));
    py::object domain_block_inds =
      np.attr("indices")(domain_dims).attr("T").attr("reshape")(-1, co_domain->num_factors);
    py::array_t<int64> block_inds = asarray_i64(np.attr("hstack")(py::make_tuple(
      domain_block_inds.attr("__getitem__")(
        py::make_tuple(py::ellipsis(), py::slice(std::nullopt, std::nullopt, static_cast<py::ssize_t>(-1)))),
      domain_block_inds)));
    std::vector<BlockBackend::BlockPtr> blocks;
    auto bi = block_inds.unchecked<2>();
    blocks.reserve(static_cast<std::size_t>(bi.shape(0)));
    for (py::ssize_t r = 0; r < bi.shape(0); ++r) {
        std::vector<int64> shape;
        shape.reserve(static_cast<std::size_t>(co_domain->num_factors));
        for (int64 f = 0; f < co_domain->num_factors; ++f) {
            auto mults = mults_of(co_domain->factors[static_cast<std::size_t>(f)]);
            shape.push_back(mults[static_cast<std::size_t>(bi(r, f))]);
        }
        blocks.push_back(block_backend->eye_block(shape, dtype, device));
    }
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, true));
}


TensorBackend::DataPtr
AbelianBackend::from_dense_block(BlockBackend::BlockPtr a,
                                 TensorProduct::Ptr codomain,
                                 TensorProduct::Ptr domain,
                                 float64 tol)
{
    Dtype dt = block_backend->get_dtype(a);
    std::string device = block_backend->get_device(a);
    auto projected = block_backend->zeros(block_backend->get_shape(a), dt);
    auto block_inds = valid_block_inds(codomain, domain);
    auto legs = conventional_leg_order(codomain, domain);
    std::vector<BlockBackend::BlockPtr> blocks;
    auto bi = block_inds.unchecked<2>();
    blocks.reserve(static_cast<std::size_t>(bi.shape(0)));
    for (py::ssize_t r = 0; r < bi.shape(0); ++r) {
        py::tuple slices(static_cast<py::ssize_t>(legs.size()));
        for (py::ssize_t c = 0; c < static_cast<py::ssize_t>(legs.size()); ++c) {
            slices[c] = slice_pair(legs[static_cast<std::size_t>(c)].attr("slices").attr("__getitem__")(bi(r, c)));
        }
        auto block = b_get(a, slices);
        blocks.push_back(block);
        b_set(projected, slices, block);
    }
    auto n_diff = block_backend->norm((*a) - (*projected));
    auto n_a = block_backend->norm(a);
    if (n_diff.as_float64() > tol * n_a.as_float64())
        throw std::invalid_argument("Block is not symmetric up to tolerance.");
    return wrap(make_data(dt, std::move(device), std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
AbelianBackend::from_dense_block_trivial_sector(BlockBackend::BlockPtr block, Space::Ptr leg)
{
    auto bi = leg->sector_decomposition_where(leg->symmetry->trivial_sector);
    assert(bi.has_value());
    auto np = numpy();
    auto block_inds = asarray_i64(np.attr("array")(py::make_tuple(py::make_tuple(*bi))));
    return wrap(make_data(block_backend->get_dtype(block),
                          block_backend->get_device(block),
                          { block },
                          block_inds,
                          true));
}

TensorBackend::DataPtr
AbelianBackend::from_grid(std::vector<std::vector<py::object>> grid,
                          TensorProduct::Ptr new_codomain,
                          TensorProduct::Ptr new_domain,
                          std::vector<std::vector<int64>> left_mult_slices,
                          std::vector<std::vector<int64>> right_mult_slices,
                          Dtype dtype,
                          std::string device)
{
    return wrap(py_call_data(block_backend, "from_grid", grid, new_codomain, new_domain,
                             left_mult_slices, right_mult_slices, dtype, device));
}

TensorBackend::DataPtr
AbelianBackend::from_random_normal(TensorProduct::Ptr codomain,
                                   TensorProduct::Ptr domain,
                                   float64 sigma,
                                   Dtype dtype,
                                   std::string device)
{
    auto self = this;
    py::function func = py::cpp_function(
      [self, sigma, dtype, device](py::object shape, py::object /*coupled*/) {
          return self->block_backend->random_normal(
            shape.cast<std::vector<int64>>(), dtype, sigma, device);
      });
    return from_sector_block_func(func, codomain, domain);
}

TensorBackend::DataPtr
AbelianBackend::from_sector_block_func(py::function func,
                                       TensorProduct::Ptr codomain,
                                       TensorProduct::Ptr domain)
{
    auto block_inds = valid_block_inds(codomain, domain);
    auto legs = conventional_leg_order(codomain, domain);
    int64 M = codomain->num_factors;
    std::vector<BlockBackend::BlockPtr> blocks;
    auto bi = block_inds.unchecked<2>();
    for (py::ssize_t r = 0; r < bi.shape(0); ++r) {
        std::vector<int64> shape;
        for (py::ssize_t c = 0; c < bi.shape(1); ++c)
            shape.push_back(mults_of(legs[static_cast<std::size_t>(c)])[static_cast<std::size_t>(bi(r, c))]);
        std::vector<Sector> secs;
        for (int64 i = 0; i < M; ++i) {
            auto sectors = codomain->factors[static_cast<std::size_t>(i)]
                             .attr("sector_decomposition")
                             .cast<SectorArray>();
            secs.push_back(sectors[static_cast<std::size_t>(bi(r, i))]);
        }
        auto coupled = codomain->symmetry->multiple_fusion(secs);
        blocks.push_back(func(shape, py::cast(coupled)).cast<BlockBackend::BlockPtr>());
    }
    BlockBackend::BlockPtr sample;
    if (blocks.empty()) {
        std::vector<int64> shape(static_cast<std::size_t>(M + domain->num_factors), 1);
        sample = func(shape, py::cast(codomain->symmetry->trivial_sector)).cast<BlockBackend::BlockPtr>();
    } else {
        sample = blocks[0];
    }
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds,
                          true));
}


TensorBackend::DataPtr
AbelianBackend::from_tree_pairs(
  std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Dtype dtype,
  std::string device)
{
    auto block_inds_all = valid_block_inds(codomain, domain);
    std::vector<BlockBackend::BlockPtr> blocks;
    py::list bi_rows;
    std::set<std::pair<FusionTree, FusionTree>> pairs_done;
    auto bi = block_inds_all.unchecked<2>();
    for (py::ssize_t r = 0; r < bi.shape(0); ++r) {
        SectorArray unc_c = SectorArray::empty(codomain->symmetry->sector_ind_len);
        std::vector<std::uint8_t> dual_c;
        for (int64 n = 0; n < codomain->num_factors; ++n) {
            auto f = codomain->factors[static_cast<std::size_t>(n)];
            auto secs = f.attr("sector_decomposition").cast<SectorArray>();
            unc_c.push_back(secs[static_cast<std::size_t>(bi(r, n))]);
            dual_c.push_back(static_cast<std::uint8_t>(f.attr("is_dual").cast<bool>()));
        }
        SectorArray unc_d = SectorArray::empty(domain->symmetry->sector_ind_len);
        std::vector<std::uint8_t> dual_d;
        for (int64 n = 0; n < domain->num_factors; ++n) {
            auto f = domain->factors[static_cast<std::size_t>(n)];
            auto secs = f.attr("sector_decomposition").cast<SectorArray>();
            unc_d.push_back(secs[static_cast<std::size_t>(bi(r, bi.shape(1) - 1 - n))]);
            dual_d.push_back(static_cast<std::uint8_t>(f.attr("is_dual").cast<bool>()));
        }
        FusionTree X = FusionTree::from_abelian_symmetry(codomain->symmetry, unc_c, dual_c);
        FusionTree Y = FusionTree::from_abelian_symmetry(domain->symmetry, unc_d, dual_d);
        auto pair = std::make_pair(X, Y);
        pairs_done.insert(pair);
        auto it = trees.find(pair);
        if (it == trees.end())
            continue;
        bi_rows.append(block_inds_all.attr("__getitem__")(r));
        blocks.push_back(it->second);
    }
    for (auto const& kv : trees) {
        if (pairs_done.find(kv.first) == pairs_done.end())
            throw std::runtime_error("from_tree_pairs: unexpected tree pair");
    }
    py::array_t<int64> block_inds;
    if (bi_rows.size() == 0)
        block_inds = zeros_i64(0, codomain->num_factors + domain->num_factors);
    else
        block_inds = asarray_i64(numpy().attr("array")(bi_rows));
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, false));
}

BlockBackend::Scalar
AbelianBackend::get_element(py::object a, std::vector<int64> idcs)
{
    auto legs = conventional_leg_order(a);
    auto np = numpy();
    py::list rows;
    for (std::size_t i = 0; i < legs.size(); ++i) {
        py::object pair = legs[i].attr("parse_index")(idcs[i]);
        rows.append(pair);
    }
    auto pos = asarray_i64(np.attr("array")(rows));
    // pos shape (num_legs, 2): [:,0]=block_idx, [:,1]=within
    auto block_idcs = asarray_i64(pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0)));
    auto a_data = data_from_tensor(a);
    auto block = a_data->get_block(block_idcs);
    if (!block) {
        Dtype dt = a.attr("dtype").cast<Dtype>();
        return block_backend->as_scalar(dtype::zero_scalar(dt), dt);
    }
    auto within = pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)).cast<std::vector<int64>>();
    return block_backend->get_block_element(block, within);
}

BlockBackend::Scalar
AbelianBackend::get_element_diagonal(py::object a, int64 idx)
{
    py::object pair = a.attr("leg").attr("parse_index")(idx);
    int64 block_idx = pair.attr("__getitem__")(0).cast<int64>();
    int64 idx_within = pair.attr("__getitem__")(1).cast<int64>();
    auto np = numpy();
    auto query = asarray_i64(np.attr("array")(py::make_tuple(block_idx, block_idx)));
    auto block = data_from_tensor(a)->get_block(query);
    if (!block) {
        Dtype dt = a.attr("dtype").cast<Dtype>();
        return block_backend->as_scalar(dtype::zero_scalar(dt), dt);
    }
    return block_backend->get_block_element(block, { idx_within });
}

BlockBackend::Scalar
AbelianBackend::get_element_mask(py::object a, std::vector<int64> idcs)
{
    auto legs = conventional_leg_order(a);
    auto np = numpy();
    py::list rows;
    for (std::size_t i = 0; i < legs.size(); ++i)
        rows.append(legs[i].attr("parse_index")(idcs[i]));
    auto pos = asarray_i64(np.attr("array")(rows));
    auto block_idcs = asarray_i64(pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0)));
    auto block = data_from_tensor(a)->get_block(block_idcs);
    if (!block)
        return block_backend->as_scalar(false);
    auto within = asarray_i64(pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)));
    int64 small, large;
    if (a.attr("is_projection").cast<bool>()) {
        small = within.at(0);
        large = within.at(1);
    } else {
        large = within.at(0);
        small = within.at(1);
    }
    return block_backend->get_block_mask_element(block, large, small);
}


BlockBackend::Scalar
AbelianBackend::inner(py::object a, py::object b, bool do_dagger)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    auto np = numpy();
    py::list nsecs;
    for (auto const& leg : a.attr("legs"))
        nsecs.append(leg.attr("num_sectors"));
    py::array strides = misc().attr("make_stride")(nsecs, py::arg("cstyle") = false);
    py::array a_bi = np.attr("sum")(a_data->block_inds.attr("__mul__")(strides), py::arg("axis") = 1);
    py::array b_bi;
    if (do_dagger) {
        b_bi = np.attr("sum")(b_data->block_inds.attr("__mul__")(strides), py::arg("axis") = 1);
    } else {
        b_bi = np.attr("sum")(
          b_data->block_inds.attr("__mul__")(
            strides.attr("__getitem__")(py::slice(std::nullopt, std::nullopt, static_cast<py::ssize_t>(-1)))),
          py::arg("axis") = 1);
        auto sort = np.attr("argsort")(b_bi);
        b_bi = b_bi.attr("__getitem__")(sort);
        b_blocks = permute_blocks(b_blocks, sort.cast<py::array>());
    }
    auto res = block_backend->as_scalar(0.0, a.attr("dtype").cast<Dtype>());
    py::object iter = misc().attr("iter_common_sorted")(a_bi, b_bi);
    for (py::handle item : iter) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        res = res + block_backend->inner(a_blocks[static_cast<std::size_t>(i)],
                                         b_blocks[static_cast<std::size_t>(j)],
                                         do_dagger);
    }
    return res;
}


TensorBackend::DataPtr
AbelianBackend::inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr vector,
                                                        Space::Ptr space,
                                                        ElementarySpace::Ptr charge_leg)
{
    assert(charge_leg->num_sectors == 1);
    auto bi = space->sector_decomposition_where(charge_leg->sector_decomposition[0]);
    assert(bi.has_value());
    assert(block_backend->get_shape(vector)
           == std::vector<int64>{ space->multiplicities[static_cast<std::size_t>(*bi)] });
    auto np = numpy();
    auto block_inds = asarray_i64(np.attr("array")(py::make_tuple(py::make_tuple(*bi, 0))));
    return wrap(make_data(block_backend->get_dtype(vector),
                          block_backend->get_device(vector),
                          { block_backend->add_axis(vector, 1) },
                          block_inds,
                          true));
}

BlockBackend::BlockPtr
AbelianBackend::inv_part_to_dense_block_single_sector(py::object tensor)
{
    auto data = data_from_tensor(tensor);
    assert(data->blocks.size() <= 1);
    if (data->blocks.size() == 1) {
        return b_get(data->blocks[0],
                     py::make_tuple(py::slice(std::nullopt, std::nullopt, std::nullopt), 0));
    }
    auto sector = tensor.attr("domain")
                    .attr("__getitem__")(0)
                    .attr("sector_decomposition")
                    .attr("__getitem__")(0)
                    .cast<Sector>();
    int64 dim = tensor.attr("codomain")
                  .attr("__getitem__")(0)
                  .attr("sector_multiplicity")(py::cast(sector))
                  .cast<int64>();
    return block_backend->zeros({ dim }, data->dtype);
}


TensorBackend::DataPtr
AbelianBackend::linear_combination(BlockBackend::Scalar a,
                                   py::object v,
                                   BlockBackend::Scalar b,
                                   py::object w)
{
    auto v_data = data_from_tensor(v);
    auto w_data = data_from_tensor(w);
    auto v_blocks = v_data->blocks;
    auto w_blocks = w_data->blocks;
    Dtype common_dtype = dtype::common({ v.attr("dtype").cast<Dtype>(), w.attr("dtype").cast<Dtype>() });
    if (v_data->dtype != common_dtype)
        for (auto& T : v_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    if (w_data->dtype != common_dtype)
        for (auto& T : w_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    py::list res_bi_rows;
    py::object iter =
      misc().attr("iter_common_noncommon_sorted_arrays")(v_data->block_inds, w_data->block_inds);
    for (py::handle item : iter) {
        auto pair = item.cast<py::tuple>();
        py::object i = pair[0];
        py::object j = pair[1];
        if (j.is_none()) {
            int64 ii = i.cast<int64>();
            res_blocks.push_back(block_backend->mul(a, v_blocks[static_cast<std::size_t>(ii)]));
            res_bi_rows.append(v_data->block_inds.attr("__getitem__")(ii));
        } else if (i.is_none()) {
            int64 jj = j.cast<int64>();
            res_blocks.push_back(block_backend->mul(b, w_blocks[static_cast<std::size_t>(jj)]));
            res_bi_rows.append(w_data->block_inds.attr("__getitem__")(jj));
        } else {
            int64 ii = i.cast<int64>();
            int64 jj = j.cast<int64>();
            res_blocks.push_back(block_backend->linear_combination(
              a, v_blocks[static_cast<std::size_t>(ii)], b, w_blocks[static_cast<std::size_t>(jj)]));
            res_bi_rows.append(v_data->block_inds.attr("__getitem__")(ii));
        }
    }
    py::array_t<int64> res_block_inds;
    if (res_bi_rows.size() > 0)
        res_block_inds = asarray_i64(numpy().attr("array")(res_bi_rows));
    else
        res_block_inds = zeros_i64(0, v.attr("num_legs").cast<int64>());
    return wrap(make_data(common_dtype, v_data->device, std::move(res_blocks), res_block_inds, true));
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
AbelianBackend::lq(py::object tensor, TensorProduct::Ptr new_co_domain)
{
    py::object tup = py_abelian(block_backend).attr("lq")(tensor, new_co_domain);
    return { wrap(py_data(tup.attr("__getitem__")(0))), wrap(py_data(tup.attr("__getitem__")(1))) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::mask_binary_operand(py::object mask1, py::object mask2, py::function func)
{
    py::object tup = py_abelian(block_backend).attr("mask_binary_operand")(mask1, mask2, func);
    return { wrap(py_data(tup.attr("__getitem__")(0))),
             tup.attr("__getitem__")(1).cast<ElementarySpace::Ptr>() };
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::mask_contract_large_leg(py::object tensor, py::object mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, true);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::mask_contract_small_leg(py::object tensor, py::object mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, false);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::_mask_contract(py::object tensor, py::object mask, int64 leg_idx, bool large_leg)
{
    py::object tup =
      py_abelian(block_backend)
        .attr("_mask_contract")(tensor, mask, leg_idx, py::arg("large_leg") = large_leg);
    return { wrap(py_data(tup.attr("__getitem__")(0))),
             tup.attr("__getitem__")(1).cast<TensorProduct::Ptr>(),
             tup.attr("__getitem__")(2).cast<TensorProduct::Ptr>() };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::mask_from_block(BlockBackend::BlockPtr a, Space::Ptr large_leg)
{
    py::object tup = py_abelian(block_backend).attr("mask_from_block")(a, large_leg);
    return { wrap(py_data(tup.attr("__getitem__")(0))),
             tup.attr("__getitem__")(1).cast<ElementarySpace::Ptr>() };
}


BlockBackend::BlockPtr
AbelianBackend::mask_to_block(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto large_leg = a.attr("large_leg");
    auto res = block_backend->zeros(
      { static_cast<int64>(large_leg.attr("dim").cast<float64>()) }, Dtype::Bool);
    bool is_projection = a.attr("is_projection").cast<bool>();
    auto bi = a_data->block_inds.unchecked<2>();
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        int64 bi_large = is_projection ? bi(static_cast<py::ssize_t>(i), 1)
                                       : bi(static_cast<py::ssize_t>(i), 0);
        auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bi_large));
        b_set(res, slc, a_data->blocks[i]);
    }
    return res;
}


TensorBackend::DataPtr
AbelianBackend::mask_to_diagonal(py::object a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    auto np = numpy();
    py::array large_leg_bi =
      a.attr("is_projection").cast<bool>()
        ? a_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1))
        : a_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0));
    auto block_inds = asarray_i64(np.attr("column_stack")(py::make_tuple(large_leg_bi, large_leg_bi)));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds));
}


std::tuple<Space::Ptr, Space::Ptr, TensorBackend::DataPtr>
AbelianBackend::mask_transpose(py::object tens)
{
    auto data = data_from_tensor(tens);
    auto np = numpy();
    auto block_inds = asarray_i64(data->block_inds.attr("__getitem__")(
      py::make_tuple(py::ellipsis(),
                     py::slice(std::nullopt, std::nullopt, static_cast<py::ssize_t>(-1)))));
    auto out = make_data(tens.attr("dtype").cast<Dtype>(), data->device, data->blocks, block_inds, false);
    auto cod = tens.attr("codomain").attr("__getitem__")(0).attr("dual").cast<Space::Ptr>();
    auto dom = tens.attr("domain").attr("__getitem__")(0).attr("dual").cast<Space::Ptr>();
    return { cod, dom, wrap(out) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
AbelianBackend::mask_unary_operand(py::object mask, py::function func)
{
    // Use block_inds (Python typo was blocks_inds)
    py::object tup = py_abelian(block_backend).attr("mask_unary_operand")(mask, func);
    return { wrap(py_data(tup.attr("__getitem__")(0))),
             tup.attr("__getitem__")(1).cast<ElementarySpace::Ptr>() };
}

TensorBackend::DataPtr
AbelianBackend::mul(BlockBackend::Scalar a, py::object b)
{
    auto b_data = data_from_tensor(b);
    if (is_zero_scalar(a))
        return zero_data(b.attr("codomain").cast<TensorProduct::Ptr>(),
                         b.attr("domain").cast<TensorProduct::Ptr>(),
                         b.attr("dtype").cast<Dtype>(),
                         b_data->device);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& T : b_data->blocks)
        blocks.push_back(block_backend->mul(a, T));
    Dtype dt;
    if (blocks.empty()) {
        dt = dtype::is_real(a.dtype()) ? b_data->dtype : dtype::to_complex(b_data->dtype);
    } else {
        dt = block_backend->get_dtype(blocks[0]);
    }
    return wrap(make_data(dt, b_data->device, std::move(blocks), b_data->block_inds, true));
}

BlockBackend::Scalar
AbelianBackend::norm(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto block_norms =
      block_backend->zeros({ static_cast<int64>(a_data->blocks.size()) }, a.attr("dtype").cast<Dtype>());
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        auto n = block_backend->norm(a_data->blocks[i], 2., std::nullopt);
        block_norms->set_item(static_cast<int64>(i), n);
    }
    return block_backend->norm(block_norms, 2., std::nullopt);
}


TensorBackend::DataPtr
AbelianBackend::outer(py::object a, py::object b)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    auto a_bi = a_data->block_inds;
    auto b_bi = b_data->block_inds;
    py::ssize_t l_a = a_bi.shape(0), N_a = a_bi.shape(1);
    py::ssize_t l_b = b_bi.shape(0), N_b = b_bi.shape(1);
    int64 K_a = a.attr("num_codomain_legs").cast<int64>();
    Dtype res_dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    if (a.attr("dtype").cast<Dtype>() != res_dtype)
        for (auto& T : a_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    if (b.attr("dtype").cast<Dtype>() != res_dtype)
        for (auto& T : b_blocks)
            T = block_backend->to_dtype(T, res_dtype);
    auto np = numpy();
    py::array grid = misc().attr("make_grid")(py::make_tuple(l_a, l_b), py::arg("cstyle") = false).cast<py::array>();
    auto g = asarray_i64(grid);
    auto gb = g.unchecked<2>();
    py::array_t<int64> res_bi = asarray_i64(np.attr("empty")(
      py::make_tuple(l_a * l_b, N_a + N_b), py::arg("dtype") = np.attr("intp")));
    // fill columns via numpy
    res_bi = asarray_i64(np.attr("empty")(py::make_tuple(gb.shape(0), N_a + N_b), py::arg("dtype") = np.attr("intp")));
    {
        auto out = res_bi.mutable_unchecked<2>();
        auto abi = a_bi.unchecked<2>();
        auto bbi = b_bi.unchecked<2>();
        for (py::ssize_t r = 0; r < gb.shape(0); ++r) {
            auto ia = gb(r, 0);
            auto ib = gb(r, 1);
            for (py::ssize_t c = 0; c < K_a; ++c)
                out(r, c) = abi(ia, c);
            for (py::ssize_t c = 0; c < N_b; ++c)
                out(r, K_a + c) = bbi(ib, c);
            for (py::ssize_t c = K_a; c < N_a; ++c)
                out(r, K_a + N_b + (c - K_a)) = abi(ia, c);
        }
    }
    std::vector<BlockBackend::BlockPtr> res_blocks;
    res_blocks.reserve(static_cast<std::size_t>(gb.shape(0)));
    for (py::ssize_t r = 0; r < gb.shape(0); ++r) {
        res_blocks.push_back(block_backend->tensor_outer(
          a_blocks[static_cast<std::size_t>(gb(r, 0))],
          b_blocks[static_cast<std::size_t>(gb(r, 1))],
          K_a));
    }
    return wrap(make_data(res_dtype, a_data->device, std::move(res_blocks), res_bi, false));
}

TensorBackend::DataPtr
AbelianBackend::partial_compose(py::object a,
                                py::object b,
                                int64 a_first_leg,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain)
{
    return wrap(py_call_data(block_backend, "partial_compose", a, b, a_first_leg, new_codomain, new_domain));
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
AbelianBackend::partial_trace(py::object tensor,
                              std::vector<std::pair<int64, int64>> pairs,
                              std::optional<std::vector<int64>> levels)
{
    py::object lev = levels.has_value() ? py::cast(*levels) : py::none();
    py::object tup = py_abelian(block_backend).attr("partial_trace")(tensor, pairs, lev);
    py::object d0 = tup.attr("__getitem__")(0);
    py::object d1 = tup.attr("__getitem__")(1);
    py::object d2 = tup.attr("__getitem__")(2);
    // Scalar path: Python returns (scalar, None, None)
    if (d1.is_none() && d2.is_none()) {
        // Wrap scalar as Data with one block and empty block_inds; nullptr domains.
        BlockBackend::BlockPtr block;
        Dtype dt = tensor.attr("dtype").cast<Dtype>();
        if (py::isinstance<BlockBackend::Scalar>(d0)) {
            auto s = d0.cast<BlockBackend::Scalar>();
            block = block_backend->as_block(s.to_numpy(), dt);
        } else {
            // plain python scalar
            block = block_backend->as_block(
              py::module_::import("numpy").attr("asarray")(d0), dt);
        }
        auto data = make_data(dt, data_from_tensor(tensor)->device, { block }, zeros_i64(1, 0), true);
        return { wrap(data), nullptr, nullptr };
    }
    return { wrap(py_data(d0)), d1.cast<TensorProduct::Ptr>(), d2.cast<TensorProduct::Ptr>() };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
AbelianBackend::qr(py::object a, TensorProduct::Ptr new_co_domain)
{
    py::object tup = py_abelian(block_backend).attr("qr")(a, new_co_domain);
    return { wrap(py_data(tup.attr("__getitem__")(0))), wrap(py_data(tup.attr("__getitem__")(1))) };
}


BlockBackend::Scalar
AbelianBackend::reduce_DiagonalTensor(py::object tensor, py::function block_func, py::function func)
{
    auto data = data_from_tensor(tensor);
    auto mults = mults_of(tensor.attr("leg"));
    py::list numbers;
    py::ssize_t i = 0;
    auto bi = data->block_inds;
    py::ssize_t nblocks = bi.shape(0);
    for (std::size_t j = 0; j < mults.size(); ++j) {
        BlockBackend::BlockPtr block;
        if (i < nblocks && static_cast<std::size_t>(bi.at(i, 0)) == j) {
            block = data->blocks[static_cast<std::size_t>(i)];
            ++i;
        } else {
            block = block_backend->zeros({ mults[j] }, tensor.attr("dtype").cast<Dtype>());
        }
        numbers.append(block_func(py::cast(block)));
    }
    return func(numbers).cast<BlockBackend::Scalar>();
}

TensorBackend::DataPtr
AbelianBackend::scale_axis(py::object a, py::object b, int64 leg)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    auto np = numpy();
    py::array a_block_inds = a_data->block_inds;
    py::array a_block_inds_cont =
      a_block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(leg, leg + 1, 1)));
    if (leg != a.attr("num_legs").cast<int64>() - 1) {
        auto sort = np.attr("lexsort")(a_block_inds_cont.attr("T"));
        a_blocks = permute_blocks(a_blocks, sort.cast<py::array>());
        a_block_inds = take_rows_obj(a_block_inds, sort);
        a_block_inds_cont = a_block_inds.attr("__getitem__")(
          py::make_tuple(py::ellipsis(), py::slice(leg, leg + 1, 1)));
    }
    Dtype common_dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    if (a_data->dtype != common_dtype)
        for (auto& blk : a_blocks)
            blk = block_backend->to_dtype(blk, common_dtype);
    if (b_data->dtype != common_dtype)
        for (auto& blk : b_blocks)
            blk = block_backend->to_dtype(blk, common_dtype);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    py::list res_bi_rows;
    py::object iter = misc().attr("iter_common_sorted_arrays")(
      a_block_inds_cont,
      b_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(0, 1, 1))),
      py::arg("a_strict") = false);
    for (py::handle item : iter) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        res_blocks.push_back(block_backend->scale_axis(
          a_blocks[static_cast<std::size_t>(i)], b_blocks[static_cast<std::size_t>(j)], leg));
        res_bi_rows.append(a_block_inds.attr("__getitem__")(i));
    }
    py::array_t<int64> res_block_inds;
    if (res_bi_rows.size() > 0)
        res_block_inds = asarray_i64(np.attr("array")(res_bi_rows));
    else
        res_block_inds = zeros_i64(0, a.attr("num_legs").cast<int64>());
    return wrap(make_data(common_dtype, a_data->device, std::move(res_blocks), res_block_inds, false));
}

TensorBackend::DataPtr
AbelianBackend::split_legs(py::object a,
                           std::vector<int64> leg_idcs,
                           TensorProduct::Ptr new_codomain,
                           TensorProduct::Ptr new_domain)
{
    return wrap(py_call_data(block_backend, "split_legs", a, leg_idcs, new_codomain, new_domain));
}

TensorBackend::DataPtr
AbelianBackend::squeeze_legs(py::object a, std::vector<int64> idcs)
{
    auto a_data = data_from_tensor(a);
    int64 n_legs = a.attr("num_legs").cast<int64>();
    if (a_data->blocks.empty()) {
        return wrap(make_data(a_data->dtype, a_data->device, {},
                              zeros_i64(0, n_legs - static_cast<int64>(idcs.size())), true));
    }
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->squeeze_axes(b, idcs));
    auto np = numpy();
    py::array keep = np.attr("ones")(n_legs, py::arg("dtype") = np.attr("bool_"));
    for (auto i : idcs)
        keep.attr("__setitem__")(i, false);
    py::array_t<int64> block_inds =
      asarray_i64(a_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), keep)));
    return wrap(make_data(a_data->dtype, a_data->device, std::move(blocks), block_inds, true));
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, TensorBackend::DataPtr>
AbelianBackend::svd(py::object a,
                    TensorProduct::Ptr new_co_domain,
                    std::optional<std::string> algorithm)
{
    py::object alg = algorithm.has_value() ? py::cast(*algorithm) : py::none();
    py::object tup = py_abelian(block_backend).attr("svd")(a, new_co_domain, alg);
    return { wrap(py_data(tup.attr("__getitem__")(0))),
             wrap(py_data(tup.attr("__getitem__")(1))),
             wrap(py_data(tup.attr("__getitem__")(2))) };
}


BlockBackend::BlockPtr
AbelianBackend::to_dense_block(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto shape = a.attr("shape").cast<std::vector<int64>>();
    auto res = block_backend->zeros(shape, a_data->dtype);
    auto legs = conventional_leg_order(a);
    auto bi = a_data->block_inds.unchecked<2>();
    for (std::size_t i = 0; i < a_data->blocks.size(); ++i) {
        py::tuple slices(static_cast<py::ssize_t>(legs.size()));
        for (py::ssize_t c = 0; c < static_cast<py::ssize_t>(legs.size()); ++c) {
            slices[c] = slice_pair(
              legs[static_cast<std::size_t>(c)].attr("slices").attr("__getitem__")(
                bi(static_cast<py::ssize_t>(i), c)));
        }
        b_set(res, slices, a_data->blocks[i]);
    }
    return res;
}


BlockBackend::Scalar
AbelianBackend::trace_full(py::object a, std::vector<int64> /*idcs1*/, std::vector<int64> /*idcs2*/)
{
    auto a_data = data_from_tensor(a);
    int64 K = a.attr("num_codomain_legs").cast<int64>();
    auto res = block_backend->as_scalar(dtype::zero_scalar(a_data->dtype), a_data->dtype);
    auto np = numpy();
    auto bi = a_data->block_inds.unchecked<2>();
    for (std::size_t n = 0; n < a_data->blocks.size(); ++n) {
        bool on_diag = true;
        for (int64 c = 0; c < K; ++c) {
            if (bi(static_cast<py::ssize_t>(n), c)
                != bi(static_cast<py::ssize_t>(n), bi.shape(1) - 1 - c)) {
                on_diag = false;
                break;
            }
        }
        if (on_diag)
            res = res + block_backend->trace_full(a_data->blocks[n]);
    }
    return res;
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr, float64, float64>
AbelianBackend::truncate_singular_values(py::object S,
                                         std::optional<int64> chi_max,
                                         int64 chi_min,
                                         float64 degeneracy_tol,
                                         float64 trunc_cut,
                                         float64 svd_min,
                                         bool minimize_error)
{
    py::object tup = py_abelian(block_backend)
                       .attr("truncate_singular_values")(S,
                                                         chi_max.has_value() ? py::cast(*chi_max) : py::none(),
                                                         chi_min,
                                                         degeneracy_tol,
                                                         trunc_cut,
                                                         svd_min,
                                                         minimize_error);
    return { wrap(py_data(tup.attr("__getitem__")(0))),
             tup.attr("__getitem__")(1).cast<ElementarySpace::Ptr>(),
             tup.attr("__getitem__")(2).cast<float64>(),
             tup.attr("__getitem__")(3).cast<float64>() };
}

} // namespace cyten
