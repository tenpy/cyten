#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/backends/fusion_tree_mapping.h>
#include <cyten/backends/fusion_tree_permute.h>

#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/trees.h>
#include <cyten/tools.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <typeinfo>
#include <utility>
#include <vector>

#include <cyten/symmetries/fusion_symbol.h>

namespace cyten {

namespace {

py::module_
numpy()
{
    return py::module_::import("numpy");
}

/// Reorder rows of a 2D array according to ``perm`` (1D index array).
py::array
take_rows(py::array const& arr, py::array const& perm)
{
    return numpy().attr("take")(arr, perm, py::arg("axis") = 0).cast<py::array>();
}

} // namespace

FusionTreeData::FusionTreeData(py::array block_inds_,
                               std::vector<BlockBackend::BlockPtr> blocks_,
                               Dtype dtype_,
                               std::string device_,
                               bool is_sorted)
  : block_inds(std::move(block_inds_))
  , blocks(std::move(blocks_))
  , dtype(dtype_)
  , device(std::move(device_))
{
    if (block_inds.ndim() != 2) {
        throw std::invalid_argument("FusionTreeData: block_inds must be 2D");
    }
    if (block_inds.shape(1) != 2) {
        throw std::invalid_argument("FusionTreeData: block_inds must have shape (N, 2)");
    }
    if (static_cast<std::size_t>(block_inds.shape(0)) != blocks.size()) {
        throw std::invalid_argument(
          "FusionTreeData: len(blocks) must equal block_inds.shape[0]");
    }

    if (!is_sorted) {
        auto np = numpy();
        // ``np.lexsort(block_inds.T)`` — last column is primary key (numpy convention).
        auto perm = np.attr("lexsort")(block_inds.attr("T")).cast<py::array>();
        block_inds = take_rows(block_inds, perm);

        auto perm_i64 =
          py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(perm);
        auto perm_buf = perm_i64.unchecked<1>();
        std::vector<BlockBackend::BlockPtr> sorted_blocks;
        sorted_blocks.reserve(blocks.size());
        for (py::ssize_t i = 0; i < perm_buf.shape(0); ++i) {
            sorted_blocks.push_back(blocks[static_cast<std::size_t>(perm_buf(i))]);
        }
        blocks = std::move(sorted_blocks);
    }
}

std::optional<int64>
FusionTreeData::block_ind_from_coupled(Sector coupled, TensorProduct::Ptr domain) const
{
    auto domain_sector_ind = domain->sector_decomposition_where(coupled);
    if (!domain_sector_ind.has_value()) {
        return std::nullopt;
    }
    return block_ind_from_domain_sector_ind(*domain_sector_ind);
}

std::optional<int64>
FusionTreeData::block_ind_from_domain_sector_ind(int64 domain_sector_ind) const
{
    auto np = numpy();
    // Column 1 only — that axis is sorted by construction (lexsort last-key-first).
    py::array col1 = block_inds[py::make_tuple(py::ellipsis(), 1)].cast<py::array>();
    int64 ind = np.attr("searchsorted")(col1, domain_sector_ind).cast<int64>();

    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(block_inds);
    auto buf = bi.unchecked<2>();
    int64 n_rows = static_cast<int64>(buf.shape(0));

    if (ind >= n_rows || buf(ind, 1) != domain_sector_ind) {
        return std::nullopt;
    }
    if (ind + 1 < n_rows && buf(ind + 1, 1) == domain_sector_ind) {
        throw std::runtime_error(
          "FusionTreeData: duplicate domain sector index in block_inds[:, 1]");
    }
    return ind;
}

void
FusionTreeData::discard_zero_blocks(std::shared_ptr<BlockBackend> backend, float64 eps)
{
    std::vector<int64> keep;
    keep.reserve(blocks.size());
    for (std::size_t i = 0; i < blocks.size(); ++i) {
        if ((backend->norm(blocks[i]) >= eps).as_bool()) {
            keep.push_back(static_cast<int64>(i));
        }
    }

    std::vector<BlockBackend::BlockPtr> kept_blocks;
    kept_blocks.reserve(keep.size());
    for (int64 i : keep) {
        kept_blocks.push_back(blocks[static_cast<std::size_t>(i)]);
    }
    blocks = std::move(kept_blocks);

    // Explicit int64 indices: empty default arrays are float64 and break np.take.
    py::array_t<int64> keep_arr(static_cast<py::ssize_t>(keep.size()));
    {
        auto buf = keep_arr.mutable_unchecked<1>();
        for (std::size_t i = 0; i < keep.size(); ++i) {
            buf(static_cast<py::ssize_t>(i)) = keep[i];
        }
    }
    block_inds = take_rows(block_inds, keep_arr);
}

void
FusionTreeData::save_hdf5(py::object hdf5_saver, py::object /*h5gr*/, std::string subpath) const
{
    hdf5_saver.attr("save")(block_inds, subpath + "block_inds");
    hdf5_saver.attr("save")(blocks, subpath + "blocks");
    hdf5_saver.attr("save")(dtype, subpath + "dtype");
    hdf5_saver.attr("save")(device, subpath + "device");
}

FusionTreeData::Ptr
FusionTreeData::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string subpath)
{
    auto block_inds = hdf5_loader.attr("load")(subpath + "block_inds").cast<py::array>();
    auto blocks =
      hdf5_loader.attr("load")(subpath + "blocks").cast<std::vector<BlockBackend::BlockPtr>>();
    auto device = hdf5_loader.attr("load")(subpath + "device").cast<std::string>();
    auto dtype = hdf5_loader.attr("load")(subpath + "dtype").cast<Dtype>();

    // Already sorted when saved; skip lexsort.
    auto obj = std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, std::move(device), /*is_sorted=*/true);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

// FusionTreeBackend method implementations (append after FusionTreeData::from_hdf5)

namespace {

py::module_
misc()
{
    return py::module_::import("cyten.tools.misc");
}

py::module_
sector_utils()
{
    return py::module_::import("cyten.symmetries.sector_utils");
}

py::array_t<int64>
zeros_i64(py::ssize_t rows, py::ssize_t cols)
{
    auto np = numpy();
    return np.attr("zeros")(py::make_tuple(rows, cols), py::arg("dtype") = np.attr("intp"))
      .cast<py::array_t<int64>>();
}

py::array_t<int64>
asarray_i64(py::object obj)
{
    auto np = numpy();
    return np.attr("asarray")(obj, py::arg("dtype") = np.attr("intp")).cast<py::array_t<int64>>();
}

py::array
block_col(py::array arr, int64 col)
{
    return arr.attr("__getitem__")(py::make_tuple(py::ellipsis(), col)).cast<py::array>();
}

py::slice
slice_from_index_slice(IndexSlice slc)
{
    return py::slice(slc.start, slc.stop, 1);
}

py::slice
slice_pair(py::object pair)
{
    return py::slice(pair.attr("__getitem__")(0).cast<py::ssize_t>(),
                     pair.attr("__getitem__")(1).cast<py::ssize_t>(),
                     1);
}

BlockBackend::BlockPtr
b_get(BlockBackend::BlockPtr const& b, py::object key)
{
    return b->get_item(key);
}

void
b_set(BlockBackend::BlockPtr const& b, py::object key, BlockBackend::BlockPtr const& v)
{
    b->set_item(key, py::cast(v));
}

void
b_set_add(BlockBackend::BlockPtr const& b, py::object key, BlockBackend::BlockPtr const& v)
{
    b_set(b, key, (*b_get(b, key)) + (*v));
}

bool
is_zero_scalar(BlockBackend::Scalar const& a)
{
    return a.as_complex128() == 0.;
}

FusionTreeData::Ptr
make_data(Dtype dtype,
          std::string device,
          std::vector<BlockBackend::BlockPtr> blocks,
          py::array block_inds,
          bool is_sorted = false)
{
    return std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, std::move(device), is_sorted);
}

int64
prod_int(std::vector<int64> const& v)
{
    int64 p = 1;
    for (int64 x : v)
        p *= x;
    return p;
}

std::vector<int64>
tp_mults(py::object tp)
{
    return tp.attr("multiplicities").cast<std::vector<int64>>();
}

std::vector<int64>
legs_flat_leg_permutation(py::object legs)
{
    std::vector<int64> out;
    int64 offset = 0;
    for (py::handle h : legs) {
        auto leg = h.cast<Leg::Ptr>();
        auto part_perm = leg->_flat_leg_permutation(offset);
        out.insert(out.end(), part_perm.begin(), part_perm.end());
        offset += leg->num_flat_legs();
    }
    return out;
}

int64
nsec(py::object leg)
{
    return leg.attr("num_sectors").cast<int64>();
}

std::vector<int64>
mults_of(py::object leg)
{
    return leg.attr("multiplicities").cast<std::vector<int64>>();
}

py::object
take_rows_obj(py::object arr, py::object perm)
{
    return numpy().attr("take")(arr, perm, py::arg("axis") = 0);
}

std::vector<BlockBackend::BlockPtr>
permute_blocks(std::vector<BlockBackend::BlockPtr> const& blocks, py::array const& perm)
{
    auto p = perm.cast<py::array_t<int64>>();
    auto buf = p.unchecked<1>();
    std::vector<BlockBackend::BlockPtr> out;
    out.reserve(static_cast<std::size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        out.push_back(blocks[static_cast<std::size_t>(buf(i))]);
    return out;
}

std::vector<std::uint8_t>
flat_are_dual(TensorProduct::Ptr tp)
{
    std::vector<std::uint8_t> out;
    for (auto const& leg : tp->flat_legs())
        out.push_back(static_cast<std::uint8_t>(leg->is_dual));
    return out;
}

} // namespace


TensorBackend::DataPtr
FusionTreeBackend::wrap(FusionTreeData::Ptr d)
{
    if (!d)
        throw std::invalid_argument("FusionTreeBackend::wrap: null");
    return d;
}

FusionTreeData::Ptr
FusionTreeBackend::unwrap(DataPtr d)
{
    if (!d)
        throw std::invalid_argument("FusionTreeBackend::unwrap: null DataPtr");
    auto* p = dynamic_cast<FusionTreeData*>(d.get());
    if (!p)
        throw std::invalid_argument(std::format(
          "FusionTreeBackend::unwrap: expected FusionTreeData, got {}", typeid(*d).name()));
    return std::static_pointer_cast<FusionTreeData>(d);
}

FusionTreeData::Ptr
FusionTreeBackend::data_from_tensor(py::object tensor)
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
                     raw.attr("block_inds").cast<py::array>(),
                     /*is_sorted=*/true);
}

FusionTreeBackend::FusionTreeBackend(std::shared_ptr<BlockBackend> block_backend_, float64 eps_)
  : TensorBackend(std::move(block_backend_))
  , eps(eps_)
{
    can_decompose_tensors = true;
    DataCls = py::none();
}

void
FusionTreeBackend::test_tensor_sanity(py::object a, bool is_diagonal)
{
    TensorBackend::test_tensor_sanity(a, is_diagonal);
    py::object raw = a.attr("data");
    FusionTreeData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    assert(a.attr("device").cast<std::string>() == data->device);
    assert(data->device == block_backend->as_device(data->device));
    assert(a.attr("dtype").cast<Dtype>() == data->dtype);
    auto np = numpy();
    assert(data->block_inds.shape(0) == static_cast<py::ssize_t>(data->blocks.size()));
    assert(data->block_inds.shape(1) == 2);
    assert(np.attr("all")(data->block_inds.attr("__ge__")(0)).cast<bool>());
    assert(np.attr("all")(data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0))
                            .attr("__lt__")(a.attr("codomain").attr("num_sectors")))
             .cast<bool>());
    assert(np.attr("all")(data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1))
                            .attr("__lt__")(a.attr("domain").attr("num_sectors")))
             .cast<bool>());
    assert(np.attr("all")(np.attr("equal")(np.attr("lexsort")(data->block_inds.attr("T")),
                                            np.attr("arange")(data->blocks.size())))
             .cast<bool>());
    py::array coupled_codomain =
      a.attr("codomain").attr("sector_decomposition").attr("__getitem__")(
        data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0)));
    py::array coupled_domain =
      a.attr("domain").attr("sector_decomposition").attr("__getitem__")(
        data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)));
    assert(np.attr("all")(coupled_codomain.attr("__eq__")(coupled_domain)).cast<bool>());
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        int64 ic = buf(i, 0);
        int64 id = buf(i, 1);
        std::vector<int64> expect_shape = {
          tp_mults(a.attr("codomain"))[static_cast<std::size_t>(ic)],
          mults_of(a.attr("domain"))[static_cast<std::size_t>(id)] };
        if (is_diagonal) {
            assert(expect_shape[0] == expect_shape[1]);
            expect_shape = { expect_shape[0] };
        }
        block_backend->test_block_sanity(
          data->blocks[static_cast<std::size_t>(i)],
          expect_shape,
          a.attr("dtype").cast<Dtype>(),
          a.attr("device").cast<std::string>());
    }
}

void
FusionTreeBackend::test_mask_sanity(py::object a)
{
    TensorBackend::test_mask_sanity(a);
    py::object raw = a.attr("data");
    FusionTreeData::Ptr data;
    try {
        data = unwrap(raw.cast<DataPtr>());
    } catch (...) {
        return;
    }
    assert(a.attr("device").cast<std::string>() == data->device);
    assert(data->dtype == Dtype::Bool);
    auto np = numpy();
    assert(data->block_inds.shape(0) == static_cast<py::ssize_t>(data->blocks.size()));
    assert(data->block_inds.shape(1) == 2);
    bool is_projection = a.attr("is_projection").cast<bool>();
    auto large_leg = a.attr("large_leg");
    auto small_leg = a.attr("small_leg");
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        int64 bi_small = is_projection ? buf(i, 0) : buf(i, 1);
        int64 bi_large = is_projection ? buf(i, 1) : buf(i, 0);
        assert(bi_large >= bi_small);
        int64 expect_len = mults_of(large_leg)[static_cast<std::size_t>(bi_large)];
        int64 expect_sum = mults_of(small_leg)[static_cast<std::size_t>(bi_small)];
        block_backend->test_block_sanity(
          data->blocks[static_cast<std::size_t>(i)],
          std::vector<int64>{ expect_len },
          Dtype::Bool,
          data->device);
        assert(block_backend->sum_all(data->blocks[static_cast<std::size_t>(i)]).as_int64() == expect_sum);
    }
}

TensorBackend::DataPtr
FusionTreeBackend::act_block_diagonal_square_matrix(py::object a,
                                                    py::function block_method,
                                                    py::object dtype_map)
{
    auto a_data = data_from_tensor(a);
    auto np = numpy();
    py::array block_inds = a_data->block_inds;
    py::array block_inds_col0 = block_col(block_inds, 0);
    std::vector<BlockBackend::BlockPtr> res_blocks;
    int64 n = 0;
    int64 bi = (block_inds.shape(0) == 0) ? -1 : block_inds_col0.attr("__getitem__")(n).cast<int64>();
    int64 num_sectors = a.attr("codomain").attr("num_sectors").cast<int64>();
    for (int64 i = 0; i < num_sectors; ++i) {
        BlockBackend::BlockPtr block;
        if (bi == i) {
            block = a_data->blocks[static_cast<std::size_t>(n)];
            ++n;
            bi = (n >= block_inds.shape(0)) ? -1 : block_inds_col0.attr("__getitem__")(n).cast<int64>();
        } else {
            block = block_backend->zeros(
              { tp_mults(a.attr("codomain"))[static_cast<std::size_t>(i)],
                tp_mults(a.attr("codomain"))[static_cast<std::size_t>(i)] },
              a.attr("dtype").cast<Dtype>());
        }
        res_blocks.push_back(block_method(py::cast(block)).cast<BlockBackend::BlockPtr>());
    }
    Dtype dt = dtype_map.is_none() ? a.attr("dtype").cast<Dtype>() : dtype_map(a.attr("dtype")).cast<Dtype>();
    py::array_t<int64> res_block_inds =
      asarray_i64(np.attr("repeat")(
        np.attr("arange")(a.attr("domain").attr("num_sectors")).attr("__getitem__")(
          py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
        2,
        py::arg("axis") = 1));
    return wrap(make_data(dt, a_data->device, std::move(res_blocks), res_block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::add_trivial_leg(py::object a,
                                   int64 /*legs_pos*/,
                                   bool /*add_to_domain*/,
                                   int64 /*co_domain_pos*/,
                                   TensorProduct::Ptr /*new_codomain*/,
                                   TensorProduct::Ptr /*new_domain*/)
{
    return wrap(data_from_tensor(a));
}

bool
FusionTreeBackend::almost_equal(py::object a, py::object b, float64 rtol, float64 atol)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    py::object iter = misc().attr("iter_common_noncommon_sorted")(
      block_col(a_data->block_inds, 0), block_col(b_data->block_inds, 0));
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
FusionTreeBackend::apply_instructions(py::object tensor,
                                      py::object instructions,
                                      std::vector<int64> codomain_idcs,
                                      std::vector<int64> domain_idcs,
                                      TensorProduct::Ptr new_codomain,
                                      TensorProduct::Ptr new_domain,
                                      bool mixes_codomain_domain)
{
    auto t_data = data_from_tensor(tensor);
    auto codomain = tensor.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain = tensor.attr("domain").cast<TensorProduct::Ptr>();
    auto instructions_vec = instructions_from_python(instructions);

    FusionTreeData::Ptr res;
    if (mixes_codomain_domain) {
        auto mapping = TreePairMapping::from_instructions(
          instructions_vec, codomain, domain, t_data->block_inds);
        res = mapping->transform_tensor(*t_data,
                                        codomain,
                                        domain,
                                        new_codomain,
                                        new_domain,
                                        codomain_idcs,
                                        domain_idcs,
                                        block_backend);
    } else {
        auto mapping = FactorizedTreeMapping::from_instructions(
          instructions_vec, codomain, domain, t_data->block_inds);
        res = mapping->transform_tensor(*t_data,
                                        codomain,
                                        domain,
                                        new_codomain,
                                        new_domain,
                                        codomain_idcs,
                                        domain_idcs,
                                        block_backend);
    }
    res->discard_zero_blocks(block_backend, eps);
    return wrap(res);
}

TensorBackend::DataPtr
FusionTreeBackend::apply_mask_to_DiagonalTensor(py::object tensor, py::object mask)
{
    auto t_data = data_from_tensor(tensor);
    auto m_data = data_from_tensor(mask);
    auto np = numpy();
    py::array tensor_block_inds_contr =
      t_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)));
    py::array mask_block_inds_contr = block_col(m_data->block_inds, 1);
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
        res_bi.push_back(m_data->block_inds.attr("__getitem__")(py::make_tuple(j, 0)).cast<int64>());
    }
    py::array_t<int64> res_block_inds;
    if (!res_bi.empty()) {
        res_block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(res_bi, res_bi)));
    } else {
        res_block_inds = zeros_i64(0, 2);
    }
    return wrap(make_data(tensor.attr("dtype").cast<Dtype>(),
                          t_data->device,
                          std::move(res_blocks),
                          res_block_inds,
                          true));
}

TensorBackend::DataPtr
FusionTreeBackend::combine_legs(py::object tensor,
                                std::vector<std::vector<int64>> /*leg_idcs_combine*/,
                                std::vector<LegPipe::Ptr> /*pipes*/,
                                TensorProduct::Ptr /*new_codomain*/,
                                TensorProduct::Ptr /*new_domain*/)
{
    return wrap(data_from_tensor(tensor));
}

TensorBackend::DataPtr
FusionTreeBackend::compose(py::object a, py::object b)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    Dtype res_dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    auto a_blocks = a_data->blocks;
    auto b_blocks = b_data->blocks;
    if (a_data->dtype != res_dtype)
        for (auto& bl : a_blocks)
            bl = block_backend->to_dtype(bl, res_dtype);
    if (b_data->dtype != res_dtype)
        for (auto& bl : b_blocks)
            bl = block_backend->to_dtype(bl, res_dtype);
    std::vector<BlockBackend::BlockPtr> blocks;
    py::list block_inds_rows;
    if (a_data->blocks.size() > 0 && b_data->blocks.size() > 0) {
        py::object iter = misc().attr("iter_common_sorted")(
          block_col(a_data->block_inds, 1), block_col(b_data->block_inds, 0));
        for (py::handle item : iter) {
            auto pair = item.cast<py::tuple>();
            int64 i = pair[0].cast<int64>();
            int64 j = pair[1].cast<int64>();
            blocks.push_back(block_backend->matrix_dot(a_blocks[static_cast<std::size_t>(i)],
                                                       b_blocks[static_cast<std::size_t>(j)]));
            block_inds_rows.append(py::make_tuple(
              a_data->block_inds.attr("__getitem__")(py::make_tuple(i, 0)).cast<int64>(),
              b_data->block_inds.attr("__getitem__")(py::make_tuple(j, 1)).cast<int64>()));
        }
    }
    py::array block_inds =
      block_inds_rows.size() > 0
        ? numpy().attr("array")(block_inds_rows, py::arg("dtype") = numpy().attr("intp")).cast<py::array>()
        : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(res_dtype, a_data->device, std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::copy_data(py::object a, std::optional<std::string> device)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& block : a_data->blocks)
        blocks.push_back(block_backend->copy_block(block, device));
    std::string dev = device.has_value() ? block_backend->as_device(*device) : a_data->device;
    return wrap(make_data(a_data->dtype,
                          std::move(dev),
                          std::move(blocks),
                          a_data->block_inds.attr("copy")().cast<py::array>(),
                          true));
}

TensorBackend::DataPtr
FusionTreeBackend::dagger(py::object a)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    blocks.reserve(a_data->blocks.size());
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->dagger(b));
    py::array block_inds =
      a_data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(std::nullopt, std::nullopt, -1)));
    return wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(blocks), block_inds));
}

BlockBackend::Scalar
FusionTreeBackend::data_item(DataPtr a)
{
    auto data = unwrap(a);
    if (data->blocks.size() > 1)
        throw std::runtime_error("Inconsistent data.");
    if (data->blocks.empty())
        return block_backend->as_scalar(dtype::zero_scalar(data->dtype), data->dtype);
    return block_backend->item(data->blocks[0]);
}

bool
FusionTreeBackend::diagonal_all(py::object a)
{
    auto data = data_from_tensor(a);
    if (static_cast<int64>(data->blocks.size()) < nsec(a.attr("domain")))
        return false;
    for (auto const& b : data->blocks)
        if (!block_backend->all(b))
            return false;
    return true;
}

bool
FusionTreeBackend::diagonal_any(py::object a)
{
    auto data = data_from_tensor(a);
    for (auto const& b : data->blocks)
        if (block_backend->any(b))
            return true;
    return false;
}


TensorBackend::DataPtr
FusionTreeBackend::diagonal_elementwise_binary(py::object a,
                                               py::object b,
                                               py::function func,
                                               py::dict func_kwargs,
                                               bool partial_zero_is_zero)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto np = numpy();
    std::vector<BlockBackend::BlockPtr> blocks;
    py::array block_inds;
    if (partial_zero_is_zero) {
        py::list rows;
        for (py::handle item : misc().attr("iter_common_sorted")(
               block_col(a_data->block_inds, 0), block_col(b_data->block_inds, 0))) {
            auto pair = item.cast<py::tuple>();
            int64 i = pair[0].cast<int64>();
            int64 j = pair[1].cast<int64>();
            rows.append(a_data->block_inds.attr("__getitem__")(i));
            blocks.push_back(func(py::cast(a_data->blocks[static_cast<std::size_t>(i)]),
                                  py::cast(b_data->blocks[static_cast<std::size_t>(j)]),
                                  **func_kwargs)
                               .cast<BlockBackend::BlockPtr>());
        }
        block_inds = rows.size() > 0 ? np.attr("array")(rows).cast<py::array>() : zeros_i64(0, 2).cast<py::array>();
    } else {
        int64 n_a = 0;
        int64 bi_a = a_data->block_inds.shape(0) == 0 ? -1
                                                      : block_col(a_data->block_inds, 0).attr("__getitem__")(n_a).cast<int64>();
        int64 n_b = 0;
        int64 bi_b = b_data->block_inds.shape(0) == 0 ? -1
                                                      : block_col(b_data->block_inds, 0).attr("__getitem__")(n_b).cast<int64>();
        int64 num_sectors = a.attr("codomain").attr("num_sectors").cast<int64>();
        for (int64 i = 0; i < num_sectors; ++i) {
            BlockBackend::BlockPtr a_block = (i == bi_a)
              ? a_data->blocks[static_cast<std::size_t>(n_a++)]
              : block_backend->zeros({ mults_of(a.attr("domain"))[static_cast<std::size_t>(i)] },
                                     a.attr("dtype").cast<Dtype>());
            if (i == bi_a - 1)
                bi_a = n_a >= a_data->block_inds.shape(0)
                         ? -1
                         : block_col(a_data->block_inds, 0).attr("__getitem__")(n_a).cast<int64>();
            BlockBackend::BlockPtr b_block = (i == bi_b)
              ? b_data->blocks[static_cast<std::size_t>(n_b++)]
              : block_backend->zeros({ mults_of(a.attr("domain"))[static_cast<std::size_t>(i)] },
                                     b.attr("dtype").cast<Dtype>());
            if (i == bi_b - 1)
                bi_b = n_b >= b_data->block_inds.shape(0)
                         ? -1
                         : block_col(b_data->block_inds, 0).attr("__getitem__")(n_b).cast<int64>();
            blocks.push_back(func(py::cast(a_block), py::cast(b_block), **func_kwargs).cast<BlockBackend::BlockPtr>());
        }
        block_inds = asarray_i64(np.attr("repeat")(
          np.attr("arange")(num_sectors).attr("__getitem__")(py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
          2,
          py::arg("axis") = 1));
    }
    Dtype dt = blocks.empty()
      ? block_backend->get_dtype(func(py::cast(block_backend->ones_block({ 1 }, a.attr("dtype").cast<Dtype>())),
                                      py::cast(block_backend->ones_block({ 1 }, b.attr("dtype").cast<Dtype>())),
                                      **func_kwargs)
                                    .cast<BlockBackend::BlockPtr>())
      : block_backend->get_dtype(blocks[0]);
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_elementwise_unary(py::object a,
                                              py::function func,
                                              py::dict func_kwargs,
                                              bool maps_zero_to_zero)
{
    auto a_data = data_from_tensor(a);
    auto np = numpy();
    std::vector<BlockBackend::BlockPtr> blocks;
    py::array block_inds;
    if (maps_zero_to_zero) {
        for (auto const& b : a_data->blocks)
            blocks.push_back(func(py::cast(b), **func_kwargs).cast<BlockBackend::BlockPtr>());
        block_inds = a_data->block_inds;
    } else {
        int64 n = 0;
        int64 bi = a_data->block_inds.shape(0) == 0 ? -1
                                                    : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
        int64 num_sectors = a.attr("codomain").attr("num_sectors").cast<int64>();
        for (int64 i = 0; i < num_sectors; ++i) {
            BlockBackend::BlockPtr block;
            if (i == bi) {
                block = a_data->blocks[static_cast<std::size_t>(n++)];
                bi = n >= a_data->block_inds.shape(0)
                       ? -1
                       : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
            } else {
                block = block_backend->zeros({ tp_mults(a.attr("codomain"))[static_cast<std::size_t>(i)] },
                                               a.attr("dtype").cast<Dtype>());
            }
            blocks.push_back(func(py::cast(block), **func_kwargs).cast<BlockBackend::BlockPtr>());
        }
        block_inds = asarray_i64(np.attr("repeat")(
          np.attr("arange")(num_sectors).attr("__getitem__")(py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
          2,
          py::arg("axis") = 1));
    }
    Dtype dt = blocks.empty()
      ? block_backend->get_dtype(func(py::cast(block_backend->ones_block({ 1 }, a.attr("dtype").cast<Dtype>())),
                                      **func_kwargs)
                                    .cast<BlockBackend::BlockPtr>())
      : block_backend->get_dtype(blocks[0]);
    return wrap(make_data(dt, a_data->device, std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_from_block(BlockBackend::BlockPtr a,
                                       TensorProduct::Ptr co_domain,
                                       float64 tol)
{
    auto leg = co_domain->factors[0];
    Dtype dt = block_backend->get_dtype(a);
    auto np = numpy();
    auto block_inds = asarray_i64(np.attr("repeat")(
      np.attr("arange")(co_domain->num_sectors).attr("__getitem__")(py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
      2,
      py::arg("axis") = 1));
    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t idx = 0; idx < co_domain->sector_decomposition.size(); ++idx) {
        Sector coupled = co_domain->sector_decomposition[idx];
        int64 dim_c = co_domain->symmetry->sector_dim(coupled);
        auto j = leg.attr("sector_decomposition_where")(py::cast(coupled)).cast<int64>();
        auto slc = slice_pair(leg.attr("slices").attr("__getitem__")(j));
        auto entries = block_backend->reshape(b_get(a, py::make_tuple(slc)), { dim_c, co_domain->multiplicities[idx] });
        auto block = block_backend->sum(entries, 0);
        block = (*(block)) / block_backend->as_scalar(static_cast<float64>(dim_c));
        auto projected = block_backend->outer(block_backend->ones_block({ dim_c }, dt), block);
        if (block_backend->norm((*entries) - (*projected)).as_float64()
            > tol * block_backend->norm(entries).as_float64())
            throw std::invalid_argument("Block is not symmetric up to tolerance.");
        blocks.push_back(block);
    }
    return wrap(make_data(dt, block_backend->get_device(a), std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_from_sector_block_func(py::function func, TensorProduct::Ptr co_domain)
{
    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t coupled_idx = 0; coupled_idx < co_domain->sector_decomposition.size(); ++coupled_idx) {
        blocks.push_back(func(py::make_tuple(co_domain->block_size(static_cast<int64>(coupled_idx))),
                              py::cast(co_domain->sector_decomposition[coupled_idx]))
                           .cast<BlockBackend::BlockPtr>());
    }
    auto np = numpy();
    auto block_inds = asarray_i64(np.attr("repeat")(
      np.attr("arange")(co_domain->num_sectors).attr("__getitem__")(py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
      2,
      py::arg("axis") = 1));
    BlockBackend::BlockPtr sample = blocks.empty()
      ? func(py::make_tuple(1), py::cast(co_domain->symmetry->trivial_sector)).cast<BlockBackend::BlockPtr>()
      : blocks[0];
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::diagonal_tensor_from_full_tensor(py::object a, std::optional<float64> tol)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->get_diagonal(b, tol));
    return wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(blocks), a_data->block_inds, true));
}

BlockBackend::Scalar
FusionTreeBackend::diagonal_tensor_trace_full(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto qdims = a.attr("domain").attr("sector_qdims").cast<std::vector<float64>>();
    auto res = block_backend->as_scalar(0.0, a.attr("dtype").cast<Dtype>());
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n)
        res = res + block_backend->as_scalar(
                      qdims[static_cast<std::size_t>(buf(n, 0))]
                        * block_backend->sum_all(a_data->blocks[static_cast<std::size_t>(n)]).as_complex128(),
                      a.attr("dtype").cast<Dtype>());
    return res;
}

BlockBackend::BlockPtr
FusionTreeBackend::diagonal_tensor_to_block(py::object a)
{
    assert(a.attr("symmetry").attr("can_be_dropped").cast<bool>());
    auto a_data = data_from_tensor(a);
    auto res = block_backend->zeros({ a.attr("leg").attr("dim").cast<int64>() }, a.attr("dtype").cast<Dtype>());
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n) {
        int64 bi_cod = buf(n, 0);
        Sector c = a.attr("codomain").attr("sector_decomposition").attr("__getitem__")(bi_cod).cast<Sector>();
        int64 dim_c = a.attr("codomain").attr("sector_dims").attr("__getitem__")(bi_cod).cast<int64>();
        auto entries = block_backend->reshape(
          block_backend->outer(block_backend->ones_block({ dim_c }, a.attr("dtype").cast<Dtype>()),
                               a_data->blocks[static_cast<std::size_t>(n)]),
          std::vector<int64>{ -1 });
        auto j = a.attr("leg").attr("sector_decomposition_where")(py::cast(c)).cast<int64>();
        b_set(res, py::make_tuple(slice_pair(a.attr("leg").attr("slices").attr("__getitem__")(j))), entries);
    }
    return res;
}

std::string
FusionTreeBackend::get_device_from_data(DataPtr a)
{
    return block_backend->as_device(unwrap(a)->device);
}

Dtype
FusionTreeBackend::get_dtype_from_data(DataPtr a)
{
    return unwrap(a)->dtype;
}

bool
FusionTreeBackend::supports_symmetry(Symmetry::Ptr symmetry)
{
    return symmetry != nullptr;
}

TensorBackend::DataPtr
FusionTreeBackend::move_to_device(py::object a, std::string device)
{
    auto a_data = data_from_tensor(a);
    for (auto& b : a_data->blocks)
        b = block_backend->as_block(py::cast(b), std::nullopt, device);
    a_data->device = block_backend->as_device(device);
    return wrap(a_data);
}

TensorBackend::DataPtr
FusionTreeBackend::to_dtype(py::object a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::to_block_backend(DataPtr data,
                                    std::shared_ptr<BlockBackend> bb,
                                    std::optional<Dtype> dtype,
                                    std::optional<std::string> device)
{
    auto d = unwrap(data);
    Dtype dt = dtype.has_value() ? *dtype : d->dtype;
    std::string dev = bb->as_device(device.has_value() ? *device : d->device);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : d->blocks)
        blocks.push_back(bb->as_block(py::cast(b), dt, dev));
    return wrap(make_data(dt, dev, std::move(blocks), d->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::full_data_from_diagonal_tensor(py::object a)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_diagonal(b));
    return wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::full_data_from_mask(py::object a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->block_from_mask(b, dtype));
    return wrap(make_data(dtype, a_data->device, std::move(blocks), a_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::mask_dagger(py::object mask)
{
    auto data = data_from_tensor(mask);
    py::array block_inds =
      data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), py::slice(std::nullopt, std::nullopt, -1)));
    return wrap(make_data(mask.attr("dtype").cast<Dtype>(), mask.attr("device").cast<std::string>(),
                          data->blocks, block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::mask_to_diagonal(py::object a, Dtype dtype)
{
    auto a_data = data_from_tensor(a);
    std::vector<BlockBackend::BlockPtr> blocks;
    for (auto const& b : a_data->blocks)
        blocks.push_back(block_backend->to_dtype(b, dtype));
    py::array large_leg_bi = a.attr("is_projection").cast<bool>() ? block_col(a_data->block_inds, 1)
                                                                  : block_col(a_data->block_inds, 0);
    py::array block_inds = numpy().attr("repeat")(
      large_leg_bi.attr("__getitem__")(py::make_tuple(py::ellipsis(), numpy().attr("newaxis"))), 2, py::arg("axis") = 1);
    return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::split_legs(py::object a,
                              std::vector<int64> /*leg_idcs*/,
                              TensorProduct::Ptr /*new_codomain*/,
                              TensorProduct::Ptr /*new_domain*/)
{
    return wrap(data_from_tensor(a));
}

TensorBackend::DataPtr
FusionTreeBackend::squeeze_legs(py::object a, std::vector<int64> /*idcs*/)
{
    return wrap(data_from_tensor(a));
}

TensorBackend::DataPtr
FusionTreeBackend::from_dense_block_trivial_sector(BlockBackend::BlockPtr /*block*/, Space::Ptr /*leg*/)
{
    throw NotImplemented("from_dense_block_trivial_sector not implemented");
}

TensorBackend::DataPtr
FusionTreeBackend::inv_part_from_dense_block_single_sector(BlockBackend::BlockPtr /*vector*/,
                                                           Space::Ptr /*space*/,
                                                           ElementarySpace::Ptr /*charge_leg*/)
{
    throw NotImplemented("inv_part_from_dense_block_single_sector not implemented");
}

BlockBackend::BlockPtr
FusionTreeBackend::inv_part_to_dense_block_single_sector(py::object /*tensor*/)
{
    throw NotImplemented("inv_part_to_dense_block_single_sector not implemented");
}

py::object
FusionTreeBackend::state_tensor_product(BlockBackend::BlockPtr /*state1*/,
                                        BlockBackend::BlockPtr /*state2*/,
                                        LegPipe::Ptr /*pipe*/)
{
    throw NotImplemented("state_tensor_product not implemented");
}

BlockBackend::BlockPtr
FusionTreeBackend::to_dense_block_trivial_sector(py::object /*tensor*/)
{
    throw NotImplemented("to_dense_block_trivial_sector not implemented");
}

TensorBackend::DataPtr
FusionTreeBackend::zero_diagonal_data(TensorProduct::Ptr /*co_domain*/, Dtype dtype, std::string device)
{
    return wrap(make_data(dtype, std::move(device), {}, zeros_i64(0, 2), true));
}

TensorBackend::DataPtr
FusionTreeBackend::zero_mask_data(Space::Ptr /*large_leg*/, std::string device)
{
    return wrap(make_data(Dtype::Bool, std::move(device), {}, zeros_i64(0, 2), true));
}

TensorBackend::DataPtr
FusionTreeBackend::zero_data(TensorProduct::Ptr codomain,
                             TensorProduct::Ptr domain,
                             Dtype dtype,
                             std::string device,
                             bool all_blocks)
{
    if (!all_blocks)
        return wrap(make_data(dtype, std::move(device), {}, zeros_i64(0, 2), true));
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> zero_blocks;
    for (std::size_t j = 0; j < domain->sector_decomposition.size(); ++j) {
        Sector coupled = domain->sector_decomposition[j];
        auto i = codomain->sector_decomposition_where(coupled);
        if (!i.has_value())
            continue;
        block_inds_rows.append(py::make_tuple(*i, static_cast<int64>(j)));
        zero_blocks.push_back(block_backend->zeros(
          { codomain->block_size(*i), domain->block_size(static_cast<int64>(j)) }, dtype, device));
    }
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(dtype, std::move(device), std::move(zero_blocks), block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::eye_data(TensorProduct::Ptr co_domain, Dtype dtype, std::string device)
{
    std::vector<BlockBackend::BlockPtr> blocks;
    for (int64 c_idx = 0; c_idx < co_domain->num_sectors; ++c_idx)
        blocks.push_back(block_backend->eye_matrix(co_domain->block_size(c_idx), dtype, device));
    auto np = numpy();
    auto block_inds = asarray_i64(np.attr("repeat")(
      np.attr("arange")(co_domain->num_sectors).attr("__getitem__")(py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
      2,
      py::arg("axis") = 1));
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::from_sector_block_func(py::function func,
                                          TensorProduct::Ptr codomain,
                                          TensorProduct::Ptr domain)
{
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    py::object codom_secs = py::cast(codomain->sector_decomposition);
    py::object dom_secs = py::cast(domain->sector_decomposition);
    for (py::handle item : misc().attr("iter_common_sorted_arrays")(codom_secs, dom_secs)) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        block_inds_rows.append(py::make_tuple(i, j));
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
        blocks.push_back(func(py::make_tuple(codomain->block_size(i), domain->block_size(j)), py::cast(coupled))
                           .cast<BlockBackend::BlockPtr>());
    }
    BlockBackend::BlockPtr sample = blocks.empty()
      ? func(py::make_tuple(1, 1), py::cast(codomain->symmetry->trivial_sector)).cast<BlockBackend::BlockPtr>()
      : blocks[0];
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(block_backend->get_dtype(sample),
                          block_backend->get_device(sample),
                          std::move(blocks),
                          block_inds));
}

TensorBackend::DataPtr
FusionTreeBackend::from_random_normal(TensorProduct::Ptr codomain,
                                      TensorProduct::Ptr domain,
                                      float64 sigma,
                                      Dtype dtype,
                                      std::string device)
{
    auto self = this;
    py::function func = py::cpp_function(
      [self, sigma, dtype, device](py::object shape, py::object /*coupled*/) {
          return self->block_backend->random_normal(shape.cast<std::vector<int64>>(), dtype, sigma, device);
      });
    return from_sector_block_func(func, codomain, domain);
}

BlockBackend::Scalar
FusionTreeBackend::inner(py::object a, py::object b, bool do_dagger)
{
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto qdims = a.attr("codomain").attr("sector_qdims").cast<std::vector<float64>>();
    py::array b_inds = do_dagger ? block_col(b_data->block_inds, 0) : block_col(b_data->block_inds, 1);
    auto res = block_backend->as_scalar(0.0, a.attr("dtype").cast<Dtype>());
    for (py::handle item : misc().attr("iter_common_sorted")(block_col(a_data->block_inds, 0), b_inds)) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        int64 bi = block_col(a_data->block_inds, 0).attr("__getitem__")(i).cast<int64>();
        auto inn = block_backend->inner(a_data->blocks[static_cast<std::size_t>(i)],
                                        b_data->blocks[static_cast<std::size_t>(j)],
                                        do_dagger);
        res = res + block_backend->as_scalar(qdims[static_cast<std::size_t>(bi)] * inn.as_complex128(),
                                             a.attr("dtype").cast<Dtype>());
    }
    return res;
}

BlockBackend::Scalar
FusionTreeBackend::trace_full(py::object a, std::vector<int64> /*idcs1*/, std::vector<int64> /*idcs2*/)
{
    auto a_data = data_from_tensor(a);
    auto qdims = a.attr("codomain").attr("sector_qdims").cast<std::vector<float64>>();
    auto res = block_backend->as_scalar(dtype::zero_scalar(a_data->dtype), a_data->dtype);
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n) {
        int64 bi_cod = buf(n, 0);
        res = res + block_backend->as_scalar(
                      qdims[static_cast<std::size_t>(bi_cod)]
                        * block_backend->trace_full(a_data->blocks[static_cast<std::size_t>(n)]).as_complex128(),
                      a_data->dtype);
    }
    return res;
}

BlockBackend::Scalar
FusionTreeBackend::norm(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto qdims = a.attr("codomain").attr("sector_qdims").cast<std::vector<float64>>();
    float64 norm_sq = 0.;
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n) {
        auto nblock = block_backend->norm(a_data->blocks[static_cast<std::size_t>(n)]);
        norm_sq += qdims[static_cast<std::size_t>(buf(n, 0))] * nblock.as_float64() * nblock.as_float64();
    }
    return block_backend->as_scalar(std::sqrt(norm_sq), dtype::to_real(a.attr("dtype").cast<Dtype>()));
}

TensorBackend::DataPtr
FusionTreeBackend::mul(BlockBackend::Scalar a, py::object b)
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
    Dtype dt = blocks.empty()
      ? (dtype::is_real(a.dtype()) ? b_data->dtype : dtype::to_complex(b_data->dtype))
      : block_backend->get_dtype(blocks[0]);
    return wrap(make_data(dt, b_data->device, std::move(blocks), b_data->block_inds, true));
}

TensorBackend::DataPtr
FusionTreeBackend::linear_combination(BlockBackend::Scalar a,
                                      py::object v,
                                      BlockBackend::Scalar b,
                                      py::object w)
{
    auto v_data = data_from_tensor(v);
    auto w_data = data_from_tensor(w);
    Dtype common_dtype = dtype::common({ v.attr("dtype").cast<Dtype>(), w.attr("dtype").cast<Dtype>() });
    auto v_blocks = v_data->blocks;
    auto w_blocks = w_data->blocks;
    if (v_data->dtype != common_dtype)
        for (auto& T : v_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    if (w_data->dtype != common_dtype)
        for (auto& T : w_blocks)
            T = block_backend->to_dtype(T, common_dtype);
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    for (py::handle item : misc().attr("iter_common_noncommon_sorted")(
           block_col(v_data->block_inds, 0), block_col(w_data->block_inds, 0))) {
        auto pair = item.cast<py::tuple>();
        py::object i = pair[0];
        py::object j = pair[1];
        if (i.is_none()) {
            blocks.push_back(block_backend->mul(b, w_blocks[static_cast<std::size_t>(j.cast<int64>())]));
            block_inds_rows.append(w_data->block_inds.attr("__getitem__")(j.cast<int64>()));
        } else if (j.is_none()) {
            blocks.push_back(block_backend->mul(a, v_blocks[static_cast<std::size_t>(i.cast<int64>())]));
            block_inds_rows.append(v_data->block_inds.attr("__getitem__")(i.cast<int64>()));
        } else {
            blocks.push_back(block_backend->linear_combination(
              a, v_blocks[static_cast<std::size_t>(i.cast<int64>())], b, w_blocks[static_cast<std::size_t>(j.cast<int64>())]));
            block_inds_rows.append(v_data->block_inds.attr("__getitem__")(i.cast<int64>()));
        }
    }
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(common_dtype, v_data->device, std::move(blocks), block_inds));
}

BlockBackend::Scalar
FusionTreeBackend::get_element_diagonal(py::object a, int64 idx)
{
    py::object pair = a.attr("leg").attr("parse_index")(idx);
    int64 sector_idx = pair.attr("__getitem__")(0).cast<int64>();
    int64 idx_within = pair.attr("__getitem__")(1).cast<int64>();
    int64 multi = mults_of(a.attr("leg"))[static_cast<std::size_t>(sector_idx)];
    if (a.attr("leg").attr("is_dual").cast<bool>()) {
        Sector sector = a.attr("leg").attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
        sector_idx = a.attr("domain").attr("sector_decomposition_where")(py::cast(sector)).cast<int64>();
    }
    auto block_idx = data_from_tensor(a)->block_ind_from_domain_sector_ind(sector_idx);
    if (!block_idx.has_value())
        return block_backend->as_scalar(dtype::zero_scalar(a.attr("dtype").cast<Dtype>()), a.attr("dtype").cast<Dtype>());
    return block_backend->get_block_element(data_from_tensor(a)->blocks[static_cast<std::size_t>(*block_idx)],
                                            { idx_within % multi });
}

BlockBackend::Scalar
FusionTreeBackend::get_element_mask(py::object a, std::vector<int64> idcs)
{
    auto legs = conventional_leg_order(a);
    auto np = numpy();
    py::list rows;
    for (std::size_t i = 0; i < legs.size(); ++i)
        rows.append(legs[i].attr("parse_index")(idcs[i]));
    auto pos = asarray_i64(np.attr("array")(rows));
    int64 sector_idx = pos.attr("__getitem__")(py::make_tuple(1, 0)).cast<int64>();
    Sector sector = a.attr("domain").attr("__getitem__")(0).attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
    if (sector != a.attr("codomain").attr("__getitem__")(0).attr("sector_decomposition")
                    .attr("__getitem__")(pos.attr("__getitem__")(py::make_tuple(0, 0)).cast<int64>())
                    .cast<Sector>())
        return block_backend->as_scalar(false);
    if (a.attr("domain").attr("__getitem__")(0).attr("is_dual").cast<bool>())
        sector_idx = a.attr("domain").attr("sector_decomposition_where")(py::cast(sector)).cast<int64>();
    auto block_idx = data_from_tensor(a)->block_ind_from_domain_sector_ind(sector_idx);
    if (!block_idx.has_value())
        return block_backend->as_scalar(false);
    int64 small, large, multi;
    if (a.attr("is_projection").cast<bool>()) {
        small = pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)).attr("__getitem__")(0).cast<int64>();
        large = pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)).attr("__getitem__")(1).cast<int64>();
        multi = mults_of(a.attr("small_leg"))[static_cast<std::size_t>(pos.attr("__getitem__")(py::make_tuple(0, 0)).cast<int64>())];
    } else {
        large = pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)).attr("__getitem__")(0).cast<int64>();
        small = pos.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)).attr("__getitem__")(1).cast<int64>();
        multi = mults_of(a.attr("small_leg"))[static_cast<std::size_t>(pos.attr("__getitem__")(py::make_tuple(1, 0)).cast<int64>())];
    }
    return block_backend->get_block_mask_element(
      data_from_tensor(a)->blocks[static_cast<std::size_t>(*block_idx)], large, small, multi);
}

TensorBackend::DataPtr
FusionTreeBackend::permute_legs(py::object a,
                                std::vector<int64> codomain_idcs,
                                std::vector<int64> domain_idcs,
                                TensorProduct::Ptr new_codomain,
                                TensorProduct::Ptr new_domain,
                                bool mixes_codomain_domain,
                                std::vector<std::optional<int64>> levels,
                                std::vector<std::optional<bool>> bend_right)
{
    std::vector<std::optional<int64>> flat_levels;
    std::vector<std::optional<bool>> flat_bend_right;
    py::list codomain_pipe_inds;
    py::list domain_pipe_inds;
    int64 flat_index = 0;
    int64 num_codomain_flat_legs = a.attr("num_codomain_flat_legs").cast<int64>();
    py::list legs = a.attr("legs");
    for (py::ssize_t i = 0; i < py::len(legs); ++i) {
        py::object leg = legs[i];
        bool is_codomain = i < a.attr("num_codomain_legs").cast<int64>();
        if (py::hasattr(leg, "num_flat_legs")) {
            int64 num = leg.attr("num_flat_legs").cast<int64>();
            py::list indices;
            for (int64 k = 0; k < num; ++k)
                indices.append(flat_index + k);
            if (is_codomain)
                codomain_pipe_inds.append(indices);
            else
                domain_pipe_inds.append(indices);
            flat_index += num;
            for (int64 k = 0; k < num; ++k) {
                flat_levels.push_back(levels[static_cast<std::size_t>(i)]);
                flat_bend_right.push_back(bend_right[static_cast<std::size_t>(i)]);
            }
        } else {
            py::list indices = py::make_tuple(flat_index);
            if (is_codomain)
                codomain_pipe_inds.append(indices);
            else
                domain_pipe_inds.append(indices);
            ++flat_index;
            flat_levels.push_back(levels[static_cast<std::size_t>(i)]);
            flat_bend_right.push_back(bend_right[static_cast<std::size_t>(i)]);
        }
    }
    py::list leg_comb;
    for (py::handle x : codomain_pipe_inds)
        leg_comb.append(x);
    for (py::handle x : domain_pipe_inds)
        leg_comb.append(x);
    std::vector<int64> new_domain_idcs;
    for (int64 idx : domain_idcs) {
        py::list rev = leg_comb[static_cast<py::ssize_t>(idx)].cast<py::list>();
        for (py::ssize_t ri = py::len(rev) - 1; ri >= 0; --ri)
            new_domain_idcs.push_back(rev[ri].cast<int64>());
    }
    std::vector<int64> new_codomain_idcs;
    for (int64 idx : codomain_idcs) {
        for (py::handle k : leg_comb[static_cast<py::ssize_t>(idx)])
            new_codomain_idcs.push_back(k.cast<int64>());
    }
    int64 const num_domain_flat_legs = a.attr("num_domain_flat_legs").cast<int64>();
    bool const has_symmetric_braid = a.attr("symmetry").attr("has_symmetric_braid").cast<bool>();
    PermuteLegsInstructionEngine engine(num_codomain_flat_legs,
                                        num_domain_flat_legs,
                                        new_codomain_idcs,
                                        new_domain_idcs,
                                        flat_levels,
                                        flat_bend_right,
                                        has_symmetric_braid);
    auto instructions = engine.evaluate_instructions();
    py::list instructions_py;
    for (Instruction const& inst : instructions) {
        std::visit(
          [&](auto const& i) { instructions_py.append(py::cast(i)); }, inst);
    }
    return apply_instructions(
      a, instructions_py, new_codomain_idcs, new_domain_idcs, new_codomain, new_domain, mixes_codomain_domain);
}

std::tuple<BlockBackend::BlockPtr, int64, int64>
FusionTreeBackend::_get_forest_block_contribution(
  BlockBackend::BlockPtr block,
  Symmetry::Ptr sym,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Sector coupled,
  py::object a_sectors_obj,
  py::object b_sectors_obj,
  std::vector<int64> a_dims,
  std::vector<int64> b_dims,
  int64 tree_block_width,
  int64 tree_block_height,
  int64 i1_init,
  int64 i2_init,
  std::vector<int64> m_mults,
  std::vector<int64> n_mults,
  Dtype dtype) const
{
    SectorArray a_sectors = a_sectors_obj.cast<SectorArray>();
    SectorArray b_sectors = b_sectors_obj.cast<SectorArray>();
    auto codomain_are_dual = flat_are_dual(codomain);
    auto domain_are_dual = flat_are_dual(domain);
    std::vector<int64> shape;
    for (int64 d : a_dims)
        shape.push_back(d);
    for (int64 d : b_dims)
        shape.push_back(d);
    for (int64 m : m_mults)
        shape.push_back(m);
    for (int64 n : n_mults)
        shape.push_back(n);
    auto entries = block_backend->zeros(shape, dtype);
    int64 i1 = i1_init;
    int64 i2 = i2_init;
    fusion_trees alpha_iter(sym, a_sectors, coupled, codomain_are_dual);
    fusion_trees beta_iter(sym, b_sectors, coupled, domain_are_dual);
    BlockBackend* bb = block_backend.get();
    for (auto const& alpha_tree : alpha_iter.all_trees()) {
        auto splitting_tree = block_backend->conj(alpha_tree.to_dense_block(bb, std::nullopt, true));
        for (auto const& beta_tree : beta_iter.all_trees()) {
            auto fusion_tree = beta_tree.to_dense_block(bb, std::nullopt, true);
            auto symmetry_data = block_backend->tdot(splitting_tree, fusion_tree, { -1 }, { -1 });
            py::slice idx1(i1, i1 + tree_block_height, 1);
            py::slice idx2(i2, i2 + tree_block_width, 1);
            auto degeneracy_data = b_get(block, py::make_tuple(idx1, idx2));
            degeneracy_data = block_backend->reshape(degeneracy_data,
                                                   [&]() {
                                                       std::vector<int64> s = m_mults;
                                                       s.insert(s.end(), n_mults.begin(), n_mults.end());
                                                       return s;
                                                   }());
            entries = (*(entries))
                      + (*(block_backend->outer(symmetry_data, degeneracy_data)));
            i2 += tree_block_width;
        }
        i2 = i2_init;
        i1 += tree_block_height;
    }
    return { entries, static_cast<int64>(alpha_iter.size()), static_cast<int64>(beta_iter.size()) };
}

void
FusionTreeBackend::_add_forest_block_entries(
  BlockBackend::BlockPtr block,
  BlockBackend::BlockPtr entries,
  Symmetry::Ptr sym,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Sector coupled,
  float64 dim_c,
  py::object a_sectors_obj,
  py::object b_sectors_obj,
  std::vector<int64> /*a_dims*/,
  std::vector<int64> /*b_dims*/,
  int64 tree_block_width,
  int64 tree_block_height,
  int64 i1_init,
  int64 i2_init,
  std::vector<int64> /*m_mults*/,
  std::vector<int64> /*n_mults*/) const
{
    SectorArray a_sectors = a_sectors_obj.cast<SectorArray>();
    SectorArray b_sectors = b_sectors_obj.cast<SectorArray>();
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    std::vector<int64> range_J;
    for (int64 i = 0; i < J; ++i)
        range_J.push_back(i);
    std::vector<int64> range_K;
    for (int64 i = 0; i < K; ++i)
        range_K.push_back(i);
    std::vector<int64> range_JK;
    for (int64 i = 0; i < J + K; ++i)
        range_JK.push_back(i);
    auto codomain_are_dual = flat_are_dual(codomain);
    auto domain_are_dual = flat_are_dual(domain);
    int64 i1 = i1_init;
    int64 i2 = i2_init;
    fusion_trees alpha_iter(sym, a_sectors, coupled, codomain_are_dual);
    fusion_trees beta_iter(sym, b_sectors, coupled, domain_are_dual);
    BlockBackend* bb = block_backend.get();
    for (auto const& alpha_tree : alpha_iter.all_trees()) {
        auto Y = alpha_tree.to_dense_block(bb, std::nullopt, true);
        auto Y_projected = block_backend->tdot(entries, Y, range_J, range_J);
        for (auto const& beta_tree : beta_iter.all_trees()) {
            auto X = block_backend->conj(beta_tree.to_dense_block(bb, std::nullopt, true));
            auto YX_projected = block_backend->tdot(Y_projected, X, range_K, range_K);
            auto tree_block = block_backend->trace_partial(YX_projected, { -2 }, { -1 }, range_JK);
            tree_block = (*(tree_block)) / block_backend->as_scalar(dim_c);
            auto ms_ns = block_backend->get_shape(tree_block);
            tree_block = block_backend->reshape(tree_block, { prod_int(std::vector<int64>(ms_ns.begin(), ms_ns.begin() + J)),
                                                              prod_int(std::vector<int64>(ms_ns.begin() + J, ms_ns.end())) });
            py::slice idx1(i1, i1 + tree_block_height, 1);
            py::slice idx2(i2, i2 + tree_block_width, 1);
            b_set(block, py::make_tuple(idx1, idx2), tree_block);
            i2 += tree_block_width;
        }
        i2 = i2_init;
        i1 += tree_block_height;
    }
}

TensorBackend::DataPtr
FusionTreeBackend::from_dense_block(BlockBackend::BlockPtr a,
                                    TensorProduct::Ptr codomain,
                                    TensorProduct::Ptr domain,
                                    float64 tol)
{
    if (codomain->has_pipes() || domain->has_pipes()) {
        py::list legs;
        for (auto const& f : codomain->factors)
            legs.append(f);
        for (auto it = domain->factors.rbegin(); it != domain->factors.rend(); ++it)
            legs.append((*it).attr("dual"));
        auto axes_perm = legs_flat_leg_permutation(legs);
        // pipe splitting omitted — dense blocks with pipes need full pipe handling
        (void)axes_perm;
    }
    assert(codomain->symmetry->can_be_dropped());
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    int64 num_legs = J + K;
    std::vector<int64> perm_axes;
    for (int64 i = 0; i < J; ++i)
        perm_axes.push_back(i);
    for (int64 i = K - 1; i >= 0; --i)
        perm_axes.push_back(J + i);
    a = block_backend->permute_axes(a, perm_axes);
    Dtype dt = dtype::common({ block_backend->get_dtype(a),
                               codomain->symmetry->fusion_tensor_dtype.value_or(Dtype::Complex128) });
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    float64 norm_sq_projected = 0.;
    py::object codom_secs = py::cast(codomain->sector_decomposition);
    py::object dom_secs = py::cast(domain->sector_decomposition);
    for (py::handle item : misc().attr("iter_common_sorted_arrays")(codom_secs, dom_secs)) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
        int64 dim_c = codomain->symmetry->sector_dim(coupled);
        auto block = block_backend->zeros({ codomain->multiplicities[static_cast<std::size_t>(i)],
                                            domain->multiplicities[static_cast<std::size_t>(j)] },
                                          dt);
        int64 i1 = 0;
        int64 i2 = 0;
        for (auto const& b_item : domain->iter_uncoupled(true)) {
            auto b_dims = codomain->symmetry->batch_sector_dim(b_item.uncoupled);
            int64 tree_block_width = domain->tree_block_size(b_item.uncoupled);
            for (auto const& a_item : codomain->iter_uncoupled(true)) {
                auto a_dims = codomain->symmetry->batch_sector_dim(a_item.uncoupled);
                int64 tree_block_height = codomain->tree_block_size(a_item.uncoupled);
                py::list j1;
                py::list j2;
                for (auto const& slc : *a_item.slices)
                    j1.append(py::make_tuple(slc.start, slc.stop));
                for (auto const& slc : *b_item.slices)
                    j2.append(py::make_tuple(slc.start, slc.stop));
                auto entries = b_get(a, py::make_tuple(*j1, *j2));
                std::vector<int64> shape(2 * num_legs);
                for (std::size_t si = 0; si < a_dims.size(); ++si) {
                    shape[2 * si] = a_dims[si];
                    shape[2 * si + 1] = a_item.multiplicities[si];
                }
                for (std::size_t si = 0; si < b_dims.size(); ++si) {
                    shape[2 * (J + si)] = b_dims[si];
                    shape[2 * (J + si) + 1] = b_item.multiplicities[si];
                }
                entries = block_backend->reshape(entries, shape);
                std::vector<int64> pperm;
                for (int64 k = 0; k < 2 * num_legs; k += 2)
                    pperm.push_back(k);
                for (int64 k = 1; k < 2 * num_legs; k += 2)
                    pperm.push_back(k);
                entries = block_backend->permute_axes(entries, pperm);
                _add_forest_block_entries(block,
                                          entries,
                                          codomain->symmetry,
                                          codomain,
                                          domain,
                                          coupled,
                                          static_cast<float64>(dim_c),
                                          py::cast(a_item.uncoupled),
                                          py::cast(b_item.uncoupled),
                                          a_dims,
                                          b_dims,
                                          tree_block_width,
                                          tree_block_height,
                                          i1,
                                          i2,
                                          a_item.multiplicities,
                                          b_item.multiplicities);
                i1 += codomain->tree_block_size(a_item.uncoupled);
            }
            i1 = 0;
            i2 += tree_block_width;
        }
        float64 block_norm = block_backend->norm(block, 2.).as_float64();
        if (block_norm <= 1e-14)
            continue;
        block_inds_rows.append(py::make_tuple(i, j));
        blocks.push_back(block);
        norm_sq_projected += dim_c * block_norm * block_norm;
    }
    if (tol != 0.) {
        float64 a_norm_sq = block_backend->norm(a, 2.).as_float64();
        a_norm_sq *= a_norm_sq;
        if (a_norm_sq - norm_sq_projected > tol * tol * a_norm_sq)
            throw std::invalid_argument("Block is not symmetric up to tolerance.");
    }
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(dt, block_backend->get_device(a), std::move(blocks), block_inds));
}

BlockBackend::BlockPtr
FusionTreeBackend::to_dense_block(py::object a)
{
    assert(a.attr("symmetry").attr("can_be_dropped").cast<bool>());
    auto a_data = data_from_tensor(a);
    auto codomain = a.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain = a.attr("domain").cast<TensorProduct::Ptr>();
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    int64 num_legs = J + K;
    Dtype dt = dtype::common({ a_data->dtype,
                               a.attr("symmetry").attr("fusion_tensor_dtype").cast<Dtype>() });
    std::vector<int64> shape;
    for (auto const& leg : codomain->flat_legs())
        shape.push_back(leg->dim);
    for (auto const& leg : domain->flat_legs())
        shape.push_back(leg->dim);
    auto res = block_backend->zeros(shape, dt);
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t row = 0; row < buf.shape(0); ++row) {
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(buf(row, 0))];
        int64 i1 = 0;
        int64 i2 = 0;
        for (auto const& b_item : domain->iter_uncoupled(true)) {
            auto b_dims = a.attr("symmetry").cast<Symmetry::Ptr>()->batch_sector_dim(b_item.uncoupled);
            int64 tree_block_width = domain->tree_block_size(b_item.uncoupled);
            for (auto const& a_item : codomain->iter_uncoupled(true)) {
                auto a_dims = a.attr("symmetry").cast<Symmetry::Ptr>()->batch_sector_dim(a_item.uncoupled);
                int64 tree_block_height = codomain->tree_block_size(a_item.uncoupled);
                auto [entries, num_alpha, num_beta] = _get_forest_block_contribution(
                  a_data->blocks[static_cast<std::size_t>(row)],
                  a.attr("symmetry").cast<Symmetry::Ptr>(),
                  codomain,
                  domain,
                  coupled,
                  py::cast(a_item.uncoupled),
                  py::cast(b_item.uncoupled),
                  a_dims,
                  b_dims,
                  tree_block_width,
                  tree_block_height,
                  i1,
                  i2,
                  a_item.multiplicities,
                  b_item.multiplicities,
                  dt);
                (void)num_alpha;
                (void)num_beta;
                std::vector<int64> pperm;
                for (int64 k = 0; k < num_legs; ++k)
                    pperm.push_back(k);
                for (int64 k = 0; k < num_legs; ++k)
                    pperm.push_back(k + num_legs);
                entries = block_backend->permute_axes(entries, pperm);
                std::vector<int64> rshape;
                for (std::size_t si = 0; si < a_dims.size(); ++si)
                    rshape.push_back(a_dims[si] * a_item.multiplicities[si]);
                for (std::size_t si = 0; si < b_dims.size(); ++si)
                    rshape.push_back(b_dims[si] * b_item.multiplicities[si]);
                entries = block_backend->reshape(entries, rshape);
                py::list j1;
                py::list j2;
                for (auto const& slc : *a_item.slices)
                    j1.append(slice_from_index_slice(slc));
                for (auto const& slc : *b_item.slices)
                    j2.append(slice_from_index_slice(slc));
                b_set_add(res, py::make_tuple(*j1, *j2), entries);
                i1 += tree_block_height;
            }
            i1 = 0;
            i2 += tree_block_width;
        }
    }
    std::vector<int64> final_perm;
    for (int64 i = 0; i < J; ++i)
        final_perm.push_back(i);
    for (int64 i = K - 1; i >= 0; --i)
        final_perm.push_back(J + i);
    res = block_backend->permute_axes(res, final_perm);
    return res;
}

BlockBackend::Scalar
FusionTreeBackend::reduce_DiagonalTensor(py::object tensor, py::function block_func, py::function func)
{
    auto data = data_from_tensor(tensor);
    py::list numbers;
    int64 n = 0;
    int64 bi = data->block_inds.shape(0) == 0 ? -1 : block_col(data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
    int64 num_sectors = tensor.attr("codomain").attr("num_sectors").cast<int64>();
    for (int64 i = 0; i < num_sectors; ++i) {
        BlockBackend::BlockPtr block;
        if (i == bi) {
            block = data->blocks[static_cast<std::size_t>(n++)];
            bi = n >= data->block_inds.shape(0) ? -1
                                                : block_col(data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
        } else {
            block = block_backend->zeros({ mults_of(tensor.attr("codomain"))[static_cast<std::size_t>(i)] },
                                           tensor.attr("dtype").cast<Dtype>());
        }
        numbers.append(block_func(py::cast(block)));
    }
    return func(numbers).cast<BlockBackend::Scalar>();
}

std::tuple<Space::Ptr, TensorBackend::DataPtr>
FusionTreeBackend::diagonal_transpose(py::object tens)
{
    auto perm = tens.attr("symmetry").attr("dual_sectors")(tens.attr("domain").attr("sector_decomposition"))
                  .attr("lexsort_indices")();
    auto inv = misc().attr("inverse_permutation")(perm);
    py::array block_inds = asarray_i64(inv.attr("__getitem__")(tens.attr("data").attr("block_inds")));
    auto data = make_data(tens.attr("dtype").cast<Dtype>(),
                          data_from_tensor(tens)->device,
                          data_from_tensor(tens)->blocks,
                          block_inds,
                          false);
    return { tens.attr("leg").attr("dual").cast<Space::Ptr>(), wrap(data) };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::eigh(py::object a, bool new_leg_dual, std::optional<std::string> sort)
{
    auto a_data = data_from_tensor(a);
    auto new_leg = a.attr("domain").attr("as_ElementarySpace")(new_leg_dual).cast<ElementarySpace::Ptr>();
    std::vector<BlockBackend::BlockPtr> v_blocks;
    std::vector<BlockBackend::BlockPtr> w_blocks;
    int64 n = 0;
    int64 bi = a_data->block_inds.shape(0) == 0 ? -1
                                                : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
    int64 num_sectors = a.attr("codomain").attr("num_sectors").cast<int64>();
    for (int64 i = 0; i < num_sectors; ++i) {
        if (i == bi) {
            auto [vals, vects] = block_backend->eigh(a_data->blocks[static_cast<std::size_t>(n)], sort);
            v_blocks.push_back(vects);
            w_blocks.push_back(vals);
            ++n;
            bi = n >= a_data->block_inds.shape(0) ? -1
                                                  : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
        } else {
            int64 block_size = tp_mults(a.attr("codomain"))[static_cast<std::size_t>(i)];
            v_blocks.push_back(block_backend->eye_matrix(block_size, a.attr("dtype").cast<Dtype>()));
        }
    }
    auto np = numpy();
    auto v_block_inds = asarray_i64(np.attr("repeat")(
      np.attr("arange")(num_sectors).attr("__getitem__")(py::make_tuple(py::ellipsis(), np.attr("newaxis"))),
      2,
      py::arg("axis") = 1));
    auto w_data = make_data(dtype::to_real(a.attr("dtype").cast<Dtype>()),
                            a_data->device,
                            std::move(w_blocks),
                            a_data->block_inds,
                            true);
    auto v_data = make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(v_blocks), v_block_inds);
    return { wrap(w_data), wrap(v_data), new_leg };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
FusionTreeBackend::lq(py::object a, TensorProduct::Ptr new_co_domain)
{
    auto a_data = data_from_tensor(a);
    py::list l_block_inds;
    py::list q_block_inds;
    std::vector<BlockBackend::BlockPtr> l_blocks;
    std::vector<BlockBackend::BlockPtr> q_blocks;
    int64 n = 0;
    int64 bi_cod = a_data->block_inds.shape(0) == 0 ? -1
                                                    : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
    int64 i_new = 0;
    py::object iter = misc().attr("iter_common_sorted_arrays")(
      a.attr("codomain").attr("sector_decomposition"), a.attr("domain").attr("sector_decomposition"));
    for (py::handle item : iter) {
        auto pair = item.cast<py::tuple>();
        int64 i_cod = pair[0].cast<int64>();
        int64 i_dom = pair[1].cast<int64>();
        q_block_inds.append(py::make_tuple(i_new, i_dom));
        if (bi_cod == i_cod) {
            auto [l, q] = block_backend->matrix_lq(a_data->blocks[static_cast<std::size_t>(n)], false);
            l_blocks.push_back(l);
            q_blocks.push_back(q);
            l_block_inds.append(py::make_tuple(i_cod, i_new));
            ++n;
            bi_cod = n >= a_data->block_inds.shape(0) ? -1
                                                      : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
        } else {
            int64 B_dom = mults_of(a.attr("domain"))[static_cast<std::size_t>(i_dom)];
            int64 B_new = mults_of(new_co_domain->factors[0])[static_cast<std::size_t>(i_new)];
            q_blocks.push_back(b_get(block_backend->eye_matrix(B_dom, a.attr("dtype").cast<Dtype>()),
                                     py::make_tuple(py::slice(0, B_new, 1), py::slice(std::nullopt, std::nullopt, 1))));
        }
        ++i_new;
    }
    auto np = numpy();
    py::array l_bi = l_blocks.empty() ? zeros_i64(0, 2).cast<py::array>()
                                      : np.attr("array")(l_block_inds).cast<py::array>();
    py::array q_bi = q_block_inds.size() == 0 ? zeros_i64(0, 2).cast<py::array>()
                                              : np.attr("array")(q_block_inds).cast<py::array>();
    return { wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(l_blocks), l_bi, true)),
             wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(q_blocks), q_bi)) };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr>
FusionTreeBackend::qr(py::object a, TensorProduct::Ptr new_co_domain)
{
    auto a_data = data_from_tensor(a);
    py::list q_block_inds;
    py::list r_block_inds;
    std::vector<BlockBackend::BlockPtr> q_blocks;
    std::vector<BlockBackend::BlockPtr> r_blocks;
    int64 n = 0;
    int64 bi_cod = a_data->block_inds.shape(0) == 0 ? -1
                                                    : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
    int64 i_new = 0;
    for (py::handle item : misc().attr("iter_common_sorted_arrays")(
           a.attr("codomain").attr("sector_decomposition"), a.attr("domain").attr("sector_decomposition"))) {
        auto pair = item.cast<py::tuple>();
        int64 i_cod = pair[0].cast<int64>();
        int64 i_dom = pair[1].cast<int64>();
        q_block_inds.append(py::make_tuple(i_cod, i_new));
        if (bi_cod == i_cod) {
            auto [q, r] = block_backend->matrix_qr(a_data->blocks[static_cast<std::size_t>(n)], false);
            q_blocks.push_back(q);
            r_blocks.push_back(r);
            r_block_inds.append(py::make_tuple(i_new, i_dom));
            ++n;
            bi_cod = n >= a_data->block_inds.shape(0) ? -1
                                                      : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
        } else {
            int64 B_cod = tp_mults(a.attr("codomain"))[static_cast<std::size_t>(i_cod)];
            int64 B_new = mults_of(new_co_domain->factors[0])[static_cast<std::size_t>(i_new)];
            q_blocks.push_back(b_get(block_backend->eye_matrix(B_cod, a.attr("dtype").cast<Dtype>()),
                                     py::make_tuple(py::slice(std::nullopt, std::nullopt, 1), py::slice(0, B_new, 1))));
        }
        ++i_new;
    }
    auto np = numpy();
    py::array q_bi = q_blocks.empty() ? zeros_i64(0, 2).cast<py::array>()
                                      : np.attr("array")(q_block_inds).cast<py::array>();
    py::array r_bi = r_block_inds.size() == 0 ? zeros_i64(0, 2).cast<py::array>()
                                              : np.attr("array")(r_block_inds).cast<py::array>();
    return { wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(q_blocks), q_bi)),
             wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(r_blocks), r_bi)) };
}

std::tuple<TensorBackend::DataPtr, TensorBackend::DataPtr, TensorBackend::DataPtr>
FusionTreeBackend::svd(py::object a,
                       TensorProduct::Ptr new_co_domain,
                       std::optional<std::string> algorithm)
{
    auto a_data = data_from_tensor(a);
    py::list u_block_inds;
    py::list s_block_inds;
    py::list vh_block_inds;
    std::vector<BlockBackend::BlockPtr> u_blocks;
    std::vector<BlockBackend::BlockPtr> s_blocks;
    std::vector<BlockBackend::BlockPtr> vh_blocks;
    int64 n = 0;
    int64 bi_cod = a_data->block_inds.shape(0) == 0 ? -1
                                                    : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
    int64 i_new = 0;
    for (py::handle item : misc().attr("iter_common_sorted_arrays")(
           a.attr("codomain").attr("sector_decomposition"), a.attr("domain").attr("sector_decomposition"))) {
        auto pair = item.cast<py::tuple>();
        int64 i_cod = pair[0].cast<int64>();
        int64 i_dom = pair[1].cast<int64>();
        u_block_inds.append(py::make_tuple(i_cod, i_new));
        vh_block_inds.append(py::make_tuple(i_new, i_dom));
        if (bi_cod == i_cod) {
            auto [u, s, vh] = block_backend->matrix_svd(a_data->blocks[static_cast<std::size_t>(n)], algorithm);
            u_blocks.push_back(u);
            s_blocks.push_back(s);
            vh_blocks.push_back(vh);
            s_block_inds.append(i_new);
            ++n;
            bi_cod = n >= a_data->block_inds.shape(0) ? -1
                                                      : block_col(a_data->block_inds, 0).attr("__getitem__")(n).cast<int64>();
        } else {
            int64 B_cod = tp_mults(a.attr("codomain"))[static_cast<std::size_t>(i_cod)];
            int64 B_dom = mults_of(a.attr("domain"))[static_cast<std::size_t>(i_dom)];
            int64 B_new = mults_of(new_co_domain->factors[0])[static_cast<std::size_t>(i_new)];
            u_blocks.push_back(b_get(block_backend->eye_matrix(B_cod, a.attr("dtype").cast<Dtype>()),
                                     py::make_tuple(py::slice(std::nullopt, std::nullopt, 1), py::slice(0, B_new, 1))));
            vh_blocks.push_back(b_get(block_backend->eye_matrix(B_dom, a.attr("dtype").cast<Dtype>()),
                                      py::make_tuple(py::slice(0, B_new, 1), py::slice(std::nullopt, std::nullopt, 1))));
        }
        ++i_new;
    }
    auto np = numpy();
    auto mk = [&](py::list const& lst) {
        return lst.size() == 0 ? zeros_i64(0, 2).cast<py::array>()
                               : asarray_i64(np.attr("array")(lst, py::arg("dtype") = np.attr("intp")));
    };
    py::list s_rows;
    for (py::handle x : s_block_inds)
        s_rows.append(py::make_tuple(x, x));
    return { wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(u_blocks), mk(u_block_inds))),
             wrap(make_data(dtype::to_real(a.attr("dtype").cast<Dtype>()),
                            a_data->device,
                            std::move(s_blocks),
                            mk(s_rows))),
             wrap(make_data(a.attr("dtype").cast<Dtype>(), a_data->device, std::move(vh_blocks), mk(vh_block_inds))) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr, float64, float64>
FusionTreeBackend::truncate_singular_values(py::object S,
                                            std::optional<int64> chi_max,
                                            int64 chi_min,
                                            float64 degeneracy_tol,
                                            float64 trunc_cut,
                                            float64 svd_min,
                                            bool minimize_error)
{
    py::array S_np = block_backend->to_numpy(diagonal_tensor_to_block(S)).cast<py::array>();
    auto [keep, err, new_norm] =
      _truncate_singular_values_selection(S_np, py::none(), chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min, minimize_error);
    auto keep_block = block_backend->as_block(keep, Dtype::Bool);
    auto [mask_data, small_leg] = mask_from_block(keep_block, S.attr("leg").cast<Space::Ptr>());
    return { mask_data, small_leg, err, new_norm };
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::mask_contract_large_leg(py::object tensor, py::object mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, true);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::mask_contract_small_leg(py::object tensor, py::object mask, int64 leg_idx)
{
    return _mask_contract(tensor, mask, leg_idx, false);
}

std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::_mask_contract(py::object tensor, py::object mask, int64 leg_idx, bool large_leg)
{
    if (tensor.attr("has_pipes").cast<bool>())
        throw NotImplemented("_mask_contract does not support pipes yet");
    py::object parsed = tensor.attr("_parse_leg_idx")(leg_idx);
    bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 co_domain_idx = parsed.attr("__getitem__")(1).cast<int64>();
    if (in_domain)
        assert(mask.attr("is_projection").cast<bool>() != large_leg);
    else
        assert(mask.attr("is_projection").cast<bool>() == large_leg);
    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    TensorProduct::Ptr target_space;
    TensorProduct::Ptr iter_space;
    if (in_domain) {
        codomain = tensor.attr("codomain").cast<TensorProduct::Ptr>();
        auto spaces = codomain->factors;
        (void)spaces;
        domain = tensor.attr("domain").cast<TensorProduct::Ptr>();
        // build target domain with swapped leg — use Python TensorProduct for now
        py::list sp = tensor.attr("domain").attr("factors");
        sp[co_domain_idx] = large_leg ? mask.attr("small_leg") : mask.attr("large_leg");
        target_space = domain = py::cast<TensorProduct::Ptr>(
          py::type::of<TensorProduct>()(sp, py::arg("symmetry") = tensor.attr("symmetry")));
        iter_space = tensor.attr("domain").cast<TensorProduct::Ptr>();
    } else {
        domain = tensor.attr("domain").cast<TensorProduct::Ptr>();
        py::list sp = tensor.attr("codomain").attr("factors");
        sp[co_domain_idx] = large_leg ? mask.attr("small_leg") : mask.attr("large_leg");
        target_space = codomain = py::cast<TensorProduct::Ptr>(
          py::type::of<TensorProduct>()(sp, py::arg("symmetry") = tensor.attr("symmetry")));
        iter_space = tensor.attr("codomain").cast<TensorProduct::Ptr>();
    }
    (void)target_space;
    auto t_data = data_from_tensor(tensor);
    auto m_data = data_from_tensor(mask);
    py::list coupled;
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(t_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        coupled.append(tensor.attr("domain").attr("sector_decomposition").attr("__getitem__")(buf(i, 1)));
    SectorArray coupled_arr = coupled.size() > 0
      ? py::cast<SectorArray>(coupled)
      : tensor.attr("symmetry").attr("empty_sector_array").cast<SectorArray>();
    std::vector<BlockBackend::BlockPtr> res_blocks;
    py::array res_block_inds = t_data->block_inds.attr("copy")().cast<py::array>();
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
        res_blocks.push_back(block_backend->zeros(
          { codomain->block_size(coupled_arr[static_cast<std::size_t>(i)]),
            domain->block_size(coupled_arr[static_cast<std::size_t>(i)]) },
          t_data->dtype));
    for (auto const& fb : iter_space->iter_forest_blocks(coupled_arr)) {
        auto unc = fb.uncoupled;
        auto dom_idx_mask = mask.attr("domain").attr("sector_decomposition_where")(py::cast(unc[static_cast<std::size_t>(co_domain_idx)]));
        if (dom_idx_mask.is_none())
            continue;
        auto j = m_data->block_ind_from_domain_sector_ind(dom_idx_mask.cast<int64>());
        if (!j.has_value())
            continue;
        auto block_slice = in_domain ? b_get(t_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                                               py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                              slice_from_index_slice(fb.slice)))
                                     : b_get(t_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                                               py::make_tuple(slice_from_index_slice(fb.slice),
                                                              py::slice(std::nullopt, std::nullopt, 1)));
        block_slice = block_backend->apply_mask(block_slice, m_data->blocks[static_cast<std::size_t>(*j)],
                                                in_domain ? co_domain_idx : 0);
        auto new_slc = (in_domain ? domain : codomain)->forest_block_slice(unc, coupled_arr[static_cast<std::size_t>(fb.coupled_idx)]);
        if (in_domain)
            b_set(res_blocks[static_cast<std::size_t>(fb.coupled_idx)],
                  py::make_tuple(py::slice(std::nullopt, std::nullopt, 1), slice_from_index_slice(new_slc)),
                  block_slice);
        else
            b_set(res_blocks[static_cast<std::size_t>(fb.coupled_idx)],
                  py::make_tuple(slice_from_index_slice(new_slc), py::slice(std::nullopt, std::nullopt, 1)),
                  block_slice);
    }
    auto res = make_data(tensor.attr("dtype").cast<Dtype>(), t_data->device, std::move(res_blocks), res_block_inds, true);
    res->discard_zero_blocks(block_backend, eps);
    return { wrap(res), codomain, domain };
}

namespace {

void
tree_block_iter(py::object a,
                std::function<void(FusionTree const&,
                                   FusionTree const&,
                                   BlockBackend::BlockPtr const&)> const& fn)
{
    auto a_data = FusionTreeBackend::data_from_tensor(a);
    auto codomain = a.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain = a.attr("domain").cast<TensorProduct::Ptr>();
    auto sym = a.attr("symmetry").cast<Symmetry::Ptr>();
    auto cod_are_dual = flat_are_dual(codomain);
    auto dom_are_dual = flat_are_dual(domain);
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t row = 0; row < buf.shape(0); ++row) {
        int64 bi_cod = buf(row, 0);
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(bi_cod)];
        auto const& block = a_data->blocks[static_cast<std::size_t>(row)];
        int64 i1_forest = 0;
        int64 i2_forest = 0;
        for (auto const& b_item : domain->iter_uncoupled(false)) {
            int64 tree_block_width = prod_int(b_item.multiplicities);
            int64 forest_block_width = 0;
            for (auto const& a_item : codomain->iter_uncoupled(false)) {
                int64 tree_block_height = prod_int(a_item.multiplicities);
                int64 i1 = i1_forest;
                int64 i2 = i2_forest;
                fusion_trees alpha_iter(sym, a_item.uncoupled, coupled, cod_are_dual);
                fusion_trees beta_iter(sym, b_item.uncoupled, coupled, dom_are_dual);
                for (auto const& alpha_tree : alpha_iter.all_trees()) {
                    i2 = i2_forest;
                    for (auto const& beta_tree : beta_iter.all_trees()) {
                        py::slice idx1(i1, i1 + tree_block_height, 1);
                        py::slice idx2(i2, i2 + tree_block_width, 1);
                        fn(alpha_tree, beta_tree, b_get(block, py::make_tuple(idx1, idx2)));
                        i2 += tree_block_width;
                    }
                    i1 += tree_block_height;
                }
                forest_block_width = std::max(forest_block_width, i2 - i2_forest);
                i1_forest += (i1 - i1_forest);
            }
            i1_forest = 0;
            i2_forest += forest_block_width;
        }
    }
}

SectorArray
as_sector_array_from_list(py::list const& sectors)
{
    return sector_utils().attr("as_sector_array")(sectors).cast<SectorArray>();
}

std::tuple<bool, complex128>
partial_trace_helper(FusionTree const& tree, std::vector<int64> const& idcs)
{
    // Native port of cyten.backends.fusion_tree_backend._partial_trace_helper
    Symmetry::Ptr const& sym = tree.symmetry;
    complex128 b_symbols = 1.0;
    for (int64 idx : idcs) {
        auto const i = static_cast<std::size_t>(idx);
        if (tree.uncoupled[i] != sym->dual_sector(tree.uncoupled[i + 1])) {
            return { false, 0.0 };
        }
        Sector left_sec;
        if (idx == 0) {
            left_sec = sym->trivial_sector;
        } else if (idx == 1) {
            left_sec = tree.uncoupled[0];
        } else {
            left_sec = tree.inner_sectors[static_cast<std::size_t>(idx - 2)];
        }
        Sector center_sec =
          (idx == 0) ? tree.uncoupled[0]
                     : tree.inner_sectors[static_cast<std::size_t>(idx - 1)];
        Sector right_sec =
          (static_cast<std::size_t>(idx) < tree.num_inner_edges)
            ? tree.inner_sectors[i]
            : tree.coupled;
        if (left_sec != right_sec) {
            return { false, 0.0 };
        }
        if (idx == 0) {
            // Match Python ``np.all(tree.multiplicities[:2] == [0, 0])``.
            // For a 2-leg tree, multiplicities has length 1; numpy broadcasts so only
            // multiplicities[0] must be 0. Requiring size >= 2 incorrectly rejects these.
            for (std::size_t k = 0; k < std::min<std::size_t>(2, tree.multiplicities.size()); ++k) {
                if (tree.multiplicities[k] != 0)
                    return { false, 0.0 };
            }
        }
        int64 mu = (idx == 0) ? 0 : tree.multiplicities[static_cast<std::size_t>(idx - 1)];
        int64 nu = tree.multiplicities[i];
        FusionSymbol B = sym->b_symbol(left_sec, tree.uncoupled[i], center_sec);
        b_symbols *= std::conj(B.get_complex(static_cast<std::size_t>(mu),
                                             static_cast<std::size_t>(nu)));
        if (tree.are_dual[i]) {
            b_symbols *= static_cast<complex128>(sym->frobenius_schur(tree.uncoupled[i]));
        }
    }
    return { true, b_symbols };
}

} // namespace

TensorBackend::DataPtr
FusionTreeBackend::outer(py::object a, py::object b)
{
    auto a_data = data_from_tensor(a);
    auto new_codomain = TensorProduct::from_partial_products(
      { a.attr("codomain").cast<TensorProduct::Ptr>(), b.attr("codomain").cast<TensorProduct::Ptr>() });
    auto new_domain = TensorProduct::from_partial_products(
      { a.attr("domain").cast<TensorProduct::Ptr>(), b.attr("domain").cast<TensorProduct::Ptr>() });
    Dtype dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    auto new_data = unwrap(zero_data(new_codomain, new_domain, dtype, a_data->device, true));
    tree_block_iter(a, [&](FusionTree const& a_codom_tree,
                           FusionTree const& a_dom_tree,
                           BlockBackend::BlockPtr const& a_tree_block) {
        tree_block_iter(b, [&](FusionTree const& b_codom_tree,
                               FusionTree const& b_dom_tree,
                               BlockBackend::BlockPtr const& b_tree_block) {
            auto new_tree_block = block_backend->outer(a_tree_block, b_tree_block);
            new_tree_block = block_backend->permute_axes(new_tree_block, { 0, 2, 1, 3 });
            new_tree_block = block_backend->combine_legs(new_tree_block, { { 0, 1 }, { 2, 3 } });
            auto new_codom_trees = a_codom_tree.outer(b_codom_tree);
            auto new_dom_trees = a_dom_tree.outer(b_dom_tree);
            for (auto const& [new_dom_tree, dom_amp] : new_dom_trees) {
                auto dom_slc = new_domain->tree_block_slice(new_dom_tree);
                auto block_idx = new_data->block_ind_from_coupled(new_dom_tree.coupled, new_domain);
                if (!block_idx.has_value())
                    continue;
                for (auto const& [new_codom_tree, codom_amp] : new_codom_trees) {
                    if (new_codom_tree.coupled != new_dom_tree.coupled)
                        continue;
                    auto codom_slc = new_codomain->tree_block_slice(new_codom_tree);
                    auto factor = block_backend->as_scalar(std::conj(codom_amp) * dom_amp, dtype);
                    auto cur = b_get(new_data->blocks[static_cast<std::size_t>(*block_idx)],
                                     py::make_tuple(slice_from_index_slice(codom_slc),
                                                    slice_from_index_slice(dom_slc)));
                    b_set(new_data->blocks[static_cast<std::size_t>(*block_idx)],
                          py::make_tuple(slice_from_index_slice(codom_slc),
                                         slice_from_index_slice(dom_slc)),
                          (*cur) + (*block_backend->mul(factor, new_tree_block)));
                }
            }
        });
    });
    new_data->discard_zero_blocks(block_backend, eps);
    return wrap(new_data);
}

TensorBackend::DataPtr
FusionTreeBackend::partial_compose(py::object a,
                                   py::object b,
                                   int64 a_first_leg,
                                   TensorProduct::Ptr new_codomain,
                                   TensorProduct::Ptr new_domain)
{
    Dtype dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    if (a_data->blocks.empty() || b_data->blocks.empty())
        return wrap(make_data(dtype, a_data->device, {}, zeros_i64(0, 2), true));

    SectorArray eff_sectors = SectorArray::empty(b.attr("symmetry").cast<Symmetry::Ptr>()->sector_ind_len);
    auto b_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(b_data->block_inds);
    auto b_buf = b_bi.unchecked<2>();
    for (py::ssize_t r = 0; r < b_buf.shape(0); ++r)
        eff_sectors.push_back(b.attr("domain").attr("sector_decomposition").attr("__getitem__")(b_buf(r, 1)).cast<Sector>());
    auto eff_space = ElementarySpace::from_defining_sectors(b.attr("symmetry").cast<Symmetry::Ptr>(), eff_sectors);

    bool in_domain;
    int64 num_contr_legs;
    int64 leg_idx;
    TensorProduct::Ptr old_space;
    TensorProduct::Ptr new_space;
    TensorProduct::Ptr iter_space;
    int64 a_num_cod = a.attr("num_codomain_legs").cast<int64>();
    int64 a_num_legs = a.attr("num_legs").cast<int64>();
    if (a_first_leg < a_num_cod) {
        in_domain = false;
        num_contr_legs = b.attr("num_domain_legs").cast<int64>();
        leg_idx = a_first_leg;
        old_space = a.attr("codomain").cast<TensorProduct::Ptr>();
        new_space = new_codomain;
        py::list factors;
        auto cod = a.attr("codomain");
        for (int64 i = 0; i < a_first_leg; ++i)
            factors.append(cod.attr("__getitem__")(i));
        factors.append(py::cast(eff_space));
        for (int64 i = a_first_leg + num_contr_legs; i < a_num_cod; ++i)
            factors.append(cod.attr("__getitem__")(i));
        iter_space = std::make_shared<TensorProduct>(factors.cast<std::vector<py::object>>(),
                                                     a.attr("symmetry").cast<Symmetry::Ptr>());
    } else {
        in_domain = true;
        num_contr_legs = b.attr("num_codomain_legs").cast<int64>();
        leg_idx = a_num_legs - a_first_leg - num_contr_legs;
        old_space = a.attr("domain").cast<TensorProduct::Ptr>();
        new_space = new_domain;
        py::list factors;
        auto dom = a.attr("domain");
        int64 n_dom = dom.attr("num_factors").cast<int64>();
        for (int64 i = 0; i < leg_idx; ++i)
            factors.append(dom.attr("__getitem__")(i));
        factors.append(py::cast(eff_space));
        for (int64 i = leg_idx + num_contr_legs; i < n_dom; ++i)
            factors.append(dom.attr("__getitem__")(i));
        iter_space = std::make_shared<TensorProduct>(factors.cast<std::vector<py::object>>(),
                                                     a.attr("symmetry").cast<Symmetry::Ptr>());
    }

    py::list new_block_inds_rows;
    std::vector<BlockBackend::BlockPtr> new_blocks;
    using TrafoMap = std::map<FusionTree, FusionTreeLinearCombination>;
    using DummyKey = std::tuple<Sector, Sector, int64, Sector>;
    std::map<DummyKey, TrafoMap> tree_transformations;

    auto a_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto a_buf = a_bi.unchecked<2>();
    for (py::ssize_t block_ind = 0; block_ind < a_buf.shape(0); ++block_ind) {
        int64 i = a_buf(block_ind, 0);
        Sector coupled = a.attr("codomain").attr("sector_decomposition").attr("__getitem__")(i).cast<Sector>();
        auto new_i = new_codomain->sector_decomposition_where(coupled);
        auto new_j = new_domain->sector_decomposition_where(coupled);
        if (!new_i.has_value() || !new_j.has_value())
            continue;
        auto new_block = block_backend->zeros(
          { new_codomain->multiplicities[*new_i], new_domain->multiplicities[*new_j] }, dtype, a.attr("device").cast<std::string>());
        new_block_inds_rows.append(py::make_tuple(*new_i, *new_j));

        SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
        for (auto const& tb : iter_space->iter_tree_blocks(coupled_arr)) {
            Sector b_coupled = tb.tree.uncoupled[static_cast<std::size_t>(leg_idx)];
            auto b_block_ind = b_data->block_ind_from_coupled(b_coupled, b.attr("domain").cast<TensorProduct::Ptr>());
            if (!b_block_ind.has_value())
                throw std::runtime_error("partial_compose: missing b block");
            std::vector<int64> dm = tb.multiplicities;
            int64 dm0 = prod_int(std::vector<int64>(dm.begin(), dm.begin() + static_cast<std::size_t>(leg_idx)));
            int64 dm2 = prod_int(std::vector<int64>(dm.begin() + static_cast<std::size_t>(leg_idx) + 1, dm.end()));
            std::vector<int64> dummy_mults = { dm0, 1, dm2 };

            DummyKey dummy_key = tb.tree.vertex_labels(std::max<int64>(0, leg_idx - 1));
            if (!tree_transformations.count(dummy_key))
                tree_transformations[dummy_key] = {};

            auto b_cod = b.attr("codomain").cast<TensorProduct::Ptr>();
            auto b_dom = b.attr("domain").cast<TensorProduct::Ptr>();
            SectorArray b_coupled_arr = SectorArray::repeat(b_coupled, 1);
            for (auto const& xb : b_cod->iter_tree_blocks(b_coupled_arr)) {
                TrafoMap& cache = tree_transformations[dummy_key];
                FusionTreeLinearCombination X_b_trafo;
                if (cache.count(xb.tree))
                    X_b_trafo = cache.at(xb.tree);
                else {
                    X_b_trafo = tb.tree.insert_at(leg_idx, xb.tree);
                    cache[xb.tree] = X_b_trafo;
                }
                for (auto const& yb : b_dom->iter_tree_blocks(b_coupled_arr)) {
                    FusionTreeLinearCombination Y_b_trafo;
                    if (cache.count(yb.tree))
                        Y_b_trafo = cache.at(yb.tree);
                    else {
                        Y_b_trafo = tb.tree.insert_at(leg_idx, yb.tree);
                        cache[yb.tree] = Y_b_trafo;
                    }
                    auto b_tree_block = b_get(b_data->blocks[static_cast<std::size_t>(*b_block_ind)],
                                              py::make_tuple(slice_from_index_slice(xb.slice),
                                                             slice_from_index_slice(yb.slice)));
                    auto b_shape = block_backend->get_shape(b_tree_block);

                    if (in_domain) {
                        for (auto const& [old_tree, amp_old_tree] : Y_b_trafo) {
                            auto old_slc = old_space->tree_block_slice(old_tree);
                            auto a_forest = b_get(a_data->blocks[static_cast<std::size_t>(block_ind)],
                                                  py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                                 slice_from_index_slice(old_slc)));
                            a_forest = block_backend->reshape(
                              a_forest, { -1, dummy_mults[0], b_shape[0], dummy_mults[2] });
                            for (auto const& [new_tree, amp_new_tree] : X_b_trafo) {
                                auto new_slc = new_space->tree_block_slice(new_tree);
                                auto contribution = block_backend->tdot(a_forest, b_tree_block, { 2 }, { 0 });
                                contribution = block_backend->permute_axes(contribution, { 0, 1, 3, 2 });
                                contribution = block_backend->reshape(
                                  contribution, { -1, dummy_mults[0] * b_shape[1] * dummy_mults[2] });
                                auto factor = block_backend->as_scalar(amp_old_tree * std::conj(amp_new_tree), dtype);
                                auto cur = b_get(new_block, py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                                           slice_from_index_slice(new_slc)));
                                b_set(new_block,
                                      py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                     slice_from_index_slice(new_slc)),
                                      (*cur) + (*block_backend->mul(factor, contribution)));
                            }
                        }
                    } else {
                        for (auto const& [old_tree, amp_old_tree] : X_b_trafo) {
                            auto old_slc = old_space->tree_block_slice(old_tree);
                            auto a_forest = b_get(a_data->blocks[static_cast<std::size_t>(block_ind)],
                                                  py::make_tuple(slice_from_index_slice(old_slc),
                                                                 py::slice(std::nullopt, std::nullopt, 1)));
                            a_forest = block_backend->reshape(
                              a_forest, { dummy_mults[0], b_shape[1], dummy_mults[2], -1 });
                            for (auto const& [new_tree, amp_new_tree] : Y_b_trafo) {
                                auto new_slc = new_space->tree_block_slice(new_tree);
                                auto contribution = block_backend->tdot(a_forest, b_tree_block, { 1 }, { 1 });
                                contribution = block_backend->permute_axes(contribution, { 0, 3, 1, 2 });
                                contribution = block_backend->reshape(
                                  contribution, { dummy_mults[0] * b_shape[0] * dummy_mults[2], -1 });
                                auto factor = block_backend->as_scalar(std::conj(amp_old_tree) * amp_new_tree, dtype);
                                auto cur = b_get(new_block, py::make_tuple(slice_from_index_slice(new_slc),
                                                                           py::slice(std::nullopt, std::nullopt, 1)));
                                b_set(new_block,
                                      py::make_tuple(slice_from_index_slice(new_slc),
                                                     py::slice(std::nullopt, std::nullopt, 1)),
                                      (*cur) + (*block_backend->mul(factor, contribution)));
                            }
                        }
                    }
                }
            }
        }
        new_blocks.push_back(new_block);
    }

    py::array block_inds = new_block_inds_rows.size() > 0
      ? numpy().attr("array")(new_block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    auto res = make_data(dtype, a_data->device, std::move(new_blocks), block_inds, true);
    res->discard_zero_blocks(block_backend, eps);
    return wrap(res);
}


std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::partial_trace(py::object tensor,
                                 std::vector<std::pair<int64, int64>> pairs,
                                 std::vector<std::optional<int64>> levels)
{
    std::sort(pairs.begin(), pairs.end(), [](auto const& a, auto const& b) {
        if (a.first != b.first)
            return a.first < b.first;
        return a.second < b.second;
    });
    for (auto& p : pairs)
        if (p.first > p.second)
            std::swap(p.first, p.second);

    int64 N = tensor.attr("num_legs").cast<int64>();
    int64 num_codom_legs = tensor.attr("num_codomain_legs").cast<int64>();
    std::vector<int64> idcs1, idcs2;
    for (auto const& [i1, i2] : pairs) {
        idcs1.push_back(i1);
        idcs2.push_back(i2);
    }
    std::set<int64> traced(idcs1.begin(), idcs1.end());
    traced.insert(idcs2.begin(), idcs2.end());
    std::vector<int64> remaining;
    for (int64 n = 0; n < N; ++n)
        if (!traced.count(n))
            remaining.push_back(n);

    auto sym = tensor.attr("symmetry").cast<Symmetry::Ptr>();
    auto codomain_tp = tensor.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain_tp = tensor.attr("domain").cast<TensorProduct::Ptr>();
    py::list new_cod_factors, new_dom_factors;
    for (int64 n = 0; n < codomain_tp->num_factors; ++n) {
        if (traced.count(n))
            continue;
        new_cod_factors.append(tensor.attr("codomain").attr("__getitem__")(n));
    }
    for (int64 n = 0; n < domain_tp->num_factors; ++n) {
        if (traced.count(N - 1 - n))
            continue;
        new_dom_factors.append(tensor.attr("domain").attr("__getitem__")(n));
    }
    auto new_codomain = std::make_shared<TensorProduct>(new_cod_factors.cast<std::vector<py::object>>(), sym);
    auto new_domain = std::make_shared<TensorProduct>(new_dom_factors.cast<std::vector<py::object>>(), sym);

    py::list insert_idcs;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        int64 pos = 0;
        for (int64 r : remaining)
            if (r < pairs[i].first)
                ++pos;
        insert_idcs.append(pos + 2 * static_cast<int64>(i));
    }
    std::vector<int64> idcs = remaining;
    int64 codom_count = num_codom_legs;
    for (std::size_t pi = 0; pi < pairs.size(); ++pi) {
        int64 idx = insert_idcs[pi].cast<int64>();
        idcs.insert(idcs.begin() + idx, pairs[pi].first);
        idcs.insert(idcs.begin() + idx + 1, pairs[pi].second);
        if (pairs[pi].first < num_codom_legs && pairs[pi].second >= num_codom_legs)
            ++codom_count;
    }
    int64 num_dom_legs = static_cast<int64>(idcs.size()) - codom_count;

    for (auto const& pair : pairs) {
        if (pair.first >= static_cast<int64>(levels.size())
            || pair.second >= static_cast<int64>(levels.size()))
            continue;
        auto const& lp0 = levels[static_cast<std::size_t>(pair.first)];
        auto const& lp1 = levels[static_cast<std::size_t>(pair.second)];
        if (!lp0.has_value() || !lp1.has_value())
            continue;
        for (int64 i = 0; i < N; ++i) {
            if (i == pair.first || i == pair.second)
                continue;
            if (i >= static_cast<int64>(levels.size()))
                continue;
            auto const& li = levels[static_cast<std::size_t>(i)];
            if (!li.has_value())
                continue;
            bool l1 = *li < *lp0;
            bool l2 = *li < *lp1;
            if (l1 != l2)
                throw std::invalid_argument("Inconsistent levels for partial_trace");
        }
    }

    std::vector<int64> codomain_idcs(idcs.begin(), idcs.begin() + codom_count);
    std::vector<int64> domain_idcs(idcs.begin() + codom_count, idcs.end());
    std::reverse(domain_idcs.begin(), domain_idcs.end());
    bool mixes = std::any_of(codomain_idcs.begin(), codomain_idcs.end(), [&](int64 i) {
                   return i >= num_codom_legs;
               })
                 || std::any_of(domain_idcs.begin(), domain_idcs.end(), [&](int64 i) {
                      return i < num_codom_legs;
                    });

    TensorProduct::Ptr codom, dom;
    if (mixes) {
        py::list cf, df;
        for (int64 i : codomain_idcs)
            cf.append(tensor.attr("_as_codomain_leg")(i));
        for (int64 i : domain_idcs)
            df.append(tensor.attr("_as_domain_leg")(i));
        codom = std::make_shared<TensorProduct>(cf.cast<std::vector<py::object>>(), sym);
        dom = std::make_shared<TensorProduct>(df.cast<std::vector<py::object>>(), sym);
    } else {
        codom = codomain_tp->permuted(codomain_idcs);
        std::vector<int64> dom_perm;
        for (int64 i : domain_idcs)
            dom_perm.push_back(N - 1 - i);
        dom = domain_tp->permuted(dom_perm);
    }

    std::vector<std::optional<int64>> level_vec = levels;
    if (level_vec.size() < static_cast<std::size_t>(N))
        level_vec.resize(static_cast<std::size_t>(N), std::nullopt);
    std::vector<std::optional<bool>> bend(static_cast<std::size_t>(N), true);
    auto data_ptr = permute_legs(tensor, codomain_idcs, domain_idcs, codom, dom, mixes, level_vec, bend);
    auto data = unwrap(data_ptr);

    py::list coupled_sectors;
    auto d_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(data->block_inds);
    auto d_buf = d_bi.unchecked<2>();
    for (py::ssize_t r = 0; r < d_buf.shape(0); ++r) {
        Sector sector = dom->sector_decomposition[static_cast<std::size_t>(d_buf(r, 1))];
        if (!new_domain->sector_decomposition_where(sector).has_value())
            continue;
        if (!new_codomain->sector_decomposition_where(sector).has_value())
            continue;
        coupled_sectors.append(py::cast(sector));
    }
    auto new_data = unwrap(zero_data(new_codomain, new_domain, data->dtype, data->device, true));
    SectorArray coupled_arr = coupled_sectors.size() > 0
      ? as_sector_array_from_list(coupled_sectors)
      : sym->empty_sector_array;

    std::vector<std::optional<int64>> old_inds, new_inds;
    for (py::handle h : coupled_sectors) {
        Sector c = h.cast<Sector>();
        old_inds.push_back(data->block_ind_from_coupled(c, dom));
        new_inds.push_back(new_data->block_ind_from_coupled(c, new_domain));
    }

    std::vector<int64> codom_unc_idcs, codom_inner_idcs, codom_multi_idcs, codom_tree_idcs;
    for (int64 i = 0; i < codom_count; ++i) {
        int64 const idx = idcs[static_cast<std::size_t>(i)];
        if (std::find(remaining.begin(), remaining.end(), idx) != remaining.end())
            codom_unc_idcs.push_back(i);
        if (std::find(idcs1.begin(), idcs1.end(), idx) != idcs1.end())
            codom_tree_idcs.push_back(i);
    }
    for (std::size_t k = 2; k < codom_unc_idcs.size(); ++k)
        codom_inner_idcs.push_back(codom_unc_idcs[k] - 2);
    for (std::size_t k = 1; k < codom_unc_idcs.size(); ++k)
        codom_multi_idcs.push_back(codom_unc_idcs[k] - 1);

    // Match Python: enumerate idcs[num_codom:] (not the reversed domain_idcs).
    std::vector<int64> dom_unc_idcs, dom_tree_idcs;
    for (int64 i = 0; i < num_dom_legs; ++i) {
        int64 const idx = idcs[static_cast<std::size_t>(codom_count + i)];
        if (std::find(remaining.begin(), remaining.end(), idx) != remaining.end())
            dom_unc_idcs.push_back(num_dom_legs - 1 - i);
        if (std::find(idcs2.begin(), idcs2.end(), idx) != idcs2.end())
            dom_tree_idcs.push_back(num_dom_legs - 1 - i);
    }
    std::reverse(dom_unc_idcs.begin(), dom_unc_idcs.end());
    std::reverse(dom_tree_idcs.begin(), dom_tree_idcs.end());
    std::vector<int64> dom_inner_idcs, dom_multi_idcs;
    for (std::size_t k = 2; k < dom_unc_idcs.size(); ++k)
        dom_inner_idcs.push_back(dom_unc_idcs[k] - 2);
    for (std::size_t k = 1; k < dom_unc_idcs.size(); ++k)
        dom_multi_idcs.push_back(dom_unc_idcs[k] - 1);

    // Python: tr_idcs = idcs[:num_codom] + idcs[num_codom:][::-1] == codomain_idcs + domain_idcs
    std::vector<int64> tr_idcs = codomain_idcs;
    tr_idcs.insert(tr_idcs.end(), domain_idcs.begin(), domain_idcs.end());
    std::vector<int64> tr_idcs1, tr_idcs2, remain_idcs;
    for (std::size_t i = 0; i < tr_idcs.size(); ++i) {
        if (std::find(idcs1.begin(), idcs1.end(), tr_idcs[i]) != idcs1.end())
            tr_idcs1.push_back(static_cast<int64>(i));
        if (std::find(idcs2.begin(), idcs2.end(), tr_idcs[i]) != idcs2.end())
            tr_idcs2.push_back(static_cast<int64>(i));
        if (std::find(remaining.begin(), remaining.end(), tr_idcs[i]) != remaining.end())
            remain_idcs.push_back(static_cast<int64>(i));
    }

    for (auto const& cod_tb : codom->iter_tree_blocks(coupled_arr)) {
        auto [on_diag, factor_codom] = partial_trace_helper(cod_tb.tree, codom_tree_idcs);
        if (!on_diag)
            continue;
        std::vector<std::uint8_t> ad(cod_tb.tree.are_dual.begin(), cod_tb.tree.are_dual.end());
        std::vector<std::uint8_t> sel_ad;
        SectorArray sel_unc = SectorArray::empty(cod_tb.tree.uncoupled.sector_ind_len());
        SectorArray sel_inner = SectorArray::empty(cod_tb.tree.inner_sectors.sector_ind_len());
        std::vector<int64> sel_mult;
        for (int64 ui : codom_unc_idcs) {
            sel_ad.push_back(cod_tb.tree.are_dual[static_cast<std::size_t>(ui)]);
            sel_unc.push_back(cod_tb.tree.uncoupled[static_cast<std::size_t>(ui)]);
        }
        for (int64 ii : codom_inner_idcs)
            sel_inner.push_back(cod_tb.tree.inner_sectors[static_cast<std::size_t>(ii)]);
        for (int64 mi : codom_multi_idcs)
            sel_mult.push_back(cod_tb.tree.multiplicities[static_cast<std::size_t>(mi)]);
        FusionTree new_codom_tree(sym, sel_unc, cod_tb.tree.coupled, sel_ad, sel_inner, sel_mult);
        auto new_codom_slc = new_codomain->tree_block_slice(new_codom_tree);
        auto old_ind = old_inds[static_cast<std::size_t>(cod_tb.coupled_idx)];
        auto new_ind = new_inds[static_cast<std::size_t>(cod_tb.coupled_idx)];
        if (!old_ind.has_value() || !new_ind.has_value())
            continue;

        for (auto const& dom_tb : dom->iter_tree_blocks(SectorArray::repeat(cod_tb.tree.coupled, 1))) {
            auto [on_diag2, factor_dom] = partial_trace_helper(dom_tb.tree, dom_tree_idcs);
            if (!on_diag2)
                continue;
            std::vector<int64> tmp_shape = cod_tb.multiplicities;
            tmp_shape.insert(tmp_shape.end(), dom_tb.multiplicities.begin(), dom_tb.multiplicities.end());
            std::vector<std::uint8_t> dad;
            SectorArray dunc = SectorArray::empty(dom_tb.tree.uncoupled.sector_ind_len());
            SectorArray dinn = SectorArray::empty(dom_tb.tree.inner_sectors.sector_ind_len());
            std::vector<int64> dmult;
            for (int64 ui : dom_unc_idcs) {
                dad.push_back(dom_tb.tree.are_dual[static_cast<std::size_t>(ui)]);
                dunc.push_back(dom_tb.tree.uncoupled[static_cast<std::size_t>(ui)]);
            }
            for (int64 ii : dom_inner_idcs)
                dinn.push_back(dom_tb.tree.inner_sectors[static_cast<std::size_t>(ii)]);
            for (int64 mi : dom_multi_idcs)
                dmult.push_back(dom_tb.tree.multiplicities[static_cast<std::size_t>(mi)]);
            FusionTree new_dom_tree(sym, dunc, dom_tb.tree.coupled, dad, dinn, dmult);
            auto new_dom_slc = new_domain->tree_block_slice(new_dom_tree);

            auto old_block = b_get(data->blocks[static_cast<std::size_t>(*old_ind)],
                                   py::make_tuple(slice_from_index_slice(cod_tb.slice),
                                                  slice_from_index_slice(dom_tb.slice)));
            old_block = block_backend->reshape(old_block, tmp_shape);
            auto contribution = block_backend->trace_partial(old_block, tr_idcs1, tr_idcs2, remain_idcs);
            contribution = block_backend->reshape(
              contribution,
              { new_codom_slc.stop - new_codom_slc.start, new_dom_slc.stop - new_dom_slc.start });
            auto factor = block_backend->as_scalar(factor_codom * std::conj(factor_dom), data->dtype);
            contribution = block_backend->mul(factor, contribution);
            auto cur = b_get(new_data->blocks[static_cast<std::size_t>(*new_ind)],
                             py::make_tuple(slice_from_index_slice(new_codom_slc),
                                            slice_from_index_slice(new_dom_slc)));
            b_set(new_data->blocks[static_cast<std::size_t>(*new_ind)],
                  py::make_tuple(slice_from_index_slice(new_codom_slc),
                                 slice_from_index_slice(new_dom_slc)),
                  (*cur) + (*contribution));
        }
    }
    new_data->discard_zero_blocks(block_backend, eps);

    if (remaining.empty()) {
        Dtype dt = tensor.attr("dtype").cast<Dtype>();
        if (new_data->blocks.empty()) {
            auto s = block_backend->as_scalar(dtype::zero_scalar(dt), dt);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            return { wrap(make_data(dt, data->device, { block }, zeros_i64(1, 0), true)), nullptr, nullptr };
        }
        if (new_data->blocks.size() == 1) {
            auto s = block_backend->item(new_data->blocks[0]);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            return { wrap(make_data(dt, data->device, { block }, zeros_i64(1, 0), true)), nullptr, nullptr };
        }
        throw std::runtime_error("partial_trace: multiple blocks for scalar result");
    }
    return { wrap(new_data), new_codomain, new_domain };
}

TensorBackend::DataPtr
FusionTreeBackend::from_tree_pairs(
  std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Dtype dtype,
  std::string device)
{
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    std::set<std::pair<FusionTree, FusionTree>> pairs_done;
    py::object codom_secs = py::cast(codomain->sector_decomposition);
    py::object dom_secs = py::cast(domain->sector_decomposition);
    for (py::handle item : misc().attr("iter_common_sorted_arrays")(codom_secs, dom_secs)) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
        auto block = block_backend->zeros(
          { codomain->multiplicities[i], domain->multiplicities[j] }, dtype, device);
        bool is_zero_block = true;
        SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
        for (auto const& xb : codomain->iter_tree_blocks(coupled_arr)) {
            for (auto const& yb : domain->iter_tree_blocks(coupled_arr)) {
                std::pair<FusionTree, FusionTree> pr{ xb.tree, yb.tree };
                auto it = trees.find(pr);
                if (it == trees.end())
                    continue;
                auto tree_block = it->second;
                std::vector<int64> expect_shape = xb.multiplicities;
                for (auto it2 = yb.multiplicities.rbegin(); it2 != yb.multiplicities.rend(); ++it2)
                    expect_shape.push_back(*it2);
                assert(block_backend->get_shape(tree_block) == expect_shape);
                std::vector<int64> perm;
                for (int64 p = 0; p < J; ++p)
                    perm.push_back(p);
                for (int64 p = J + K - 1; p >= J; --p)
                    perm.push_back(p);
                tree_block = block_backend->permute_axes(tree_block, perm);
                tree_block = block_backend->reshape(
                  tree_block, { prod_int(xb.multiplicities), prod_int(yb.multiplicities) });
                b_set(block,
                      py::make_tuple(slice_from_index_slice(xb.slice), slice_from_index_slice(yb.slice)),
                      tree_block);
                is_zero_block = false;
                pairs_done.insert(pr);
            }
        }
        if (is_zero_block)
            continue;
        block_inds_rows.append(py::make_tuple(i, j));
        blocks.push_back(block);
    }
    for (auto const& kv : trees)
        if (!pairs_done.count(kv.first))
            throw std::runtime_error("from_tree_pairs: uncovered tree pair");
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, true));
}

BlockBackend::Scalar
FusionTreeBackend::get_element(py::object a, std::vector<int64> idcs)
{
    py::module_::import("warnings").attr("warn")(
      "Accessing individual entries in the FusionTreeBackend is comparably expensive.",
      py::arg("category") = py::module_::import("builtins").attr("UserWarning"),
      py::arg("stacklevel") = 2);
    auto np = numpy();
    py::list flat_idcs;
    py::list legs = a.attr("legs");
    for (py::ssize_t li = 0; li < py::len(legs); ++li) {
        py::object leg = legs[li];
        py::list dims;
        for (py::handle fl : leg.attr("flat_legs"))
            dims.append(fl.attr("dim"));
        auto unr = np.attr("unravel_index")(idcs[static_cast<std::size_t>(li)], dims);
        for (py::handle u : unr)
            flat_idcs.append(u);
    }
    int64 num_cod_legs = a.attr("num_codomain_flat_legs").cast<int64>();
    int64 num_legs = a.attr("num_flat_legs").cast<int64>();
    py::list a_legs;
    for (py::handle l : a.attr("codomain").attr("flat_legs"))
        a_legs.append(l);
    for (py::handle l : a.attr("domain").attr("flat_legs"))
        a_legs.append(l);
    py::list rev_domain;
    for (py::ssize_t i = py::len(flat_idcs) - 1; i >= num_cod_legs; --i)
        rev_domain.append(flat_idcs[i]);
    py::list flat_ordered;
    for (py::ssize_t i = 0; i < num_cod_legs; ++i)
        flat_ordered.append(flat_idcs[i]);
    for (py::handle h : rev_domain)
        flat_ordered.append(h);
    py::list rows;
    for (py::ssize_t i = 0; i < py::len(a_legs); ++i)
        rows.append(a_legs[i].attr("parse_index")(flat_ordered[i]));
    auto pos = asarray_i64(np.attr("array")(rows));
    py::list uncoupled_list;
    for (py::ssize_t i = 0; i < py::len(a_legs); ++i)
        uncoupled_list.append(a_legs[i].attr("sector_decomposition").attr("__getitem__")(
          pos.attr("__getitem__")(py::make_tuple(i, 0))));
    SectorArray uncoupled = as_sector_array_from_list(uncoupled_list);
    SectorArray codom_uncoupled = uncoupled.slice(0, static_cast<std::size_t>(num_cod_legs));
    SectorArray dom_uncoupled = uncoupled.slice(static_cast<std::size_t>(num_cod_legs), uncoupled.size());
    std::vector<int64> mults, dims_vec;
    for (py::ssize_t i = 0; i < py::len(a_legs); ++i) {
        int64 si = pos.attr("__getitem__")(py::make_tuple(i, 0)).cast<int64>();
        mults.push_back(mults_of(a_legs[i])[static_cast<std::size_t>(si)]);
    }
    auto sym = a.attr("symmetry").cast<Symmetry::Ptr>();
    auto batch_dims = sym->batch_sector_dim(uncoupled);
    std::vector<int64> codom_dims(batch_dims.begin(), batch_dims.begin() + num_cod_legs);
    std::vector<int64> dom_dims(batch_dims.begin() + num_cod_legs, batch_dims.end());
    std::vector<int64> codom_mults(mults.begin(), mults.begin() + num_cod_legs);
    std::vector<int64> dom_mults(mults.begin() + num_cod_legs, mults.end());
    std::vector<int64> shape;
    for (std::size_t i = 0; i < batch_dims.size(); ++i)
        shape.push_back(batch_dims[i] * mults[i]);
    auto a_data = data_from_tensor(a);
    Dtype dt = dtype::common({ a_data->dtype, sym->fusion_tensor_dtype.value_or(Dtype::Complex128) });
    auto forest_block = block_backend->zeros(shape, dt, a.attr("device").cast<std::string>());
    auto codomain = a.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain = a.attr("domain").cast<TensorProduct::Ptr>();
    int64 tree_block_height = codomain->tree_block_size(codom_uncoupled);
    int64 tree_block_width = domain->tree_block_size(dom_uncoupled);
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n) {
        int64 bi_cod = buf(n, 0);
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(bi_cod)];
        int64 i1 = codomain->forest_block_slice(codom_uncoupled, coupled).start;
        int64 i2 = domain->forest_block_slice(dom_uncoupled, coupled).start;
        auto [entries, _, __] = _get_forest_block_contribution(
          a_data->blocks[static_cast<std::size_t>(n)], sym, codomain, domain, coupled,
          py::cast(codom_uncoupled), py::cast(dom_uncoupled), codom_dims, dom_dims,
          tree_block_width, tree_block_height, i1, i2, codom_mults, dom_mults, dt);
        std::vector<int64> perm;
        for (int64 i = 0; i < num_legs; ++i) {
            perm.push_back(i);
            perm.push_back(i + num_legs);
        }
        entries = block_backend->permute_axes(entries, perm);
        entries = block_backend->reshape(entries, shape);
        forest_block = (*forest_block) + (*entries);
    }
    py::list idx_within;
    for (py::ssize_t i = 0; i < pos.attr("shape").attr("__getitem__")(0).cast<py::ssize_t>(); ++i)
        idx_within.append(pos.attr("__getitem__")(py::make_tuple(i, 1)));
    return block_backend->get_block_element(forest_block, idx_within.cast<std::vector<int64>>());
}


TensorBackend::DataPtr
FusionTreeBackend::from_grid(std::vector<std::vector<py::object>> grid,
                             TensorProduct::Ptr new_codomain,
                             TensorProduct::Ptr new_domain,
                             std::vector<std::vector<int64>> left_mult_slices,
                             std::vector<std::vector<int64>> right_mult_slices,
                             Dtype dtype,
                             std::string device)
{
    auto data = unwrap(zero_data(new_codomain, new_domain, dtype, device, true));
    auto new_codomain_legs = new_codomain->flat_legs();
    auto new_domain_legs = new_domain->flat_legs();
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid[i].size(); ++j) {
            py::object op = grid[i][j];
            if (op.is_none())
                continue;
            auto op_data = data_from_tensor(op);
            auto op_codomain = op.attr("codomain").cast<TensorProduct::Ptr>();
            auto op_domain = op.attr("domain").cast<TensorProduct::Ptr>();
            py::list op_coupled_list;
            auto op_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(op_data->block_inds);
            auto op_buf = op_bi.unchecked<2>();
            for (py::ssize_t r = 0; r < op_buf.shape(0); ++r)
                op_coupled_list.append(
                  op_domain->sector_decomposition[static_cast<std::size_t>(op_buf(r, 1))]);
            SectorArray op_coupled = as_sector_array_from_list(op_coupled_list);
            for (auto const& fb : op_codomain->iter_forest_blocks(op_coupled)) {
                Sector coupled = op_coupled[static_cast<std::size_t>(fb.coupled_idx)];
                auto block_idx = data->block_ind_from_coupled(coupled, new_domain);
                if (!block_idx.has_value())
                    continue;
                std::vector<int64> op_codom_shape;
                for (std::size_t l = 0; l < fb.uncoupled.size(); ++l)
                    op_codom_shape.push_back(
                      py::cast(op_codomain->flat_legs()[l])
                        .attr("sector_multiplicity")(py::cast(fb.uncoupled[l]))
                        .cast<int64>());
                int64 op_codom0 = op_codom_shape.empty() ? 1 : op_codom_shape[0];
                int64 op_codom_rest = prod_int(std::vector<int64>(op_codom_shape.begin() + 1, op_codom_shape.end()));
                op_codom_shape = { op_codom0, op_codom_rest };
                int64 op_num_codom_trees =
                  (fb.slice.stop - fb.slice.start) / std::max<int64>(1, prod_int(op_codom_shape));
                std::vector<int64> codom_shape = {
                  py::cast(new_codomain_legs[0])
                    .attr("sector_multiplicity")(py::cast(fb.uncoupled[0]))
                    .cast<int64>(),
                  op_codom_rest };
                auto codom_slc = new_codomain->forest_block_slice(fb.uncoupled, coupled);
                int64 num_codom_trees =
                  (codom_slc.stop - codom_slc.start) / std::max<int64>(1, prod_int(codom_shape));
                auto codom_leg_idx = py::cast(new_codomain_legs[0])
                                       .attr("sector_decomposition_where")(py::cast(fb.uncoupled[0]))
                                       .cast<int64>();
                py::slice codom_leg_slc(left_mult_slices[static_cast<std::size_t>(codom_leg_idx)][i],
                                        left_mult_slices[static_cast<std::size_t>(codom_leg_idx)][i + 1],
                                        1);
                for (auto const& fb_dom : op_domain->iter_forest_blocks(SectorArray::repeat(coupled, 1))) {
                    std::vector<int64> op_dom_shape;
                    for (std::size_t l = 0; l < fb_dom.uncoupled.size(); ++l)
                        op_dom_shape.push_back(
                          py::cast(op_domain->flat_legs()[l])
                            .attr("sector_multiplicity")(py::cast(fb_dom.uncoupled[l]))
                            .cast<int64>());
                    int64 op_dom_last = op_dom_shape.empty() ? 1 : op_dom_shape.back();
                    int64 op_dom_rest = prod_int(std::vector<int64>(op_dom_shape.begin(), op_dom_shape.end() - 1));
                    op_dom_shape = { op_dom_rest, op_dom_last };
                    int64 op_num_dom_trees =
                      (fb_dom.slice.stop - fb_dom.slice.start) / std::max<int64>(1, prod_int(op_dom_shape));
                    std::vector<int64> op_new_shape = { op_num_codom_trees, op_codom0, op_codom_rest,
                                                        op_num_dom_trees, op_dom_rest, op_dom_last };
                    std::vector<int64> dom_shape = { op_dom_rest,
                      py::cast(new_domain_legs.back())
                        .attr("sector_multiplicity")(
                          py::cast(fb_dom.uncoupled[fb_dom.uncoupled.size() - 1]))
                        .cast<int64>() };
                    auto dom_slc = new_domain->forest_block_slice(fb_dom.uncoupled, coupled);
                    int64 num_dom_trees =
                      (dom_slc.stop - dom_slc.start) / std::max<int64>(1, prod_int(dom_shape));
                    std::vector<int64> new_shape = { num_codom_trees, codom_shape[0], codom_shape[1],
                                                     num_dom_trees, dom_shape[0], dom_shape[1] };
                    auto dom_leg_idx = py::cast(new_domain_legs.back())
                                         .attr("sector_decomposition_where")(
                                           py::cast(fb_dom.uncoupled[fb_dom.uncoupled.size() - 1]))
                                         .cast<int64>();
                    py::slice dom_leg_slc(right_mult_slices[static_cast<std::size_t>(dom_leg_idx)][j],
                                          right_mult_slices[static_cast<std::size_t>(dom_leg_idx)][j + 1],
                                          1);
                    auto op_block = b_get(op_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                                          py::make_tuple(slice_from_index_slice(fb.slice),
                                                         slice_from_index_slice(fb_dom.slice)));
                    op_block = block_backend->reshape(op_block, op_new_shape);
                    auto block = block_backend->copy_block(
                      b_get(data->blocks[static_cast<std::size_t>(*block_idx)],
                            py::make_tuple(slice_from_index_slice(codom_slc),
                                           slice_from_index_slice(dom_slc))),
                      device);
                    auto final_shape = block_backend->get_shape(block);
                    block = block_backend->reshape(block, new_shape);
                    auto cur = b_get(block, py::make_tuple(py::slice(0, std::nullopt, 1),
                                                           codom_leg_slc,
                                                           py::slice(0, std::nullopt, 1),
                                                           py::slice(0, std::nullopt, 1),
                                                           py::slice(0, std::nullopt, 1),
                                                           dom_leg_slc));
                    b_set(block,
                          py::make_tuple(py::slice(0, std::nullopt, 1),
                                         codom_leg_slc,
                                         py::slice(0, std::nullopt, 1),
                                         py::slice(0, std::nullopt, 1),
                                         py::slice(0, std::nullopt, 1),
                                         dom_leg_slc),
                          (*cur) + (*op_block));
                    block = block_backend->reshape(block, final_shape);
                    b_set(data->blocks[static_cast<std::size_t>(*block_idx)],
                          py::make_tuple(slice_from_index_slice(codom_slc),
                                         slice_from_index_slice(dom_slc)),
                          block);
                }
            }
        }
    }
    data->discard_zero_blocks(block_backend, eps);
    return wrap(data);
}

TensorBackend::DataPtr
FusionTreeBackend::scale_axis(py::object a, py::object b, int64 leg)
{
    Dtype dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    py::object parsed = a.attr("_parse_leg_idx")(leg);
    bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 co_domain_idx = parsed.attr("__getitem__")(1).cast<int64>();
    int64 ax_a = in_domain ? 1 : 0;
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto a_buf = a_bi.unchecked<2>();
    py::array b_block_inds_col = b_data->block_inds.attr("__getitem__")(
      py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)));
    int64 num_cod = a.attr("num_codomain_legs").cast<int64>();
    int64 num_dom = a.attr("domain").attr("num_factors").cast<int64>();
    if ((in_domain && num_dom == 1) || (!in_domain && num_cod == 1)) {
        py::list block_inds_rows;
        std::vector<BlockBackend::BlockPtr> blocks;
        py::array a_contr = in_domain ? a_data->block_inds.attr("__getitem__")(
                                          py::make_tuple(py::ellipsis(), py::slice(1, 2, 1)))
                                      : a_data->block_inds.attr("__getitem__")(
                                          py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)));
        py::array a_open = in_domain ? a_data->block_inds.attr("__getitem__")(
                                       py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)))
                                     : a_data->block_inds.attr("__getitem__")(
                                         py::make_tuple(py::ellipsis(), py::slice(1, 2, 1)));
        if (a_buf.shape(0) > 0 && b_data->block_inds.shape(0) > 0) {
            for (py::handle item : misc().attr("iter_common_sorted")(a_contr, b_block_inds_col)) {
                auto pair = item.cast<py::tuple>();
                int64 n_a = pair[0].cast<int64>();
                int64 n_b = pair[1].cast<int64>();
                blocks.push_back(block_backend->scale_axis(a_data->blocks[static_cast<std::size_t>(n_a)],
                                                           b_data->blocks[static_cast<std::size_t>(n_b)],
                                                           ax_a));
                if (in_domain)
                    block_inds_rows.append(py::make_tuple(a_buf(n_a, 0), n_b));
                else
                    block_inds_rows.append(py::make_tuple(n_b, a_buf(n_a, 1)));
            }
        }
        py::array block_inds = block_inds_rows.size() > 0
          ? numpy().attr("array")(block_inds_rows).cast<py::array>()
          : zeros_i64(0, 2).cast<py::array>();
        return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
    }
    TensorProduct::Ptr iter_space = in_domain ? a.attr("domain").cast<TensorProduct::Ptr>()
                                              : a.attr("codomain").cast<TensorProduct::Ptr>();
    if (a.attr("has_pipes").cast<bool>()) {
        for (int64 i = 0; i < co_domain_idx; ++i)
            co_domain_idx += static_cast<int64>(iter_space->flat_leg_idcs(i).size()) - 1;
        py::list flat_factors;
        for (auto const& fl : iter_space->flat_legs())
            flat_factors.append(py::cast(fl));
        iter_space = std::make_shared<TensorProduct>(
          flat_factors.cast<std::vector<py::object>>(),
          iter_space->symmetry,
          iter_space->sector_decomposition,
          iter_space->multiplicities);
    }
    py::list coupled_list;
    for (py::ssize_t r = 0; r < a_buf.shape(0); ++r)
        coupled_list.append(a.attr("codomain").attr("sector_decomposition").attr("__getitem__")(a_buf(r, 0)));
    SectorArray coupled_sectors = as_sector_array_from_list(coupled_list);
    std::vector<BlockBackend::BlockPtr> blocks;
    py::list block_inds_rows;
    std::map<int64, std::size_t> ind_mapping;
    std::set<int64> domain_inds_seen;
    for (auto const& fb : iter_space->iter_forest_blocks(coupled_sectors)) {
        int64 ind = a_buf(fb.coupled_idx, 1);
        auto ind_b = b_data->block_ind_from_coupled(fb.uncoupled[static_cast<std::size_t>(co_domain_idx)],
                                                    b.attr("domain").cast<TensorProduct::Ptr>());
        if (!ind_b.has_value())
            continue;
        if (!domain_inds_seen.count(ind)) {
            domain_inds_seen.insert(ind);
            ind_mapping[fb.coupled_idx] = blocks.size();
            block_inds_rows.append(py::make_tuple(a_buf(fb.coupled_idx, 0), ind));
            blocks.push_back(block_backend->zeros(block_backend->get_shape(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)]),
                                                  dtype));
        }
        std::vector<int64> reshape;
        for (std::size_t li = 0; li < fb.uncoupled.size(); ++li)
            reshape.push_back(py::cast(iter_space->flat_legs()[li])
                                .attr("sector_multiplicity")(py::cast(fb.uncoupled[li]))
                                .cast<int64>());
        BlockBackend::BlockPtr forest;
        std::vector<int64> initial_shape;
        py::object slcs0, slcs1;
        if (in_domain) {
            forest = b_get(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                           py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                          slice_from_index_slice(fb.slice)));
            initial_shape = block_backend->get_shape(forest);
            std::vector<int64> rshape = { initial_shape[0], -1 };
            rshape.insert(rshape.end(), reshape.begin(), reshape.end());
            forest = block_backend->reshape(forest, rshape);
            slcs0 = py::slice(0, initial_shape[0], 1);
            slcs1 = slice_from_index_slice(fb.slice);
        } else {
            forest = b_get(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                           py::make_tuple(slice_from_index_slice(fb.slice),
                                          py::slice(std::nullopt, std::nullopt, 1)));
            initial_shape = block_backend->get_shape(forest);
            std::vector<int64> rshape = { -1 };
            rshape.insert(rshape.end(), reshape.begin(), reshape.end());
            rshape.push_back(initial_shape[1]);
            forest = block_backend->reshape(forest, rshape);
            slcs0 = slice_from_index_slice(fb.slice);
            slcs1 = py::slice(0, initial_shape[1], 1);
        }
        forest = block_backend->scale_axis(forest,
                                           b_data->blocks[static_cast<std::size_t>(*ind_b)],
                                           ax_a + co_domain_idx + 1);
        forest = block_backend->reshape(forest, initial_shape);
        b_set(blocks[ind_mapping.at(fb.coupled_idx)],
              py::make_tuple(slcs0, slcs1),
              forest);
    }
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::diagonal_to_mask(py::object tens)
{
    auto tens_data = data_from_tensor(tens);
    py::object large_leg = tens.attr("leg");
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> codom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(tens_data->block_inds);
    auto buf = bi.unchecked<2>();
    auto sym = tens.attr("symmetry").cast<Symmetry::Ptr>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n) {
        auto const& diag_block = tens_data->blocks[static_cast<std::size_t>(n)];
        if (!block_backend->any(diag_block))
            continue;
        int64 bii = buf(n, 0);
        Sector sector = tens.attr("codomain").attr("sector_decomposition").attr("__getitem__")(bii).cast<Sector>();
        if (large_leg.attr("is_dual").cast<bool>())
            sector = sym->dual_sector(sector);
        blocks.push_back(diag_block);
        codom_block_inds.push_back(bii);
        sectors_vec.push_back(sector);
        multiplicities.push_back(block_backend->sum_all(diag_block).as_int64());
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(sector);
            py::array mask = numpy().attr("tile")(
              block_backend->to_numpy(diag_block, py::module_::import("builtins").attr("bool")), dim);
            if (large_leg.attr("is_dual").cast<bool>()) {
                auto j = large_leg.attr("sector_decomposition_where")(sector);
                bii = j.cast<int64>();
            }
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bii));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    auto np = numpy();
    SectorArray sectors = sym->empty_sector_array;
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    py::array_t<int64> block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!is_sorted) {
            auto perm = sectors.lexsort_indices();
            sectors = sectors.take(perm);
            std::vector<int64> new_mults, new_cod_inds;
            py::list new_ranks;
            for (std::size_t p : perm) {
                new_mults.push_back(multiplicities[p]);
                new_cod_inds.push_back(codom_block_inds[p]);
                if (!basis_perm.is_none())
                    new_ranks.append(basis_perm_ranks[p]);
            }
            multiplicities = std::move(new_mults);
            codom_block_inds = std::move(new_cod_inds);
            if (!basis_perm.is_none())
                basis_perm_ranks = new_ranks;
        }
        if (!basis_perm.is_none()) {
            auto ranked = misc().attr("rank_data")(np.attr("concatenate")(basis_perm_ranks));
            basis_perm_opt = ranked.cast<std::vector<int64>>();
        }
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(sectors.size()), codom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, tens_data->device, std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      sym, std::move(sectors), multiplicities, large_leg.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_binary_operand(py::object mask1, py::object mask2, py::function func)
{
    py::object large_leg = mask1.attr("large_leg");
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    auto mask1_data = data_from_tensor(mask1);
    auto mask2_data = data_from_tensor(mask2);
    auto mask1_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(mask1_data->block_inds);
    auto mask2_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(mask2_data->block_inds);
    auto m1 = mask1_bi.unchecked<2>();
    auto m2 = mask2_bi.unchecked<2>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    int64 i1 = 0, i2 = 0;
    int64 b1_i1 = m1.shape(0) == 0 ? -1 : m1(0, 1);
    int64 b2_i2 = m2.shape(0) == 0 ? -1 : m2(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto mults = mults_of(large_leg);
    auto sym = mask1.attr("symmetry").cast<Symmetry::Ptr>();
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block1, block2;
        bool block1_found = false, block2_found = false;
        if (is_sorted && static_cast<int64>(sector_idx) == b1_i1) {
            block1 = mask1_data->blocks[static_cast<std::size_t>(i1)];
            ++i1;
            b1_i1 = i1 >= m1.shape(0) ? -1 : m1(i1, 1);
            block1_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
            auto dom_idx = mask1.attr("domain").attr("sector_decomposition_where")(dual_sec);
            if (!dom_idx.is_none()) {
                auto idx = mask1_data->block_ind_from_domain_sector_ind(dom_idx.cast<int64>());
                if (idx.has_value()) {
                    block1 = mask1_data->blocks[static_cast<std::size_t>(*idx)];
                    block1_found = true;
                }
            }
        }
        if (!block1_found)
            block1 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        if (is_sorted && static_cast<int64>(sector_idx) == b2_i2) {
            block2 = mask2_data->blocks[static_cast<std::size_t>(i2)];
            ++i2;
            b2_i2 = i2 >= m2.shape(0) ? -1 : m2(i2, 1);
            block2_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
            auto dom_idx = mask2.attr("domain").attr("sector_decomposition_where")(dual_sec);
            if (!dom_idx.is_none()) {
                auto idx = mask2_data->block_ind_from_domain_sector_ind(dom_idx.cast<int64>());
                if (idx.has_value()) {
                    block2 = mask2_data->blocks[static_cast<std::size_t>(*idx)];
                    block2_found = true;
                }
            }
        }
        if (!block2_found)
            block2 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        auto new_block = func(py::cast(block1), py::cast(block2)).cast<BlockBackend::BlockPtr>();
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        dom_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(defining[sector_idx]);
            py::array mask = numpy().attr("tile")(
              block_backend->to_numpy(new_block, py::module_::import("builtins").attr("bool")), dim);
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    auto np = numpy();
    SectorArray sectors = sym->empty_sector_array;
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
          py::make_tuple(np.attr("arange")(sectors.size()), dom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, mask1.attr("device").cast<std::string>(), std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      sym, std::move(sectors), multiplicities, large_leg.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_from_block(BlockBackend::BlockPtr a, Space::Ptr large_leg)
{
    py::object large_leg_obj = py::cast(large_leg);
    py::object basis_perm = large_leg_obj.attr("_basis_perm");
    bool is_sorted = !large_leg_obj.attr("is_dual").cast<bool>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto defining = large_leg_obj.attr("defining_sectors").cast<SectorArray>();
    auto slices = large_leg_obj.attr("slices");
    py::object domain;
    if (!is_sorted) {
        auto perm = large_leg_obj.attr("sector_decomposition").attr("lexsort_indices")();
        auto sorted_duals = large_leg_obj.attr("sector_decomposition").attr("__getitem__")(perm);
        auto multis = large_leg_obj.attr("multiplicities").attr("__getitem__")(perm);
        domain = py::module_::import("cyten.symmetries.spaces")
                   .attr("TensorProduct")(
                     py::make_tuple(large_leg_obj), py::arg("symmetry") = large_leg->symmetry,
                     py::arg("_sector_decomposition") = sorted_duals,
                     py::arg("_multiplicities") = multis);
    }
    for (std::size_t bi_large = 0; bi_large < defining.size(); ++bi_large) {
        auto slc = slice_pair(slices.attr("__getitem__")(static_cast<int64>(bi_large)));
        auto block = b_get(a, slc);
        int64 mult = block_backend->sum_all(block).as_int64();
        if (mult == 0)
            continue;
        Sector sector = defining[bi_large];
        int64 bi_out = static_cast<int64>(bi_large);
        if (!is_sorted) {
            Sector dual_sector = large_leg->symmetry->dual_sector(sector);
            bi_out = domain.attr("sector_decomposition_where")(py::cast(dual_sector)).cast<int64>();
        }
        dom_block_inds.push_back(bi_out);
        sectors_vec.push_back(sector);
        int64 dim = large_leg->symmetry->sector_dim(sector);
        int64 stop = block_backend->get_shape(block)[0] / dim;
        blocks.push_back(b_get(block, py::slice(0, stop, 1)));
        multiplicities.push_back(mult / dim);
        if (!basis_perm.is_none()) {
            py::array mask = block_backend->to_numpy(block).cast<py::array>();
            basis_perm_ranks.append(
              large_leg_obj.attr("basis_perm").attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    auto np = numpy();
    SectorArray sectors = large_leg->symmetry->empty_sector_array;
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
        if (!is_sorted) {
            auto perm = np.attr("argsort")(dom_block_inds);
            py::list new_dom, new_blocks;
            std::vector<int64> new_mults;
            for (py::handle p : perm) {
                int64 pi = p.cast<int64>();
                new_dom.append(dom_block_inds[static_cast<std::size_t>(pi)]);
                new_blocks.append(blocks[static_cast<std::size_t>(pi)]);
                new_mults.push_back(multiplicities[static_cast<std::size_t>(pi)]);
            }
            dom_block_inds.clear();
            for (py::handle h : new_dom)
                dom_block_inds.push_back(h.cast<int64>());
            blocks.clear();
            for (py::handle h : new_blocks)
                blocks.push_back(h.cast<BlockBackend::BlockPtr>());
            multiplicities = std::move(new_mults);
        }
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(sectors.size()), dom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, block_backend->get_device(a), std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      large_leg->symmetry, std::move(sectors), multiplicities,
      large_leg_obj.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}

BlockBackend::BlockPtr
FusionTreeBackend::mask_to_block(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto large_leg = a.attr("large_leg");
    auto res = block_backend->zeros({ large_leg.attr("dim").cast<int64>() }, Dtype::Bool);
    bool is_projection = a.attr("is_projection").cast<bool>();
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    auto co_dom = is_projection ? a.attr("domain") : a.attr("codomain");
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        int64 bi_large = is_projection ? buf(i, 1) : buf(i, 0);
        Sector sector = co_dom.attr("sector_decomposition").attr("__getitem__")(bi_large).cast<Sector>();
        int64 dim = co_dom.attr("symmetry").attr("sector_dim")(sector).cast<int64>();
        if (large_leg.attr("is_dual").cast<bool>())
            bi_large = large_leg.attr("sector_decomposition_where")(sector).cast<int64>();
        auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bi_large));
        b_set(res, slc, block_backend->tile(a_data->blocks[static_cast<std::size_t>(i)], dim));
    }
    return res;
}

std::tuple<Space::Ptr, Space::Ptr, TensorBackend::DataPtr>
FusionTreeBackend::mask_transpose(py::object tens)
{
    auto data = data_from_tensor(tens);
    auto sym = tens.attr("symmetry").cast<Symmetry::Ptr>();
    auto perm_dom = sym->dual_sectors(tens.attr("domain").attr("sector_decomposition").cast<SectorArray>())
                      .lexsort_indices();
    auto perm_codom = sym->dual_sectors(tens.attr("codomain").attr("sector_decomposition").cast<SectorArray>())
                        .lexsort_indices();
    auto inv_dom = misc().attr("inverse_permutation")(py::cast(perm_dom));
    auto inv_codom = misc().attr("inverse_permutation")(py::cast(perm_codom));
    auto np = numpy();
    py::array col1 = inv_dom.attr("__getitem__")(
      data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)));
    py::array col0 = inv_codom.attr("__getitem__")(
      data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0)));
    auto block_inds = asarray_i64(np.attr("column_stack")(py::make_tuple(col1, col0)));
    auto out = make_data(tens.attr("dtype").cast<Dtype>(), data->device, data->blocks, block_inds, false);
    return { tens.attr("codomain").attr("__getitem__")(0).attr("dual").cast<Space::Ptr>(),
             tens.attr("domain").attr("__getitem__")(0).attr("dual").cast<Space::Ptr>(),
             wrap(out) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_unary_operand(py::object mask, py::function func)
{
    py::object large_leg = mask.attr("large_leg");
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    auto mask_data = data_from_tensor(mask);
    auto mask_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(mask_data->block_inds);
    auto mbuf = mask_bi.unchecked<2>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    int64 i = 0;
    int64 b_i = mbuf.shape(0) == 0 ? -1 : mbuf(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto mults = mults_of(large_leg);
    auto sym = mask.attr("symmetry").cast<Symmetry::Ptr>();
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block;
        bool block_found = false;
        if (is_sorted && static_cast<int64>(sector_idx) == b_i) {
            block = mask_data->blocks[static_cast<std::size_t>(i)];
            ++i;
            b_i = i >= mbuf.shape(0) ? -1 : mbuf(i, 1);
            block_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
            auto idx = mask_data->block_ind_from_coupled(dual_sec, mask.attr("domain").cast<TensorProduct::Ptr>());
            if (idx.has_value()) {
                block = mask_data->blocks[static_cast<std::size_t>(*idx)];
                block_found = true;
            }
        }
        if (!block_found)
            block = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        auto new_block = func(py::cast(block)).cast<BlockBackend::BlockPtr>();
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        dom_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(defining[sector_idx]);
            py::array mask_np = numpy().attr("tile")(
              block_backend->to_numpy(new_block, py::module_::import("builtins").attr("bool")), dim);
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask_np));
        }
    }
    auto np = numpy();
    SectorArray sectors = sym->empty_sector_array;
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
          py::make_tuple(np.attr("arange")(sectors.size()), dom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, mask.attr("device").cast<std::string>(), std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      sym, std::move(sectors), multiplicities, large_leg.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}

} // namespace cyten
