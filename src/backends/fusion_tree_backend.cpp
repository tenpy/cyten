#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/backends/fusion_tree_mapping.h>
#include <cyten/backends/fusion_tree_permute.h>

#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/trees.h>
#include <cyten/tools.h>

#include <algorithm>
#include <cmath>
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

py::module_
ft_py()
{
    return py::module_::import("cyten.backends.fusion_tree_backend");
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

#include "_ft_native_methods.inc"

} // namespace cyten
