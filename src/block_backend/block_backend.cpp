#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/block_backend/numpy.h>
#include <cyten/cyten.h>
#include <cyten/tools.h>

#include <algorithm>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace cyten {

// Product of elements in [first, last)
int64
prod_range(const std::vector<int64>& shape, size_t first, size_t last)
{
    int64 p = 1;
    for (size_t i = first; i < last; ++i)
        p *= shape[i];
    return p;
}

BlockBackend::BlockBackend(std::string default_device)
  : default_device(std::move(default_device))
{
}

BlockBackend*
BlockBackend::from_factory(std::string /* device */)
{
    throw NotImplemented(
      "from_factory needs to be called on a subclass (e.g. NumpyBlockBackend::from_factory)");
}

std::string
BlockBackend::get_backend_name() const
{
    return "BlockBackend";
}

BlockPtr
BlockBackend::Block::operator+(const BlockCPtr& other) const
{
    return get_backend()->linear_combination(Scalar(1.0), shared_from_this(), Scalar(1.0), other);
}

BlockPtr
BlockBackend::Block::operator*(Scalar s) const
{
    return get_backend()->mul(s.to_numpy(), shared_from_this());
}

std::shared_ptr<BlockBackend::Block>
BlockBackend::Block::operator[](py::object key)
{
    return get_item(key);
}
std::shared_ptr<const BlockBackend::Block>
BlockBackend::Block::operator[](py::object key) const
{
    return get_item(key);
}

BlockBackend::Block&
BlockBackend::Block::operator=(py::object rhs)
{
    set_item(py::slice(py::none(), py::none(), py::none()), rhs);
    return *this;
}

BlockPtr
BlockBackend::apply_basis_perm(const BlockCPtr& block,
                               const std::vector<py::object>& legs,
                               bool inv)
{
    std::vector<py::array_t<int64>> perms;
    perms.reserve(legs.size());
    for (const py::object& leg : legs) {
        py::object perm = inv ? leg.attr("inverse_basis_perm") : leg.attr("basis_perm");
        perms.push_back(py::array_t<int64>::ensure(perm));
    }
    return apply_leg_permutations(block, perms);
}

BlockPtr
BlockBackend::argsort(const BlockCPtr& block, std::optional<std::string> sort, int64 axis)
{
    BlockCPtr work = block;
    if (sort) {
        if (*sort == "m<" || *sort == "SM") {
            work = abs(block);
        } else if (*sort == "m>" || *sort == "LM") {
            work = mul(py::float_(-1.0), abs(block));
        } else if (*sort == "<" || *sort == "SR" || *sort == "SA") {
            work = real(block);
        } else if (*sort == ">" || *sort == "LR" || *sort == "LA") {
            work = mul(py::float_(-1.0), real(block));
        } else if (*sort == "SI") {
            work = imag(block);
        } else if (*sort == "LI") {
            work = mul(py::float_(-1.0), imag(block));
        } else {
            throw std::invalid_argument(std::string("unknown sort option ") + *sort);
        }
    }
    return _argsort(work, axis);
}

BlockPtr
BlockBackend::combine_legs(const BlockCPtr& a,
                           const std::vector<std::vector<int64>>& leg_idcs_combine,
                           const std::vector<bool>& cstyles_in)
{
    std::vector<bool> cstyles = cstyles_in;
    if (cstyles.size() == 1u)
        cstyles.resize(leg_idcs_combine.size(), cstyles[0]);

    std::vector<int64> const old_shape = get_shape(a);
    size_t const ndim = old_shape.size();
    std::vector<int64> axes_perm(ndim);
    std::iota(axes_perm.begin(), axes_perm.end(), int64(0));

    std::vector<int64> new_shape;
    size_t last_stop = 0;

    for (size_t g = 0; g < leg_idcs_combine.size(); ++g) {
        const std::vector<int64>& group = leg_idcs_combine[g];
        int64 const start = group.front();
        int64 const stop = group.back() + 1;

        if (start < static_cast<int64>(last_stop))
            throw std::invalid_argument("The groups in leg_idcs_combine must not overlap");
        for (size_t i = 0; i < group.size(); ++i)
            if (group[i] != static_cast<int64>(start + i))
                throw std::invalid_argument(
                  "Each group in leg_idcs_combine must be contiguous and ascending");

        for (size_t i = last_stop; i < static_cast<size_t>(start); ++i)
            new_shape.push_back(old_shape[i]);

        int64 combined = 1;
        for (int64 i = start; i < stop; ++i)
            combined *= old_shape[i];
        new_shape.push_back(combined);

        if (!cstyles[g])
            std::reverse(axes_perm.begin() + start, axes_perm.begin() + stop);

        last_stop = static_cast<size_t>(stop);
    }
    for (size_t i = last_stop; i < ndim; ++i)
        new_shape.push_back(old_shape[i]);

    return reshape(permute_axes(a, axes_perm), new_shape);
}

BlockPtr
BlockBackend::combine_legs(const BlockCPtr& a,
                           const std::vector<std::vector<int64>>& leg_idcs_combine,
                           bool cstyles)
{
    return combine_legs(a, leg_idcs_combine, std::vector<bool>(1, cstyles));
}

BlockPtr
BlockBackend::dagger(const BlockCPtr& a)
{
    std::vector<int64> const sh = get_shape(a);
    int64 const num_legs = static_cast<int64>(sh.size());
    std::vector<int64> rev(num_legs);
    for (int64 i = 0; i < num_legs; ++i)
        rev[i] = num_legs - 1 - i;
    return conj(permute_axes(a, rev));
}

bool
BlockBackend::is_real(const BlockCPtr& a)
{
    return dtype::is_real(get_dtype(a));
}

BlockPtr
BlockBackend::permute_combined_matrix(const BlockCPtr& block,
                                      const std::vector<int64>& dims1,
                                      const std::vector<int64>& idcs1,
                                      const std::vector<int64>& dims2,
                                      const std::vector<int64>& idcs2)
{
    std::vector<int64> shape = dims1;
    shape.insert(shape.end(), dims2.begin(), dims2.end());
    BlockPtr b = reshape(block, shape);

    // idcs1 and idcs2 are absolute indices into [0..ndim-1] (same as Python)
    std::vector<int64> perm;
    perm.reserve(idcs1.size() + idcs2.size());
    for (int64 i : idcs1)
        perm.push_back(i);
    for (int64 i : idcs2)
        perm.push_back(i);
    b = permute_axes(b, perm);

    std::vector<int64> const sh = get_shape(b);
    size_t const n1 = idcs1.size();
    int64 M_new = 1;
    for (size_t i = 0; i < n1; ++i)
        M_new *= sh[i];
    int64 N_new = 1;
    for (size_t i = n1; i < sh.size(); ++i)
        N_new *= sh[i];
    return reshape(b, { M_new, N_new });
}

BlockPtr
BlockBackend::permute_combined_idx(const BlockCPtr& block,
                                   int64 axis,
                                   const std::vector<int64>& dims,
                                   const std::vector<int64>& idcs)
{
    std::vector<int64> const sh = get_shape(block);
    if (sh.size() != 2)
        throw std::runtime_error("permute_combined_idx: block must be 2D");
    int64 const M = sh[0];
    int64 const N = sh[1];
    int64 ax = axis;
    if (ax < 0)
        ax += 2;

    if (ax == 0) {
        std::vector<int64> new_shape = dims;
        new_shape.push_back(N);
        BlockPtr b = reshape(block, new_shape);
        std::vector<int64> perm;
        for (int64 i : idcs)
            perm.push_back(i);
        perm.push_back(static_cast<int64>(idcs.size()));
        b = permute_axes(b, perm);
        return reshape(b, { M, N });
    }
    if (ax == 1) {
        std::vector<int64> new_shape = { M };
        new_shape.insert(new_shape.end(), dims.begin(), dims.end());
        BlockPtr b = reshape(block, new_shape);
        std::vector<int64> perm = { 0 };
        for (int64 i : idcs)
            perm.push_back(1 + i);
        b = permute_axes(b, perm);
        return reshape(b, { M, N });
    }
    throw std::runtime_error("permute_combined_idx: invalid axis");
}

BlockPtr
BlockBackend::split_legs(const BlockCPtr& a,
                         const std::vector<int64>& idcs,
                         const std::vector<std::vector<int64>>& dims,
                         const std::vector<bool>& cstyles)
{
    if (idcs.size() != dims.size() || idcs.size() != cstyles.size())
        throw std::invalid_argument("idcs, dims, and cstyles must have the same length");

    std::vector<int64> const old_shape = get_shape(a);
    size_t new_ndim = old_shape.size();
    for (auto& dim : dims)
        new_ndim += dim.size();
    std::vector<int64> new_shape;
    std::vector<int64> axes_perm;
    new_shape.reserve(new_ndim);
    axes_perm.reserve(new_ndim);

    size_t start_old_shape = 0;
    for (size_t g = 0; g < idcs.size(); ++g) {
        size_t stop_old_shape = static_cast<size_t>(idcs[g]);
        for (size_t k = start_old_shape; k < stop_old_shape; ++k)
            new_shape.push_back(old_shape[k]);
        for (size_t j = axes_perm.size(); j < new_shape.size(); ++j)
            axes_perm.push_back(static_cast<int64>(j));
        // now insert the split dimensions instead of the old dimensions
        size_t replace_dims = 1;
        for (int64 d : dims[g]) {
            new_shape.push_back(d);
            replace_dims *= d;
        }
        if (old_shape[stop_old_shape] != replace_dims)
            throw std::invalid_argument(
              "The dimensions of the split legs do not match the old dimensions");

        size_t n_axes_now = axes_perm.size();
        size_t n_axes_after = new_shape.size();
        if (cstyles[g]) {
            for (size_t j = n_axes_now; j < n_axes_after; ++j)
                axes_perm.push_back(static_cast<int64>(j));
        } else {
            // reverse the order of the axes
            for (size_t j = n_axes_after; j > n_axes_now; --j)
                axes_perm.push_back(static_cast<int64>(j - 1));
        }
        start_old_shape = stop_old_shape + 1;
    }
    size_t stop_old_shape = new_ndim;
    for (size_t k = start_old_shape; k < stop_old_shape; ++k)
        new_shape.push_back(old_shape[k]);
    for (size_t j = axes_perm.size(); j < new_shape.size(); ++j)
        axes_perm.push_back(static_cast<int64>(j));

    return permute_axes(reshape(a, new_shape), axes_perm);
}

BlockPtr
BlockBackend::split_legs(const BlockCPtr& a,
                         const std::vector<int64>& idcs,
                         const std::vector<std::vector<int64>>& dims,
                         bool cstyles)
{
    return split_legs(a, idcs, dims, std::vector<bool>(1, cstyles));
}

BlockPtr
BlockBackend::tensor_outer(const BlockCPtr& a, const BlockCPtr& b, int64 K)
{
    BlockPtr res = outer(a, b);
    std::vector<int64> const sh_a = get_shape(a);
    std::vector<int64> const sh_b = get_shape(b);
    int64 const N = static_cast<int64>(sh_a.size());
    int64 const M = static_cast<int64>(sh_b.size());

    std::vector<int64> perm;
    for (int64 i = 0; i < K; ++i)
        perm.push_back(i);
    for (int64 i = 0; i < M; ++i)
        perm.push_back(N + i);
    for (int64 i = K; i < N; ++i)
        perm.push_back(i);
    return permute_axes(res, perm);
}

BlockPtr
BlockBackend::eye_block(const std::vector<int64>& legs,
                        Dtype dtype,
                        std::optional<std::string> device)
{
    int64 dim = 1;
    for (int64 d : legs)
        dim *= d;
    BlockPtr eye = eye_matrix(static_cast<int64>(dim), dtype, device);
    std::vector<int64> shape = legs;
    shape.insert(shape.end(), legs.begin(), legs.end());
    eye = reshape(eye, shape);
    int64 const J = static_cast<int64>(legs.size());
    std::vector<int64> perm;
    for (int64 i = 0; i < J; ++i)
        perm.push_back(i);
    for (int64 i = J - 1; i >= 0; --i)
        perm.push_back(J + i);
    return permute_axes(eye, perm);
}

std::tuple<BlockPtr, BlockPtr>
BlockBackend::matrix_lq(const BlockCPtr& a, bool full)
{
    std::vector<int64> perm = { 1, 0 };
    BlockPtr at = permute_axes(a, perm);
    auto [q, r] = matrix_qr(at, full);
    return { permute_axes(r, perm), permute_axes(q, perm) };
}

void
BlockBackend::synchronize()
{
}

void
BlockBackend::test_block_sanity(const BlockCPtr& block,
                                std::optional<std::vector<int64>> expect_shape,
                                std::optional<Dtype> expect_dtype,
                                std::optional<std::string> expect_device)
{
    if (!is_correct_block_type(block)) {
        throw std::runtime_error("wrong block type");
    }
    if (expect_shape) {
        std::vector<int64> const got = get_shape(block);
        if (got != *expect_shape) {
            std::ostringstream msg;
            msg << "wrong block shape ";
            msg << "(";
            for (size_t i = 0; i < got.size(); ++i)
                msg << (i ? ", " : "") << got[i];
            msg << ") != (";
            for (size_t i = 0; i < expect_shape->size(); ++i)
                msg << (i ? ", " : "") << (*expect_shape)[i];
            msg << ")";
            throw std::runtime_error(msg.str());
        }
    }
    if (expect_dtype) {
        if (get_dtype(block) != *expect_dtype) {
            throw std::runtime_error("wrong block dtype");
        }
    }
    if (expect_device) {
        if (get_device(block) != *expect_device) {
            throw std::runtime_error("wrong block device");
        }
    }
}

complex128
BlockBackend::inner(const BlockCPtr& a, const BlockCPtr& b, bool do_dagger)
{
    BlockCPtr ac;
    if (do_dagger) {
        ac = conj(a);
    } else {
        std::vector<int64> const sh = get_shape(a);
        std::vector<int64> rev(sh.size());
        for (size_t i = 0; i < sh.size(); ++i)
            rev[i] = static_cast<int64>(sh.size() - 1 - i);
        ac = permute_axes(a, rev);
    }
    return sum_all(multiply_blocks(ac, b));
}

void
BlockBackend::save_hdf5(py::object hdf5_saver, py::object h5gr, const std::string& subpath)
{
    hdf5_saver.attr("save")(default_device, subpath + std::string("default_device"));
}

std::shared_ptr<BlockBackend>
BlockBackend::from_hdf5(py::object hdf5_loader, py::object h5gr, const std::string& subpath)
{
    throw NotImplemented(
      "Needs to be implemented in Subclass, since we don't know the subclass type here!");
}

} // namespace cyten
