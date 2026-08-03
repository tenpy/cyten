#include <cyten/symmetries/symmetry.h>

#include <cyten/symmetries/sector_numpy.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cyten {

namespace {

py::module_
numpy()
{
    return py::module_::import("numpy");
}

} // namespace

FusionStyle
Symmetry::max_fusion_style(std::vector<SymmetryFactor::Ptr> const& factors)
{
    if (factors.empty()) {
        return FusionStyle::single;
    }
    auto style = factors[0]->fusion_style;
    for (std::size_t i = 1; i < factors.size(); ++i) {
        if (factors[i]->fusion_style > style) {
            style = factors[i]->fusion_style;
        }
    }
    return style;
}

BraidingStyle
Symmetry::max_braiding_style(std::vector<SymmetryFactor::Ptr> const& factors)
{
    if (factors.empty()) {
        return BraidingStyle::bosonic;
    }
    auto style = factors[0]->braiding_style;
    for (std::size_t i = 1; i < factors.size(); ++i) {
        if (factors[i]->braiding_style > style) {
            style = factors[i]->braiding_style;
        }
    }
    return style;
}

Sector
Symmetry::concat_trivial_sectors(std::vector<SymmetryFactor::Ptr> const& factors)
{
    std::size_t len = 0;
    for (auto const& f : factors) {
        len += f->sector_ind_len;
    }
    if (len > max_sector_ind_len) {
        throw std::invalid_argument("Symmetry: product sector_ind_len exceeds max_sector_ind_len");
    }
    Sector out;
    out.len = static_cast<std::uint8_t>(len);
    std::size_t off = 0;
    for (auto const& f : factors) {
        for (std::uint8_t j = 0; j < f->sector_ind_len; ++j) {
            out.q[off++] = f->trivial_sector.q[j];
        }
    }
    return out;
}

float64
Symmetry::prod_num_sectors(std::vector<SymmetryFactor::Ptr> const& factors)
{
    float64 prod = 1.0;
    for (auto const& f : factors) {
        prod *= f->num_sectors;
        if (!std::isfinite(prod)) {
            return std::numeric_limits<float64>::infinity();
        }
    }
    return prod;
}

Symmetry::Symmetry(std::vector<SymmetryFactor::Ptr> factors_in)
  : BaseSymmetry(max_fusion_style(factors_in),
                 max_braiding_style(factors_in),
                 concat_trivial_sectors(factors_in),
                 prod_num_sectors(factors_in),
                 false,
                 true)
  , factors(std::move(factors_in))
{
    // Flatten nested Symmetry factors (should already be SymmetryFactor::Ptr only).
    for (auto const& f : factors) {
        if (!f) {
            throw std::invalid_argument("Symmetry: null factor");
        }
        if (dynamic_cast<Symmetry const*>(f.get()) != nullptr) {
            throw std::invalid_argument("Symmetry: nested Symmetry factors must be flattened before construct");
        }
    }

    sector_slices.assign(1, 0);
    for (auto const& f : factors) {
        auto next = static_cast<std::uint8_t>(sector_slices.back() + f->sector_ind_len);
        sector_slices.push_back(next);
    }

    has_complex_topological_data =
      std::any_of(factors.begin(), factors.end(), [](auto const& f) {
          return f->has_complex_topological_data;
      });
    trivial_shift =
      std::all_of(factors.begin(), factors.end(), [](auto const& f) { return f->trivial_shift; });

    std::vector<Dtype> dtypes;
    dtypes.reserve(factors.size());
    bool any_none = false;
    for (auto const& f : factors) {
        // Python subclasses often set fusion_tensor_dtype as a class attribute; the C++
        // optional member may still be empty. Read via the Python object.
        py::object dt_py = py::cast(f).attr("fusion_tensor_dtype");
        if (dt_py.is_none()) {
            any_none = true;
            break;
        }
        dtypes.push_back(dt_py.cast<Dtype>());
    }
    if (any_none || factors.empty()) {
        fusion_tensor_dtype = std::nullopt;
    } else {
        fusion_tensor_dtype = dtype::common(dtypes);
    }

    // Multiple fermionic factors: warn via Python warnings module.
    int num_fermionic = 0;
    for (auto const& f : factors) {
        // Type check via Python isinstance once bindings exist; use group_name heuristic + RTTI later.
        auto const& name = f->group_name;
        if (name.find("Fermion") != std::string::npos) {
            ++num_fermionic;
        }
    }
    if (num_fermionic > 1) {
        auto warnings = py::module_::import("warnings");
        warnings.attr("warn")(
          "Symmetry with multiple fermionic factors probably does not do what you "
          "expect. See docstring of FermionParity for details.",
          py::arg("stacklevel") = 2);
    }
}

Sector
Symmetry::factor_sector(Sector const& a, std::size_t i) const
{
    auto const begin = sector_slices[i];
    auto const end = sector_slices[i + 1];
    Sector out;
    out.len = static_cast<std::uint8_t>(end - begin);
    for (std::uint8_t j = 0; j < out.len; ++j) {
        out.q[j] = a.q[begin + j];
    }
    return out;
}

SectorArray
Symmetry::factor_sectors(SectorArray const& a, std::size_t i) const
{
    auto const begin = sector_slices[i];
    auto const end = sector_slices[i + 1];
    auto const flen = static_cast<std::uint8_t>(end - begin);
    SectorArray out(a.num_sectors, flen);
    for (std::size_t r = 0; r < a.num_sectors; ++r) {
        for (std::uint8_t j = 0; j < flen; ++j) {
            out.data[r * flen + j] = a.data[r * a.sector_ind_len + begin + j];
        }
    }
    return out;
}

std::size_t
Symmetry::factor_where(std::string const& descriptive_name) const
{
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (factors[i]->descriptive_name && *factors[i]->descriptive_name == descriptive_name) {
            return i;
        }
    }
    throw std::invalid_argument("Name not found: " + descriptive_name);
}

bool
Symmetry::has_factor(SymmetryFactor const& other) const
{
    for (auto const& f : factors) {
        if (f->equals(other)) {
            return true;
        }
    }
    return false;
}

bool
Symmetry::is_equivalent_to(Symmetry const& other, bool strict_ordering) const
{
    if (num_factors() != other.num_factors()) {
        return false;
    }
    if (strict_ordering) {
        for (std::size_t i = 0; i < factors.size(); ++i) {
            if (!factors[i]->_is_equivalent_factor(*other.factors[i])) {
                return false;
            }
        }
        return true;
    }
    std::vector<bool> matched(other.factors.size(), false);
    for (auto const& f1 : factors) {
        bool found = false;
        for (std::size_t i = 0; i < other.factors.size(); ++i) {
            if (matched[i]) {
                continue;
            }
            if (f1->_is_equivalent_factor(*other.factors[i])) {
                matched[i] = true;
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

py::object
Symmetry::as_Symmetry()
{
    return py::cast(std::static_pointer_cast<Symmetry>(shared_from_this()));
}

bool
Symmetry::is_valid_sector(Sector a) const
{
    if (a.len != sector_ind_len) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!factors[i]->is_valid_sector(factor_sector(a, i))) {
            return false;
        }
    }
    return true;
}

bool
Symmetry::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len != sector_ind_len) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!factors[i]->are_valid_sectors(factor_sectors(sectors, i))) {
            return false;
        }
    }
    return true;
}

SectorArray
Symmetry::fusion_outcomes(Sector a, Sector b) const
{
    auto np = numpy();
    std::vector<py::array> all_outcomes;
    std::vector<ssize_t> num_possibilities;
    all_outcomes.reserve(factors.size());
    num_possibilities.reserve(factors.size());

    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto c_i = factors[i]->fusion_outcomes(factor_sector(a, i), factor_sector(b, i));
        all_outcomes.push_back(sector_array_to_numpy(c_i));
        num_possibilities.push_back(static_cast<ssize_t>(c_i.num_sectors));
    }

    if (factors.empty()) {
        return SectorArray::empty(0);
    }

    py::list shape_list;
    for (auto n : num_possibilities) {
        shape_list.append(n);
    }
    shape_list.append(static_cast<int>(sector_ind_len));
    auto zeros = np.attr("zeros");
    auto result = zeros(py::tuple(shape_list), py::arg("dtype") = np.attr("int64"));

    py::object colon = py::slice(py::none(), py::none(), py::none());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        py::list res_idx;
        for (std::size_t k = 0; k < factors.size(); ++k) {
            res_idx.append(colon);
        }
        res_idx.append(py::slice(static_cast<int>(sector_slices[i]),
                                 static_cast<int>(sector_slices[i + 1]),
                                 1));
        py::list c_i_idx;
        for (std::size_t k = 0; k < i; ++k) {
            c_i_idx.append(py::none());
        }
        c_i_idx.append(colon);
        for (std::size_t k = i + 1; k < factors.size(); ++k) {
            c_i_idx.append(py::none());
        }
        c_i_idx.append(colon);
        result[py::tuple(res_idx)] = all_outcomes[i][py::tuple(c_i_idx)];
    }

    ssize_t n_rows = 1;
    for (auto n : num_possibilities) {
        n_rows *= n;
    }
    auto reshaped =
      result.attr("reshape")(py::make_tuple(n_rows, static_cast<int>(sector_ind_len)));
    return sector_array_from_numpy(reshaped);
}

SectorArray
Symmetry::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    if (!is_abelian()) {
        PyErr_SetString(PyExc_AssertionError, "fusion_outcomes_broadcast requires an abelian symmetry");
        throw py::error_already_set();
    }
    std::vector<SectorArray> components;
    components.reserve(factors.size());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        components.push_back(
          factors[i]->fusion_outcomes_broadcast(factor_sectors(a, i), factor_sectors(b, i)));
    }
    // concatenate along last axis
    SectorArray out(a.num_sectors, sector_ind_len);
    for (std::size_t r = 0; r < a.num_sectors; ++r) {
        std::size_t off = 0;
        for (std::size_t i = 0; i < factors.size(); ++i) {
            auto const flen = components[i].sector_ind_len;
            for (std::uint8_t j = 0; j < flen; ++j) {
                out.data[r * sector_ind_len + off + j] =
                  components[i].data[r * flen + j];
            }
            off += flen;
        }
    }
    return out;
}

SectorArray
Symmetry::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    std::vector<SectorArray> components;
    components.reserve(factors.size());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        std::vector<SectorArray> sectors_i;
        sectors_i.reserve(sectors.size());
        for (auto const& s : sectors) {
            sectors_i.push_back(factor_sectors(s, i));
        }
        components.push_back(factors[i]->_multiple_fusion_broadcast(sectors_i));
    }
    auto const n = components.empty() ? 0 : components[0].num_sectors;
    SectorArray out(n, sector_ind_len);
    for (std::size_t r = 0; r < n; ++r) {
        std::size_t off = 0;
        for (std::size_t i = 0; i < factors.size(); ++i) {
            auto const flen = components[i].sector_ind_len;
            for (std::uint8_t j = 0; j < flen; ++j) {
                out.data[r * sector_ind_len + off + j] =
                  components[i].data[r * flen + j];
            }
            off += flen;
        }
    }
    return out;
}

Sector
Symmetry::dual_sector(Sector a) const
{
    Sector res;
    res.len = a.len;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto d = factors[i]->dual_sector(factor_sector(a, i));
        auto const begin = sector_slices[i];
        for (std::uint8_t j = 0; j < d.len; ++j) {
            res.q[begin + j] = d.q[j];
        }
    }
    return res;
}

SectorArray
Symmetry::dual_sectors(SectorArray const& sectors) const
{
    SectorArray res(sectors.num_sectors, sectors.sector_ind_len);
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto d = factors[i]->dual_sectors(factor_sectors(sectors, i));
        auto const begin = sector_slices[i];
        auto const flen = d.sector_ind_len;
        for (std::size_t r = 0; r < sectors.num_sectors; ++r) {
            for (std::uint8_t j = 0; j < flen; ++j) {
                res.data[r * sector_ind_len + begin + j] = d.data[r * flen + j];
            }
        }
    }
    return res;
}

int64
Symmetry::_n_symbol(Sector a, Sector b, Sector c) const
{
    if (has_unique_fusion()) {
        return 1;
    }
    int64 res = 1;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        res *= factors[i]->_n_symbol(factor_sector(a, i), factor_sector(b, i), factor_sector(c, i));
    }
    return res;
}

SectorArray
Symmetry::all_sectors() const
{
    if (!std::isfinite(num_sectors)) {
        throw SymmetryError("symmetry has infinitely many sectors.");
    }
    if (factors.empty()) {
        return SectorArray::empty(0);
    }

    auto np = numpy();
    py::list shape_list;
    for (auto const& f : factors) {
        shape_list.append(static_cast<long long>(f->num_sectors));
    }
    shape_list.append(static_cast<int>(sector_ind_len));
    auto results = np.attr("zeros")(py::tuple(shape_list), py::arg("dtype") = np.attr("int64"));

    py::object colon = py::slice(py::none(), py::none(), py::none());
    for (std::size_t i = 0; i < factors.size(); ++i) {
        py::list lhs_idx;
        for (std::size_t k = 0; k < factors.size(); ++k) {
            lhs_idx.append(colon);
        }
        lhs_idx.append(py::slice(static_cast<int>(sector_slices[i]),
                                 static_cast<int>(sector_slices[i + 1]),
                                 1));
        py::list rhs_idx;
        for (std::size_t k = 0; k < i; ++k) {
            rhs_idx.append(py::none());
        }
        rhs_idx.append(colon);
        for (std::size_t k = i + 1; k < factors.size(); ++k) {
            rhs_idx.append(py::none());
        }
        rhs_idx.append(colon);
        auto secs = sector_array_to_numpy(factors[i]->all_sectors());
        results[py::tuple(lhs_idx)] = secs[py::tuple(rhs_idx)];
    }

    long long n_rows = 1;
    for (auto const& f : factors) {
        n_rows *= static_cast<long long>(f->num_sectors);
    }
    auto reshaped =
      results.attr("reshape")(py::make_tuple(n_rows, static_cast<int>(sector_ind_len)));
    return sector_array_from_numpy(reshaped);
}

int64
Symmetry::sector_dim(Sector a) const
{
    if (is_abelian()) {
        return 1;
    }
    int64 dim = 1;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        dim *= factors[i]->sector_dim(factor_sector(a, i));
    }
    return dim;
}

py::array
Symmetry::batch_sector_dim(SectorArray const& a) const
{
    auto np = numpy();
    if (is_abelian()) {
        return np.attr("ones")(py::make_tuple(static_cast<ssize_t>(a.num_sectors)),
                               py::arg("dtype") = np.attr("int64"))
          .cast<py::array>();
    }
    auto dims = np.attr("ones")(py::make_tuple(static_cast<ssize_t>(a.num_sectors)),
                                py::arg("dtype") = np.attr("int64"));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        dims = np.attr("multiply")(dims, factors[i]->batch_sector_dim(factor_sectors(a, i)));
    }
    return dims.cast<py::array>();
}

py::array
Symmetry::batch_qdim(SectorArray const& a) const
{
    auto np = numpy();
    if (is_abelian()) {
        return np.attr("ones")(py::make_tuple(static_cast<ssize_t>(a.num_sectors)),
                               py::arg("dtype") = np.attr("int64"))
          .cast<py::array>();
    }
    auto dims = np.attr("ones")(py::make_tuple(static_cast<ssize_t>(a.num_sectors)));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        dims = np.attr("multiply")(dims, factors[i]->batch_qdim(factor_sectors(a, i)));
    }
    return dims.cast<py::array>();
}

float64
Symmetry::qdim(Sector a) const
{
    if (is_abelian()) {
        return 1.0;
    }
    float64 dim = 1.0;
    for (std::size_t i = 0; i < factors.size(); ++i) {
        dim *= factors[i]->qdim(factor_sector(a, i));
    }
    return dim;
}

std::string
Symmetry::sector_str(Sector a) const
{
    std::string out = "[";
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += factors[i]->sector_str(factor_sector(a, i));
    }
    out += "]";
    return out;
}

py::array
Symmetry::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    auto np = numpy();
    py::object res = np.attr("ones")(py::make_tuple(1, 1, 1, 1));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Fi = factors[i]->_f_symbol(factor_sector(a, i),
                                        factor_sector(b, i),
                                        factor_sector(c, i),
                                        factor_sector(d, i),
                                        factor_sector(e, i),
                                        factor_sector(f, i));
        res = np.attr("kron")(res, Fi);
    }
    return res.cast<py::array>();
}

py::array
Symmetry::_r_symbol(Sector a, Sector b, Sector c) const
{
    auto np = numpy();
    py::object res = np.attr("ones")(py::make_tuple(1));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Ri = factors[i]->_r_symbol(factor_sector(a, i), factor_sector(b, i), factor_sector(c, i));
        res = np.attr("kron")(res, Ri);
    }
    return res.cast<py::array>();
}

py::array
Symmetry::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("fusion tensor can not be written as array for this symmetry");
    }
    auto np = numpy();
    py::object res = np.attr("ones")(py::make_tuple(1, 1, 1, 1));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Xi = factors[i]->_fusion_tensor(
          factor_sector(a, i), factor_sector(b, i), factor_sector(c, i), Z_a, Z_b);
        res = np.attr("kron")(res, Xi);
    }
    return res.cast<py::array>();
}

py::array
Symmetry::swap_gate(Sector a, Sector b) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("fusion tensor can not be written as array for this symmetry");
    }
    auto np = numpy();
    py::object res = np.attr("ones")(py::make_tuple(1, 1, 1, 1));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Si = factors[i]->swap_gate(factor_sector(a, i), factor_sector(b, i));
        res = np.attr("kron")(res, Si);
    }
    return res.cast<py::array>();
}

py::array
Symmetry::Z_iso(Sector a) const
{
    if (!can_be_dropped()) {
        throw SymmetryError("Z iso can not be written as array for this symmetry");
    }
    auto np = numpy();
    py::object res = np.attr("ones")(py::make_tuple(1, 1));
    for (std::size_t i = 0; i < factors.size(); ++i) {
        auto Zi = factors[i]->Z_iso(factor_sector(a, i));
        res = np.attr("kron")(res, Zi);
    }
    return res.cast<py::array>();
}

std::string
Symmetry::repr() const
{
    if (num_factors() == 0) {
        return "Symmetry([])";
    }
    if (num_factors() == 1) {
        return "Symmetry([" + factors[0]->repr() + "])";
    }
    std::string out = factors[0]->repr();
    for (std::size_t i = 1; i < factors.size(); ++i) {
        out += " * " + factors[i]->repr();
    }
    return out;
}

std::string
Symmetry::str() const
{
    if (num_factors() == 0) {
        return "Symmetry([])";
    }
    if (num_factors() == 1) {
        return "Symmetry([" + factors[0]->str() + "])";
    }
    std::string out = factors[0]->str();
    for (std::size_t i = 1; i < factors.size(); ++i) {
        out += " ⨉ " + factors[i]->str();
    }
    return out;
}

bool
Symmetry::equals(Symmetry const& other) const
{
    if (num_factors() != other.num_factors()) {
        return false;
    }
    for (std::size_t i = 0; i < factors.size(); ++i) {
        if (!factors[i]->equals(*other.factors[i])) {
            return false;
        }
    }
    return true;
}

Symmetry::Ptr
Symmetry::mul(SymmetryFactor::Ptr other) const
{
    auto out = factors;
    out.push_back(std::move(other));
    return std::make_shared<Symmetry>(std::move(out));
}

Symmetry::Ptr
Symmetry::mul(Symmetry const& other) const
{
    auto out = factors;
    out.insert(out.end(), other.factors.begin(), other.factors.end());
    return std::make_shared<Symmetry>(std::move(out));
}

void
Symmetry::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    py::list factors_py;
    for (auto const& f : factors) {
        factors_py.append(py::cast(f));
    }
    hdf5_saver.attr("save")(factors_py, subpath + "factors");
    auto np = numpy();
    py::array slices = np.attr("array")(sector_slices, py::arg("dtype") = np.attr("int64"));
    hdf5_saver.attr("save")(slices, subpath + "sector_slices");
    if (fusion_tensor_dtype.has_value()) {
        hdf5_saver.attr("save")(static_cast<int>(*fusion_tensor_dtype), subpath + "fusion_tensor_dtype");
    } else {
        hdf5_saver.attr("save")(py::none(), subpath + "fusion_tensor_dtype");
    }
    hdf5_saver.attr("save")(static_cast<int>(fusion_style), subpath + "fusion_style");
    hdf5_saver.attr("save")(static_cast<int>(braiding_style), subpath + "braiding_style");
    hdf5_saver.attr("save")(sector_to_numpy(trivial_sector), subpath + "trivial_sector");
    hdf5_saver.attr("save")(num_sectors, subpath + "num_sectors");
    hdf5_saver.attr("save")(static_cast<int>(sector_ind_len), subpath + "sector_ind_len");
    h5gr.attr("attrs")["has_complex_topological_data"] = has_complex_topological_data;
}

} // namespace cyten
