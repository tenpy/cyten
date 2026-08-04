#include <cyten/symmetries/sun.h>

#include <cyten/symmetries/sector_numpy.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>

namespace cyten {

namespace {

py::module_
numpy()
{
    return py::module_::import("numpy");
}

std::string
sector_slash_path(Sector const& a)
{
    std::string s;
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        if (i != 0) {
            s += '/';
        }
        s += std::to_string(a.q[i]);
    }
    s += '/';
    return s;
}

std::string
sector_concat(Sector const& a)
{
    std::string s;
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        s += std::to_string(a.q[i]);
    }
    return s;
}

std::string
sector_bracket(Sector const& a)
{
    std::string s = "[";
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        if (i != 0) {
            s += ", ";
        }
        s += std::to_string(static_cast<int>(a.q[i]));
    }
    s += "]";
    return s;
}

std::string
cg_key(int N, Sector const& a, Sector const& b)
{
    return "/N_" + std::to_string(N) + "/" + sector_slash_path(a) + sector_slash_path(b);
}

bool
cg_key_usable(py::object CGfile, std::string const& key)
{
    return CGfile.contains(key) && py::len(CGfile[py::str(key)]) > 0;
}

int64
binomial(int n, int k)
{
    if (k < 0 || k > n) {
        return 0;
    }
    if (k == 0 || k == n) {
        return 1;
    }
    if (k > n - k) {
        k = n - k;
    }
    int64 r = 1;
    for (int i = 1; i <= k; ++i) {
        r = r * (n - k + i) / i;
    }
    return r;
}

Sector
zeros_sector(int N)
{
    std::array<int16_t, max_sector_ind_len> z{};
    return Sector::from_span(std::span<const int16_t>(z.data(), static_cast<std::size_t>(N)));
}

} // namespace

SUN::SUN(int N_,
         py::object CGfile_,
         py::object Ffile_,
         py::object Rfile_,
         std::optional<std::string> descriptive_name)
  : Group(FusionStyle::general,
          zeros_sector(N_),
          "SU(" + std::to_string(N_) + ")",
          std::numeric_limits<float64>::infinity(),
          /*has_complex_topological_data=*/false,
          std::move(descriptive_name),
          /*trivial_shift=*/true)
  , N(N_)
  , CGfile(std::move(CGfile_))
  , Ffile(std::move(Ffile_))
  , Rfile(std::move(Rfile_))
{
    if (N <= 1) {
        throw std::invalid_argument("Invalid N!");
    }
    if (static_cast<std::size_t>(N) > max_sector_ind_len) {
        throw std::invalid_argument("SUN: N exceeds max_sector_ind_len");
    }
    auto n_cg = CGfile.attr("attrs")["N"].cast<int>();
    auto n_f = Ffile.attr("attrs")["N"].cast<int>();
    auto n_r = Rfile.attr("attrs")["N"].cast<int>();
    if (N != n_cg || N != n_f || N != n_r) {
        throw std::invalid_argument("Files must contain data for same N!");
    }
    sanity_check_hdf5(CGfile);
    sanity_check_hdf5(Ffile);
    sanity_check_hdf5(Rfile);
    fusion_tensor_dtype = Dtype::Float64;
}

bool
SUN::is_valid_sector(Sector a) const
{
    if (a.len() != static_cast<std::uint8_t>(N)) {
        return false;
    }
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        if (a.q[i] < 0) {
            return false;
        }
    }
    for (std::uint8_t i = 0; i + 1 < a.len(); ++i) {
        if (a.q[i] < a.q[i + 1]) {
            return false;
        }
    }
    return a.q[a.len() - 1] == 0;
}

bool
SUN::_is_equivalent_factor(SymmetryFactor const& other) const
{
    if (auto const* sun = dynamic_cast<SUN const*>(&other)) {
        return sun->N == N;
    }
    return false;
}

int64
SUN::sector_dim(Sector a) const
{
    assert(is_valid_sector(a));
    float64 dim = 1.0;
    for (int kp = 2; kp <= N; ++kp) {
        for (int k = 1; k < kp; ++k) {
            dim *= 1.0 + (static_cast<float64>(a.q[k - 1] - a.q[kp - 1]) / (kp - k));
        }
    }
    return static_cast<int64>(dim);
}

std::string
SUN::repr() const
{
    return "SUNSymmetry(N=" + std::to_string(N) + ")";
}

Sector
SUN::dual_sector(Sector a) const
{
    int16_t mx = a.q[0];
    for (std::uint8_t i = 1; i < a.len(); ++i) {
        mx = std::max(mx, a.q[i]);
    }
    std::array<int16_t, max_sector_ind_len> buf{};
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        auto v = static_cast<int16_t>(a.q[i] - mx);
        buf[a.len() - 1 - i] = static_cast<int16_t>(std::abs(v));
    }
    return Sector::from_span(std::span<const int16_t>(buf.data(), a.len()));
}

int64
SUN::hweight_from_CG_hdf5() const
{
    return CGfile.attr("attrs")["Highest_Weight"].cast<int64>();
}

int64
SUN::hweight_from_F_hdf5() const
{
    return Ffile.attr("attrs")["Highest_Weight"].cast<int64>();
}

int64
SUN::hweight_from_R_hdf5() const
{
    return Rfile.attr("attrs")["Highest_Weight"].cast<int64>();
}

bool
SUN::can_fuse_to(Sector a, Sector b, Sector c) const
{
    auto const hmax = hweight_from_CG_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    if (c.q[0] > a.q[0] + b.q[0]) {
        return false;
    }
    auto key = cg_key(N, a, b);
    if (!cg_key_usable(CGfile, key)) {
        key = cg_key(N, b, a);
    }
    auto grp = CGfile[py::str(key)];
    for (auto item : grp) {
        auto child = grp[item];
        auto label = child.attr("attrs")["Irreplabel"];
        auto arr = py::array::ensure(label);
        Sector lab = sector_from_numpy(arr);
        if (lab == c) {
            return true;
        }
    }
    return false;
}

int64
SUN::_n_symbol(Sector a, Sector b, Sector c) const
{
    auto key = cg_key(N, a, b);
    if (!cg_key_usable(CGfile, key)) {
        key = cg_key(N, b, a);
    }
    auto grp = CGfile[py::str(key)];
    auto ckey = std::string("Irrep") + sector_concat(c) + "a1";
    if (!grp.contains(ckey)) {
        return 0;
    }
    return grp[py::str(ckey)].attr("attrs")["Outer Multiplicity"].cast<int64>();
}

int64
SUN::S_index_irrep_weight(Sector a) const
{
    int64 S = 0;
    for (int k = 1; k < N; ++k) {
        S += binomial(N - k + a.q[k - 1] - 1, N - k);
    }
    return S;
}

Sector
SUN::highest_irrep_in_decomp(Sector a, Sector b) const
{
    assert(a.len() == b.len());
    std::array<int16_t, max_sector_ind_len> buf{};
    for (std::uint8_t i = 0; i < a.len(); ++i) {
        buf[i] = static_cast<int16_t>(a.q[i] + b.q[i]);
    }
    return Sector::from_span(std::span<const int16_t>(buf.data(), a.len()));
}

SectorArray
SUN::fusion_outcomes(Sector a, Sector b) const
{
    auto const hmax = hweight_from_CG_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto key = cg_key(N, a, b);
    if (!cg_key_usable(CGfile, key)) {
        key = cg_key(N, b, a);
    }
    auto grp = CGfile[py::str(key)];
    py::list dec;
    for (auto item : grp) {
        auto child = grp[item];
        dec.append(child.attr("attrs")["Irreplabel"]);
    }
    return sector_array_from_numpy(numpy().attr("array")(dec));
}

py::dict
SUN::dims_of_irreps(Sector a, Sector b) const
{
    auto outcomes = fusion_outcomes(a, b);
    auto key = cg_key(N, a, b);
    // Python uses key without swap fallback for lookup after fusion_outcomes resolved order.
    if (!cg_key_usable(CGfile, key)) {
        key = cg_key(N, b, a);
    }
    // Match Python: always N+a+b (not swapped) for dims_of_irreps.
    key = cg_key(N, a, b);
    auto grp = CGfile[py::str(key)];
    py::dict C;
    for (std::size_t i = 0; i < outcomes.num_sectors; ++i) {
        Sector ir = outcomes[i];
        py::tuple k(ir.len());
        for (std::uint8_t j = 0; j < ir.len(); ++j) {
            k[j] = ir.q[j];
        }
        auto obj = std::string("Irrep") + sector_concat(ir) + "a1";
        C[k] = grp[py::str(obj)].attr("attrs")["Dimension"].cast<int64>();
    }
    return C;
}

py::dict
SUN::outer_multiplicity_from_CG(Sector a, Sector b) const
{
    auto outcomes = fusion_outcomes(a, b);
    auto key = cg_key(N, a, b);
    auto grp = CGfile[py::str(key)];
    py::dict C;
    for (std::size_t i = 0; i < outcomes.num_sectors; ++i) {
        Sector ir = outcomes[i];
        py::tuple k(ir.len());
        for (std::uint8_t j = 0; j < ir.len(); ++j) {
            k[j] = ir.q[j];
        }
        auto obj = std::string("Irrep") + sector_concat(ir) + "a1";
        C[k] = grp[py::str(obj)].attr("attrs")["Outer Multiplicity"].cast<int64>();
    }
    return C;
}

float64
SUN::clebschgordan(Sector a, int64 q_a, Sector b, int64 q_b, Sector c, int64 q_c, int64 mu) const
{
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto key1 = cg_key(N, a, b);
    auto key2 = std::string("Irrep") + sector_concat(c) + "a" + std::to_string(mu);
    py::array arr;
    py::list ms;
    if (cg_key_usable(CGfile, key1)) {
        arr = py::array(CGfile[py::str(key1)][py::str(key2)])[py::int_(0)].cast<py::array>();
        ms.append(static_cast<float64>(q_a));
        ms.append(static_cast<float64>(q_b));
        ms.append(static_cast<float64>(q_c));
    } else {
        key1 = cg_key(N, b, a);
        arr = py::array(CGfile[py::str(key1)][py::str(key2)])[py::int_(0)].cast<py::array>();
        ms.append(static_cast<float64>(q_b));
        ms.append(static_cast<float64>(q_a));
        ms.append(static_cast<float64>(q_c));
    }
    auto np = numpy();
    auto ms_arr = np.attr("array")(ms);
    auto n = arr.attr("shape").attr("__getitem__")(0).cast<py::ssize_t>();
    for (py::ssize_t i = 0; i < n; ++i) {
        auto row = arr[py::int_(i)];
        auto head = row[py::slice(0, 3, 1)];
        if (py::bool_(np.attr("array_equal")(head, ms_arr))) {
            return row[py::int_(3)].cast<float64>();
        }
    }
    return 0.0;
}

py::array
SUN::_fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const
{
    if (Z_a || Z_b) {
        PyErr_SetString(PyExc_NotImplementedError, "");
        throw py::error_already_set();
    }
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto dim_Sa = sector_dim(a);
    auto dim_Sb = sector_dim(b);
    auto dim_Sc = sector_dim(c);
    auto dim_mu = _n_symbol(a, b, c);
    auto np = numpy();
    if (dim_mu == 0) {
        return np
          .attr("zeros")(py::make_tuple(dim_Sa, dim_Sb, dim_Sc, 1),
                         py::arg("dtype") = np.attr("float64"))
          .cast<py::array>();
    }
    auto X = np.attr("zeros")(py::make_tuple(dim_Sa, dim_Sb, dim_Sc, dim_mu),
                              py::arg("dtype") = np.attr("float64"));
    for (int64 m_a = 1; m_a <= dim_Sa; ++m_a) {
        for (int64 m_b = 1; m_b <= dim_Sb; ++m_b) {
            for (int64 m_c = 1; m_c <= dim_Sc; ++m_c) {
                for (int64 mu = 1; mu <= dim_mu; ++mu) {
                    auto rr = clebschgordan(a, m_a, b, m_b, c, m_c, mu);
                    X[py::make_tuple(m_a - 1, m_b - 1, m_c - 1, mu - 1)] = rr;
                }
            }
        }
    }
    return X.attr("transpose")(py::make_tuple(3, 0, 1, 2)).cast<py::array>();
}

py::array
SUN::_f_symbol_from_CG(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw || d.q[0] > hw || e.q[0] > hw || f.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto np = numpy();
    auto X1 = _fusion_tensor(a, b, f, false, false).attr("transpose")(py::make_tuple(1, 2, 3, 0));
    auto X2 = _fusion_tensor(f, c, d, false, false).attr("transpose")(py::make_tuple(1, 2, 3, 0));
    auto X3 = _fusion_tensor(b, c, e, false, false).attr("transpose")(py::make_tuple(1, 2, 3, 0));
    auto X4 = _fusion_tensor(a, e, d, false, false).attr("transpose")(py::make_tuple(1, 2, 3, 0));
    if (!py::bool_(X1.attr("any")()) || !py::bool_(X2.attr("any")()) ||
        !py::bool_(X3.attr("any")()) || !py::bool_(X4.attr("any")())) {
        return np
          .attr("zeros")(py::make_tuple(1, 1, 1, 1), py::arg("dtype") = np.attr("complex128"))
          .cast<py::array>();
    }
    auto X12 = np.attr("tensordot")(X1, X2, py::arg("axes") = py::make_tuple(2, 0));
    X12 = X12.attr("transpose")(py::make_tuple(0, 1, 3, 4, 2, 5));
    auto X34 = np.attr("tensordot")(X3, X4, py::arg("axes") = py::make_tuple(2, 1));
    X34 = X34.attr("transpose")(py::make_tuple(3, 0, 1, 4, 2, 5));
    auto F = np.attr("tensordot")(
      X12,
      np.attr("conj")(X34),
      py::arg("axes") = py::make_tuple(py::make_tuple(0, 1, 2, 3), py::make_tuple(0, 1, 2, 3)));
    F = F.attr("transpose")(py::make_tuple(2, 3, 0, 1));
    F = np.attr("where")(np.attr("abs")(F).attr("__lt__")(1e-12), 0, F);
    auto denom = complex128{ static_cast<float64>(sector_dim(d)), 0.0 };
    return F.attr("__truediv__")(denom).cast<py::array>();
}

py::array
SUN::_f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const
{
    auto const hmax = hweight_from_F_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax || c.q[0] > hmax || d.q[0] > hmax || e.q[0] > hmax ||
        f.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    std::string key = "F";
    for (Sector const& s : { a, b, c, d, e, f }) {
        key += sector_bracket(s);
    }
    std::string keybar = "F";
    for (Sector const& s : { a, b, c, d, e, f }) {
        keybar += sector_bracket(dual_sector(s));
    }
    auto fsym = Ffile[py::str("/F_sym/")];
    if (fsym.contains(key)) {
        return py::array(fsym[py::str(key)]);
    }
    if (fsym.contains(keybar)) {
        return py::array(fsym[py::str(keybar)]);
    }
    auto np = numpy();
    return np.attr("zeros")(py::make_tuple(1, 1, 1, 1), py::arg("dtype") = np.attr("complex128"))
      .cast<py::array>();
}

py::array
SUN::_r_symbol_from_CG(Sector a, Sector b, Sector c) const
{
    auto const hw = hweight_from_CG_hdf5();
    if (a.q[0] > hw || b.q[0] > hw || c.q[0] > hw) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    auto np = numpy();
    auto X1 = fusion_tensor(a, b, c);
    auto Y1 = fusion_tensor(b, a, c).attr("conj")();
    if (!py::bool_(X1.attr("any")()) || !py::bool_(Y1.attr("any")())) {
        auto mult = n_symbol(a, b, c);
        return np.attr("zeros")(py::make_tuple(mult), py::arg("dtype") = np.attr("complex128"))
          .cast<py::array>();
    }
    auto R = np.attr("tensordot")(
      X1, Y1, py::arg("axes") = py::make_tuple(py::make_tuple(0, 1, 2), py::make_tuple(1, 0, 2)));
    auto denom = complex128{ static_cast<float64>(sector_dim(c)), 0.0 };
    R = R.attr("transpose")(py::make_tuple(1, 0)).attr("__truediv__")(denom);
    return np.attr("diag")(R).cast<py::array>();
}

py::array
SUN::_r_symbol(Sector a, Sector b, Sector c) const
{
    auto const hmax = hweight_from_R_hdf5();
    if (a.q[0] > hmax || b.q[0] > hmax || c.q[0] > hmax) {
        throw std::invalid_argument(
          "Input irreps have higher weight than highest weight irrep in HDF5-file");
    }
    std::string key = "R";
    for (Sector const& s : { a, b, c }) {
        key += sector_bracket(s);
    }
    auto rsym = Rfile[py::str("/R_sym/")];
    if (rsym.contains(key)) {
        return py::array(rsym[py::str(key)]);
    }
    auto np = numpy();
    return np.attr("zeros")(py::make_tuple(1), py::arg("dtype") = np.attr("complex128"))
      .cast<py::array>();
}

int64
SUN::frobenius_schur(Sector a) const
{
    if (N == 2) {
        return 1 - 2 * (static_cast<int64>(a.q[0]) % 2);
    }
    auto F = _f_symbol(a, dual_sector(a), a, a, trivial_sector, trivial_sector);
    auto val = F[py::make_tuple(0, 0, 0, 0)];
    // Match Python ``int(np.sign(F))``.
    return py::int_(numpy().attr("sign")(val)).cast<int64>();
}

bool
SUN::has_data_in_group(py::object group) const
{
    auto h5py = py::module_::import("h5py");
    if (py::isinstance(group, h5py.attr("Dataset"))) {
        return group.attr("size").cast<py::ssize_t>() > 0;
    }
    if (py::isinstance(group, h5py.attr("Group"))) {
        for (auto key : group) {
            if (has_data_in_group(group[key])) {
                return true;
            }
        }
    }
    return false;
}

void
SUN::sanity_check_hdf5(py::object file) const
{
    auto H = file.attr("attrs")["Highest_Weight"];
    auto Nattr = file.attr("attrs")["N"];
    auto keys0 = py::list(file.attr("keys")());
    auto filetype = std::string(py::str(keys0[0]));
    char ft = filetype.empty() ? '?' : filetype[0];

    if (ft == 'F') {
        if (!file.contains("/F_sym/")) {
            throw std::invalid_argument("HDF5 file does not contain '/F_sym/' group.");
        }
        auto keys = py::list(file[py::str("/F_sym/")].attr("keys")());
        py::list valid_keys;
        for (auto key : keys) {
            auto ks = std::string(py::str(key));
            if (ks.rfind("F[", 0) == 0) {
                valid_keys.append(key);
            }
        }
        if (py::len(valid_keys) == 0) {
            throw std::invalid_argument("No valid F-symbol keys found in '/F_sym/'.");
        }
        auto first_key = std::string(py::str(valid_keys[0]));
        auto num_lists = static_cast<int>(std::count(first_key.begin(), first_key.end(), '['));
        auto commas = static_cast<int>(std::count(first_key.begin(), first_key.end(), ','));
        std::string zero_key = "F";
        for (int i = 0; i < num_lists; ++i) {
            zero_key += "[0";
            for (int j = 0; j < commas / num_lists; ++j) {
                zero_key += ", 0";
            }
            zero_key += "]";
        }
        bool found_zero = false;
        for (auto key : keys) {
            if (std::string(py::str(key)) == zero_key) {
                found_zero = true;
                break;
            }
        }
        if (!found_zero) {
            throw std::invalid_argument("Missing key for all-trivial-sector F-symbol: " +
                                        zero_key);
        }
        auto h_key =
          std::string("[") + std::string(py::str(H)) + ", " + std::string(py::str(H)) + ", 0]";
        bool found_h = false;
        for (auto key : keys) {
            if (std::string(py::str(key)).find(h_key) != std::string::npos) {
                found_h = true;
                break;
            }
        }
        if (!found_h) {
            throw std::invalid_argument("No key found containing " + h_key + ".");
        }
    } else if (ft == 'R') {
        if (!file.contains("/R_sym/")) {
            throw std::invalid_argument("HDF5 file does not contain '/R_sym/' group.");
        }
        auto keys = py::list(file[py::str("/R_sym/")].attr("keys")());
        py::list valid_keys;
        for (auto key : keys) {
            auto ks = std::string(py::str(key));
            if (ks.rfind("R[", 0) == 0) {
                valid_keys.append(key);
            }
        }
        if (py::len(valid_keys) == 0) {
            throw std::invalid_argument("No valid R-symbol keys found in '/R_sym/'.");
        }
        auto first_key = std::string(py::str(valid_keys[0]));
        auto num_lists = static_cast<int>(std::count(first_key.begin(), first_key.end(), '['));
        auto commas = static_cast<int>(std::count(first_key.begin(), first_key.end(), ','));
        std::string zero_key = "R";
        for (int i = 0; i < num_lists; ++i) {
            zero_key += "[0";
            for (int j = 0; j < commas / num_lists; ++j) {
                zero_key += ", 0";
            }
            zero_key += "]";
        }
        bool found_zero = false;
        for (auto key : keys) {
            if (std::string(py::str(key)) == zero_key) {
                found_zero = true;
                break;
            }
        }
        if (!found_zero) {
            throw std::invalid_argument("Missing key for all-trivial-sector R-symbol: " +
                                        zero_key);
        }
        auto h_key =
          std::string("[") + std::string(py::str(H)) + ", " + std::string(py::str(H)) + ", 0]";
        bool found_h = false;
        for (auto key : keys) {
            if (std::string(py::str(key)).find(h_key) != std::string::npos) {
                found_h = true;
                break;
            }
        }
        if (!found_h) {
            throw std::invalid_argument("No key found containing " + h_key + ".");
        }
    } else if (ft == 'N') {
        auto path = std::string("/N_") + std::string(py::str(Nattr)) + "/";
        if (!file.contains(path)) {
            throw std::invalid_argument("HDF5 file does not contain " + path + " group.");
        }
        auto keys = py::list(file[py::str(path)].attr("keys")());
        if (static_cast<int64>(py::len(keys)) != H.cast<int64>() + 1) {
            throw std::runtime_error("SUN sanity_check_hdf5: unexpected CG key count");
        }
        auto high = file[py::str(path)][keys[py::len(keys) - 1]];
        auto low = file[py::str(path)][keys[0]];
        for (auto group : { high, low }) {
            if (py::len(group.attr("keys")()) == 0) {
                throw std::runtime_error("SUN sanity_check_hdf5: empty weight group");
            }
            if (!has_data_in_group(group)) {
                throw std::invalid_argument("Key exists but contains no data.");
            }
        }
    }
    (void)Nattr;
}

void
SUN::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    SymmetryFactor::save_hdf5(hdf5_saver, h5gr, subpath);
    hdf5_saver.attr("save")(N, subpath + "N");
    // Persist paths so from_hdf5 can reopen (h5py.File is not Hdf5Exportable).
    hdf5_saver.attr("save")(py::str(CGfile.attr("filename")), subpath + "CGfile");
    hdf5_saver.attr("save")(py::str(Ffile.attr("filename")), subpath + "Ffile");
    hdf5_saver.attr("save")(py::str(Rfile.attr("filename")), subpath + "Rfile");
}

SUN::Ptr
SUN::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    int N = hdf5_loader.attr("load")(subpath + "N").cast<int>();
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    auto h5py = py::module_::import("h5py");
    py::object CGfile = h5py.attr("File")(hdf5_loader.attr("load")(subpath + "CGfile"), "r");
    py::object Ffile = h5py.attr("File")(hdf5_loader.attr("load")(subpath + "Ffile"), "r");
    py::object Rfile = h5py.attr("File")(hdf5_loader.attr("load")(subpath + "Rfile"), "r");
    auto obj = std::make_shared<SUN>(N, CGfile, Ffile, Rfile, name);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
