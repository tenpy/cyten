#include <cyten/symmetries/fermion_parity.h>

#include <cyten/symmetries/topo_ones.h>

#include <utility>

namespace cyten {

Sector const FermionParity::even{ 0 };
Sector const FermionParity::odd{ 1 };

namespace {

int64_t
fermion_sign(int16_t a, int16_t b)
{
    return 1 - 2 * static_cast<int64_t>(a) * static_cast<int64_t>(b);
}

py::module_
numpy()
{
    return py::module_::import("numpy");
}

} // namespace

FermionParity::FermionParity(std::optional<std::string> descriptive_name, bool trivial_shift)
  : SymmetryFactor(FusionStyle::single,
                   BraidingStyle::fermionic,
                   Sector{ 0 },
                   "FermionParity",
                   2.0,
                   false,
                   std::move(descriptive_name),
                   trivial_shift)
{
    fusion_tensor_dtype = Dtype::Float64;
}

bool
FermionParity::is_valid_sector(Sector a) const
{
    return a.len() == 1 && a.q[0] >= 0 && a.q[0] < 2;
}

bool
FermionParity::are_valid_sectors(SectorArray const& sectors) const
{
    if (sectors.sector_ind_len() != 1) {
        return false;
    }
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        auto q = sectors[i][0];
        if (q < 0 || q >= 2) {
            return false;
        }
    }
    return true;
}

SectorArray
FermionParity::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 1);
    SectorArray bb(1, 1);
    aa[0] = a;
    bb[0] = b;
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
FermionParity::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    SectorArray out(a.size(), 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i][0] = topo_ones::mod_n(static_cast<int32_t>(a[i][0]) + b[i][0], 2);
    }
    return out;
}

SectorArray
FermionParity::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i][0] = topo_ones::mod_n(static_cast<int32_t>(out[i][0]) + sectors[s][i][0], 2);
        }
    }
    return out;
}

int64
FermionParity::sector_dim(Sector /*a*/) const
{
    return 1;
}

py::array
FermionParity::batch_sector_dim(SectorArray const& a) const
{
    return numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.size())),
                    py::arg("dtype") = numpy().attr("intp"))
      .cast<py::array>();
}

py::array
FermionParity::batch_qdim(SectorArray const& a) const
{
    return batch_sector_dim(a);
}

std::string
FermionParity::sector_str(Sector a) const
{
    return a.q[0] == 0 ? "even" : "odd";
}

bool
FermionParity::_is_equivalent_factor(SymmetryFactor const& other) const
{
    return dynamic_cast<FermionParity const*>(&other) != nullptr;
}

Sector
FermionParity::dual_sector(Sector a) const
{
    return a;
}

SectorArray
FermionParity::dual_sectors(SectorArray const& sectors) const
{
    return sectors;
}

int64
FermionParity::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

py::array
FermionParity::_f_symbol(Sector /*a*/,
                         Sector /*b*/,
                         Sector /*c*/,
                         Sector /*d*/,
                         Sector /*e*/,
                         Sector /*f*/) const
{
    return topo_ones::one_4D();
}

int64
FermionParity::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
FermionParity::qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
FermionParity::sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
FermionParity::inv_sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

py::array
FermionParity::_b_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return topo_ones::one_2D();
}

py::array
FermionParity::_r_symbol(Sector a, Sector b, Sector /*c*/) const
{
    py::array_t<int64_t> out(1);
    out.mutable_at(0) = fermion_sign(a.q[0], b.q[0]);
    return out;
}

py::array
FermionParity::_c_symbol(Sector a, Sector /*b*/, Sector c, Sector /*d*/, Sector e, Sector /*f*/)
  const
{
    auto const C = fermion_sign(e.q[0], c.q[0]) * fermion_sign(a.q[0], c.q[0]);
    py::array_t<int64_t> out({ 1, 1, 1, 1 });
    out.mutable_at(0, 0, 0, 0) = C;
    return out;
}

py::array
FermionParity::_fusion_tensor(Sector /*a*/, Sector /*b*/, Sector /*c*/, bool /*Z_a*/, bool /*Z_b*/)
  const
{
    return topo_ones::one_4D_float();
}

py::array
FermionParity::swap_gate(Sector a, Sector b) const
{
    auto const sign = fermion_sign(a.q[0], b.q[0]);
    return (py::cast(sign) * topo_ones::one_4D_float()).cast<py::array>();
}

complex128
FermionParity::topological_twist(Sector a) const
{
    auto const sign = static_cast<float64>(1 - 2 * static_cast<int64_t>(a.q[0]));
    return complex128{ sign, 0.0 };
}

SectorArray
FermionParity::all_sectors() const
{
    SectorArray out(2, 1);
    out[0][0] = 0;
    out[1][0] = 1;
    return out;
}

py::array
FermionParity::Z_iso(Sector /*a*/) const
{
    return topo_ones::one_2D_float();
}

std::string
FermionParity::repr() const
{
    if (!descriptive_name.has_value()) {
        return "FermionParity()";
    }
    return std::string("FermionParity(\"") + *descriptive_name + "\")";
}

FermionParity::Ptr
FermionParity::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    bool trivial_shift = trivial_shift_from_hdf5(hdf5_loader, subpath);
    auto obj = std::make_shared<FermionParity>(name, trivial_shift);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
