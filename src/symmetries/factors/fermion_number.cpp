#include <cyten/symmetries/fermion_number.h>

#include <cyten/symmetries/topo_ones.h>

#include <limits>
#include <utility>

namespace cyten {

namespace {

int64_t
mod2(int16_t x)
{
    return static_cast<int64_t>(topo_ones::mod_n(static_cast<int32_t>(x), 2));
}

int64_t
fermion_sign(int16_t a, int16_t b)
{
    return 1 - 2 * mod2(a) * mod2(b);
}

py::module_
numpy()
{
    return py::module_::import("numpy");
}

} // namespace

FermionNumber::FermionNumber(std::optional<std::string> descriptive_name, bool trivial_shift)
  : SymmetryFactor(FusionStyle::single,
                   BraidingStyle::fermionic,
                   Sector{ 0 },
                   "FermionNumber",
                   std::numeric_limits<float64>::infinity(),
                   false,
                   std::move(descriptive_name),
                   trivial_shift)
{
    fusion_tensor_dtype = Dtype::Float64;
}

bool
FermionNumber::is_valid_sector(Sector a) const
{
    return a.len() == 1;
}

bool
FermionNumber::are_valid_sectors(SectorArray const& sectors) const
{
    return sectors.sector_ind_len() == 1;
}

SectorArray
FermionNumber::fusion_outcomes(Sector a, Sector b) const
{
    SectorArray aa(1, 1);
    SectorArray bb(1, 1);
    aa[0] = a;
    bb[0] = b;
    return fusion_outcomes_broadcast(aa, bb);
}

SectorArray
FermionNumber::fusion_outcomes_broadcast(SectorArray const& a, SectorArray const& b) const
{
    SectorArray out(a.size(), 1);
    for (std::size_t i = 0; i < a.size(); ++i) {
        out[i][0] = static_cast<int16_t>(a[i][0] + b[i][0]);
    }
    return out;
}

SectorArray
FermionNumber::_multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const
{
    SectorArray out = sectors[0];
    for (std::size_t s = 1; s < sectors.size(); ++s) {
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i][0] = static_cast<int16_t>(out[i][0] + sectors[s][i][0]);
        }
    }
    return out;
}

int64
FermionNumber::sector_dim(Sector /*a*/) const
{
    return 1;
}

py::array
FermionNumber::batch_sector_dim(SectorArray const& a) const
{
    return numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.size())),
                    py::arg("dtype") = numpy().attr("intp"))
      .cast<py::array>();
}

py::array
FermionNumber::batch_qdim(SectorArray const& a) const
{
    return batch_sector_dim(a);
}

bool
FermionNumber::_is_equivalent_factor(SymmetryFactor const& other) const
{
    return dynamic_cast<FermionNumber const*>(&other) != nullptr;
}

Sector
FermionNumber::dual_sector(Sector a) const
{
    return Sector{ static_cast<int16_t>(-a.q[0]) };
}

SectorArray
FermionNumber::dual_sectors(SectorArray const& sectors) const
{
    SectorArray out(sectors.size(), 1);
    for (std::size_t i = 0; i < sectors.size(); ++i) {
        out[i][0] = static_cast<int16_t>(-sectors[i][0]);
    }
    return out;
}

int64
FermionNumber::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

py::array
FermionNumber::_f_symbol(Sector /*a*/,
                         Sector /*b*/,
                         Sector /*c*/,
                         Sector /*d*/,
                         Sector /*e*/,
                         Sector /*f*/) const
{
    return topo_ones::one_4D();
}

int64
FermionNumber::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
FermionNumber::qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
FermionNumber::sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
FermionNumber::inv_sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

py::array
FermionNumber::_b_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return topo_ones::one_2D();
}

py::array
FermionNumber::_r_symbol(Sector a, Sector b, Sector /*c*/) const
{
    py::array_t<int64_t> out(1);
    out.mutable_at(0) = fermion_sign(a.q[0], b.q[0]);
    return out;
}

py::array
FermionNumber::_c_symbol(Sector a, Sector /*b*/, Sector c, Sector /*d*/, Sector e, Sector /*f*/)
  const
{
    auto const C = fermion_sign(c.q[0], e.q[0]) * fermion_sign(a.q[0], c.q[0]);
    py::array_t<int64_t> out({ 1, 1, 1, 1 });
    out.mutable_at(0, 0, 0, 0) = C;
    return out;
}

py::array
FermionNumber::_fusion_tensor(Sector /*a*/, Sector /*b*/, Sector /*c*/, bool /*Z_a*/, bool /*Z_b*/)
  const
{
    return topo_ones::one_4D_float();
}

py::array
FermionNumber::swap_gate(Sector a, Sector b) const
{
    auto const sign = fermion_sign(a.q[0], b.q[0]);
    return (py::cast(sign) * topo_ones::one_4D_float()).cast<py::array>();
}

complex128
FermionNumber::topological_twist(Sector a) const
{
    auto const sign = static_cast<float64>(1 - 2 * mod2(a.q[0]));
    return complex128{ sign, 0.0 };
}

py::array
FermionNumber::Z_iso(Sector /*a*/) const
{
    return topo_ones::one_2D_float();
}

std::string
FermionNumber::repr() const
{
    if (!descriptive_name.has_value()) {
        return "FermionNumber()";
    }
    return std::string("FermionNumber(\"") + *descriptive_name + "\")";
}

FermionNumber::Ptr
FermionNumber::from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath)
{
    auto name = descriptive_name_from_hdf5_attrs(h5gr);
    bool trivial_shift = trivial_shift_from_hdf5(hdf5_loader, subpath);
    auto obj = std::make_shared<FermionNumber>(name, trivial_shift);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

} // namespace cyten
