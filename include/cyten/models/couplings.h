#pragma once

#include <cyten/models/degrees_of_freedom.h>
#include <cyten/symmetries/sector.h>
#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/symmetric_tensor.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace cyten {

[[nodiscard]] std::map<std::string, py::object> space_to_dict(ElementarySpace::Ptr space);
[[nodiscard]] std::vector<int64> adjacent_transpositions(std::vector<int64> const& permutation);
[[nodiscard]] py::object freeze(py::object obj);

class Coupling
{
  public:
    std::vector<Site::Ptr> sites;
    std::vector<SymmetricTensorPtr> factorization;
    std::optional<std::string> name;

    Coupling(std::vector<Site::Ptr> sites,
             std::vector<SymmetricTensorPtr> factorization,
             std::optional<std::string> name = std::nullopt);

    [[nodiscard]] static Coupling from_dense_block(
      py::object operator_,
      std::vector<Site::Ptr> sites,
      std::optional<std::string> name = std::nullopt,
      std::optional<Dtype> dtype = std::nullopt,
      bool understood_braiding = false,
      std::optional<float64> cutoff_singular_values = std::nullopt);

    [[nodiscard]] static Coupling from_tensor(SymmetricTensorPtr operator_,
                                              std::vector<Site::Ptr> sites,
                                              std::optional<std::string> name = std::nullopt,
                                              std::optional<float64> cutoff = std::nullopt);

    [[nodiscard]] SymmetricTensorPtr to_tensor() const;
    [[nodiscard]] py::array to_numpy(
      std::optional<std::vector<std::variant<int64, std::string>>> leg_order = std::nullopt,
      py::object dtype = py::none(),
      bool understood_braiding = false) const;

    [[nodiscard]] Coupling stretch_with_identities(
      std::vector<Site::Ptr> const& all_sites,
      std::vector<int64> const& coupling_positions) const;

    [[nodiscard]] Coupling permute(
      std::vector<int64> const& permutation,
      std::optional<LevelsSpec> levels = std::nullopt,
      std::optional<std::vector<std::optional<bool>>> over_braid = std::nullopt) const;

    [[nodiscard]] std::tuple<py::object, std::vector<Site::Ptr>, std::vector<SymmetricTensorPtr>>
    key() const;

    [[nodiscard]] bool operator==(Coupling const& other) const;
    [[nodiscard]] size_t hash() const;

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] int64 num_sites() const { return static_cast<int64>(sites.size()); }

    void test_sanity() const;

  private:
    mutable std::vector<std::pair<std::vector<int64>, Coupling>> _permuted;
    std::vector<int64> _levels;
};

[[nodiscard]] Coupling spin_spin_coupling(std::vector<Site::Ptr> sites,
                                          float64 Jx = 0,
                                          float64 Jy = 0,
                                          float64 Jz = 0,
                                          py::object backend = py::none(),
                                          py::object device = py::none(),
                                          py::object name = py::none());

[[nodiscard]] Coupling spin_field_coupling(std::vector<Site::Ptr> sites,
                                           float64 hx = 0,
                                           float64 hy = 0,
                                           float64 hz = 0,
                                           py::object backend = py::none(),
                                           py::object device = py::none(),
                                           py::object name = py::none());

[[nodiscard]] Coupling aklt_coupling(std::vector<Site::Ptr> sites,
                                     float64 J = 1,
                                     py::object backend = py::none(),
                                     py::object device = py::none(),
                                     py::object name = py::none());

[[nodiscard]] Coupling heisenberg_coupling(std::vector<Site::Ptr> sites,
                                           float64 J = 1,
                                           py::object backend = py::none(),
                                           py::object device = py::none(),
                                           py::object name = py::none());

[[nodiscard]] Coupling chiral_3spin_coupling(std::vector<Site::Ptr> sites,
                                             float64 chi = 1,
                                             py::object backend = py::none(),
                                             py::object device = py::none(),
                                             py::object name = py::none());

[[nodiscard]] Coupling chemical_potential(std::vector<Site::Ptr> sites,
                                          float64 mu,
                                          py::object species = py::none(),
                                          py::object backend = py::none(),
                                          py::object device = py::none(),
                                          py::object name = py::none());

[[nodiscard]] Coupling onsite_interaction(std::vector<Site::Ptr> sites,
                                          float64 U = 1,
                                          py::object species = py::none(),
                                          py::object backend = py::none(),
                                          py::object device = py::none(),
                                          py::object name = py::none());

[[nodiscard]] Coupling density_density_interaction(std::vector<Site::Ptr> sites,
                                                   float64 V = 1,
                                                   py::object species_i = py::none(),
                                                   py::object species_j = py::none(),
                                                   py::object backend = py::none(),
                                                   py::object device = py::none(),
                                                   py::object name = py::none());

[[nodiscard]] Coupling hopping(std::vector<Site::Ptr> sites,
                               float64 t = 1,
                               py::object species = py::none(),
                               py::object backend = py::none(),
                               py::object device = py::none(),
                               py::object name = py::none());

[[nodiscard]] Coupling pairing(std::vector<Site::Ptr> sites,
                               float64 Delta = 1,
                               py::object species = py::none(),
                               py::object backend = py::none(),
                               py::object device = py::none(),
                               py::object name = py::none());

[[nodiscard]] Coupling onsite_pairing(std::vector<Site::Ptr> sites,
                                      float64 Delta = 1,
                                      py::object species = py::none(),
                                      py::object backend = py::none(),
                                      py::object device = py::none(),
                                      py::object name = py::none());

[[nodiscard]] Coupling clock_clock_coupling(std::vector<Site::Ptr> sites,
                                            float64 Jx = 0,
                                            float64 Jz = 0,
                                            py::object backend = py::none(),
                                            py::object device = py::none(),
                                            py::object name = py::none());

[[nodiscard]] Coupling clock_field_coupling(std::vector<Site::Ptr> sites,
                                            std::optional<float64> hx = std::nullopt,
                                            std::optional<float64> hz = std::nullopt,
                                            py::object backend = py::none(),
                                            py::object device = py::none(),
                                            py::object name = py::none());

[[nodiscard]] Coupling sector_projection_coupling(std::vector<Site::Ptr> sites,
                                                  float64 J,
                                                  Sector sector,
                                                  py::object name,
                                                  py::object backend = py::none(),
                                                  py::object device = py::none());

[[nodiscard]] Coupling gold_coupling(std::vector<Site::Ptr> sites,
                                     float64 J = 1,
                                     py::object backend = py::none(),
                                     py::object device = py::none(),
                                     py::object name = py::none());

} // namespace cyten
