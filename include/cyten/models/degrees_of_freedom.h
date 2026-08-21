#pragma once

#include <cyten/backends/backend_factory.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/cyten.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/symmetry.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/forward_declare.h>
#include <cyten/tensors/symmetric_tensor.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace cyten {

/// Sentinel for summing over all species in fermion/boson couplings.
/// Python exposes ``ALL_SPECIES = object()``; :func:`all_species_sentinel` is the unique
/// identity used for ``is`` comparison and pybind default arguments (process-lifetime).
struct AllSpeciesTag
{};
inline AllSpeciesTag const ALL_SPECIES{};

[[nodiscard]] py::object& all_species_sentinel();
[[nodiscard]] bool is_all_species(py::object const& species);

/// Collects necessary information about a local site of a lattice model.
class Site : public virtual std::enable_shared_from_this<Site>
{
  public:
    using Ptr = std::shared_ptr<Site>;
    using CPtr = std::shared_ptr<const Site>;

    ElementarySpace::Ptr leg;
    std::map<std::string, int64> state_labels;
    TensorBackend::Ptr backend;
    std::string default_device;
    std::map<std::string, SymmetricTensorPtr> onsite_operators;

    Site(ElementarySpace::Ptr leg,
         std::map<std::string, int64> state_labels = {},
         std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
         TensorBackend::Ptr backend = nullptr,
         std::optional<std::string> default_device = std::nullopt);

    virtual ~Site() = default;

    virtual void test_sanity();

    [[nodiscard]] Symmetry::Ptr symmetry() const;
    [[nodiscard]] float64 dim() const;

    void add_onsite_operator(std::string const& name,
                             py::object op,
                             std::optional<bool> is_diagonal = std::nullopt,
                             bool understood_braiding = false);

    [[nodiscard]] bool valid_opname(std::string const& name) const;
    [[nodiscard]] SymmetricTensorPtr get_op(std::string const& name);
    [[nodiscard]] std::string multiply_op_names(std::vector<std::string> const& names) const;
    [[nodiscard]] SymmetricTensorPtr multiply_operators(std::vector<py::object> const& operators);
    [[nodiscard]] SymmetricTensorPtr identity_tensor(ElementarySpace::Ptr w,
                                                     bool overbraid = true);
    [[nodiscard]] int64 state_index(py::object label) const;
    [[nodiscard]] std::vector<int64> state_indices(std::vector<py::object> const& labels) const;
    [[nodiscard]] std::string repr() const;

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;
    static py::object from_hdf5(py::object cls,
                                py::object hdf5_loader,
                                py::object h5gr,
                                std::string const& subpath);

  protected:
    /// Constructor keyword arguments used by :meth:`save_hdf5` / :meth:`from_hdf5`.
    [[nodiscard]] virtual py::dict hdf5_init_kwargs() const;
    [[nodiscard]] py::dict hdf5_backend_kwargs() const;
};

class SpinDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<SpinDOF>;

    py::array spin_vector;

    SpinDOF(ElementarySpace::Ptr leg,
            py::array spin_vector,
            std::map<std::string, int64> state_labels = {},
            std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
            TensorBackend::Ptr backend = nullptr,
            std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    [[nodiscard]] static py::array spin_vector_from_Sp(py::array Sz, py::array Sp);
    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(
      std::optional<std::string> conserve);
};

class ClockDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<ClockDOF>;

    py::array clock_operators;

    ClockDOF(ElementarySpace::Ptr leg,
             py::array clock_operators,
             std::map<std::string, int64> state_labels = {},
             std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
             TensorBackend::Ptr backend = nullptr,
             std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(
      std::optional<std::string> conserve);
};

class AnyonDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<AnyonDOF>;

    std::vector<std::string> sector_names;

    AnyonDOF(ElementarySpace::Ptr leg,
             std::vector<std::string> sector_names = {},
             std::map<std::string, int64> state_labels = {},
             std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
             TensorBackend::Ptr backend = nullptr,
             std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;
};

class OccupationDOF : public virtual Site
{
  public:
    using Ptr = std::shared_ptr<OccupationDOF>;

    int64 num_species{};
    py::array creators;
    py::array annihilators;
    int64 anti_commute_sign{};
    std::vector<std::optional<std::string>> species_names;
    py::array number_operators;
    py::array n_tot;

    OccupationDOF(ElementarySpace::Ptr leg,
                  py::array creators,
                  py::array annihilators,
                  int64 anti_commute_sign,
                  std::vector<std::optional<std::string>> species_names = {},
                  std::map<std::string, int64> state_labels = {},
                  std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
                  TensorBackend::Ptr backend = nullptr,
                  std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    virtual void add_individual_occupation_ops();
    void add_total_occupation_ops();

    [[nodiscard]] virtual py::array get_annihilator_numpy(py::object species,
                                                          bool include_JW = false) = 0;
    [[nodiscard]] virtual py::array get_creator_numpy(py::object species,
                                                      bool include_JW = false) = 0;

    [[nodiscard]] py::array get_occupation_numpy(py::object species = py::none());
    [[nodiscard]] int64 get_species_idx(py::object species) const;

  protected:
    std::map<std::string, int64> species_name_to_idx;
};

class BosonicDOF : public OccupationDOF
{
  public:
    using Ptr = std::shared_ptr<BosonicDOF>;

    py::array Nmax;
    py::array JW;

    BosonicDOF(ElementarySpace::Ptr leg,
               py::array Nmax,
               py::array creators,
               py::array annihilators,
               std::vector<std::optional<std::string>> species_names = {},
               std::map<std::string, int64> state_labels = {},
               std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
               TensorBackend::Ptr backend = nullptr,
               std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    void add_individual_occupation_ops() override;

    [[nodiscard]] py::array get_annihilator_numpy(py::object species,
                                                  bool include_JW = false) override;
    [[nodiscard]] py::array get_creator_numpy(py::object species,
                                              bool include_JW = false) override;

    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(py::object conserve);

    [[nodiscard]] static std::pair<py::array, py::array> creation_annihilation_op_from_single_Nmax(
      int64 Nmax,
      int64 dim);

    [[nodiscard]] static std::pair<py::array, py::array> creation_annihilation_ops_from_Nmax(
      py::array Nmax,
      int64 dim);

    [[nodiscard]] static std::pair<py::array, py::array>
    creation_annihilation_ops(int64 num_species, py::array Nmax, int64 dim);
};

class FermionicDOF : public OccupationDOF
{
  public:
    using Ptr = std::shared_ptr<FermionicDOF>;

    py::array partial_JWs;
    py::array JW;

    FermionicDOF(ElementarySpace::Ptr leg,
                 py::array creators,
                 py::array annihilators,
                 std::vector<std::optional<std::string>> species_names = {},
                 std::map<std::string, int64> state_labels = {},
                 std::map<std::string, SymmetricTensorPtr> onsite_operators = {},
                 TensorBackend::Ptr backend = nullptr,
                 std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    [[nodiscard]] py::array get_annihilator_numpy(py::object species,
                                                  bool include_JW = false) override;
    [[nodiscard]] py::array get_creator_numpy(py::object species,
                                              bool include_JW = false) override;

    [[nodiscard]] static Symmetry::Ptr conservation_law_to_symmetry(
      std::optional<std::string> conserve);

    [[nodiscard]] static std::pair<py::array, py::array> creation_annihilation_ops(
      int64 num_species);
};

[[nodiscard]] py::array as_immutable_array(py::object a, py::object dtype = py::none());

} // namespace cyten
