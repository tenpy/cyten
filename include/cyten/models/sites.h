#pragma once

#include <cyten/models/degrees_of_freedom.h>

#include <map>
#include <optional>
#include <string>

namespace cyten {

/// Class for sites that have a single spin degree of freedom.
///
/// TODO find a good format to doc the onsite operators that exist in a site
///
/// Attributes:
///
/// S : float
///     The total spin.
/// double_total_spin : int
///     Twice the `S`. We store this in addition because it is an integer.
/// conserve : Literal['SU(2)', 'Sz', 'parity', 'None']
///     The symmetry to be conserved. We can conserve::
///
///         - SU(2), the full spin rotation symmetry.
///         - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``.
///         - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``.
///         - nothing.
///
///     Conserves nothing by default.
class SpinSite : public SpinDOF
{
  public:
    using Ptr = std::shared_ptr<SpinSite>;

    float64 S{};
    int64 double_total_spin{};
    std::optional<std::string> conserve;

    SpinSite(float64 S = 0.5,
             std::optional<std::string> conserve = std::nullopt,
             TensorBackend::Ptr backend = nullptr,
             std::optional<std::string> default_device = std::nullopt);

/// Perform sanity checks.
    void test_sanity() override;
    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    /// Temporary data for virtual-base initialization; lives only during construction.
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array spin_vector;
        std::map<std::string, int64> state_labels;
        Symmetry::Ptr sym;
        int64 two_S;
    };
    static Prepared prepare(float64 S, std::optional<std::string> conserve);
    SpinSite(Prepared&& prepared,
             float64 S,
             std::optional<std::string> conserve,
             TensorBackend::Ptr backend,
             std::optional<std::string> default_device);
};

/// Site for (possibly multiple) spinless bosons.
///
/// TODO describe onsite operators
///
/// @param Nmax The maximum occupation of each of the boson species. An `int` corresponds to a single boson species. Otherwise, the number of boson species corresponds to `len(Nmax)`.
/// @param conserve The symmetry to be conserved. We can conserve::  - total particle number sum_k N_k (``conserve == 'N'``). - individual particle numbers N_k (``conserve[i] == 'N'``). - total parity (sum_i N_k) % 2 (``conserve == 'parity'``). - individual parities N_k % 2 (``conserve[i] == 'parity'``). - nothing (``conserve == 'None'`` or ``conserve[i] == 'None'``).  A `Literal` corresponds to symmetries involving all boson species, such as the total particle number (``conserve == 'N'``) or the total parity (``conserve == 'parity'``). For a sequence, the entry ``conserve[i]`` corresponds to the symmetry of boson species `k`, such that, e.g., ``conserve[k] == 'N'`` signifies that its particle number is conserved.  Conserves nothing by default.
/// @param filling Average total filling (that is, filling of all species together). Used to define the on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.
///
/// Attributes:
///
/// conserve : Literal['N', 'parity', 'None'] | list[Literal['N', 'parity', 'None']]
///     The conserved symmetry, see above.
/// filling : float | None
///     Average total filling.
/// num_species, Nmax, creators, annihilators
///     see `BosonicDOF`
class SpinlessBosonSite : public BosonicDOF
{
  public:
    using Ptr = std::shared_ptr<SpinlessBosonSite>;

    py::object conserve;
    std::optional<float64> filling;

    SpinlessBosonSite(py::object Nmax = py::int_(1),
                      py::object conserve = py::none(),
                      std::optional<float64> filling = std::nullopt,
                      TensorBackend::Ptr backend = nullptr,
                      std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array Nmax_arr;
        py::array creators;
        py::array annihilators;
        std::map<std::string, int64> state_labels;
        int64 total_dim;
    };
    static Prepared prepare(py::object Nmax, py::object conserve);
    SpinlessBosonSite(Prepared&& prepared,
                      py::object conserve,
                      std::optional<float64> filling,
                      TensorBackend::Ptr backend,
                      std::optional<std::string> default_device);
};

/// Site for (possibly multiple) spinless fermions.
///
/// TODO describe onsite operators
///
/// .. todo ::
///     For now, assume that the symmetry needs to capture the fermionic statistics.
///     Do not think about JW strings yet...
///     That is also the reason why NoSymmetry is not an option here
///
/// @param num_species Number of fermion species.
/// @param conserve The symmetry to be conserved. We can conserve::  - total fermion number sum_i N_k (``conserve == 'N'``). - individual fermion numbers N_k (``conserve[i] == 'N'``). - total fermion parity (sum_i N_k) % 2 (``conserve == 'parity'``). - individual fermion parities N_k % 2 (``conserve[i] == 'parity'``). - nothing for an individual fermion (``conserve[i] == 'None'``); .  A `Literal` corresponds to symmetries involving all fermion species, such as the total fermion number (``conserve == 'N'``) or the total fermion parity (``conserve == 'parity'``). For a sequence, the entry ``conserve[k]`` corresponds to the symmetry of fermion species `k`, such that, e.g., ``conserve[k] == 'N'`` signifies that its fermion number is conserved.  Note that the total fermion parity is always conserved. It is thus always part of the symmetry. Hence, ``conserve == 'None'`` is not a valid value. On the other hand, ``conserve = ['None']`` is interpreted as valid and the resulting symmetry conserves the fermionic parity.  Conserves total fermion parity by default.
/// @param filling Average total filling (that is, filling of all species together). Used to define the on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.
///
/// Attributes:
///
/// num_species : int
///     Number of fermion species.
/// conserve : Literal['N', 'parity'] | list[Literal['N', 'parity', 'None']]
///     The conserved symmetry, see above.
/// filling : float, optional
///     Average total filling.
/// creators, annihilators
///     see `FermionicDOF`
class SpinlessFermionSite : public FermionicDOF
{
  public:
    using Ptr = std::shared_ptr<SpinlessFermionSite>;

    int64 num_species{};
    py::object conserve;
    std::optional<float64> filling;

    SpinlessFermionSite(int64 num_species = 1,
                        py::object conserve = py::str("parity"),
                        std::optional<float64> filling = std::nullopt,
                        TensorBackend::Ptr backend = nullptr,
                        std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array creators;
        py::array annihilators;
        std::map<std::string, int64> state_labels;
    };
    static Prepared prepare(int64 num_species, py::object conserve);
    SpinlessFermionSite(Prepared&& prepared,
                        int64 num_species,
                        py::object conserve,
                        std::optional<float64> filling,
                        TensorBackend::Ptr backend,
                        std::optional<std::string> default_device);
};

/// Site for spin-1/2 fermions.
///
/// TODO describe onsite operators
///
/// @param conserve_N The fermion symmetry to be conserved. We can conserve::  - total fermion number N_up + N_down (``conserve == 'N'``). - total fermion parity (N_up + N_down) % 2 (``conserve == 'parity'``).  Note that the total fermion parity is always conserved and is thus always part of the total symmetry. Hence, ``conserve == 'None'`` is not a valid choice. Conserves total fermion parity by default.
/// @param conserve_S The spin symmetry to be conserved. We can conserve::  - SU(2), the full spin rotation symmetry. - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``. - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``. - nothing.  Conserves nothing by default.
/// @param filling Average total filling (that is, filling of spin up and spin down fermions together). Used to define the on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.
///
/// Attributes:
///
/// conserve_N : Literal['N', 'parity']
///     The conserved symmetry, see above.
/// conserve_S : Literal['SU(2)', 'Sz', 'parity', 'None']
///     The conserved spin symmetry, see above.
/// filling : float, optional
///     Average total filling.
/// creators, annihilators
///     see `FermionicDOF`
/// spin_vector
///     see `SpinDOF`
class SpinHalfFermionSite
  : public SpinDOF
  , public FermionicDOF
{
  public:
    using Ptr = std::shared_ptr<SpinHalfFermionSite>;

    std::string conserve_N;
    std::optional<std::string> conserve_S;
    std::optional<float64> filling;

    SpinHalfFermionSite(std::string conserve_N = "parity",
                        std::optional<std::string> conserve_S = std::nullopt,
                        std::optional<float64> filling = std::nullopt,
                        TensorBackend::Ptr backend = nullptr,
                        std::optional<std::string> default_device = std::nullopt);

    void test_sanity() override;

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array spin_vector;
        py::array creators;
        py::array annihilators;
        std::map<std::string, int64> state_labels;
        SymmetryFactor::Ptr sym_S_factor;
    };
    static Prepared prepare(std::string const& conserve_N,
                            std::optional<std::string> const& conserve_S);
    SpinHalfFermionSite(Prepared&& prepared,
                        std::string conserve_N,
                        std::optional<std::string> conserve_S,
                        std::optional<float64> filling,
                        TensorBackend::Ptr backend,
                        std::optional<std::string> default_device);
};

/// Class for sites that have a single quantum clock degree of freedom.
///
/// TODO describe onsite operators
///
/// @param q Number of states per site.
/// @param conserve The symmetry to be conserved. We can conserve::  - Z_N symmetry. - nothing.
///
/// Attributes:
///
/// conserve : Literal['Z_N', 'None']
///     The conserved symmetry, see above.
/// q, clock_operators
///     see `ClockDOF`
class ClockSite : public ClockDOF
{
  public:
    using Ptr = std::shared_ptr<ClockSite>;

    int64 q;
    std::optional<std::string> conserve;

    ClockSite(int64 q,
              std::optional<std::string> conserve = std::nullopt,
              TensorBackend::Ptr backend = nullptr,
              std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        ElementarySpace::Ptr leg;
        py::array clock_operators;
        std::map<std::string, int64> state_labels;
    };
    static Prepared prepare(int64 q, std::optional<std::string> conserve);
    ClockSite(Prepared&& prepared,
              int64 q,
              std::optional<std::string> conserve,
              TensorBackend::Ptr backend,
              std::optional<std::string> default_device);
};

/// Class for anyon models where the local Hilbert space contains all sectors once.
///
/// @param symmetry The symmetry describing the anyons.
/// @param sector_names The sector names that appear in the onsite projection operators. The `i`th operator is called `f'P_{sector_names[i]}'` and projects onto the `i`th sector in `leg.sector_decomposition`. For `None` entries (default), no projection operators are constructed.
class AnyonSite : public AnyonDOF
{
  public:
    using Ptr = std::shared_ptr<AnyonSite>;

    AnyonSite(Symmetry::Ptr symmetry,
              TensorBackend::Ptr backend = nullptr,
              std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  protected:
    /// Initialize from an already-built local space (shared by Site and AnyonDOF).
    AnyonSite(ElementarySpace::Ptr leg,
              TensorBackend::Ptr backend,
              std::optional<std::string> default_device);
};

/// Class for sites containing the trivial and the Fibonacci / tau sectors.
///
/// Projectors onto the onsite vacuum and tau sectors are automatically constructed
/// and are named `'P_vac'` and `'P_tau'`, respectively.
///
/// @param handedness The handedness of the anyons.
class FibonacciAnyonSite : public AnyonSite
{
  public:
    using Ptr = std::shared_ptr<FibonacciAnyonSite>;

    FibonacciAnyonSite(TensorBackend::Ptr backend = nullptr,
                       std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        Symmetry::Ptr symmetry;
        ElementarySpace::Ptr leg;
    };
    static Prepared prepare();
    FibonacciAnyonSite(Prepared&& prepared,
                       TensorBackend::Ptr backend,
                       std::optional<std::string> default_device);
};

/// Class for sites containing the trivial, the Ising / sigma, and the fermion / psi sectors.
///
/// Projectors onto the onsite vacuum, sigma and psi sectors are automatically constructed and are
/// named `'P_vac'`, `'P_sigma'`, and `'P_psi'`, respectively.
///
/// @param `nu` Specifies the Ising anyons as different `nu` correspond to different topological twists.
class IsingAnyonSite : public AnyonSite
{
  public:
    using Ptr = std::shared_ptr<IsingAnyonSite>;

    IsingAnyonSite(int nu = 1,
                   TensorBackend::Ptr backend = nullptr,
                   std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    struct Prepared
    {
        Symmetry::Ptr symmetry;
        ElementarySpace::Ptr leg;
    };
    static Prepared prepare(int nu);
    IsingAnyonSite(Prepared&& prepared,
                   TensorBackend::Ptr backend,
                   std::optional<std::string> default_device);
};

/// Class for Fibonacci anyon models where the local Hilbert space only contains the tau sector.
///
/// @param handedness The handedness of the anyons.
class GoldenSite : public AnyonDOF
{
  public:
    using Ptr = std::shared_ptr<GoldenSite>;

    GoldenSite(std::string handedness = "left",
               TensorBackend::Ptr backend = nullptr,
               std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    GoldenSite(ElementarySpace::Ptr leg,
               TensorBackend::Ptr backend,
               std::optional<std::string> default_device);
};

/// Class for SU(2)_k anyon models where the local Hilbert space only contains the spin-1 sector.
///
/// @param k Level of the SU(2)_k anyon model / symmetry.
/// @param handedness The handedness of the anyons.
class SU2kSpin1Site : public AnyonDOF
{
  public:
    using Ptr = std::shared_ptr<SU2kSpin1Site>;

    int64 k;

    SU2kSpin1Site(int64 k,
                  TensorBackend::Ptr backend = nullptr,
                  std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;

    [[nodiscard]] py::dict hdf5_init_kwargs() const override;

  private:
    SU2kSpin1Site(ElementarySpace::Ptr leg,
                  int64 k,
                  TensorBackend::Ptr backend,
                  std::optional<std::string> default_device);
};

} // namespace cyten
