#pragma once

#include <cyten/models/degrees_of_freedom.h>

#include <map>
#include <optional>
#include <string>

namespace cyten {

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
