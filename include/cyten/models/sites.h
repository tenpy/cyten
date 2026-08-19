#pragma once

#include <cyten/models/degrees_of_freedom.h>

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
};

class AnyonSite : public AnyonDOF
{
  public:
    using Ptr = std::shared_ptr<AnyonSite>;

    AnyonSite(Symmetry::Ptr symmetry,
              TensorBackend::Ptr backend = nullptr,
              std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;
};

class FibonacciAnyonSite : public AnyonSite
{
  public:
    using Ptr = std::shared_ptr<FibonacciAnyonSite>;

    FibonacciAnyonSite(TensorBackend::Ptr backend = nullptr,
                       std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;
};

class IsingAnyonSite : public AnyonSite
{
  public:
    using Ptr = std::shared_ptr<IsingAnyonSite>;

    IsingAnyonSite(TensorBackend::Ptr backend = nullptr,
                   std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;
};

class GoldenSite : public AnyonDOF
{
  public:
    using Ptr = std::shared_ptr<GoldenSite>;

    GoldenSite(TensorBackend::Ptr backend = nullptr,
               std::optional<std::string> default_device = std::nullopt);

    [[nodiscard]] std::string repr() const;
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
};

} // namespace cyten
