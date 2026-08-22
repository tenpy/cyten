// Build a two-site Heisenberg Hamiltonian and print its eigenvalues.

#include <iostream>
#include <memory>
#include <stdexcept>

#include <pybind11/embed.h>

#include <cyten/models/couplings.h>
#include <cyten/models/sites.h>
#include <cyten/tensors/decompositions.h>
#include <cyten/tensors/diagonal_tensor.h>

namespace py = pybind11;
using namespace cyten;

int
main()
{
    py::scoped_interpreter guard{};
    py::module_::import("cyten");

    // Local spin-1/2 Hilbert space (no conserved quantum numbers).
    auto site0 = std::make_shared<SpinSite>(0.5);
    auto site1 = std::make_shared<SpinSite>(0.5);

    // Two-site coupling  H = J S_0 · S_1  with antiferromagnetic J = 1.
    Coupling coupling = heisenberg_coupling({ site0, site1 }, /*J=*/1);
    std::cout << coupling.repr() << '\n';

    // Contract the MPO-like factors to a two-site Hamiltonian tensor.
    SymmetricTensorPtr H = coupling.to_tensor();
    std::cout << H->ascii_diagram() << '\n';

    // Hermitian eigen-decomposition. W is diagonal (the eigenvalues);
    // V contains the eigenvectors.
    auto [W, V] = eigh(H, { "e" }, /*new_leg_dual=*/false);
    (void)V;
    auto np = py::module_::import("numpy");
    py::object evals = np.attr("sort")(np.attr("real")(W->diagonal_as_numpy()));
    std::cout << "eigenvalues: " << py::str(evals).cast<std::string>() << '\n';

    // Singlet  E = -3/4  and triplet  E = +1/4.
    py::object expect = np.attr("array")(py::make_tuple(-0.75, 0.25, 0.25, 0.25));
    if (!np.attr("allclose")(evals, expect, py::arg("atol") = 1e-12).cast<bool>()) {
        throw std::runtime_error("unexpected Heisenberg spectrum");
    }
    return 0;
}
