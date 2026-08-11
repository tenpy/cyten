#include "py_cyten_pybind11.h"

using namespace cyten;

PYBIND11_MODULE(_core, m)
{
    m.doc() = "Cyten python bindings using pybind11"; // optional module docstring

    bind_version(m);
    bind_tools(m);
    bind_cost_polynomials(m);
    bind_config(m);
    bind_block_backend(m);
    bind_symmetries(m);
    bind_mappings(m); // after FusionTree registration
    bind_tensor_backend(m);
    bind_no_symmetry_backend(m);
    bind_block_inds(m);
    bind_abelian_backend_data(m);
    bind_abelian_backend(m);
    bind_fusion_tree_data(m);
    bind_fusion_tree_backend(m);
    bind_fusion_tree_mapping(m);
    bind_backend_factory(m);
    bind_tensors_labels(m);
    bind_tensors_tensor(m);
    bind_tensors_symmetric_tensor(m);
    bind_tensors_diagonal_tensor(m);
    bind_tensors_mask(m);
    bind_tensors_charged_tensor(m);

    bind_check(m); // TODO: remove check
}
