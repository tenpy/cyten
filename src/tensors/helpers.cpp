#include <cyten/tensors/helpers.h>

#include <cyten/backends/no_symmetry.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tools.h>

#include <cassert>
#include <format>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace cyten {

namespace {

py::object
tensors_mod()
{
    return py::module_::import("cyten.tensors._tensors");
}

LegLabels
leg_labels_from_py(py::object seq)
{
    LegLabels out;
    for (auto item : py::reinterpret_borrow<py::iterable>(seq)) {
        if (item.is_none()) {
            out.push_back(std::nullopt);
        } else {
            out.push_back(item.cast<std::string>());
        }
    }
    return out;
}

LegLabels
apply_relabel(LegLabels labels, std::optional<std::map<std::string, std::string>> const& relabel)
{
    if (!relabel.has_value()) {
        return labels;
    }
    for (auto& lab : labels) {
        if (!lab.has_value()) {
            continue;
        }
        auto it = relabel->find(*lab);
        if (it != relabel->end()) {
            lab = it->second;
        }
    }
    return labels;
}

std::unordered_set<std::string>
duplicate_label_entries(LegLabels const& labels)
{
    std::unordered_set<std::string> seen;
    std::unordered_set<std::string> dups;
    for (auto const& lab : labels) {
        if (!lab.has_value()) {
            continue;
        }
        if (!seen.insert(*lab).second) {
            dups.insert(*lab);
        }
    }
    return dups;
}

py::object
data_as_python(TensorBackend::DataPtr data, TensorBackend::Ptr const& backend)
{
    // NoSymmetry stores BlockData in C++ while Python tensors store the Block directly.
    if (std::dynamic_pointer_cast<NoSymmetryBackend>(backend)) {
        return py::cast(NoSymmetryBackend::unwrap(std::move(data)));
    }
    return py::cast(std::move(data));
}

py::object
make_python_symmetric_tensor(TensorBackend::DataPtr data,
                             py::object codomain,
                             py::object domain,
                             TensorBackend::Ptr backend,
                             py::object labels)
{
    return tensors_mod().attr("SymmetricTensor")(data_as_python(std::move(data), backend),
                                                 codomain,
                                                 domain,
                                                 py::arg("backend") = py::cast(backend),
                                                 py::arg("labels") = labels);
}

py::object
make_python_charged_tensor(py::object invariant_part, py::object charged_state)
{
    return tensors_mod().attr("ChargedTensor")(invariant_part, charged_state);
}

bool
is_python_instance(py::object obj, char const* class_name)
{
    return py::isinstance(obj, tensors_mod().attr(class_name));
}

int64
prod_i64(std::vector<int64> const& vals)
{
    return std::accumulate(vals.begin(), vals.end(), int64{ 1 }, std::multiplies<int64>{});
}

/// Odometer-style cartesian product over ``ranges[i] == [0, ranges[i])``.
class SectorIndexProduct
{
  public:
    explicit SectorIndexProduct(std::vector<int64> ranges)
      : ranges_(std::move(ranges))
      , cur_(ranges_.size(), 0)
      , done_(false)
    {
        for (auto r : ranges_) {
            if (r <= 0) {
                done_ = true;
                return;
            }
        }
        if (ranges_.empty()) {
            // one empty combination
            done_ = false;
        }
    }

    [[nodiscard]] bool done() const { return done_; }

    [[nodiscard]] std::vector<int64> const& current() const { return cur_; }

    void next()
    {
        if (done_ || ranges_.empty()) {
            done_ = true;
            return;
        }
        for (std::size_t i = ranges_.size(); i-- > 0;) {
            ++cur_[i];
            if (cur_[i] < ranges_[i]) {
                return;
            }
            cur_[i] = 0;
        }
        done_ = true;
    }

  private:
    std::vector<int64> ranges_;
    std::vector<int64> cur_;
    bool done_;
};

std::vector<Space::Ptr>
spaces_of_product(TensorProduct::Ptr const& tp)
{
    std::vector<Space::Ptr> out;
    out.reserve(static_cast<std::size_t>(tp->num_factors));
    for (auto const& f : tp->factors) {
        out.push_back(f.cast<Space::Ptr>());
    }
    return out;
}

std::vector<int64>
num_sectors_per_leg(std::vector<Space::Ptr> const& legs)
{
    std::vector<int64> out;
    out.reserve(legs.size());
    for (auto const& leg : legs) {
        out.push_back(leg->num_sectors);
    }
    return out;
}

std::vector<Sector>
sectors_at(std::vector<Space::Ptr> const& legs, std::vector<int64> const& idcs)
{
    std::vector<Sector> out;
    out.reserve(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) {
        out.push_back(legs[i]->sector_decomposition[static_cast<std::size_t>(idcs[i])]);
    }
    return out;
}

std::vector<int64>
mults_at(std::vector<Space::Ptr> const& legs, std::vector<int64> const& idcs)
{
    std::vector<int64> out;
    out.reserve(legs.size());
    for (std::size_t i = 0; i < legs.size(); ++i) {
        out.push_back(legs[i]->multiplicities[static_cast<std::size_t>(idcs[i])]);
    }
    return out;
}

using BlockIndex = BlockBackend::BlockIndex;
using AxisSlice = BlockBackend::AxisSlice;

void
assign_block_slice(BlockBackend::BlockPtr& dest,
                   std::initializer_list<BlockIndex> key,
                   BlockBackend::BlockPtr const& value)
{
    dest->set_item(std::span<const BlockIndex>(key.begin(), key.size()), *value);
}

BlockBackend::BlockPtr
get_block_slice(BlockBackend::BlockPtr const& src, std::initializer_list<BlockIndex> key)
{
    return (*src)[std::span<const BlockIndex>(key.begin(), key.size())];
}

} // namespace

void
_check_compatible_legs(py::sequence legs1, py::sequence legs2, bool expect_equal)
{
    if (py::len(legs1) != py::len(legs2)) {
        throw std::invalid_argument("Different number of legs");
    }
    auto n = static_cast<py::ssize_t>(py::len(legs1));
    for (py::ssize_t i = 0; i < n; ++i) {
        py::object l1 = legs1[i];
        py::object l2 = legs2[i];
        if (!l1.attr("symmetry")
               .attr("is_equivalent_to")(l2.attr("symmetry"))
               .cast<bool>()) {
            throw std::invalid_argument("Different symmetries");
        }
        py::object rhs = expect_equal ? l2 : py::object(l2.attr("dual"));
        // Use Python ``__eq__`` so Space/Leg bindings apply (``py::object::operator==`` is pointer identity).
        py::object eq = l1.attr("__eq__")(rhs);
        if (eq.is(py::reinterpret_borrow<py::object>(Py_NotImplemented)) || !eq.cast<bool>()) {
            throw std::invalid_argument("Incompatible legs.");
        }
    }
}

py::object
_compose_with_Mask(py::object tensor, py::object mask, int64 leg_idx)
{
    // Match Python: ``in_domain, co_domain_idx, leg_idx = tensor._parse_leg_idx(leg_idx)``
    auto parsed = tensor.attr("_parse_leg_idx")(leg_idx);
    bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 co_domain_idx = parsed.attr("__getitem__")(1).cast<int64>();
    leg_idx = parsed.attr("__getitem__")(2).cast<int64>();

    if (in_domain) {
        py::list a;
        a.append(tensor.attr("domain").attr("__getitem__")(co_domain_idx));
        py::list b;
        b.append(mask.attr("codomain").attr("__getitem__")(0));
        _check_compatible_legs(a, b);
    } else {
        py::list a;
        a.append(tensor.attr("codomain").attr("__getitem__")(co_domain_idx));
        py::list b;
        b.append(mask.attr("domain").attr("__getitem__")(0));
        _check_compatible_legs(a, b);
    }

    if (is_python_instance(tensor, "ChargedTensor") || py::isinstance<ChargedTensor>(tensor)) {
        py::object invariant_part =
          _compose_with_Mask(tensor.attr("invariant_part"), mask, leg_idx);
        return make_python_charged_tensor(invariant_part, tensor.attr("charged_state"));
    }
    if (is_python_instance(tensor, "Mask") || py::isinstance<Mask>(tensor)) {
        throw NotImplemented("tensors._compose_with_Mask not implemented for Mask");
    }
    tensor = tensor.attr("as_SymmetricTensor")(py::arg("warning") = "Converting to SymmetricTensor.");

    auto backend = get_same_backend({ tensor, mask });
    bool mask_is_projection = mask.attr("is_projection").cast<bool>();
    std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr> contracted;
    if (in_domain == mask_is_projection) {
        contracted = backend->mask_contract_small_leg(tensor, mask, leg_idx);
    } else {
        contracted = backend->mask_contract_large_leg(tensor, mask, leg_idx);
    }
    auto& [data, codomain, domain] = contracted;
    return make_python_symmetric_tensor(std::move(data),
                                        py::cast(codomain),
                                        py::cast(domain),
                                        backend,
                                        tensor.attr("labels"));
}

py::object
_compose_SymmetricTensors(py::object tensor1,
                          py::object tensor2,
                          std::optional<std::map<std::string, std::string>> relabel1,
                          std::optional<std::map<std::string, std::string>> relabel2)
{
    if (tensor1.attr("num_codomain_legs").cast<int64>() == 0
        && tensor2.attr("num_domain_legs").cast<int64>() == 0) {
        return tensors_mod().attr("inner")(tensor1, tensor2, py::arg("do_dagger") = false);
    }

    LegLabels labels_codomain =
      apply_relabel(leg_labels_from_py(tensor1.attr("codomain_labels")), relabel1);
    LegLabels labels_domain =
      apply_relabel(leg_labels_from_py(tensor2.attr("domain_labels")), relabel2);

    LegLabels labels = labels_codomain;
    for (auto it = labels_domain.rbegin(); it != labels_domain.rend(); ++it) {
        labels.push_back(*it);
    }

    auto duplicates = duplicate_label_entries(labels);
    int64 dup_counter = 0;
    for (auto& lab : labels) {
        if (lab.has_value() && duplicates.contains(*lab)) {
            lab = std::format("?{}", dup_counter);
            ++dup_counter;
        }
    }

    auto backend = get_same_backend({ tensor1, tensor2 });
    auto data = backend->compose(tensor1, tensor2);
    return make_python_symmetric_tensor(std::move(data),
                                        tensor1.attr("codomain"),
                                        tensor2.attr("domain"),
                                        backend,
                                        py::cast(labels));
}

FusionTreeData::Ptr
_convert_abelian_to_FT(py::object tensor,
                       FusionTreeBackend::Ptr backend,
                       Dtype dtype,
                       std::string device)
{
    auto codomain = tensor.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain = tensor.attr("domain").cast<TensorProduct::Ptr>();
    auto symmetry = tensor.attr("symmetry").cast<Symmetry::Ptr>();
    auto ab_data = AbelianBackend::data_from_tensor(tensor);
    auto old_bb = tensor.attr("backend").attr("block_backend").cast<std::shared_ptr<BlockBackend>>();

    int64 num_codomain_legs = tensor.attr("num_codomain_legs").cast<int64>();
    int64 num_domain_legs = tensor.attr("num_domain_legs").cast<int64>();
    int64 num_legs = tensor.attr("num_legs").cast<int64>();

    // Start with all allowed blocks initialized with zeros
    // OPTIMIZE create the blocks on-demand instead?
    auto res = FusionTreeBackend::unwrap(
      backend->zero_data(codomain, domain, dtype, device, /*all_blocks=*/true));
    std::vector<bool> blocks_touched(res->blocks.size(), false);

    auto cod_legs = spaces_of_product(codomain);
    auto dom_legs = spaces_of_product(domain);

    if (num_codomain_legs == 0) {
        int64 i2 = 0;
        for (SectorIndexProduct prod(num_sectors_per_leg(dom_legs)); !prod.done(); prod.next()) {
            auto const& dom_sector_idcs = prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            if (c != symmetry->trivial_sector) {
                continue;
            }
            int64 tree_block_width = prod_i64(b_mults);
            std::vector<int64> ab_block_inds(dom_sector_idcs.rbegin(), dom_sector_idcs.rend());
            auto ab_i = ab_data->get_block_num(BlockInds::from_row(ab_block_inds));
            if (!ab_i.has_value()) {
                // block is missing (zero) -> nothing to do
                i2 += tree_block_width;
                continue;
            }
            auto ab_block = ab_data->blocks[static_cast<std::size_t>(*ab_i)];
            std::vector<int64> all_axes(static_cast<std::size_t>(num_legs));
            std::iota(all_axes.begin(), all_axes.end(), int64{ 0 });
            auto tree_block =
              old_bb->combine_legs(ab_block, std::vector<std::vector<int64>>{ all_axes }, false);
            tree_block = backend->block_backend->as_block(py::cast(tree_block), dtype, device);
            assign_block_slice(res->blocks[0],
                               { int64{ 0 }, AxisSlice{ i2, i2 + tree_block_width, {} } },
                               tree_block);
            blocks_touched[0] = true;
            i2 += tree_block_width;
        }
    } else if (num_domain_legs == 0) {
        int64 i1 = 0;
        for (SectorIndexProduct prod(num_sectors_per_leg(cod_legs)); !prod.done(); prod.next()) {
            auto const& cod_sector_idcs = prod.current();
            auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
            auto a_mults = mults_at(cod_legs, cod_sector_idcs);
            Sector c = symmetry->multiple_fusion(a_sectors);
            if (c != symmetry->trivial_sector) {
                continue;
            }
            int64 tree_block_height = prod_i64(a_mults);
            auto ab_i = ab_data->get_block_num(BlockInds::from_row(cod_sector_idcs));
            if (!ab_i.has_value()) {
                // block is missing (zero) -> nothing to do
                i1 += tree_block_height;
                continue;
            }
            auto ab_block = ab_data->blocks[static_cast<std::size_t>(*ab_i)];
            std::vector<int64> all_axes(static_cast<std::size_t>(num_legs));
            std::iota(all_axes.begin(), all_axes.end(), int64{ 0 });
            auto tree_block =
              old_bb->combine_legs(ab_block, std::vector<std::vector<int64>>{ all_axes }, true);
            tree_block = backend->block_backend->as_block(py::cast(tree_block), dtype, device);
            assign_block_slice(res->blocks[0],
                               { AxisSlice{ i1, i1 + tree_block_height, {} }, int64{ 0 } },
                               tree_block);
            blocks_touched[0] = true;
            i1 += tree_block_height;
        }
    } else {
        int64 i1 = 0;
        std::vector<int64> i2_per_coupled(res->blocks.size(), 0);
        std::vector<std::vector<int64>> combine{
            [&] {
                std::vector<int64> v(static_cast<std::size_t>(num_codomain_legs));
                std::iota(v.begin(), v.end(), int64{ 0 });
                return v;
            }(),
            [&] {
                std::vector<int64> v(static_cast<std::size_t>(num_domain_legs));
                std::iota(v.begin(), v.end(), num_codomain_legs);
                return v;
            }(),
        };
        std::vector<bool> cstyles{ true, false };

        for (SectorIndexProduct dom_prod(num_sectors_per_leg(dom_legs)); !dom_prod.done();
             dom_prod.next()) {
            auto const& dom_sector_idcs = dom_prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            auto ft_bi = res->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // this can happen if c does not appear in the codomain at all -> no block
            }
            int64 tree_block_width = prod_i64(b_mults);
            int64 i2 = i2_per_coupled[static_cast<std::size_t>(*ft_bi)];

            for (SectorIndexProduct cod_prod(num_sectors_per_leg(cod_legs)); !cod_prod.done();
                 cod_prod.next()) {
                auto const& cod_sector_idcs = cod_prod.current();
                auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
                auto a_mults = mults_at(cod_legs, cod_sector_idcs);
                Sector c2 = symmetry->multiple_fusion(a_sectors);
                int64 tree_block_height = prod_i64(a_mults);
                if (c2 != c) {
                    continue; // sector combination violates fusion rules -> no contributions
                }

                std::vector<int64> ab_block_inds = cod_sector_idcs;
                ab_block_inds.insert(ab_block_inds.end(),
                                     dom_sector_idcs.rbegin(),
                                     dom_sector_idcs.rend());

                // OPTIMIZE use that the data.block_inds are lexsorted for this lookup (also above)
                auto ab_i = ab_data->get_block_num(BlockInds::from_row(ab_block_inds));
                if (!ab_i.has_value()) {
                    // block is missing (zero) -> nothing to do
                    i1 += tree_block_height;
                    continue;
                }

                auto ab_block = ab_data->blocks[static_cast<std::size_t>(*ab_i)];
                // cstyle combine in the codomain, Fstyle in the domain
                auto tree_block = old_bb->combine_legs(ab_block, combine, cstyles);
                tree_block = backend->block_backend->as_block(py::cast(tree_block), dtype, device);
                assign_block_slice(
                  res->blocks[static_cast<std::size_t>(*ft_bi)],
                  { AxisSlice{ i1, i1 + tree_block_height, {} },
                    AxisSlice{ i2, i2 + tree_block_width, {} } },
                  tree_block);
                blocks_touched[static_cast<std::size_t>(*ft_bi)] = true;

                i1 += tree_block_height; // move down by one tree-block
            }

            // reset to the top
            i1 = 0;
            // move to the right by one tree-block, for the next time we visit this block
            i2_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_width;
        }
    }

    std::vector<BlockBackend::BlockPtr> blocks;
    for (std::size_t n = 0; n < res->blocks.size(); ++n) {
        if (blocks_touched[n]) {
            blocks.push_back(res->blocks[n]);
        }
    }
    BlockInds block_inds = res->block_inds.take_mask(blocks_touched);
    return std::make_shared<FusionTreeData>(
      std::move(block_inds), std::move(blocks), dtype, std::move(device), /*is_sorted=*/true);
}

AbelianBackendData::Ptr
_convert_FT_to_abelian(py::object tensor,
                       AbelianBackend::Ptr backend,
                       Dtype dtype,
                       std::string device)
{
    auto domain = tensor.attr("domain").cast<TensorProduct::Ptr>();
    auto codomain = tensor.attr("codomain").cast<TensorProduct::Ptr>();
    auto symmetry = tensor.attr("symmetry").cast<Symmetry::Ptr>();
    auto ft_data = FusionTreeBackend::data_from_tensor(tensor);
    auto old_bb = tensor.attr("backend").attr("block_backend").cast<std::shared_ptr<BlockBackend>>();

    int64 num_codomain_legs = tensor.attr("num_codomain_legs").cast<int64>();
    int64 num_domain_legs = tensor.attr("num_domain_legs").cast<int64>();
    int64 num_legs = tensor.attr("num_legs").cast<int64>();

    auto cod_legs = spaces_of_product(codomain);
    auto dom_legs = spaces_of_product(domain);

    std::vector<BlockBackend::BlockPtr> res_blocks;
    std::vector<std::vector<int64>> res_block_inds;

    if (num_codomain_legs == 0) {
        std::vector<int64> i2_per_coupled(ft_data->blocks.size(), 0);
        for (SectorIndexProduct prod(num_sectors_per_leg(dom_legs)); !prod.done(); prod.next()) {
            auto const& dom_sector_idcs = prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            if (c != symmetry->trivial_sector) {
                continue; // fusion rule violated
            }
            auto ft_bi = ft_data->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // no block for this coupled sector -> dont need to add a result block either
            }
            int64 tree_block_width = prod_i64(b_mults);
            int64 i2 = i2_per_coupled[static_cast<std::size_t>(*ft_bi)];
            auto tree_block = get_block_slice(
              ft_data->blocks[static_cast<std::size_t>(*ft_bi)],
              { int64{ 0 }, AxisSlice{ i2, i2 + tree_block_width, {} } });
            auto ab_block =
              old_bb->split_legs(tree_block, { 0 }, { b_mults }, false);
            // convert to new block_backend
            ab_block = backend->block_backend->as_block(py::cast(ab_block), dtype, device);
            res_blocks.push_back(ab_block);
            res_block_inds.emplace_back(dom_sector_idcs.rbegin(), dom_sector_idcs.rend());
            i2_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_width;
        }
    } else if (num_domain_legs == 0) {
        std::vector<int64> i1_per_coupled(ft_data->blocks.size(), 0);
        for (SectorIndexProduct prod(num_sectors_per_leg(cod_legs)); !prod.done(); prod.next()) {
            auto const& cod_sector_idcs = prod.current();
            auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
            auto a_mults = mults_at(cod_legs, cod_sector_idcs);
            Sector c = symmetry->multiple_fusion(a_sectors);
            if (c != symmetry->trivial_sector) {
                continue; // fusion rule violated
            }
            auto ft_bi = ft_data->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // no block for this coupled sector -> dont need to add a result block either
            }
            int64 tree_block_height = prod_i64(a_mults);
            int64 i1 = i1_per_coupled[static_cast<std::size_t>(*ft_bi)];
            auto tree_block = get_block_slice(
              ft_data->blocks[static_cast<std::size_t>(*ft_bi)],
              { AxisSlice{ i1, i1 + tree_block_height, {} }, int64{ 0 } });
            auto ab_block =
              old_bb->split_legs(tree_block, { 0 }, { a_mults }, true);
            // convert to new block_backend
            ab_block = backend->block_backend->as_block(py::cast(ab_block), dtype, device);
            res_blocks.push_back(ab_block);
            res_block_inds.push_back(cod_sector_idcs);
            i1_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_height;
        }
    } else {
        std::vector<int64> i2_per_coupled(ft_data->blocks.size(), 0);
        int64 i1 = 0;
        for (SectorIndexProduct dom_prod(num_sectors_per_leg(dom_legs)); !dom_prod.done();
             dom_prod.next()) {
            auto const& dom_sector_idcs = dom_prod.current();
            auto b_sectors = sectors_at(dom_legs, dom_sector_idcs);
            auto b_mults = mults_at(dom_legs, dom_sector_idcs);
            Sector c = symmetry->multiple_fusion(b_sectors);
            auto ft_bi = ft_data->block_ind_from_coupled(c, domain);
            if (!ft_bi.has_value()) {
                continue; // no block for this coupled sector -> dont need to add a result block either
            }
            int64 tree_block_width = prod_i64(b_mults);
            int64 i2 = i2_per_coupled[static_cast<std::size_t>(*ft_bi)];
            for (SectorIndexProduct cod_prod(num_sectors_per_leg(cod_legs)); !cod_prod.done();
                 cod_prod.next()) {
                auto const& cod_sector_idcs = cod_prod.current();
                auto a_sectors = sectors_at(cod_legs, cod_sector_idcs);
                auto a_mults = mults_at(cod_legs, cod_sector_idcs);
                Sector c2 = symmetry->multiple_fusion(a_sectors);
                int64 tree_block_height = prod_i64(a_mults);
                if (c2 != c) {
                    continue; // sector combination violates fusion rules -> no contributions
                }
                auto tree_block = get_block_slice(
                  ft_data->blocks[static_cast<std::size_t>(*ft_bi)],
                  { AxisSlice{ i1, i1 + tree_block_height, {} },
                    AxisSlice{ i2, i2 + tree_block_width, {} } });
                auto ab_block = old_bb->split_legs(
                  tree_block, { 0, 1 }, { a_mults, b_mults }, std::vector<bool>{ true, false });
                // convert to new block_backend
                ab_block = backend->block_backend->as_block(py::cast(ab_block), dtype, device);
                res_blocks.push_back(ab_block);
                std::vector<int64> row = cod_sector_idcs;
                row.insert(row.end(), dom_sector_idcs.rbegin(), dom_sector_idcs.rend());
                res_block_inds.push_back(std::move(row));
                i1 += tree_block_height; // move down by one tree-block
            }
            // reset to the top
            i1 = 0;
            // move to the right by one tree-block, for the next time we visit this block
            i2_per_coupled[static_cast<std::size_t>(*ft_bi)] += tree_block_width;
        }
    }

    BlockInds block_inds = res_block_inds.empty()
                             ? BlockInds::zeros(0, static_cast<std::size_t>(num_legs))
                             : BlockInds::from_rows(res_block_inds);
    return std::make_shared<AbelianBackendData>(
      dtype, std::move(device), std::move(res_blocks), std::move(block_inds));
}

std::tuple<py::object, TensorProduct::Ptr, bool, bool>
_decomposition_prepare(py::object tensor, bool new_leg_dual)
{
    if (tensor.attr("num_codomain_legs").cast<int64>() <= 0) {
        throw std::runtime_error("empty codomain");
    }
    if (tensor.attr("num_domain_legs").cast<int64>() <= 0) {
        throw std::runtime_error("empty domain");
    }

    if (is_python_instance(tensor, "ChargedTensor") || py::isinstance<ChargedTensor>(tensor)) {
        // do not define decompositions for ChargedTensors.
        throw NotImplemented("_decomposition_prepare for ChargedTensor");
    }
    tensor = tensor.attr("as_SymmetricTensor")();

    auto codomain = tensor.attr("codomain").cast<Space::Ptr>();
    auto domain = tensor.attr("domain").cast<Space::Ptr>();
    auto new_leg =
      ElementarySpace::from_largest_common_subspace({ codomain, domain }, new_leg_dual);
    auto new_co_domain = std::make_shared<TensorProduct>(std::vector<py::object>{ py::cast(new_leg) });

    bool combine_codomain = false;
    bool combine_domain = false;
    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    if (!backend->can_decompose_tensors) {
        combine_codomain = tensor.attr("num_codomain_legs").cast<int64>() > 1;
        combine_domain = tensor.attr("num_domain_legs").cast<int64>() > 1;
        auto combine_legs = tensors_mod().attr("combine_legs");
        int64 n_cod = tensor.attr("num_codomain_legs").cast<int64>();
        int64 n_legs = tensor.attr("num_legs").cast<int64>();
        if (combine_codomain && combine_domain) {
            py::list cod_range;
            for (int64 i = 0; i < n_cod; ++i) {
                cod_range.append(i);
            }
            py::list dom_range;
            for (int64 i = n_cod; i < n_legs; ++i) {
                dom_range.append(i);
            }
            tensor = combine_legs(tensor, cod_range, dom_range);
        } else if (combine_codomain) {
            py::list cod_range;
            for (int64 i = 0; i < n_cod; ++i) {
                cod_range.append(i);
            }
            tensor = combine_legs(tensor, cod_range);
        } else if (combine_domain) {
            py::list dom_range;
            for (int64 i = n_cod; i < n_legs; ++i) {
                dom_range.append(i);
            }
            tensor = combine_legs(tensor, dom_range);
        }
    }
    return { tensor, new_co_domain, combine_codomain, combine_domain };
}

std::pair<LegLabel, LegLabel>
_decomposition_labels(py::object new_labels)
{
    LegLabels labels = leg_labels_from_py(to_iterable(new_labels));
    if (labels.size() == 1) {
        LegLabel a = labels[0];
        return { a, _dual_leg_label(a) };
    }
    if (labels.size() == 2) {
        return { labels[0], labels[1] };
    }
    throw std::invalid_argument(
      std::format("Expected 1 or 2 labels. Got {}", labels.size()));
}

std::tuple<LegLabel, LegLabel, LegLabel, LegLabel>
_svd_new_labels(py::object new_labels)
{
    if (new_labels.is_none()) {
        return { std::nullopt, std::nullopt, std::nullopt, std::nullopt };
    }
    LegLabels labels = leg_labels_from_py(to_iterable(new_labels));
    LegLabel a, b, c, d;
    if (labels.size() == 1) {
        a = c = labels[0];
        b = d = _dual_leg_label(labels[0]);
    } else if (labels.size() == 2) {
        a = c = labels[0];
        b = d = labels[1];
    } else if (labels.size() == 4) {
        a = labels[0];
        b = labels[1];
        c = labels[2];
        d = labels[3];
    } else {
        throw std::invalid_argument(
          std::format("Expected 1, 2 or 4 new_labels. Got {}", labels.size()));
    }
    assert(!(b.has_value() && c.has_value() && *b == *c));
    return { a, b, c, d };
}

} // namespace cyten
