
namespace {

void
tree_block_iter(py::object a,
                std::function<void(FusionTree const&,
                                   FusionTree const&,
                                   BlockBackend::BlockPtr const&)> const& fn)
{
    auto a_data = FusionTreeBackend::data_from_tensor(a);
    auto codomain = a.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain = a.attr("domain").cast<TensorProduct::Ptr>();
    auto sym = a.attr("symmetry").cast<Symmetry::Ptr>();
    auto cod_are_dual = flat_are_dual(codomain);
    auto dom_are_dual = flat_are_dual(domain);
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t row = 0; row < buf.shape(0); ++row) {
        int64 bi_cod = buf(row, 0);
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(bi_cod)];
        auto const& block = a_data->blocks[static_cast<std::size_t>(row)];
        int64 i1_forest = 0;
        int64 i2_forest = 0;
        for (auto const& b_item : domain->iter_uncoupled(false)) {
            int64 tree_block_width = prod_int(b_item.multiplicities);
            int64 forest_block_width = 0;
            for (auto const& a_item : codomain->iter_uncoupled(false)) {
                int64 tree_block_height = prod_int(a_item.multiplicities);
                int64 i1 = i1_forest;
                int64 i2 = i2_forest;
                fusion_trees alpha_iter(sym, a_item.uncoupled, coupled, cod_are_dual);
                fusion_trees beta_iter(sym, b_item.uncoupled, coupled, dom_are_dual);
                for (auto const& alpha_tree : alpha_iter.all_trees()) {
                    i2 = i2_forest;
                    for (auto const& beta_tree : beta_iter.all_trees()) {
                        py::slice idx1(i1, i1 + tree_block_height, 1);
                        py::slice idx2(i2, i2 + tree_block_width, 1);
                        fn(alpha_tree, beta_tree, b_get(block, py::make_tuple(idx1, idx2)));
                        i2 += tree_block_width;
                    }
                    i1 += tree_block_height;
                }
                forest_block_width = std::max(forest_block_width, i2 - i2_forest);
                i1_forest += (i1 - i1_forest);
            }
            i1_forest = 0;
            i2_forest += forest_block_width;
        }
    }
}

SectorArray
as_sector_array_from_list(py::list const& sectors)
{
    return sector_utils().attr("as_sector_array")(sectors).cast<SectorArray>();
}

std::tuple<bool, complex128>
partial_trace_helper(FusionTree const& tree, std::vector<int64> const& idcs)
{
    py::object res = ft_py().attr("_partial_trace_helper")(py::cast(tree), idcs);
    auto tup = res.cast<py::tuple>();
    return { tup[0].cast<bool>(), tup[1].cast<complex128>() };
}

} // namespace

TensorBackend::DataPtr
FusionTreeBackend::outer(py::object a, py::object b)
{
    auto a_data = data_from_tensor(a);
    auto new_codomain = TensorProduct::from_partial_products(
      { a.attr("codomain").cast<TensorProduct::Ptr>(), b.attr("codomain").cast<TensorProduct::Ptr>() });
    auto new_domain = TensorProduct::from_partial_products(
      { a.attr("domain").cast<TensorProduct::Ptr>(), b.attr("domain").cast<TensorProduct::Ptr>() });
    Dtype dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    auto new_data = unwrap(zero_data(new_codomain, new_domain, dtype, a_data->device, true));
    tree_block_iter(a, [&](FusionTree const& a_codom_tree,
                           FusionTree const& a_dom_tree,
                           BlockBackend::BlockPtr const& a_tree_block) {
        tree_block_iter(b, [&](FusionTree const& b_codom_tree,
                               FusionTree const& b_dom_tree,
                               BlockBackend::BlockPtr const& b_tree_block) {
            auto new_tree_block = block_backend->outer(a_tree_block, b_tree_block);
            new_tree_block = block_backend->permute_axes(new_tree_block, { 0, 2, 1, 3 });
            new_tree_block = block_backend->combine_legs(new_tree_block, { { 0, 1 }, { 2, 3 } });
            auto new_codom_trees = a_codom_tree.outer(b_codom_tree);
            auto new_dom_trees = a_dom_tree.outer(b_dom_tree);
            for (auto const& [new_dom_tree, dom_amp] : new_dom_trees) {
                auto dom_slc = new_domain->tree_block_slice(new_dom_tree);
                auto block_idx = new_data->block_ind_from_coupled(new_dom_tree.coupled, new_domain);
                if (!block_idx.has_value())
                    continue;
                for (auto const& [new_codom_tree, codom_amp] : new_codom_trees) {
                    if (new_codom_tree.coupled != new_dom_tree.coupled)
                        continue;
                    auto codom_slc = new_codomain->tree_block_slice(new_codom_tree);
                    auto factor = block_backend->as_scalar(std::conj(codom_amp) * dom_amp, dtype);
                    auto cur = b_get(new_data->blocks[static_cast<std::size_t>(*block_idx)],
                                     py::make_tuple(slice_from_index_slice(codom_slc),
                                                    slice_from_index_slice(dom_slc)));
                    b_set(new_data->blocks[static_cast<std::size_t>(*block_idx)],
                          py::make_tuple(slice_from_index_slice(codom_slc),
                                         slice_from_index_slice(dom_slc)),
                          (*cur) + (*block_backend->mul(factor, new_tree_block)));
                }
            }
        });
    });
    new_data->discard_zero_blocks(block_backend, eps);
    return wrap(new_data);
}

TensorBackend::DataPtr
FusionTreeBackend::partial_compose(py::object a,
                                   py::object b,
                                   int64 a_first_leg,
                                   TensorProduct::Ptr new_codomain,
                                   TensorProduct::Ptr new_domain)
{
    Dtype dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    if (a_data->blocks.empty() || b_data->blocks.empty())
        return wrap(make_data(dtype, a_data->device, {}, zeros_i64(0, 2), true));

    SectorArray eff_sectors = SectorArray::empty(b.attr("symmetry").cast<Symmetry::Ptr>()->sector_ind_len);
    auto b_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(b_data->block_inds);
    auto b_buf = b_bi.unchecked<2>();
    for (py::ssize_t r = 0; r < b_buf.shape(0); ++r)
        eff_sectors.push_back(b.attr("domain").attr("sector_decomposition").attr("__getitem__")(b_buf(r, 1)).cast<Sector>());
    auto eff_space = ElementarySpace::from_defining_sectors(b.attr("symmetry").cast<Symmetry::Ptr>(), eff_sectors);

    bool in_domain;
    int64 num_contr_legs;
    int64 leg_idx;
    TensorProduct::Ptr old_space;
    TensorProduct::Ptr new_space;
    TensorProduct::Ptr iter_space;
    int64 a_num_cod = a.attr("num_codomain_legs").cast<int64>();
    int64 a_num_legs = a.attr("num_legs").cast<int64>();
    if (a_first_leg < a_num_cod) {
        in_domain = false;
        num_contr_legs = b.attr("num_domain_legs").cast<int64>();
        leg_idx = a_first_leg;
        old_space = a.attr("codomain").cast<TensorProduct::Ptr>();
        new_space = new_codomain;
        py::list factors;
        auto cod = a.attr("codomain");
        for (int64 i = 0; i < a_first_leg; ++i)
            factors.append(cod.attr("__getitem__")(i));
        factors.append(py::cast(eff_space));
        for (int64 i = a_first_leg + num_contr_legs; i < a_num_cod; ++i)
            factors.append(cod.attr("__getitem__")(i));
        iter_space = std::make_shared<TensorProduct>(factors.cast<std::vector<py::object>>(),
                                                     a.attr("symmetry").cast<Symmetry::Ptr>());
    } else {
        in_domain = true;
        num_contr_legs = b.attr("num_codomain_legs").cast<int64>();
        leg_idx = a_num_legs - a_first_leg - num_contr_legs;
        old_space = a.attr("domain").cast<TensorProduct::Ptr>();
        new_space = new_domain;
        py::list factors;
        auto dom = a.attr("domain");
        int64 n_dom = dom.attr("num_factors").cast<int64>();
        for (int64 i = 0; i < leg_idx; ++i)
            factors.append(dom.attr("__getitem__")(i));
        factors.append(py::cast(eff_space));
        for (int64 i = leg_idx + num_contr_legs; i < n_dom; ++i)
            factors.append(dom.attr("__getitem__")(i));
        iter_space = std::make_shared<TensorProduct>(factors.cast<std::vector<py::object>>(),
                                                     a.attr("symmetry").cast<Symmetry::Ptr>());
    }

    py::list new_block_inds_rows;
    std::vector<BlockBackend::BlockPtr> new_blocks;
    using TrafoMap = std::map<FusionTree, FusionTreeLinearCombination>;
    using DummyKey = std::tuple<Sector, Sector, int64, Sector>;
    std::map<DummyKey, TrafoMap> tree_transformations;

    auto a_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto a_buf = a_bi.unchecked<2>();
    for (py::ssize_t block_ind = 0; block_ind < a_buf.shape(0); ++block_ind) {
        int64 i = a_buf(block_ind, 0);
        Sector coupled = a.attr("codomain").attr("sector_decomposition").attr("__getitem__")(i).cast<Sector>();
        auto new_i = new_codomain->sector_decomposition_where(coupled);
        auto new_j = new_domain->sector_decomposition_where(coupled);
        if (!new_i.has_value() || !new_j.has_value())
            continue;
        auto new_block = block_backend->zeros(
          { new_codomain->multiplicities[*new_i], new_domain->multiplicities[*new_j] }, dtype, a.attr("device").cast<std::string>());
        new_block_inds_rows.append(py::make_tuple(*new_i, *new_j));

        SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
        for (auto const& tb : iter_space->iter_tree_blocks(coupled_arr)) {
            Sector b_coupled = tb.tree.uncoupled[static_cast<std::size_t>(leg_idx)];
            auto b_block_ind = b_data->block_ind_from_coupled(b_coupled, b.attr("domain").cast<TensorProduct::Ptr>());
            if (!b_block_ind.has_value())
                throw std::runtime_error("partial_compose: missing b block");
            std::vector<int64> dm = tb.multiplicities;
            int64 dm0 = prod_int(std::vector<int64>(dm.begin(), dm.begin() + static_cast<std::size_t>(leg_idx)));
            int64 dm2 = prod_int(std::vector<int64>(dm.begin() + static_cast<std::size_t>(leg_idx) + 1, dm.end()));
            std::vector<int64> dummy_mults = { dm0, 1, dm2 };

            DummyKey dummy_key = tb.tree.vertex_labels(std::max<int64>(0, leg_idx - 1));
            if (!tree_transformations.count(dummy_key))
                tree_transformations[dummy_key] = {};

            auto b_cod = b.attr("codomain").cast<TensorProduct::Ptr>();
            auto b_dom = b.attr("domain").cast<TensorProduct::Ptr>();
            SectorArray b_coupled_arr = SectorArray::repeat(b_coupled, 1);
            for (auto const& xb : b_cod->iter_tree_blocks(b_coupled_arr)) {
                TrafoMap& cache = tree_transformations[dummy_key];
                FusionTreeLinearCombination X_b_trafo;
                if (cache.count(xb.tree))
                    X_b_trafo = cache.at(xb.tree);
                else {
                    X_b_trafo = tb.tree.insert_at(leg_idx, xb.tree);
                    cache[xb.tree] = X_b_trafo;
                }
                for (auto const& yb : b_dom->iter_tree_blocks(b_coupled_arr)) {
                    FusionTreeLinearCombination Y_b_trafo;
                    if (cache.count(yb.tree))
                        Y_b_trafo = cache.at(yb.tree);
                    else {
                        Y_b_trafo = tb.tree.insert_at(leg_idx, yb.tree);
                        cache[yb.tree] = Y_b_trafo;
                    }
                    auto b_tree_block = b_get(b_data->blocks[static_cast<std::size_t>(*b_block_ind)],
                                              py::make_tuple(slice_from_index_slice(xb.slice),
                                                             slice_from_index_slice(yb.slice)));
                    auto b_shape = block_backend->get_shape(b_tree_block);

                    if (in_domain) {
                        for (auto const& [old_tree, amp_old_tree] : Y_b_trafo) {
                            auto old_slc = old_space->tree_block_slice(old_tree);
                            auto a_forest = b_get(a_data->blocks[static_cast<std::size_t>(block_ind)],
                                                  py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                                 slice_from_index_slice(old_slc)));
                            a_forest = block_backend->reshape(
                              a_forest, { -1, dummy_mults[0], b_shape[0], dummy_mults[2] });
                            for (auto const& [new_tree, amp_new_tree] : X_b_trafo) {
                                auto new_slc = new_space->tree_block_slice(new_tree);
                                auto contribution = block_backend->tdot(a_forest, b_tree_block, { 2 }, { 0 });
                                contribution = block_backend->permute_axes(contribution, { 0, 1, 3, 2 });
                                contribution = block_backend->reshape(
                                  contribution, { -1, dummy_mults[0] * b_shape[1] * dummy_mults[2] });
                                auto factor = block_backend->as_scalar(amp_old_tree * std::conj(amp_new_tree), dtype);
                                auto cur = b_get(new_block, py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                                           slice_from_index_slice(new_slc)));
                                b_set(new_block,
                                      py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                                     slice_from_index_slice(new_slc)),
                                      (*cur) + (*block_backend->mul(factor, contribution)));
                            }
                        }
                    } else {
                        for (auto const& [old_tree, amp_old_tree] : X_b_trafo) {
                            auto old_slc = old_space->tree_block_slice(old_tree);
                            auto a_forest = b_get(a_data->blocks[static_cast<std::size_t>(block_ind)],
                                                  py::make_tuple(slice_from_index_slice(old_slc),
                                                                 py::slice(std::nullopt, std::nullopt, 1)));
                            a_forest = block_backend->reshape(
                              a_forest, { dummy_mults[0], b_shape[1], dummy_mults[2], -1 });
                            for (auto const& [new_tree, amp_new_tree] : Y_b_trafo) {
                                auto new_slc = new_space->tree_block_slice(new_tree);
                                auto contribution = block_backend->tdot(a_forest, b_tree_block, { 1 }, { 1 });
                                contribution = block_backend->permute_axes(contribution, { 0, 3, 1, 2 });
                                contribution = block_backend->reshape(
                                  contribution, { dummy_mults[0] * b_shape[0] * dummy_mults[2], -1 });
                                auto factor = block_backend->as_scalar(std::conj(amp_old_tree) * amp_new_tree, dtype);
                                auto cur = b_get(new_block, py::make_tuple(slice_from_index_slice(new_slc),
                                                                           py::slice(std::nullopt, std::nullopt, 1)));
                                b_set(new_block,
                                      py::make_tuple(slice_from_index_slice(new_slc),
                                                     py::slice(std::nullopt, std::nullopt, 1)),
                                      (*cur) + (*block_backend->mul(factor, contribution)));
                            }
                        }
                    }
                }
            }
        }
        new_blocks.push_back(new_block);
    }

    py::array block_inds = new_block_inds_rows.size() > 0
      ? numpy().attr("array")(new_block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    auto res = make_data(dtype, a_data->device, std::move(new_blocks), block_inds, true);
    res->discard_zero_blocks(block_backend, eps);
    return wrap(res);
}


std::tuple<TensorBackend::DataPtr, TensorProduct::Ptr, TensorProduct::Ptr>
FusionTreeBackend::partial_trace(py::object tensor,
                                 std::vector<std::pair<int64, int64>> pairs,
                                 std::optional<std::vector<int64>> levels)
{
    std::sort(pairs.begin(), pairs.end(), [](auto const& a, auto const& b) {
        if (a.first != b.first)
            return a.first < b.first;
        return a.second < b.second;
    });
    for (auto& p : pairs)
        if (p.first > p.second)
            std::swap(p.first, p.second);

    int64 N = tensor.attr("num_legs").cast<int64>();
    int64 num_codom_legs = tensor.attr("num_codomain_legs").cast<int64>();
    std::vector<int64> idcs1, idcs2;
    for (auto const& [i1, i2] : pairs) {
        idcs1.push_back(i1);
        idcs2.push_back(i2);
    }
    std::set<int64> traced(idcs1.begin(), idcs1.end());
    traced.insert(idcs2.begin(), idcs2.end());
    std::vector<int64> remaining;
    for (int64 n = 0; n < N; ++n)
        if (!traced.count(n))
            remaining.push_back(n);

    py::list new_cod_factors, new_dom_factors;
    for (int64 n : remaining) {
        if (n < num_codom_legs)
            new_cod_factors.append(tensor.attr("codomain").attr("__getitem__")(n));
        if (N - 1 - n < tensor.attr("domain").attr("num_factors").cast<int64>())
            new_dom_factors.append(tensor.attr("domain").attr("__getitem__")(N - 1 - n));
    }
    auto sym = tensor.attr("symmetry").cast<Symmetry::Ptr>();
    auto new_codomain = std::make_shared<TensorProduct>(new_cod_factors.cast<std::vector<py::object>>(), sym);
    auto new_domain = std::make_shared<TensorProduct>(new_dom_factors.cast<std::vector<py::object>>(), sym);

    auto np = numpy();
    py::list insert_idcs;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        int64 pos = 0;
        for (int64 r : remaining)
            if (r < pairs[i].first)
                ++pos;
        insert_idcs.append(pos + 2 * static_cast<int64>(i));
    }
    std::vector<int64> idcs = remaining;
    int64 codom_count = num_codom_legs;
    for (std::size_t pi = 0; pi < pairs.size(); ++pi) {
        int64 idx = insert_idcs[pi].cast<int64>();
        idcs.insert(idcs.begin() + idx, pairs[pi].first);
        idcs.insert(idcs.begin() + idx + 1, pairs[pi].second);
        if (pairs[pi].first < num_codom_legs && pairs[pi].second >= num_codom_legs)
            ++codom_count;
    }
    int64 num_dom_legs = static_cast<int64>(idcs.size()) - codom_count;

    if (levels.has_value()) {
        for (auto const& pair : pairs) {
            if (pair.first >= static_cast<int64>(levels->size())
                || pair.second >= static_cast<int64>(levels->size()))
                continue;
            for (int64 i = 0; i < N; ++i) {
                if (i == pair.first || i == pair.second)
                    continue;
                if (i >= static_cast<int64>(levels->size()))
                    continue;
                bool l1 = (*levels)[static_cast<std::size_t>(i)]
                          < (*levels)[static_cast<std::size_t>(pair.first)];
                bool l2 = (*levels)[static_cast<std::size_t>(i)]
                          < (*levels)[static_cast<std::size_t>(pair.second)];
                if (l1 != l2)
                    throw std::invalid_argument("Inconsistent levels for partial_trace");
            }
        }
    }

    std::vector<int64> codomain_idcs(idcs.begin(), idcs.begin() + codom_count);
    std::vector<int64> domain_idcs(idcs.begin() + codom_count, idcs.end());
    std::reverse(domain_idcs.begin(), domain_idcs.end());
    bool mixes = std::any_of(codomain_idcs.begin(), codomain_idcs.end(), [&](int64 i) {
                   return i >= num_codom_legs;
               })
                 || std::any_of(domain_idcs.begin(), domain_idcs.end(), [&](int64 i) {
                      return i < num_codom_legs;
                    });

    TensorProduct::Ptr codom, dom;
    if (mixes) {
        py::list cf, df;
        for (int64 i : codomain_idcs)
            cf.append(tensor.attr("_as_codomain_leg")(i));
        for (int64 i : domain_idcs)
            df.append(tensor.attr("_as_domain_leg")(i));
        codom = std::make_shared<TensorProduct>(cf.cast<std::vector<py::object>>(), sym);
        dom = std::make_shared<TensorProduct>(df.cast<std::vector<py::object>>(), sym);
    } else {
        codom = tensor.attr("codomain").cast<TensorProduct::Ptr>()->permuted(codomain_idcs);
        std::vector<int64> dom_perm;
        for (int64 i : domain_idcs)
            dom_perm.push_back(N - 1 - i);
        dom = tensor.attr("domain").cast<TensorProduct::Ptr>()->permuted(dom_perm);
    }

    std::vector<std::optional<int64>> level_vec(static_cast<std::size_t>(N), std::nullopt);
    if (levels.has_value()) {
        for (std::size_t i = 0; i < levels->size() && i < level_vec.size(); ++i)
            level_vec[i] = (*levels)[i];
    }
    std::vector<std::optional<bool>> bend(static_cast<std::size_t>(N), true);
    auto data_ptr = permute_legs(tensor, codomain_idcs, domain_idcs, codom, dom, mixes, level_vec, bend);
    auto data = unwrap(data_ptr);

    py::list coupled_sectors;
    auto d_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(data->block_inds);
    auto d_buf = d_bi.unchecked<2>();
    for (py::ssize_t r = 0; r < d_buf.shape(0); ++r) {
        Sector sector = dom->sector_decomposition[static_cast<std::size_t>(d_buf(r, 1))];
        if (!new_domain->sector_decomposition_where(sector).has_value())
            continue;
        if (!new_codomain->sector_decomposition_where(sector).has_value())
            continue;
        coupled_sectors.append(py::cast(sector));
    }
    auto new_data = unwrap(zero_data(new_codomain, new_domain, data->dtype, data->device, true));
    SectorArray coupled_arr = coupled_sectors.size() > 0
      ? as_sector_array_from_list(coupled_sectors)
      : sym->empty_sector_array;

    std::vector<std::optional<int64>> old_inds, new_inds;
    for (py::handle h : coupled_sectors) {
        Sector c = h.cast<Sector>();
        old_inds.push_back(data->block_ind_from_coupled(c, dom));
        new_inds.push_back(new_data->block_ind_from_coupled(c, new_domain));
    }

    std::vector<int64> codom_unc_idcs, codom_inner_idcs, codom_multi_idcs, codom_tree_idcs;
    for (std::size_t i = 0; i < codomain_idcs.size(); ++i)
        if (std::find(remaining.begin(), remaining.end(), codomain_idcs[i]) != remaining.end())
            codom_unc_idcs.push_back(static_cast<int64>(i));
    for (std::size_t k = 2; k < codom_unc_idcs.size(); ++k)
        codom_inner_idcs.push_back(codom_unc_idcs[k] - 2);
    for (std::size_t k = 1; k < codom_unc_idcs.size(); ++k)
        codom_multi_idcs.push_back(codom_unc_idcs[k] - 1);
    for (std::size_t i = 0; i < codomain_idcs.size(); ++i)
        if (std::find(idcs1.begin(), idcs1.end(), codomain_idcs[i]) != idcs1.end())
            codom_tree_idcs.push_back(static_cast<int64>(i));

    std::vector<int64> dom_unc_idcs;
    for (std::size_t i = 0; i < domain_idcs.size(); ++i)
        if (std::find(remaining.begin(), remaining.end(), domain_idcs[i]) != remaining.end())
            dom_unc_idcs.push_back(static_cast<int64>(num_dom_legs - 1 - i));
    std::reverse(dom_unc_idcs.begin(), dom_unc_idcs.end());
    std::vector<int64> dom_inner_idcs, dom_multi_idcs, dom_tree_idcs;
    for (std::size_t k = 2; k < dom_unc_idcs.size(); ++k)
        dom_inner_idcs.push_back(dom_unc_idcs[k] - 2);
    for (std::size_t k = 1; k < dom_unc_idcs.size(); ++k)
        dom_multi_idcs.push_back(dom_unc_idcs[k] - 1);
    for (std::size_t i = 0; i < domain_idcs.size(); ++i)
        if (std::find(idcs2.begin(), idcs2.end(), domain_idcs[i]) != idcs2.end())
            dom_tree_idcs.push_back(num_dom_legs - 1 - static_cast<int64>(i));
    std::reverse(dom_tree_idcs.begin(), dom_tree_idcs.end());

    std::vector<int64> tr_idcs = codomain_idcs;
    tr_idcs.insert(tr_idcs.end(), domain_idcs.rbegin(), domain_idcs.rend());
    std::vector<int64> tr_idcs1, tr_idcs2, remain_idcs;
    for (std::size_t i = 0; i < tr_idcs.size(); ++i) {
        if (std::find(idcs1.begin(), idcs1.end(), tr_idcs[i]) != idcs1.end())
            tr_idcs1.push_back(static_cast<int64>(i));
        if (std::find(idcs2.begin(), idcs2.end(), tr_idcs[i]) != idcs2.end())
            tr_idcs2.push_back(static_cast<int64>(i));
        if (std::find(remaining.begin(), remaining.end(), tr_idcs[i]) != remaining.end())
            remain_idcs.push_back(static_cast<int64>(i));
    }

    for (auto const& cod_tb : codom->iter_tree_blocks(coupled_arr)) {
        auto [on_diag, factor_codom] = partial_trace_helper(cod_tb.tree, codom_tree_idcs);
        if (!on_diag)
            continue;
        std::vector<std::uint8_t> ad(cod_tb.tree.are_dual.begin(), cod_tb.tree.are_dual.end());
        std::vector<std::uint8_t> sel_ad;
        SectorArray sel_unc = SectorArray::empty(cod_tb.tree.uncoupled.sector_ind_len());
        SectorArray sel_inner = SectorArray::empty(cod_tb.tree.inner_sectors.sector_ind_len());
        std::vector<int64> sel_mult;
        for (int64 ui : codom_unc_idcs) {
            sel_ad.push_back(cod_tb.tree.are_dual[static_cast<std::size_t>(ui)]);
            sel_unc.push_back(cod_tb.tree.uncoupled[static_cast<std::size_t>(ui)]);
        }
        for (int64 ii : codom_inner_idcs)
            sel_inner.push_back(cod_tb.tree.inner_sectors[static_cast<std::size_t>(ii)]);
        for (int64 mi : codom_multi_idcs)
            sel_mult.push_back(cod_tb.tree.multiplicities[static_cast<std::size_t>(mi)]);
        FusionTree new_codom_tree(sym, sel_unc, cod_tb.tree.coupled, sel_ad, sel_inner, sel_mult);
        auto new_codom_slc = new_codomain->tree_block_slice(new_codom_tree);
        auto old_ind = old_inds[static_cast<std::size_t>(cod_tb.coupled_idx)];
        auto new_ind = new_inds[static_cast<std::size_t>(cod_tb.coupled_idx)];
        if (!old_ind.has_value() || !new_ind.has_value())
            continue;

        for (auto const& dom_tb : dom->iter_tree_blocks(SectorArray::repeat(cod_tb.tree.coupled, 1))) {
            auto [on_diag2, factor_dom] = partial_trace_helper(dom_tb.tree, dom_tree_idcs);
            if (!on_diag2)
                continue;
            std::vector<int64> tmp_shape = cod_tb.multiplicities;
            tmp_shape.insert(tmp_shape.end(), dom_tb.multiplicities.begin(), dom_tb.multiplicities.end());
            std::vector<std::uint8_t> dad;
            SectorArray dunc = SectorArray::empty(dom_tb.tree.uncoupled.sector_ind_len());
            SectorArray dinn = SectorArray::empty(dom_tb.tree.inner_sectors.sector_ind_len());
            std::vector<int64> dmult;
            for (int64 ui : dom_unc_idcs) {
                dad.push_back(dom_tb.tree.are_dual[static_cast<std::size_t>(ui)]);
                dunc.push_back(dom_tb.tree.uncoupled[static_cast<std::size_t>(ui)]);
            }
            for (int64 ii : dom_inner_idcs)
                dinn.push_back(dom_tb.tree.inner_sectors[static_cast<std::size_t>(ii)]);
            for (int64 mi : dom_multi_idcs)
                dmult.push_back(dom_tb.tree.multiplicities[static_cast<std::size_t>(mi)]);
            FusionTree new_dom_tree(sym, dunc, dom_tb.tree.coupled, dad, dinn, dmult);
            auto new_dom_slc = new_domain->tree_block_slice(new_dom_tree);

            auto old_block = b_get(data->blocks[static_cast<std::size_t>(*old_ind)],
                                   py::make_tuple(slice_from_index_slice(cod_tb.slice),
                                                  slice_from_index_slice(dom_tb.slice)));
            old_block = block_backend->reshape(old_block, tmp_shape);
            auto contribution = block_backend->trace_partial(old_block, tr_idcs1, tr_idcs2, remain_idcs);
            contribution = block_backend->reshape(
              contribution,
              { new_codom_slc.stop - new_codom_slc.start, new_dom_slc.stop - new_dom_slc.start });
            auto factor = block_backend->as_scalar(factor_codom * std::conj(factor_dom), data->dtype);
            contribution = block_backend->mul(factor, contribution);
            auto cur = b_get(new_data->blocks[static_cast<std::size_t>(*new_ind)],
                             py::make_tuple(slice_from_index_slice(new_codom_slc),
                                            slice_from_index_slice(new_dom_slc)));
            b_set(new_data->blocks[static_cast<std::size_t>(*new_ind)],
                  py::make_tuple(slice_from_index_slice(new_codom_slc),
                                 slice_from_index_slice(new_dom_slc)),
                  (*cur) + (*contribution));
        }
    }
    new_data->discard_zero_blocks(block_backend, eps);

    if (remaining.empty()) {
        Dtype dt = tensor.attr("dtype").cast<Dtype>();
        if (new_data->blocks.empty()) {
            auto s = block_backend->as_scalar(dtype::zero_scalar(dt), dt);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            return { wrap(make_data(dt, data->device, { block }, zeros_i64(1, 0), true)), nullptr, nullptr };
        }
        if (new_data->blocks.size() == 1) {
            auto s = block_backend->item(new_data->blocks[0]);
            auto block = block_backend->as_block(s.to_numpy(), dt);
            return { wrap(make_data(dt, data->device, { block }, zeros_i64(1, 0), true)), nullptr, nullptr };
        }
        throw std::runtime_error("partial_trace: multiple blocks for scalar result");
    }
    return { wrap(new_data), new_codomain, new_domain };
}

TensorBackend::DataPtr
FusionTreeBackend::from_tree_pairs(
  std::map<std::pair<FusionTree, FusionTree>, BlockBackend::BlockPtr> trees,
  TensorProduct::Ptr codomain,
  TensorProduct::Ptr domain,
  Dtype dtype,
  std::string device)
{
    int64 J = codomain->num_flat_legs();
    int64 K = domain->num_flat_legs();
    py::list block_inds_rows;
    std::vector<BlockBackend::BlockPtr> blocks;
    std::set<std::pair<FusionTree, FusionTree>> pairs_done;
    py::object codom_secs = py::cast(codomain->sector_decomposition);
    py::object dom_secs = py::cast(domain->sector_decomposition);
    for (py::handle item : misc().attr("iter_common_sorted_arrays")(codom_secs, dom_secs)) {
        auto pair = item.cast<py::tuple>();
        int64 i = pair[0].cast<int64>();
        int64 j = pair[1].cast<int64>();
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(i)];
        auto block = block_backend->zeros(
          { codomain->multiplicities[i], domain->multiplicities[j] }, dtype, device);
        bool is_zero_block = true;
        SectorArray coupled_arr = SectorArray::repeat(coupled, 1);
        for (auto const& xb : codomain->iter_tree_blocks(coupled_arr)) {
            for (auto const& yb : domain->iter_tree_blocks(coupled_arr)) {
                std::pair<FusionTree, FusionTree> pr{ xb.tree, yb.tree };
                auto it = trees.find(pr);
                if (it == trees.end())
                    continue;
                auto tree_block = it->second;
                std::vector<int64> expect_shape = xb.multiplicities;
                for (auto it2 = yb.multiplicities.rbegin(); it2 != yb.multiplicities.rend(); ++it2)
                    expect_shape.push_back(*it2);
                assert(block_backend->get_shape(tree_block) == expect_shape);
                std::vector<int64> perm;
                for (int64 p = 0; p < J; ++p)
                    perm.push_back(p);
                for (int64 p = J + K - 1; p >= J; --p)
                    perm.push_back(p);
                tree_block = block_backend->permute_axes(tree_block, perm);
                tree_block = block_backend->reshape(
                  tree_block, { prod_int(xb.multiplicities), prod_int(yb.multiplicities) });
                b_set(block,
                      py::make_tuple(slice_from_index_slice(xb.slice), slice_from_index_slice(yb.slice)),
                      tree_block);
                is_zero_block = false;
                pairs_done.insert(pr);
            }
        }
        if (is_zero_block)
            continue;
        block_inds_rows.append(py::make_tuple(i, j));
        blocks.push_back(block);
    }
    for (auto const& kv : trees)
        if (!pairs_done.count(kv.first))
            throw std::runtime_error("from_tree_pairs: uncovered tree pair");
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(dtype, std::move(device), std::move(blocks), block_inds, true));
}

BlockBackend::Scalar
FusionTreeBackend::get_element(py::object a, std::vector<int64> idcs)
{
    py::module_::import("warnings").attr("warn")(
      "Accessing individual entries in the FusionTreeBackend is comparably expensive.",
      py::arg("category") = py::module_::import("builtins").attr("UserWarning"),
      py::arg("stacklevel") = 2);
    auto np = numpy();
    py::list flat_idcs;
    py::list legs = a.attr("legs");
    for (py::ssize_t li = 0; li < py::len(legs); ++li) {
        py::object leg = legs[li];
        py::list dims;
        for (py::handle fl : leg.attr("flat_legs"))
            dims.append(fl.attr("dim"));
        auto unr = np.attr("unravel_index")(idcs[static_cast<std::size_t>(li)], dims);
        for (py::handle u : unr)
            flat_idcs.append(u);
    }
    int64 num_cod_legs = a.attr("num_codomain_flat_legs").cast<int64>();
    int64 num_legs = a.attr("num_flat_legs").cast<int64>();
    py::list a_legs;
    for (py::handle l : a.attr("codomain").attr("flat_legs"))
        a_legs.append(l);
    for (py::handle l : a.attr("domain").attr("flat_legs"))
        a_legs.append(l);
    py::list rev_domain;
    for (py::ssize_t i = py::len(flat_idcs) - 1; i >= num_cod_legs; --i)
        rev_domain.append(flat_idcs[i]);
    py::list flat_ordered;
    for (py::ssize_t i = 0; i < num_cod_legs; ++i)
        flat_ordered.append(flat_idcs[i]);
    for (py::handle h : rev_domain)
        flat_ordered.append(h);
    py::list rows;
    for (py::ssize_t i = 0; i < py::len(a_legs); ++i)
        rows.append(a_legs[i].attr("parse_index")(flat_ordered[i]));
    auto pos = asarray_i64(np.attr("array")(rows));
    py::list uncoupled_list;
    for (py::ssize_t i = 0; i < py::len(a_legs); ++i)
        uncoupled_list.append(a_legs[i].attr("sector_decomposition").attr("__getitem__")(
          pos.attr("__getitem__")(py::make_tuple(i, 0))));
    SectorArray uncoupled = as_sector_array_from_list(uncoupled_list);
    SectorArray codom_uncoupled = uncoupled.slice(0, static_cast<std::size_t>(num_cod_legs));
    SectorArray dom_uncoupled = uncoupled.slice(static_cast<std::size_t>(num_cod_legs), uncoupled.size());
    std::vector<int64> mults, dims_vec;
    for (py::ssize_t i = 0; i < py::len(a_legs); ++i) {
        int64 si = pos.attr("__getitem__")(py::make_tuple(i, 0)).cast<int64>();
        mults.push_back(mults_of(a_legs[i])[static_cast<std::size_t>(si)]);
    }
    auto sym = a.attr("symmetry").cast<Symmetry::Ptr>();
    auto batch_dims = sym->batch_sector_dim(uncoupled);
    std::vector<int64> codom_dims(batch_dims.begin(), batch_dims.begin() + num_cod_legs);
    std::vector<int64> dom_dims(batch_dims.begin() + num_cod_legs, batch_dims.end());
    std::vector<int64> codom_mults(mults.begin(), mults.begin() + num_cod_legs);
    std::vector<int64> dom_mults(mults.begin() + num_cod_legs, mults.end());
    std::vector<int64> shape;
    for (std::size_t i = 0; i < batch_dims.size(); ++i)
        shape.push_back(batch_dims[i] * mults[i]);
    auto a_data = data_from_tensor(a);
    Dtype dt = dtype::common({ a_data->dtype, sym->fusion_tensor_dtype.value_or(Dtype::Complex128) });
    auto forest_block = block_backend->zeros(shape, dt, a.attr("device").cast<std::string>());
    auto codomain = a.attr("codomain").cast<TensorProduct::Ptr>();
    auto domain = a.attr("domain").cast<TensorProduct::Ptr>();
    int64 tree_block_height = codomain->tree_block_size(codom_uncoupled);
    int64 tree_block_width = domain->tree_block_size(dom_uncoupled);
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n) {
        int64 bi_cod = buf(n, 0);
        Sector coupled = codomain->sector_decomposition[static_cast<std::size_t>(bi_cod)];
        int64 i1 = codomain->forest_block_slice(codom_uncoupled, coupled).start;
        int64 i2 = domain->forest_block_slice(dom_uncoupled, coupled).start;
        auto [entries, _, __] = _get_forest_block_contribution(
          a_data->blocks[static_cast<std::size_t>(n)], sym, codomain, domain, coupled,
          py::cast(codom_uncoupled), py::cast(dom_uncoupled), codom_dims, dom_dims,
          tree_block_width, tree_block_height, i1, i2, codom_mults, dom_mults, dt);
        std::vector<int64> perm;
        for (int64 i = 0; i < num_legs; ++i) {
            perm.push_back(i);
            perm.push_back(i + num_legs);
        }
        entries = block_backend->permute_axes(entries, perm);
        entries = block_backend->reshape(entries, shape);
        forest_block = (*forest_block) + (*entries);
    }
    py::list idx_within;
    for (py::ssize_t i = 0; i < pos.attr("shape").attr("__getitem__")(0).cast<py::ssize_t>(); ++i)
        idx_within.append(pos.attr("__getitem__")(py::make_tuple(i, 1)));
    return block_backend->get_block_element(forest_block, idx_within.cast<std::vector<int64>>());
}


TensorBackend::DataPtr
FusionTreeBackend::from_grid(std::vector<std::vector<py::object>> grid,
                             TensorProduct::Ptr new_codomain,
                             TensorProduct::Ptr new_domain,
                             std::vector<std::vector<int64>> left_mult_slices,
                             std::vector<std::vector<int64>> right_mult_slices,
                             Dtype dtype,
                             std::string device)
{
    auto data = unwrap(zero_data(new_codomain, new_domain, dtype, device, true));
    auto new_codomain_legs = new_codomain->flat_legs();
    auto new_domain_legs = new_domain->flat_legs();
    for (std::size_t i = 0; i < grid.size(); ++i) {
        for (std::size_t j = 0; j < grid[i].size(); ++j) {
            py::object op = grid[i][j];
            if (op.is_none())
                continue;
            auto op_data = data_from_tensor(op);
            auto op_codomain = op.attr("codomain").cast<TensorProduct::Ptr>();
            auto op_domain = op.attr("domain").cast<TensorProduct::Ptr>();
            py::list op_coupled_list;
            auto op_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(op_data->block_inds);
            auto op_buf = op_bi.unchecked<2>();
            for (py::ssize_t r = 0; r < op_buf.shape(0); ++r)
                op_coupled_list.append(
                  op_domain->sector_decomposition[static_cast<std::size_t>(op_buf(r, 1))]);
            SectorArray op_coupled = as_sector_array_from_list(op_coupled_list);
            for (auto const& fb : op_codomain->iter_forest_blocks(op_coupled)) {
                Sector coupled = op_coupled[static_cast<std::size_t>(fb.coupled_idx)];
                auto block_idx = data->block_ind_from_coupled(coupled, new_domain);
                if (!block_idx.has_value())
                    continue;
                std::vector<int64> op_codom_shape;
                for (std::size_t l = 0; l < fb.uncoupled.size(); ++l)
                    op_codom_shape.push_back(
                      py::cast(op_codomain->flat_legs()[l])
                        .attr("sector_multiplicity")(py::cast(fb.uncoupled[l]))
                        .cast<int64>());
                int64 op_codom0 = op_codom_shape.empty() ? 1 : op_codom_shape[0];
                int64 op_codom_rest = prod_int(std::vector<int64>(op_codom_shape.begin() + 1, op_codom_shape.end()));
                op_codom_shape = { op_codom0, op_codom_rest };
                int64 op_num_codom_trees =
                  (fb.slice.stop - fb.slice.start) / std::max<int64>(1, prod_int(op_codom_shape));
                std::vector<int64> codom_shape = {
                  py::cast(new_codomain_legs[0])
                    .attr("sector_multiplicity")(py::cast(fb.uncoupled[0]))
                    .cast<int64>(),
                  op_codom_rest };
                auto codom_slc = new_codomain->forest_block_slice(fb.uncoupled, coupled);
                int64 num_codom_trees =
                  (codom_slc.stop - codom_slc.start) / std::max<int64>(1, prod_int(codom_shape));
                auto codom_leg_idx = py::cast(new_codomain_legs[0])
                                       .attr("sector_decomposition_where")(py::cast(fb.uncoupled[0]))
                                       .cast<int64>();
                py::slice codom_leg_slc(left_mult_slices[static_cast<std::size_t>(codom_leg_idx)][i],
                                        left_mult_slices[static_cast<std::size_t>(codom_leg_idx)][i + 1],
                                        1);
                for (auto const& fb_dom : op_domain->iter_forest_blocks(SectorArray::repeat(coupled, 1))) {
                    std::vector<int64> op_dom_shape;
                    for (std::size_t l = 0; l < fb_dom.uncoupled.size(); ++l)
                        op_dom_shape.push_back(
                          py::cast(op_domain->flat_legs()[l])
                            .attr("sector_multiplicity")(py::cast(fb_dom.uncoupled[l]))
                            .cast<int64>());
                    int64 op_dom_last = op_dom_shape.empty() ? 1 : op_dom_shape.back();
                    int64 op_dom_rest = prod_int(std::vector<int64>(op_dom_shape.begin(), op_dom_shape.end() - 1));
                    op_dom_shape = { op_dom_rest, op_dom_last };
                    int64 op_num_dom_trees =
                      (fb_dom.slice.stop - fb_dom.slice.start) / std::max<int64>(1, prod_int(op_dom_shape));
                    std::vector<int64> op_new_shape = { op_num_codom_trees, op_codom0, op_codom_rest,
                                                        op_num_dom_trees, op_dom_rest, op_dom_last };
                    std::vector<int64> dom_shape = { op_dom_rest,
                      py::cast(new_domain_legs.back())
                        .attr("sector_multiplicity")(
                          py::cast(fb_dom.uncoupled[fb_dom.uncoupled.size() - 1]))
                        .cast<int64>() };
                    auto dom_slc = new_domain->forest_block_slice(fb_dom.uncoupled, coupled);
                    int64 num_dom_trees =
                      (dom_slc.stop - dom_slc.start) / std::max<int64>(1, prod_int(dom_shape));
                    std::vector<int64> new_shape = { num_codom_trees, codom_shape[0], codom_shape[1],
                                                     num_dom_trees, dom_shape[0], dom_shape[1] };
                    auto dom_leg_idx = py::cast(new_domain_legs.back())
                                         .attr("sector_decomposition_where")(
                                           py::cast(fb_dom.uncoupled[fb_dom.uncoupled.size() - 1]))
                                         .cast<int64>();
                    py::slice dom_leg_slc(right_mult_slices[static_cast<std::size_t>(dom_leg_idx)][j],
                                          right_mult_slices[static_cast<std::size_t>(dom_leg_idx)][j + 1],
                                          1);
                    auto op_block = b_get(op_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                                          py::make_tuple(slice_from_index_slice(fb.slice),
                                                         slice_from_index_slice(fb_dom.slice)));
                    op_block = block_backend->reshape(op_block, op_new_shape);
                    auto block = block_backend->copy_block(
                      b_get(data->blocks[static_cast<std::size_t>(*block_idx)],
                            py::make_tuple(slice_from_index_slice(codom_slc),
                                           slice_from_index_slice(dom_slc))),
                      device);
                    auto final_shape = block_backend->get_shape(block);
                    block = block_backend->reshape(block, new_shape);
                    auto cur = b_get(block, py::make_tuple(py::slice(0, std::nullopt, 1),
                                                           codom_leg_slc,
                                                           py::slice(0, std::nullopt, 1),
                                                           py::slice(0, std::nullopt, 1),
                                                           py::slice(0, std::nullopt, 1),
                                                           dom_leg_slc));
                    b_set(block,
                          py::make_tuple(py::slice(0, std::nullopt, 1),
                                         codom_leg_slc,
                                         py::slice(0, std::nullopt, 1),
                                         py::slice(0, std::nullopt, 1),
                                         py::slice(0, std::nullopt, 1),
                                         dom_leg_slc),
                          (*cur) + (*op_block));
                    block = block_backend->reshape(block, final_shape);
                    b_set(data->blocks[static_cast<std::size_t>(*block_idx)],
                          py::make_tuple(slice_from_index_slice(codom_slc),
                                         slice_from_index_slice(dom_slc)),
                          block);
                }
            }
        }
    }
    data->discard_zero_blocks(block_backend, eps);
    return wrap(data);
}

TensorBackend::DataPtr
FusionTreeBackend::scale_axis(py::object a, py::object b, int64 leg)
{
    Dtype dtype = dtype::common({ a.attr("dtype").cast<Dtype>(), b.attr("dtype").cast<Dtype>() });
    py::object parsed = a.attr("_parse_leg_idx")(leg);
    bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 co_domain_idx = parsed.attr("__getitem__")(1).cast<int64>();
    int64 ax_a = in_domain ? 1 : 0;
    auto a_data = data_from_tensor(a);
    auto b_data = data_from_tensor(b);
    auto a_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto a_buf = a_bi.unchecked<2>();
    py::array b_block_inds_col = b_data->block_inds.attr("__getitem__")(
      py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)));
    int64 num_cod = a.attr("num_codomain_legs").cast<int64>();
    int64 num_dom = a.attr("domain").attr("num_factors").cast<int64>();
    if ((in_domain && num_dom == 1) || (!in_domain && num_cod == 1)) {
        py::list block_inds_rows;
        std::vector<BlockBackend::BlockPtr> blocks;
        py::array a_contr = in_domain ? a_data->block_inds.attr("__getitem__")(
                                          py::make_tuple(py::ellipsis(), py::slice(1, 2, 1)))
                                      : a_data->block_inds.attr("__getitem__")(
                                          py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)));
        py::array a_open = in_domain ? a_data->block_inds.attr("__getitem__")(
                                       py::make_tuple(py::ellipsis(), py::slice(0, 1, 1)))
                                     : a_data->block_inds.attr("__getitem__")(
                                         py::make_tuple(py::ellipsis(), py::slice(1, 2, 1)));
        if (a_buf.shape(0) > 0 && b_data->block_inds.shape(0) > 0) {
            for (py::handle item : misc().attr("iter_common_sorted")(a_contr, b_block_inds_col)) {
                auto pair = item.cast<py::tuple>();
                int64 n_a = pair[0].cast<int64>();
                int64 n_b = pair[1].cast<int64>();
                blocks.push_back(block_backend->scale_axis(a_data->blocks[static_cast<std::size_t>(n_a)],
                                                           b_data->blocks[static_cast<std::size_t>(n_b)],
                                                           ax_a));
                if (in_domain)
                    block_inds_rows.append(py::make_tuple(a_buf(n_a, 0), n_b));
                else
                    block_inds_rows.append(py::make_tuple(n_b, a_buf(n_a, 1)));
            }
        }
        py::array block_inds = block_inds_rows.size() > 0
          ? numpy().attr("array")(block_inds_rows).cast<py::array>()
          : zeros_i64(0, 2).cast<py::array>();
        return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
    }
    TensorProduct::Ptr iter_space = in_domain ? a.attr("domain").cast<TensorProduct::Ptr>()
                                              : a.attr("codomain").cast<TensorProduct::Ptr>();
    if (a.attr("has_pipes").cast<bool>()) {
        for (int64 i = 0; i < co_domain_idx; ++i)
            co_domain_idx += static_cast<int64>(iter_space->flat_leg_idcs(i).size()) - 1;
        py::list flat_factors;
        for (auto const& fl : iter_space->flat_legs())
            flat_factors.append(py::cast(fl));
        iter_space = std::make_shared<TensorProduct>(
          flat_factors.cast<std::vector<py::object>>(),
          iter_space->symmetry,
          iter_space->sector_decomposition,
          iter_space->multiplicities);
    }
    py::list coupled_list;
    for (py::ssize_t r = 0; r < a_buf.shape(0); ++r)
        coupled_list.append(a.attr("codomain").attr("sector_decomposition").attr("__getitem__")(a_buf(r, 0)));
    SectorArray coupled_sectors = as_sector_array_from_list(coupled_list);
    std::vector<BlockBackend::BlockPtr> blocks;
    py::list block_inds_rows;
    std::map<int64, std::size_t> ind_mapping;
    std::set<int64> domain_inds_seen;
    for (auto const& fb : iter_space->iter_forest_blocks(coupled_sectors)) {
        int64 ind = a_buf(fb.coupled_idx, 1);
        auto ind_b = b_data->block_ind_from_coupled(fb.uncoupled[static_cast<std::size_t>(co_domain_idx)],
                                                    b.attr("domain").cast<TensorProduct::Ptr>());
        if (!ind_b.has_value())
            continue;
        if (!domain_inds_seen.count(ind)) {
            domain_inds_seen.insert(ind);
            ind_mapping[fb.coupled_idx] = blocks.size();
            block_inds_rows.append(py::make_tuple(a_buf(fb.coupled_idx, 0), ind));
            blocks.push_back(block_backend->zeros(block_backend->get_shape(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)]),
                                                  dtype));
        }
        std::vector<int64> reshape;
        for (std::size_t li = 0; li < fb.uncoupled.size(); ++li)
            reshape.push_back(py::cast(iter_space->flat_legs()[li])
                                .attr("sector_multiplicity")(py::cast(fb.uncoupled[li]))
                                .cast<int64>());
        BlockBackend::BlockPtr forest;
        std::vector<int64> initial_shape;
        py::object slcs0, slcs1;
        if (in_domain) {
            forest = b_get(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                           py::make_tuple(py::slice(std::nullopt, std::nullopt, 1),
                                          slice_from_index_slice(fb.slice)));
            initial_shape = block_backend->get_shape(forest);
            std::vector<int64> rshape = { initial_shape[0], -1 };
            rshape.insert(rshape.end(), reshape.begin(), reshape.end());
            forest = block_backend->reshape(forest, rshape);
            slcs0 = py::slice(0, initial_shape[0], 1);
            slcs1 = slice_from_index_slice(fb.slice);
        } else {
            forest = b_get(a_data->blocks[static_cast<std::size_t>(fb.coupled_idx)],
                           py::make_tuple(slice_from_index_slice(fb.slice),
                                          py::slice(std::nullopt, std::nullopt, 1)));
            initial_shape = block_backend->get_shape(forest);
            std::vector<int64> rshape = { -1 };
            rshape.insert(rshape.end(), reshape.begin(), reshape.end());
            rshape.push_back(initial_shape[1]);
            forest = block_backend->reshape(forest, rshape);
            slcs0 = slice_from_index_slice(fb.slice);
            slcs1 = py::slice(0, initial_shape[1], 1);
        }
        forest = block_backend->scale_axis(forest,
                                           b_data->blocks[static_cast<std::size_t>(*ind_b)],
                                           ax_a + co_domain_idx + 1);
        forest = block_backend->reshape(forest, initial_shape);
        b_set(blocks[ind_mapping.at(fb.coupled_idx)],
              py::make_tuple(slcs0, slcs1),
              forest);
    }
    py::array block_inds = block_inds_rows.size() > 0
      ? numpy().attr("array")(block_inds_rows).cast<py::array>()
      : zeros_i64(0, 2).cast<py::array>();
    return wrap(make_data(dtype, a_data->device, std::move(blocks), block_inds, true));
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::diagonal_to_mask(py::object tens)
{
    auto tens_data = data_from_tensor(tens);
    py::object large_leg = tens.attr("leg");
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> codom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(tens_data->block_inds);
    auto buf = bi.unchecked<2>();
    auto sym = tens.attr("symmetry").cast<Symmetry::Ptr>();
    for (py::ssize_t n = 0; n < buf.shape(0); ++n) {
        auto const& diag_block = tens_data->blocks[static_cast<std::size_t>(n)];
        if (!block_backend->any(diag_block))
            continue;
        int64 bii = buf(n, 0);
        Sector sector = tens.attr("codomain").attr("sector_decomposition").attr("__getitem__")(bii).cast<Sector>();
        if (large_leg.attr("is_dual").cast<bool>())
            sector = sym->dual_sector(sector);
        blocks.push_back(diag_block);
        codom_block_inds.push_back(bii);
        sectors_vec.push_back(sector);
        multiplicities.push_back(block_backend->sum_all(diag_block).as_int64());
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(sector);
            py::array mask = numpy().attr("tile")(
              block_backend->to_numpy(diag_block, py::module_::import("builtins").attr("bool")), dim);
            if (large_leg.attr("is_dual").cast<bool>()) {
                auto j = large_leg.attr("sector_decomposition_where")(sector);
                bii = j.cast<int64>();
            }
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bii));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    auto np = numpy();
    SectorArray sectors = sym->empty_sector_array;
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    py::array_t<int64> block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!is_sorted) {
            auto perm = sectors.lexsort_indices();
            sectors = sectors.take(perm);
            std::vector<int64> new_mults, new_cod_inds;
            py::list new_ranks;
            for (std::size_t p : perm) {
                new_mults.push_back(multiplicities[p]);
                new_cod_inds.push_back(codom_block_inds[p]);
                if (!basis_perm.is_none())
                    new_ranks.append(basis_perm_ranks[p]);
            }
            multiplicities = std::move(new_mults);
            codom_block_inds = std::move(new_cod_inds);
            if (!basis_perm.is_none())
                basis_perm_ranks = new_ranks;
        }
        if (!basis_perm.is_none()) {
            auto ranked = misc().attr("rank_data")(np.attr("concatenate")(basis_perm_ranks));
            basis_perm_opt = ranked.cast<std::vector<int64>>();
        }
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(sectors.size()), codom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, tens_data->device, std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      sym, std::move(sectors), multiplicities, large_leg.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_binary_operand(py::object mask1, py::object mask2, py::function func)
{
    py::object large_leg = mask1.attr("large_leg");
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    auto mask1_data = data_from_tensor(mask1);
    auto mask2_data = data_from_tensor(mask2);
    auto mask1_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(mask1_data->block_inds);
    auto mask2_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(mask2_data->block_inds);
    auto m1 = mask1_bi.unchecked<2>();
    auto m2 = mask2_bi.unchecked<2>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    int64 i1 = 0, i2 = 0;
    int64 b1_i1 = m1.shape(0) == 0 ? -1 : m1(0, 1);
    int64 b2_i2 = m2.shape(0) == 0 ? -1 : m2(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto mults = mults_of(large_leg);
    auto sym = mask1.attr("symmetry").cast<Symmetry::Ptr>();
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block1, block2;
        bool block1_found = false, block2_found = false;
        if (is_sorted && static_cast<int64>(sector_idx) == b1_i1) {
            block1 = mask1_data->blocks[static_cast<std::size_t>(i1)];
            ++i1;
            b1_i1 = i1 >= m1.shape(0) ? -1 : m1(i1, 1);
            block1_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
            auto dom_idx = mask1.attr("domain").attr("sector_decomposition_where")(dual_sec);
            if (!dom_idx.is_none()) {
                auto idx = mask1_data->block_ind_from_domain_sector_ind(dom_idx.cast<int64>());
                if (idx.has_value()) {
                    block1 = mask1_data->blocks[static_cast<std::size_t>(*idx)];
                    block1_found = true;
                }
            }
        }
        if (!block1_found)
            block1 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        if (is_sorted && static_cast<int64>(sector_idx) == b2_i2) {
            block2 = mask2_data->blocks[static_cast<std::size_t>(i2)];
            ++i2;
            b2_i2 = i2 >= m2.shape(0) ? -1 : m2(i2, 1);
            block2_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
            auto dom_idx = mask2.attr("domain").attr("sector_decomposition_where")(dual_sec);
            if (!dom_idx.is_none()) {
                auto idx = mask2_data->block_ind_from_domain_sector_ind(dom_idx.cast<int64>());
                if (idx.has_value()) {
                    block2 = mask2_data->blocks[static_cast<std::size_t>(*idx)];
                    block2_found = true;
                }
            }
        }
        if (!block2_found)
            block2 = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        auto new_block = func(py::cast(block1), py::cast(block2)).cast<BlockBackend::BlockPtr>();
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        dom_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(defining[sector_idx]);
            py::array mask = numpy().attr("tile")(
              block_backend->to_numpy(new_block, py::module_::import("builtins").attr("bool")), dim);
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    auto np = numpy();
    SectorArray sectors = sym->empty_sector_array;
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    py::array_t<int64> block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = misc().attr("rank_data")(np.attr("concatenate")(basis_perm_ranks));
            basis_perm_opt = ranked.cast<std::vector<int64>>();
        }
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(sectors.size()), dom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, mask1.attr("device").cast<std::string>(), std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      sym, std::move(sectors), multiplicities, large_leg.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_from_block(BlockBackend::BlockPtr a, Space::Ptr large_leg)
{
    py::object large_leg_obj = py::cast(large_leg);
    py::object basis_perm = large_leg_obj.attr("_basis_perm");
    bool is_sorted = !large_leg_obj.attr("is_dual").cast<bool>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    auto defining = large_leg_obj.attr("defining_sectors").cast<SectorArray>();
    auto slices = large_leg_obj.attr("slices");
    py::object domain;
    if (!is_sorted) {
        auto perm = large_leg_obj.attr("sector_decomposition").attr("lexsort_indices")();
        auto sorted_duals = large_leg_obj.attr("sector_decomposition").attr("__getitem__")(perm);
        auto multis = large_leg_obj.attr("multiplicities").attr("__getitem__")(perm);
        domain = py::module_::import("cyten.symmetries.spaces")
                   .attr("TensorProduct")(
                     py::make_tuple(large_leg_obj), py::arg("symmetry") = large_leg->symmetry,
                     py::arg("_sector_decomposition") = sorted_duals,
                     py::arg("_multiplicities") = multis);
    }
    for (std::size_t bi_large = 0; bi_large < defining.size(); ++bi_large) {
        auto slc = slice_pair(slices.attr("__getitem__")(static_cast<int64>(bi_large)));
        auto block = b_get(a, slc);
        int64 mult = block_backend->sum_all(block).as_int64();
        if (mult == 0)
            continue;
        Sector sector = defining[bi_large];
        int64 bi_out = static_cast<int64>(bi_large);
        if (!is_sorted) {
            Sector dual_sector = large_leg->symmetry->dual_sector(sector);
            bi_out = domain.attr("sector_decomposition_where")(py::cast(dual_sector)).cast<int64>();
        }
        dom_block_inds.push_back(bi_out);
        sectors_vec.push_back(sector);
        int64 dim = large_leg->symmetry->sector_dim(sector);
        int64 stop = block_backend->get_shape(block)[0] / dim;
        blocks.push_back(b_get(block, py::slice(0, stop, 1)));
        multiplicities.push_back(mult / dim);
        if (!basis_perm.is_none()) {
            py::array mask = block_backend->to_numpy(block).cast<py::array>();
            basis_perm_ranks.append(
              large_leg_obj.attr("basis_perm").attr("__getitem__")(slc).attr("__getitem__")(mask));
        }
    }
    auto np = numpy();
    SectorArray sectors = large_leg->symmetry->empty_sector_array;
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    py::array_t<int64> block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = misc().attr("rank_data")(np.attr("concatenate")(basis_perm_ranks));
            basis_perm_opt = ranked.cast<std::vector<int64>>();
        }
        if (!is_sorted) {
            auto perm = np.attr("argsort")(dom_block_inds);
            py::list new_dom, new_blocks;
            std::vector<int64> new_mults;
            for (py::handle p : perm) {
                int64 pi = p.cast<int64>();
                new_dom.append(dom_block_inds[static_cast<std::size_t>(pi)]);
                new_blocks.append(blocks[static_cast<std::size_t>(pi)]);
                new_mults.push_back(multiplicities[static_cast<std::size_t>(pi)]);
            }
            dom_block_inds.clear();
            for (py::handle h : new_dom)
                dom_block_inds.push_back(h.cast<int64>());
            blocks.clear();
            for (py::handle h : new_blocks)
                blocks.push_back(h.cast<BlockBackend::BlockPtr>());
            multiplicities = std::move(new_mults);
        }
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(sectors.size()), dom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, block_backend->get_device(a), std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      large_leg->symmetry, std::move(sectors), multiplicities,
      large_leg_obj.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}

BlockBackend::BlockPtr
FusionTreeBackend::mask_to_block(py::object a)
{
    auto a_data = data_from_tensor(a);
    auto large_leg = a.attr("large_leg");
    auto res = block_backend->zeros({ large_leg.attr("dim").cast<int64>() }, Dtype::Bool);
    bool is_projection = a.attr("is_projection").cast<bool>();
    auto bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(a_data->block_inds);
    auto buf = bi.unchecked<2>();
    auto co_dom = is_projection ? a.attr("domain") : a.attr("codomain");
    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        int64 bi_large = is_projection ? buf(i, 1) : buf(i, 0);
        Sector sector = co_dom.attr("sector_decomposition").attr("__getitem__")(bi_large).cast<Sector>();
        int64 dim = co_dom.attr("symmetry").attr("sector_dim")(sector).cast<int64>();
        if (large_leg.attr("is_dual").cast<bool>())
            bi_large = large_leg.attr("sector_decomposition_where")(sector).cast<int64>();
        auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(bi_large));
        b_set(res, slc, block_backend->tile(a_data->blocks[static_cast<std::size_t>(i)], dim));
    }
    return res;
}

std::tuple<Space::Ptr, Space::Ptr, TensorBackend::DataPtr>
FusionTreeBackend::mask_transpose(py::object tens)
{
    auto data = data_from_tensor(tens);
    auto sym = tens.attr("symmetry").cast<Symmetry::Ptr>();
    auto perm_dom = sym->dual_sectors(tens.attr("domain").attr("sector_decomposition").cast<SectorArray>())
                      .lexsort_indices();
    auto perm_codom = sym->dual_sectors(tens.attr("codomain").attr("sector_decomposition").cast<SectorArray>())
                        .lexsort_indices();
    auto inv_dom = misc().attr("inverse_permutation")(py::cast(perm_dom));
    auto inv_codom = misc().attr("inverse_permutation")(py::cast(perm_codom));
    auto np = numpy();
    py::array col1 = inv_dom.attr("__getitem__")(
      data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 1)));
    py::array col0 = inv_codom.attr("__getitem__")(
      data->block_inds.attr("__getitem__")(py::make_tuple(py::ellipsis(), 0)));
    auto block_inds = asarray_i64(np.attr("column_stack")(py::make_tuple(col1, col0)));
    auto out = make_data(tens.attr("dtype").cast<Dtype>(), data->device, data->blocks, block_inds, false);
    return { tens.attr("codomain").attr("__getitem__")(0).attr("dual").cast<Space::Ptr>(),
             tens.attr("domain").attr("__getitem__")(0).attr("dual").cast<Space::Ptr>(),
             wrap(out) };
}

std::tuple<TensorBackend::DataPtr, ElementarySpace::Ptr>
FusionTreeBackend::mask_unary_operand(py::object mask, py::function func)
{
    py::object large_leg = mask.attr("large_leg");
    py::object basis_perm = large_leg.attr("_basis_perm");
    bool is_sorted = !large_leg.attr("is_dual").cast<bool>();
    auto mask_data = data_from_tensor(mask);
    auto mask_bi = py::array_t<int64, py::array::c_style | py::array::forcecast>::ensure(mask_data->block_inds);
    auto mbuf = mask_bi.unchecked<2>();
    std::vector<BlockBackend::BlockPtr> blocks;
    std::vector<int64> dom_block_inds;
    std::vector<Sector> sectors_vec;
    std::vector<int64> multiplicities;
    py::list basis_perm_ranks;
    int64 i = 0;
    int64 b_i = mbuf.shape(0) == 0 ? -1 : mbuf(0, 1);
    auto defining = large_leg.attr("defining_sectors").cast<SectorArray>();
    auto mults = mults_of(large_leg);
    auto sym = mask.attr("symmetry").cast<Symmetry::Ptr>();
    for (std::size_t sector_idx = 0; sector_idx < defining.size(); ++sector_idx) {
        BlockBackend::BlockPtr block;
        bool block_found = false;
        if (is_sorted && static_cast<int64>(sector_idx) == b_i) {
            block = mask_data->blocks[static_cast<std::size_t>(i)];
            ++i;
            b_i = i >= mbuf.shape(0) ? -1 : mbuf(i, 1);
            block_found = true;
        } else if (!is_sorted) {
            Sector dual_sec = large_leg.attr("sector_decomposition").attr("__getitem__")(sector_idx).cast<Sector>();
            auto idx = mask_data->block_ind_from_coupled(dual_sec, mask.attr("domain").cast<TensorProduct::Ptr>());
            if (idx.has_value()) {
                block = mask_data->blocks[static_cast<std::size_t>(*idx)];
                block_found = true;
            }
        }
        if (!block_found)
            block = block_backend->zeros({ mults[sector_idx] }, Dtype::Bool);
        auto new_block = func(py::cast(block)).cast<BlockBackend::BlockPtr>();
        int64 mult = block_backend->sum_all(new_block).as_int64();
        if (mult == 0)
            continue;
        blocks.push_back(new_block);
        dom_block_inds.push_back(static_cast<int64>(sector_idx));
        sectors_vec.push_back(defining[sector_idx]);
        multiplicities.push_back(mult);
        if (!basis_perm.is_none()) {
            int64 dim = sym->sector_dim(defining[sector_idx]);
            py::array mask_np = numpy().attr("tile")(
              block_backend->to_numpy(new_block, py::module_::import("builtins").attr("bool")), dim);
            auto slc = slice_pair(large_leg.attr("slices").attr("__getitem__")(static_cast<int64>(sector_idx)));
            basis_perm_ranks.append(basis_perm.attr("__getitem__")(slc).attr("__getitem__")(mask_np));
        }
    }
    auto np = numpy();
    SectorArray sectors = sym->empty_sector_array;
    std::optional<std::vector<int64>> basis_perm_opt = std::nullopt;
    py::array_t<int64> block_inds;
    if (blocks.empty()) {
        multiplicities.clear();
        block_inds = zeros_i64(0, 2);
    } else {
        sectors = SectorArray::empty(sectors_vec[0].len());
        for (auto const& s : sectors_vec)
            sectors.push_back(s);
        if (!basis_perm.is_none()) {
            auto ranked = misc().attr("rank_data")(np.attr("concatenate")(basis_perm_ranks));
            basis_perm_opt = ranked.cast<std::vector<int64>>();
        }
        block_inds = asarray_i64(np.attr("column_stack")(
          py::make_tuple(np.attr("arange")(sectors.size()), dom_block_inds)));
    }
    auto data = make_data(Dtype::Bool, mask.attr("device").cast<std::string>(), std::move(blocks), block_inds, true);
    auto small_leg = std::make_shared<ElementarySpace>(
      sym, std::move(sectors), multiplicities, large_leg.attr("is_dual").cast<bool>(), basis_perm_opt);
    return { wrap(data), small_leg };
}
