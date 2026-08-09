#include <cyten/backends/fusion_tree_permute.h>

#include <cyten/tools.h>

#include <algorithm>
#include <cassert>
#include <numeric>
#include <set>
#include <stdexcept>

namespace cyten {

namespace {

[[nodiscard]] std::vector<int64>
inverse_permutation(std::vector<int64> const& perm)
{
    std::vector<int64> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        inv[static_cast<std::size_t>(perm[i])] = static_cast<int64>(i);
    }
    return inv;
}

[[nodiscard]] std::vector<int64>
permutation_as_swaps(std::vector<int64> const& permutation)
{
    int64 const N = static_cast<int64>(permutation.size());
    std::set<int64> seen(permutation.begin(), permutation.end());
    if (static_cast<int64>(seen.size()) != N) {
        throw std::invalid_argument("permutation_as_swaps: not a permutation");
    }
    for (int64 x : seen) {
        if (x < 0 || x >= N) {
            throw std::invalid_argument("permutation_as_swaps: not a permutation");
        }
    }

    std::vector<int64> current_positions(N);
    std::iota(current_positions.begin(), current_positions.end(), 0);
    std::vector<int64> swaps;
    for (int64 target_pos = 0; target_pos < N - 1; ++target_pos) {
        int64 const original_pos = permutation[static_cast<std::size_t>(target_pos)];
        int64 const current_pos = current_positions[static_cast<std::size_t>(original_pos)];
        for (int64 j = current_pos; j > target_pos; --j) {
            swaps.push_back(j - 1);
        }
        std::vector<int64> perm(N);
        std::iota(perm.begin(), perm.end(), 0);
        for (int64 j = target_pos; j <= current_pos; ++j) {
            if (j < current_pos) {
                perm[static_cast<std::size_t>(j)] = j + 1;
            } else if (j == current_pos) {
                perm[static_cast<std::size_t>(j)] = target_pos;
            }
        }
        std::vector<int64> new_current(N);
        for (int64 p = 0; p < N; ++p) {
            new_current[static_cast<std::size_t>(p)] =
              perm[static_cast<std::size_t>(current_positions[static_cast<std::size_t>(p)])];
        }
        current_positions = std::move(new_current);
    }
    return swaps;
}

} // namespace

PermuteLegsInstructionEngine::PermuteLegsInstructionEngine(
  int64 num_codomain_legs_,
  int64 num_domain_legs_,
  std::vector<int64> codomain_idcs,
  std::vector<int64> domain_idcs,
  std::vector<std::optional<int64>> levels_,
  std::vector<std::optional<bool>> bend_right,
  bool has_symmetric_braid_)
  : num_legs(num_codomain_legs_ + num_domain_legs_)
  , has_symmetric_braid(has_symmetric_braid_)
  , num_codomain_legs(num_codomain_legs_)
  , num_domain_legs(num_domain_legs_)
  , target_positions(num_legs, std::nullopt)
  , should_bend(num_legs, ShouldBend::None)
  , levels(std::move(levels_))
{
    for (std::size_t new_codom_idx = 0; new_codom_idx < codomain_idcs.size(); ++new_codom_idx) {
        int64 const old_idx = codomain_idcs[new_codom_idx];
        target_positions[static_cast<std::size_t>(old_idx)] = static_cast<int64>(new_codom_idx);
        if (old_idx >= num_codomain_legs_) {
            should_bend[static_cast<std::size_t>(old_idx)] =
              bend_right[static_cast<std::size_t>(old_idx)].value() ? ShouldBend::Right : ShouldBend::Left;
        }
    }
    for (std::size_t new_dom_idx = 0; new_dom_idx < domain_idcs.size(); ++new_dom_idx) {
        int64 const old_idx = domain_idcs[new_dom_idx];
        target_positions[static_cast<std::size_t>(old_idx)] = num_legs - 1 - static_cast<int64>(new_dom_idx);
        if (old_idx < static_cast<std::size_t>(num_codomain_legs_)) {
            should_bend[static_cast<std::size_t>(old_idx)] =
              bend_right[static_cast<std::size_t>(old_idx)].value() ? ShouldBend::Right : ShouldBend::Left;
        }
    }
}

std::vector<Instruction>
PermuteLegsInstructionEngine::evaluate_instructions()
{
    assert(instructions.empty());

    auto [num_left_cod, num_right_cod] = do_initial_codomain_permutation();
    do_codomain_bends(num_left_cod, num_right_cod);
    auto [num_left_dom, num_right_dom] = do_domain_permutation();
    do_domain_bends(num_left_dom, num_right_dom);
    do_final_codomain_permutation();

    for (std::size_t i = 0; i < target_positions.size(); ++i) {
        assert(target_positions[i].has_value());
        assert(*target_positions[i] == static_cast<int64>(i));
    }
    assert(std::all_of(should_bend.begin(), should_bend.end(), [](ShouldBend b) {
        return b == ShouldBend::None;
    }));

    return instructions;
}

void
PermuteLegsInstructionEngine::verify(int64 num_codomain_legs_,
                                     int64 num_domain_legs_,
                                     std::vector<int64> const& codomain_idcs,
                                     std::vector<int64> const& domain_idcs) const
{
    std::vector<int64> codomain(num_codomain_legs_);
    std::iota(codomain.begin(), codomain.end(), 0);
    std::vector<int64> domain(num_domain_legs_);
    for (int64 i = 0; i < num_domain_legs_; ++i) {
        domain[static_cast<std::size_t>(i)] = num_codomain_legs_ + num_domain_legs_ - 1 - i;
    }

    for (Instruction const& inst : instructions) {
        std::visit(
          [&](auto const& i) {
              using T = std::decay_t<decltype(i)>;
              if constexpr (std::is_same_v<T, BraidInstruction>) {
                  if (i.codomain) {
                      assert(i.idx >= 0 && i.idx + 1 < static_cast<int64>(codomain.size()));
                      std::swap(codomain[static_cast<std::size_t>(i.idx)],
                                codomain[static_cast<std::size_t>(i.idx + 1)]);
                  } else {
                      assert(i.idx >= 0 && i.idx + 1 < static_cast<int64>(domain.size()));
                      std::swap(domain[static_cast<std::size_t>(i.idx)],
                                domain[static_cast<std::size_t>(i.idx + 1)]);
                  }
              } else if constexpr (std::is_same_v<T, BendInstruction>) {
                  if (i.bend_down) {
                      assert(!domain.empty());
                      codomain.push_back(domain.back());
                      domain.pop_back();
                  } else {
                      assert(!codomain.empty());
                      domain.push_back(codomain.back());
                      codomain.pop_back();
                  }
              } else if constexpr (std::is_same_v<T, TwistInstruction>) {
                  if (i.codomain) {
                      assert(!i.idcs.empty());
                      assert(*std::min_element(i.idcs.begin(), i.idcs.end()) >= 0);
                      assert(*std::max_element(i.idcs.begin(), i.idcs.end())
                             < static_cast<int64>(codomain.size()));
                  } else {
                      assert(!i.idcs.empty());
                      assert(*std::min_element(i.idcs.begin(), i.idcs.end()) >= 0);
                      assert(*std::max_element(i.idcs.begin(), i.idcs.end())
                             < static_cast<int64>(domain.size()));
                  }
              }
          },
          inst);
    }

    assert(codomain == codomain_idcs);
    assert(domain == domain_idcs);
}

bool
PermuteLegsInstructionEngine::compare_levels(int64 idx_1, int64 idx_2) const
{
    if (has_symmetric_braid) {
        return true;
    }
    auto const level_1 = levels[static_cast<std::size_t>(idx_1)];
    auto const level_2 = levels[static_cast<std::size_t>(idx_2)];
    if (!level_1.has_value() || !level_2.has_value()) {
        throw BraidChiralityUnspecifiedError("Legs that braid must have specified levels.");
    }
    if (*level_1 == *level_2) {
        throw BraidChiralityUnspecifiedError("Legs that braid can not have the same level.");
    }
    return *level_1 > *level_2;
}

std::pair<int64, int64>
PermuteLegsInstructionEngine::do_initial_codomain_permutation()
{
    int64 num_left_bends = 0;
    for (int64 leg = 0; leg < num_codomain_legs; ++leg) {
        if (should_bend[static_cast<std::size_t>(leg)] == ShouldBend::Left) {
            move_leg(leg, num_left_bends);
            ++num_left_bends;
        }
    }
    int64 num_right_bends = 0;
    for (int64 leg = num_codomain_legs - 1; leg >= 0; --leg) {
        if (should_bend[static_cast<std::size_t>(leg)] == ShouldBend::Right) {
            move_leg(leg, num_codomain_legs - 1 - num_right_bends);
            ++num_right_bends;
        }
    }
    return { num_left_bends, num_right_bends };
}

void
PermuteLegsInstructionEngine::do_codomain_bends(int64 num_left_bends, int64 num_right_bends)
{
    for (int64 n = 0; n < num_right_bends; ++n) {
        bend(false);
    }
    if (num_left_bends > 0) {
        std::vector<int64> idcs(static_cast<std::size_t>(num_left_bends));
        std::iota(idcs.begin(), idcs.end(), 0);
        instructions.push_back(TwistInstruction{ true, idcs, true });
    }
    for (int64 n = num_left_bends - 1; n >= 0; --n) {
        move_leg(n, num_codomain_legs - 1, true);
        bend(false);
        move_leg(num_codomain_legs, n - num_left_bends, true);
    }
}

std::pair<int64, int64>
PermuteLegsInstructionEngine::do_domain_permutation()
{
    std::vector<int64> perm;
    perm.reserve(static_cast<std::size_t>(num_legs));
    for (int64 i = 0; i < num_codomain_legs; ++i) {
        perm.push_back(i);
    }
    int64 num_right_bends = 0;
    for (int64 i = 0; i < num_legs; ++i) {
        if (should_bend[static_cast<std::size_t>(i)] == ShouldBend::Right) {
            perm.push_back(i);
            ++num_right_bends;
        }
    }
    std::vector<int64> remain_in_domain;
    for (int64 i = num_codomain_legs; i < num_legs; ++i) {
        if (should_bend[static_cast<std::size_t>(i)] == ShouldBend::None) {
            remain_in_domain.push_back(i);
        }
    }
    std::vector<std::size_t> order(remain_in_domain.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return *target_positions[static_cast<std::size_t>(remain_in_domain[a])]
             < *target_positions[static_cast<std::size_t>(remain_in_domain[b])];
    });
    for (std::size_t n : order) {
        perm.push_back(remain_in_domain[n]);
    }
    int64 num_left_bends = 0;
    for (int64 i = 0; i < num_legs; ++i) {
        if (should_bend[static_cast<std::size_t>(i)] == ShouldBend::Left) {
            perm.push_back(i);
            ++num_left_bends;
        }
    }

    for (int64 swap_idx : permutation_as_swaps(perm)) {
        swap(swap_idx);
    }
    return { num_left_bends, num_right_bends };
}

void
PermuteLegsInstructionEngine::do_domain_bends(int64 num_left_bends, int64 num_right_bends)
{
    for (int64 n = 0; n < num_right_bends; ++n) {
        bend(true);
    }
    if (num_left_bends > 0) {
        std::vector<int64> idcs(static_cast<std::size_t>(num_left_bends));
        std::iota(idcs.begin(), idcs.end(), 0);
        instructions.push_back(TwistInstruction{ false, idcs, false });
    }
    for (int64 n = num_left_bends - 1; n >= 0; --n) {
        move_leg(-1 - n, num_codomain_legs, true);
        bend(true);
        move_leg(num_codomain_legs - 1, num_left_bends - 1 - n, true);
    }
}

void
PermuteLegsInstructionEngine::do_final_codomain_permutation()
{
    std::vector<int64> target_cod;
    target_cod.reserve(static_cast<std::size_t>(num_codomain_legs));
    for (int64 j = 0; j < num_codomain_legs; ++j) {
        target_cod.push_back(*target_positions[static_cast<std::size_t>(j)]);
    }
    auto perm = inverse_permutation(target_cod);
    for (int64 swap_idx : permutation_as_swaps(perm)) {
        swap(swap_idx);
    }
}

void
PermuteLegsInstructionEngine::bend(bool bend_down)
{
    instructions.push_back(BendInstruction{ bend_down });
    if (bend_down) {
        assert(should_bend[static_cast<std::size_t>(num_codomain_legs)] != ShouldBend::None);
        should_bend[static_cast<std::size_t>(num_codomain_legs)] = ShouldBend::None;
        ++num_codomain_legs;
        --num_domain_legs;
    } else {
        assert(should_bend[static_cast<std::size_t>(num_codomain_legs - 1)] != ShouldBend::None);
        should_bend[static_cast<std::size_t>(num_codomain_legs - 1)] = ShouldBend::None;
        --num_codomain_legs;
        ++num_domain_legs;
    }
}

void
PermuteLegsInstructionEngine::move_leg(int64 start, int64 goal, std::optional<bool> over)
{
    start = to_valid_idx(start, num_legs);
    goal = to_valid_idx(goal, num_legs);
    assert((start < num_codomain_legs) == (goal < num_codomain_legs));

    if (start < goal) {
        for (int64 j = start; j < goal; ++j) {
            swap(j, over);
        }
    } else if (start > goal) {
        std::optional<bool> over_rev = over;
        if (over_rev.has_value()) {
            over_rev = !*over_rev;
        }
        for (int64 j = goal; j < start; ++j) {
            swap(j, over_rev);
        }
    }
}

void
PermuteLegsInstructionEngine::swap(int64 idx, std::optional<bool> over)
{
    idx = to_valid_idx(idx, num_legs);
    bool const over_val = over.has_value() ? *over : compare_levels(idx, idx + 1);
    if (idx < num_codomain_legs) {
        assert(idx + 1 < num_codomain_legs);
        instructions.push_back(BraidInstruction{ true, idx, over_val });
    } else {
        instructions.push_back(
          BraidInstruction{ false, num_legs - 2 - idx, over_val });
    }

    std::size_t const ia = static_cast<std::size_t>(idx);
    std::size_t const ib = static_cast<std::size_t>(idx + 1);
    std::swap(levels[ia], levels[ib]);
    std::swap(target_positions[ia], target_positions[ib]);
    std::swap(should_bend[ia], should_bend[ib]);
}

} // namespace cyten
