#include <cassert>
#include <iostream>
#include <optional>
#include <vector>

#include <cyten/backends/block_inds.h>

using namespace cyten;

int
test_block_inds(int /*argc*/, char** /*args*/)
{
    // empty (0, L)
    {
        auto e = BlockInds::empty(3);
        assert(e.nrows() == 0);
        assert(e.ncols() == 3);
        assert(e.empty());
        assert(e.lexsort_indices().empty());
    }

    // scalar (1, 0)
    {
        auto s = BlockInds::zeros(1, 0);
        assert(s.nrows() == 1);
        assert(s.ncols() == 0);
        assert(s.lexsort_indices().size() == 1);
        assert(s.lexsort_indices()[0] == 0);
    }

    // lexsort: last column primary (np.lexsort(T))
    {
        // rows: (0,2), (1,1), (0,1) → sorted by col1 then col0: (0,1), (1,1), (0,2)
        BlockInds a(3, 2);
        a(0, 0) = 0;
        a(0, 1) = 2;
        a(1, 0) = 1;
        a(1, 1) = 1;
        a(2, 0) = 0;
        a(2, 1) = 1;
        auto [sorted, perm] = a.sorted();
        assert(perm.size() == 3);
        assert(perm[0] == 2);
        assert(perm[1] == 1);
        assert(perm[2] == 0);
        assert(sorted(0, 0) == 0 && sorted(0, 1) == 1);
        assert(sorted(1, 0) == 1 && sorted(1, 1) == 1);
        assert(sorted(2, 0) == 0 && sorted(2, 1) == 2);
    }

    // row_where / take / reverse_columns / pack
    {
        BlockInds a(2, 2);
        a(0, 0) = 1;
        a(0, 1) = 2;
        a(1, 0) = 3;
        a(1, 1) = 4;
        std::vector<int64> q{ 3, 4 };
        assert(a.row_where(q).value() == 1);
        auto rev = a.reverse_columns();
        assert(rev(0, 0) == 2 && rev(0, 1) == 1);
        std::vector<int64> strides{ 10, 1 };
        auto packed = a.pack(strides);
        assert(packed[0] == 12);
        assert(packed[1] == 34);
    }

    // iter_common_sorted / noncommon
    {
        BlockInds a(3, 1);
        a(0, 0) = 1;
        a(1, 0) = 3;
        a(2, 0) = 5;
        BlockInds b(3, 1);
        b(0, 0) = 2;
        b(1, 0) = 3;
        b(2, 0) = 6;
        std::vector<std::pair<std::ptrdiff_t, std::ptrdiff_t>> common;
        BlockInds::iter_common_sorted(a, b, true, true, [&](std::ptrdiff_t i, std::ptrdiff_t j) {
            common.emplace_back(i, j);
        });
        assert(common.size() == 1);
        assert(common[0].first == 1 && common[0].second == 1);

        int n_only_a = 0, n_only_b = 0, n_both = 0;
        BlockInds::iter_common_noncommon_sorted(
          a, b, [&](std::optional<std::ptrdiff_t> i, std::optional<std::ptrdiff_t> j) {
              if (i && j) {
                  ++n_both;
              } else if (i) {
                  ++n_only_a;
              } else {
                  ++n_only_b;
              }
          });
        assert(n_both == 1);
        assert(n_only_a == 2);
        assert(n_only_b == 2);
    }

    // column algebra
    {
        BlockInds a = BlockInds::zeros(2, 1);
        a(0, 0) = 7;
        a(1, 0) = 8;
        auto stacked = a.repeat_columns(2);
        assert(stacked.ncols() == 2);
        assert(stacked(0, 0) == 7 && stacked(0, 1) == 7);
        auto ins = a.insert_column(0, 0);
        assert(ins.ncols() == 2);
        assert(ins(0, 0) == 0 && ins(0, 1) == 7);
        auto [left, right] = stacked.hsplit(1);
        assert(left.ncols() == 1 && right.ncols() == 1);
    }

    std::cout << "test_block_inds passed." << std::endl;
    return 0;
}
