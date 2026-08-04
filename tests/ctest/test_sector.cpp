#include <array>
#include <cassert>
#include <iostream>
#include <unordered_set>

#include <cyten/symmetries/sector.h>

using namespace cyten;

int
test_sector(int /*argc*/, char** /*args*/)
{
    static_assert(sizeof(Sector) == 16);
    static_assert(max_sector_ind_len == 7);

    Sector a{ 1, -2, 3 };
    assert(a.len() == 3);
    assert(a[0] == 1);
    assert(a[1] == -2);
    assert(a[2] == 3);

    auto sp = a.as_span<3>();
    assert(sp[0] == 1);
    assert(sp.size() == 3);

    Sector prod{ 10, 20, 30, 40 };
    auto factor = prod.subspan<2>(1);
    assert(factor[0] == 20);
    assert(factor[1] == 30);

    Sector b{ 1, -2, 3 };
    assert(a == b);
    assert((a <=> b) == 0);
    assert((a < Sector{ 1, -2, 4 }));

    std::unordered_set<Sector> set;
    set.insert(a);
    assert(set.contains(b));

    SectorArray arr(2, 3);
    arr.set(0, Sector{ 1, 2, 3 });
    arr.set(1, Sector{ 4, 5, 6 });
    assert((arr[0] == Sector{ 1, 2, 3 }));
    assert(arr.row_as_span<3>(1)[2] == 6);

    SectorArray empty = SectorArray::empty(2);
    assert(empty.num_sectors == 0);
    assert(empty.sector_ind_len == 2);

    bool threw = false;
    try {
        Sector too_long{ 0, 1, 2, 3, 4, 5, 6, 7 };
        (void)too_long;
    } catch (std::invalid_argument const&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        std::array<int16_t, 8> buf{};
        (void)Sector::from_span(buf);
    } catch (std::invalid_argument const&) {
        threw = true;
    }
    assert(threw);

    std::cout << "test_sector passed." << std::endl;
    return 0;
}
