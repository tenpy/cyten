
#include <iostream>
#include <cassert>
#include <bitset>

#include "cyten/symmetries.h"

using namespace cyten;


int test_symmetries(int argc, char ** args) {
    std::cout << "Hello world" << std::endl;
    // std::cout << (BraidingStyle::bosonic < BraidingStyle::anyonic) << std::endl;
    Sector a =-1, b=-5;
    std::array<int, 2> bit_lengths {4,4};

    Sector c = compress_Sector<2>({a,b}, bit_lengths);
    auto dec = decompress_Sector<2>(c, bit_lengths);
    std::cout << std::bitset<64>(a) << ", " << std::bitset<64>(b) << std::endl
              << " -> " << std::bitset<64>(c) << std::endl
              << " -> " << std::bitset<64>(dec[0]) << ", " << std::bitset<64>(dec[1]) << std::endl;
    assert(a == dec[0]);
    assert(b == dec[1]);
    return 0;
}


