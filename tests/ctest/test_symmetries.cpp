
#include "symmetries.h"
#include <iostream>

using namespace cyten;


int test_symmetries(int argc, char ** args) {
    std::cout << "Hello world" << std::endl;
    // std::cout << (BraidingStyle::bosonic < BraidingStyle::anyonic) << std::endl;
    SectorArray x(2);
    x.push_back({0, 0});
    x.push_back({0, 1});
    x.push_back({1, 0});
    for (int i = 0; i < x.size() ; ++ i)
    {
        for (auto j : x[i])
           std::cout << j << " ";
       std::cout<< std::endl;
    }
    return 0;
}


