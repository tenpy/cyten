
#include <cassert>
#include <iostream>

#include "check.h"

using namespace cyten;

int
test_setup_check(int argc, char** args)
{
    std::cout << "Testing check setup. Adding 1 and 2 gives:" << std::endl;
    std::cout << cyten::add(1, 2) << std::endl;
    assert(cyten::add(1, 2) == 3);
    assert(cyten::add(1, -1) == 0);
    std::cout << "Test check setup passed." << std::endl;
    return 0;
}
