
#include "check.h"
#include "cyten/cyten.h"

// #include <iostream>

namespace cyten {

int64
add(int64 i, int64 j)
{
    // if (j < 0)
    // {
    //     std::cout << "j is negative, testing C++ exception handling" << std::endl;
    //     throw std::runtime_error("test exception from cyten/check.cpp:add");
    // }
    return i + j;
}

int64
apply_check_op(const CheckBaseOp& op, int64 i, int64 j)
{
    return op.check(i, j);
}

}
