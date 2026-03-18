#pragma once

#include <cyten/cyten.h>

namespace cyten {

class CheckBaseOp
{
  public:
    virtual ~CheckBaseOp() = default;
    virtual int64 check(int64 i, int64 j) const = 0;
};

class CheckAdd : public CheckBaseOp
{
  public:
    int64 check(int64 i, int64 j) const override { return i + j; }
};
class CheckSub : public CheckBaseOp
{
  public:
    int64 check(int64 i, int64 j) const override { return i - j; }
};

/**
 * A function adding the two arguments. Just there as a test of the surrounding setup.
 */
int64 add(int64 i, int64 j);

int64 apply_check_op(const CheckBaseOp& op, int64 i, int64 j);

}
