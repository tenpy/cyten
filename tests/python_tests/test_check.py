import pytest

import cyten


def test_check():
    x = cyten.add(1, 1)
    assert x == 2
    assert cyten.add(1, -1) == 0


def test_check_exception():
    with pytest.raises(RuntimeError):
        cyten.add(1, -1)


if __name__ == '__main__':
    test_check_exception()
    test_check()
