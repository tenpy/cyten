import cytnx

def test_check():
    x = cytnx.add(1, 1)
    assert x == 2
    assert cytnx.add(1, -1) == 0

