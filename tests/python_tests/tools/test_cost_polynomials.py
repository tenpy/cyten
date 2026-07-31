"""Tests for cyten.tools.cost_polynomials (C++ BigOMonomial / BigOPolynomial)."""
# Copyright (C) TeNPy Developers, Apache license

import pytest

from cyten.tools.cost_polynomials import BigOMonomial, BigOPolynomial


def test_monomial_from_str_and_str():
    m = BigOMonomial.from_str('x^2 y^3')
    assert m.factors == {'x': 2, 'y': 3}
    assert str(m) == 'x^2 y^3'
    assert BigOMonomial.from_str(m) == m


def test_monomial_mul_add_eq():
    a = BigOMonomial.from_str('x^2')
    b = BigOMonomial.from_str('y')
    assert a * b == BigOMonomial.from_str('x^2 y')
    assert a * b == BigOMonomial.from_str('y^1 x^2')  # map equality ignores insertion order

    poly = a + b
    assert isinstance(poly, BigOPolynomial)
    assert poly == BigOPolynomial.from_str('x^2 + y')


def test_monomial_is_negligible():
    small = BigOMonomial.from_str('x')
    large = BigOMonomial.from_str('x^2 y')
    other = BigOMonomial.from_str('z')
    assert small.is_negligible(large)
    assert not large.is_negligible(small)
    assert not small.is_negligible(other)
    with pytest.raises(Exception):
        small.is_negligible(large, relations=[])


def test_polynomial_from_str_add_mul():
    p = BigOPolynomial.from_str('x^2 y^3 + x^4')
    assert len(p.terms) == 2
    assert p == BigOPolynomial.from_str('x^4 + x^2 y^3')  # order-independent equality

    q = BigOPolynomial.from_str('y')
    assert p + q == BigOPolynomial.from_str('x^2 y^3 + x^4 + y')
    assert p + 'y' == p + q
    assert 'y' + p == p + q

    assert q * BigOPolynomial.from_str('x') == BigOPolynomial.from_str('x y')
    assert q * 'x' == BigOPolynomial.from_str('x y')


def test_polynomial_equality_ignores_term_order():
    a = BigOPolynomial([BigOMonomial.from_str('x'), BigOMonomial.from_str('y^2')])
    b = BigOPolynomial([BigOMonomial.from_str('y^2'), BigOMonomial.from_str('x')])
    assert a == b
    assert a != BigOPolynomial.from_str('x')
    assert BigOPolynomial.from_str('x') == BigOMonomial.from_str('x')
    assert BigOPolynomial.from_str('x + y') != BigOMonomial.from_str('x')


def test_polynomial_simplify_drops_negligible():
    terms = {
        BigOMonomial.from_str('x^2'),
        BigOMonomial.from_str('x'),  # negligible vs x^2
        BigOMonomial.from_str('y'),
    }
    simplified = BigOPolynomial.simplify_terms(terms)
    assert BigOMonomial.from_str('x') not in simplified
    assert BigOMonomial.from_str('x^2') in simplified
    assert BigOMonomial.from_str('y') in simplified

    p = BigOPolynomial(terms)
    assert p == BigOPolynomial.from_str('x^2 + y')


def test_polynomial_no_duplicate_terms():
    x = BigOMonomial.from_str('x')
    p = BigOPolynomial([x, x, x])
    assert len(p.terms) == 1
    assert p == BigOPolynomial.from_str('x')
    assert BigOPolynomial.from_str('x + x') == BigOPolynomial.from_str('x')


def test_polynomial_prod():
    x = BigOPolynomial.from_str('x')
    y = BigOPolynomial.from_str('y')
    z = BigOPolynomial.from_str('z')
    assert x.prod() == x
    assert x.prod(y) == BigOPolynomial.from_str('x y')
    assert BigOPolynomial.prod(x, y, z) == BigOPolynomial.from_str('x y z')
