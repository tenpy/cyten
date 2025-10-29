"""TODO"""
from __future__ import annotations
# Copyright (C) TeNPy Developers, Apache license


class BigOMonomial:
    """A symbolic representation of an algorithmic cost as a monomial.

    TODO elaborate
    """

    def __init__(self, factors: dict[str, int]):
        self.factors = factors

    @classmethod
    def from_str(cls, mono: str):
        mono = str(mono).strip()
        str_factors = ' '.split(mono)
        factors = {}
        for f in str_factors:
            f = f.split('^')
            if len(f) == 1:
                dim = f[0]
                exp = 1
            elif len(f) == 2:
                dim = f[0]
                exp = int(f[1])
                assert exp > 0
            else:
                raise ValueError(f'Invalid monomial: "{mono}"')
            factors[dim] = factors.get(dim, 0) + exp
        return cls(factors=factors)

    def leq(self, *others: BigOMonomial, relations: list[tuple[BigOMonomial, BigOMonomial]] = None):
        """ <= such that O(A + B) = O(B)"""
        if relations is not None:
            raise NotImplementedError
        for o in others:
            ...

    def __repr__(self):
        return f'<{type(self).__name__} {str(self)} >'

    def __str__(self):
        return ' '.join(dim + '^' + exp for dim, exp in self.factors.items())


class BigOPolynomial:
    def __init__(self, terms: list[BigOMonomial]):
        self.terms = self.simplify_terms(terms)

    @staticmethod
    def simplify_terms(terms: list[BigOMonomial],
                       relations: list[tuple[BigOMonomial, BigOMonomial]] = None):
        is_negligible = [t.is_negligible(*terms[:n], *terms[n + 1:], relations=relations)
                         for n, t in enumerate(terms)]
        non_negligible = [t for t, neg in zip(terms, is_negligible) if not neg]
        return non_negligible

    @classmethod
    def from_str(cls, poly: str):
        terms = '+'.split(poly)
        return cls(terms=[BigOMonomial.from_str(t) for t in terms])

    def __repr__(self):
        return f'<{type(self).__name__} {str(self)} >'

    def __str__(self):
        return ' + '.join(str(t) for t in self.terms)

    def __add__(self, other):
        if isinstance(other, str):
            other = BigOPolynomial.from_str(other)
        if isinstance(other, BigOMonomial):
            other = BigOPolynomial([other])
        if not isinstance(other, BigOPolynomial):
            return NotImplemented
        
