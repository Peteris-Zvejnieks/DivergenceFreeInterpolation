from sympy import product, symbols, Rational


class RBF:
    r, i, f = symbols('r i f')

    def __init__(self, nu: int, k: int):
        self.nu = nu
        self.k = k
        eq = 0
        for incr in range(0, k + 1):
            eq += self._beta(incr, k) * RBF.r ** incr * self.p(nu + 2 * k - incr)
        self.eq = eq

    def _beta(self, j: int, k: int):
        if j == 0 and k == 0:
            return Rational(1)
        else:
            coefficient = 0
            for incr in range(max([0, j - 1]), k):
                coefficient += self._beta(incr, k - 1) * self._square_bracket(incr - j + 1).subs(RBF.f,
                                                                                                 incr + 1) / self._round_bracket(
                    incr - j + 2).subs(RBF.f, self.nu + 2 * (k - 1) - incr + 1)
            return coefficient

    @staticmethod
    def p(nu: int):
        return (1 - RBF.r) ** nu

    @staticmethod
    def _square_bracket(l: int):
        if l == -1:
            return 1 / (RBF.f + 1)
        elif l == 0:
            return Rational(1)
        else:
            return product(RBF.f - RBF.i, (RBF.i, 0, l - 1))

    @staticmethod
    def _round_bracket(l: int):
        if l == 0:
            return Rational(0)
        else:
            return product(RBF.f + RBF.i, (RBF.i, 0, l - 1))
