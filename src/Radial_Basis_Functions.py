from sympy import Rational, Symbol, product, symbols, var
from functools import cache

'''
Function which generates radial basis functions as defined in:
    Wendland, H. Piecewise polynomial, positive definite and compactly supported radial functions of minimal degree. 
    Adv Comput Math 4, 389–396 (1995). https://doi.org/10.1007/BF02123482
'''


def radial_basis_function(nu: int, k: int, arg='r', provide_argument=False):
    if type(arg) is str:
        arg = var(arg)
    elif type(arg) is not Symbol:
        raise (TypeError(f'arg has to be a string or sympy Symbol, not {type(arg)}'))

    i, f = symbols('i, f')

    def p(degree: int):
        return (1 - arg) ** degree

    @cache
    def square_bracket(argument, degree: int):
        if argument < degree - 1:
            return Rational(0)
        elif degree == -1:
            print('hello')
            return 1 / (argument + 1)
        elif degree == 0:
            return Rational(1)
        else:
            return product(argument - i, (i, 0, degree - 1))

    @cache
    def round_bracket(argument, degree: int):
        if degree == 0:
            return Rational(0)
        else:
            return product(argument + i, (i, 0, degree - 1))

    @cache
    def beta(m: int, n: int):
        if m == 0 and n == 0:
            return Rational(1)
        else:
            coefficient = Rational(0)
            for _j in range(max([0, m - 1]), n):
                a = beta(_j, n - 1)
                b = square_bracket(_j + 1, _j - m + 1)
                c = round_bracket(nu + 2 * (n - 1) - _j + 1, _j - m + 2)
                coefficient += a * b / c
            return coefficient

    expression = Rational(0)
    for j in range(0, k + 1):
        expression += beta(j, k) * arg ** j * p(nu + 2 * k - j)
    expression = expression
    if provide_argument:
        return expression, arg
    else:
        return expression
