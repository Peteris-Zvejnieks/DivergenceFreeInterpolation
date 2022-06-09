from sympy import product, symbols, Rational, var, lambdify, sqrt, Matrix, diff

class RBF_kernel():
    def __init__(self, nu: int = 5, k: int = 3 , dim: int = 3):
        if dim not in [2, 3]: raise ValueError(f'dimensionalit {dim} is not supported, on 2D or 3D are available')
        self.rbf = RBF(nu, k).eq
        self.dim = dim
        if dim == 2: self.kernel_2d()
        elif dim == 3: self.kernel_3d()
        
    def kernel_2d(self):
        x0, x1 = var('x0'), var('x1')
        r = self.rbf.free_symbols.pop()
        r_x = sqrt(x0**2 + x1**2)
        
        f = self.rbf.subs(r, r_x)
        
        d00, d11 = diff(f, x0, x0), diff(f, x1, x1)
        d01 = diff(f, x0, x1)
        
        dd0, dd1 = - d11, -d00
        
        self.kernel = Matrix([[dd0, d01],
                              [d01, dd1]]).subs(r_x, r)
        
        self.kernel_numpy = lambdify((x0, x1, r), self.kernel, ['numpy'])
        
    def kernel_3d(self):
        x0, x1, x2 = var('x0'), var('x1'), var('x2')
        r = self.rbf.free_symbols.pop()
        r_x = sqrt(x0**2 + x1**2 + x2**2)
        
        f = self.rbf.subs(r, r_x)
        
        d00, d11, d22 = diff(f, x0, x0), diff(f, x1, x1), diff(f, x2, x2)
        d01, d02, d12 = diff(f, x0, x1), diff(f, x0, x2), diff(f, x1, x2)
        
        dd0, dd1, dd2 = -d11 - d22, -d00-d22, -d00 - d11
        
        self.kernel = Matrix([[dd0, d01, d02],
                              [d01, dd1, d12],
                              [d02, d12, dd2]]).subs(r_x, r)
        
        self.kernel_numpy = lambdify((x0, x1, x2, r), self.kernel, ['numpy'])

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