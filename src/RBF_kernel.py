from sympy import Rational, var, sqrt, Matrix, diff
from Radial_Basis_Functions import radial_basis_function


class RadialBasisFunctionKernel:
    def __init__(self, nu: int = 5, k: int = 3, dim: int = 3):
        if dim <= 1:
            raise ValueError(f'{dim}D vector space does not make sense')
        self.rbf, self.arg = radial_basis_function(nu, k, provide_argument=True)
        self.nu, self.k = nu, k
        self.dim = dim
        self.variables = [var(f'x_{i}') for i in range(dim)]
        self.r_x = sqrt(sum([variable ** 2 for variable in self.variables]))
        self.unique_elements = []
        self.kernel = [[Rational(0) for _ in range(dim)] for _ in range(dim)]
        self.populate_kernel()

    def populate_kernel(self):
        rbf_x = self.rbf.subs(self.arg, self.r_x)

        for i, var_i in enumerate(self.variables):
            for j, var_j in enumerate(self.variables):
                if i == j:
                    elem = Rational(0)
                    for k, var_k in enumerate(self.variables):
                        if k == i:
                            continue
                        else:
                            elem -= diff(rbf_x, var_i, var_j)
                    elem = elem
                    self.kernel[i][j] = elem
                    self.unique_elements.append(elem)
                elif i < j:
                    elem = diff(rbf_x, var_i, var_j)
                    self.kernel[i][j] = elem
                    self.kernel[j][i] = elem
                    self.unique_elements.append(elem)

        self.kernel = Matrix(self.kernel)


if __name__ == '__main__':
    kernel = RadialBasisFunctionKernel(5, 3)
    print(kernel.kernel)
