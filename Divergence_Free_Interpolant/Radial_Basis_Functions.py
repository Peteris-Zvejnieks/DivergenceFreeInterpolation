from sympy import product, symbols, Rational, var, lambdify, sqrt, Matrix, diff, Array, cse
from sympy.printing import cxxcode    

class RBF_kernel():
    def __init__(self, nu: int = 5, k: int = 3 , dim: int = 3):
        if dim not in [2, 3]: raise ValueError(f'dimensionalit {dim} is not supported, on 2D or 3D are available')
        self.rbf = RBF(nu, k).eq
        self.nu, self.k = nu, k
        self.dim = dim
        self.variables = [var(f'x_{i}') for i in range(dim)]
        
        arguments = [f'float {var}, ' for var in self.variables]
        arguments = "".join(map(lambda x: x, arguments))[:-2]
        
        self.h_string = f'array<array<float, {dim}>, {dim}> kernel_{nu}_{k}_{dim}D({arguments});\n'
        
        self.cpp_string = f'array<array<float, {dim}>, {dim}>  kernel_{nu}_{k}_{dim}D({arguments})\n{{\n'
        
        self.cpp_string += ' '*4 + f'array<array<float, {dim}>, {dim}> kernel;\n'
        self.cpp_string += ' '*4 +  'kernel.fill({{}});\n\n'
        
        self.make_kernel()
        
        self.cpp_string = self.cpp_string.replace('std::', '')
        
    def make_kernel(self):
        sym_to_c = lambda sym: cxxcode(sym, standard='C++11')
        
        Vars = self.variables
        r_x = sqrt(sum([var**2 for var in Vars]))
        sym_to_c = lambda sym: cxxcode(sym, standard='C++11')
        r = self.rbf.free_symbols.pop()
        f = self.rbf.subs(r, r_x)
        
        # self.cpp_string += ' '*4 + f'float r = {sym_to_c(r_x)};\n'
        # self.cpp_string += ' '*4 +  'if (r > 1) return kernel;\n\n' 
        
        
        unique_elements = []
        mapping_to_kernel = [[0 for j in range(self.dim)] for i in range(self.dim)]

        for i, var_i in enumerate(Vars):
            for j, var_j in enumerate(Vars):
                if i == j:
                    elem = Rational(0)
                    for k, var_k in enumerate(Vars):
                        if k == i: continue
                        else: elem -= diff(f, var_i, var_j)#.factor()     
                    mapping_to_kernel[i][j] = len(unique_elements)
                    unique_elements.append(elem.factor()) 
                elif i < j: 
                    mapping_to_kernel[i][j] = len(unique_elements)
                    mapping_to_kernel[j][i] = len(unique_elements)
                    unique_elements.append(diff(f, var_i, var_j).factor())
                else: continue

        substitutions = cse(unique_elements)
        
        for sub in substitutions[0]:
            r_x = r_x.subs(*reversed(sub))
        
        self.cpp_string += ' '*4 + '{\n'
        
        for sub_expr in substitutions[0]:
            self.cpp_string += ' '*8 + f'float {sym_to_c(sub_expr[0])} = {sym_to_c(sub_expr[1])};\n'
            if sub_expr[0] == r_x: self.cpp_string += ' '*8 + f'if ({sym_to_c(sub_expr[0])} > 1) return kernel;\n'
         
        self.cpp_string += '\n'
        
        for i, kernel_elem in enumerate(substitutions[1]):
            self.cpp_string += ' '*8 + f'float kernel{i} = {sym_to_c(kernel_elem)};\n'
            
        
        self.cpp_string += '\n'
        
        for i, row in enumerate(mapping_to_kernel):
            for j, elem in enumerate(row):
                self.cpp_string += ' '*8 + f'kernel[{i}][{j}] = kernel{elem}; '
            self.cpp_string += '\n'
        
        self.cpp_string += ' '*4 + '}\n'
        
        self.cpp_string += ' '*4 + 'return kernel;\n}\n\n'
        
        
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


def kernel_c_code_maker(dim_list = [2, 3], nu_list = [5,7,9], k_list = [3, 5, 7]):   
    for dim in dim_list:
        
        header_string = f'''
#pragma once

/*
 * Radial basis function kernels
 * {dim}D
 */

#include <array>
using namespace std;
#ifndef KERNELS_{dim}D_H_
#define KERNELS_{dim}D_H_\n\n'''
        
        cpp_string = f'''
/*
 * Radial basis function kernels
 * {dim}D
 */

#include <cmath>
#include <array>
using namespace std;\n
'''
        
        for nu in nu_list:
            for k in k_list:
                kernel = RBF_kernel(nu, k, dim)
                header_string += kernel.h_string
                cpp_string += kernel.cpp_string
        
        header_string += '\n#endif'
        
        with open(f'./C++_kernels/kernels_{dim}D.cpp', 'w+') as f:
            f.write(cpp_string)

        with open(f'./C++_kernels/kernels_{dim}D.h', 'w+') as f:
            f.write(header_string)
            
if __name__ == '__main__':
    kernel_c_code_maker(nu_list = [5,7], k_list = [3, 5])