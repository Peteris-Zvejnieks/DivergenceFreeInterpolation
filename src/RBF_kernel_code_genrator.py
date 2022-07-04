from RBF_kernel import RadialBasisFunctionKernel
from sympy.printing import cxxcode
from sympy import cse
import os

def _initialize_header_cpp_and_kernel_list(path: str = './'):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    header_string = f'''/*
 * Radial basis function kernels
 */
 
#pragma once
#include <Eigen/Dense>
using Eigen::Matrix;
using namespace std;
#ifndef RBF_KERNELS_H
#define RBF_KERNELS_H


#endif
'''

    with open(f'{path}/RBF_kernels.h', 'w') as f:
        f.seek(0)
        f.write(header_string)
        f.truncate()

    cpp_string = f'''/*
 * Radial basis function kernels
 */

#include <Eigen/Dense>
#include <cmath>
using Eigen::MatrixXd;
using Eigen::Matrix;
using Eigen::Ref;
using namespace std;


'''

    with open(f'{path}/RBF_kernels.cpp', 'w') as f:
        f.seek(0)
        f.write(cpp_string)
        f.truncate()

    kernel_list = f'''/*
 * List of generated kernels
 */

nu, k, dim
'''

    with open(f'{path}/RBF_kernels.csv', 'w') as f:
        f.seek(0)
        f.write(kernel_list)
        f.truncate()

def generate_cpp_code_from_kernel(rbf_kernel):
    variables = rbf_kernel.variables
    nu, k = rbf_kernel.nu, rbf_kernel.k
    dim = rbf_kernel.dim

    def sym_to_c(expression):
        return cxxcode(expression, standard='C++11')

    def indent(n: int = 1):
        return ' ' * 4 * n

    function_head = f'Matrix<double, {dim}, {dim}> kernel_{nu}_{k}_{dim}D(Matrix<double, {dim}, 1> &vector)'

    h_string = f'{function_head};\n'
    cpp_string = f'{function_head}{{\n'
    cpp_string += f'{indent()}//{dim}D kernel\n'
    cpp_string += f'{indent()}//nu = {nu}, k = {k}\n'
    cpp_string += f'{indent()}//RBF = {sym_to_c(rbf_kernel.rbf.factor())}\n\n'
    cpp_string += f'{indent()}Matrix<double, {dim}, {dim}> kernel = MatrixXd::Zero({dim}, {dim});\n'
    for i, variable in enumerate(variables):
        cpp_string += f'{indent()}double {variable} = vector({i});\n'
    cpp_string += '\n'

    r_x = rbf_kernel.r_x

    mapping_to_kernel = [[0 for _ in range(dim)] for _ in range(dim)]
    k = 0
    for i in range(dim):
        for j in range(dim):
            if i == j:
                mapping_to_kernel[i][j] = k
                k += 1
            elif i < j:
                mapping_to_kernel[i][j] = k
                mapping_to_kernel[j][i] = k
                k += 1

    substitutions = cse(rbf_kernel.unique_elements)

    for sub in substitutions[0]:
        r_x = r_x.subs(*reversed(sub))

    cpp_string += f'{indent()}{{\n'

    for sub_expr in substitutions[0]:
        cpp_string += f'{indent(2)}double {sym_to_c(sub_expr[0])} = {sym_to_c(sub_expr[1])};\n'
        if sub_expr[0] == r_x:
            cpp_string += f'{indent(2)}if ({sym_to_c(sub_expr[0])} > 1) return kernel;\n'

    cpp_string += '\n'

    for i, kernel_elem in enumerate(substitutions[1]):
        cpp_string += f'{indent(2)}double kernel_{i} = {sym_to_c(kernel_elem)};\n'

    cpp_string += '\n'
    for j, row in enumerate(mapping_to_kernel):
        for i, elem in enumerate(row):
            cpp_string += f'{indent(2)}kernel({i}, {j}) = kernel_{elem}; '
        cpp_string += '\n'

    cpp_string += f'{indent()}}}\n'

    cpp_string += f'{indent()}return kernel;\n}}\n\n'

    cpp_string = cpp_string.replace('std::', '')

    return cpp_string, h_string


def append_kernel(nu: int, k: int, dim: int, path: str = './'):
    def listizer(line):
        return list(map(int, line))

    with open(f'{path}/RBF_kernels.csv', 'r+') as f:
        csv = f.readlines()
        kernel_string = f'{nu}, {k}, {dim}'
        if kernel_string + '\n' in csv:
            return
        else:
            # csv.append(kernel_string+'\n')
            f.write(kernel_string+'\n')

    rbf_kernel = RadialBasisFunctionKernel(nu, k, dim)
    cpp_string, h_string = generate_cpp_code_from_kernel(rbf_kernel)

    with open(f'{path}/RBF_kernels.cpp', 'a') as f:
        f.write(cpp_string)

    with open(f'{path}/RBF_kernels.h', 'r+') as f:
        h_lines = f.readlines()
        h_lines.insert(-2, h_string)
        f.seek(0)
        f.writelines(h_lines)
    return

if __name__ == '__main__':
    path = './RBF_kernels'
    _initialize_header_cpp_and_kernel_list(path)
    dimension_list = [2, 3]
    nu_list = [5]
    k_list = [3, 5]
    for nu in nu_list:
        for k in k_list:
            for dim in dimension_list:
                append_kernel(nu, k, dim, path)

