import unittest
from sympy import sympify
from ..Radial_Basis_Functions import radial_basis_function

functions_from_paper = [[1, 0, '(1 - r)'],
                        [2, 1, '(1 - r)**3 * (3*r + 1)'],
                        [3, 2, '(1 - r)**5 * (8*r**2 + 5*r + 1)'],
                        [2, 0, '(1 - r)**2'],
                        [3, 1, '(1 - r)**4 * (4*r + 1)'],
                        [4, 2, '(1 - r)**6 * (35*r**2 + 18*r + 3)'],
                        [5, 3, '(1 - r)**8 * (32*r**3 + 25*r**2 + 8*r + 1)'],
                        [3, 0, '(1 - r)**3'],
                        [4, 1, '(1 - r)**5 * (5*r + 1)'],
                        [5, 2, '(1 - r)**7 * (16*r**2 + 7*r + 1)']]


class TestUtils(unittest.TestCase):
    def test_function_generator(self):
        for precalculated_function in functions_from_paper:
            nu, _k, paper_function = precalculated_function
            paper_function = sympify(paper_function)
            generated_function = radial_basis_function(nu, _k, 'r').factor()
            self.assertTrue((paper_function / generated_function).simplify().is_constant())


if __name__ == '__main__':
    unittest.main()
