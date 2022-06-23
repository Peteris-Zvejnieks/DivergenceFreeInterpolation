#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cmath>

namespace py = pybind11;
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Ref;
using Eigen::Dynamic;
using Eigen::Matrix;

array<array<double, 3>, 3>  kernel_5_3_3D(double x_0, double x_1, double x_2)
{
    array<array<double, 3>, 3> kernel;
    kernel.fill({{}});

    {
        double x0 = pow(x_0, 2);
        double x1 = pow(x_1, 2);
        double x2 = pow(x_2, 2);
        double x3 = x0 + x1 + x2;
        double x4 = sqrt(x3);
        if (x4 >= 1) return kernel;
        double x5 = x4 - 1;
        double x6 = 24*x5;
        double x7 = 168*x4;
        double x8 = x3*x6 - 9*x4*pow(x5, 2) + pow(x5, 3);
        double x9 = pow(x5, 6);
        double x10 = (1.0/504.0)*x9;
        double x11 = 6*x4 + 1;
        double x12 = (1.0/42.0)*x11*x9*x_0;

        double kernel0 = -x10*(-x0*x6 + x0*x7 + x8);
        double kernel1 = x12*x_1;
        double kernel2 = x12*x_2;
        double kernel3 = -x10*(-x1*x6 + x1*x7 + x8);
        double kernel4 = (1.0/42.0)*x11*x9*x_1*x_2;
        double kernel5 = -x10*(-x2*x6 + x2*x7 + x8);

        kernel[0][0] = kernel0;         kernel[0][1] = kernel1;         kernel[0][2] = kernel2;
        kernel[1][0] = kernel1;         kernel[1][1] = kernel3;         kernel[1][2] = kernel4;
        kernel[2][0] = kernel2;         kernel[2][1] = kernel4;         kernel[2][2] = kernel5;
    }
    return kernel;
}

class Interpolant3D {
    const int dim = 3;
    int nu, k;
    double support_radii;
    int N;
    VectorXd b;
    VectorXd solution;
    VectorXd X;
    VectorXd Y;
    VectorXd Z;
public:
    MatrixXd A;
    Interpolant3D(int nu_ = 5, int k_ = 3, double support_radii_ = 1) : nu(nu_), k(k_), support_radii(support_radii_){}

    void setSampleCoordinates(const Ref<const Matrix<double, Dynamic, 1>> &x,
                              const Ref<const Matrix<double, Dynamic, 1>> &y,
                              const Ref<const Matrix<double, Dynamic, 1>> &z) {
        X = x;
        Y = y;
        Z = z;

        N = X.size();
        b.resize(dim * N);
        solution.resize(dim * N);
    }

    void condition(const Ref<const VectorXd> &U,
                   const Ref<const VectorXd> &V,
                   const Ref<const VectorXd> &W) {
        A.resize(dim * N, dim * N);

        for (int i = 0; i < N; i++) {
            double xi = X(i);
            double yi = Y(i);
            double zi = Z(i);

            for (int j = 0; j < N; j++) {
                double dx = (xi - X(j)) / support_radii;
                double dy = (yi - Y(j)) / support_radii;
                double dz = (zi - Z(j)) / support_radii;

                array<array<double, 3>, 3> kernel = kernel_5_3_3D(dx, dz, dy);

                for (int m = 0; m < dim; m++) {
                    for (int n = 0; n < dim; n++) {
                        A(dim * i + m, dim * j + n) = pow(support_radii, -dim)*kernel[m][n];
                    }
                }
            }

            {
                b(dim * i + 0) = U[i];
                b(dim * i + 1) = V[i];
                b(dim * i + 2) = W[i];
            }
        }
        solution = A.colPivHouseholderQr().solve(b);
    }

    VectorXd interpolate(double x, double y, double z) {
        MatrixXd interpolationKernels(dim, dim * N);

        for (int i = 0; i < N; i++) {
            double dx = (x - X(i)) / support_radii;
            double dy = (y - Y(i)) / support_radii;
            double dz = (z - Z(i)) / support_radii;
            auto kernel = kernel_5_3_3D(dx, dy, dz);

            for (int m = 0; m < dim; m++) {
                for (int n = 0; n < dim; n++) {
                    interpolationKernels(m, dim * i + n) = pow(support_radii, -dim)*kernel[m][n];
                }
            }
        }

        VectorXd result = interpolationKernels * solution;
        return result;

    }

    MatrixXd getArray(){return A;}
};

PYBIND11_MODULE(Divergence_Free_Interpolant, m) {
    m.doc()  = "I'm a docstring hehe";
    py::class_<Interpolant3D>(m,  "Interpolant3D")
        .def(py::init<int, int, double>(), py::arg("nu") = 5, py::arg("k") = 3, py::arg("support_radii") = 1)
        .def("array", &Interpolant3D::getArray)
        .def("setSampleCoordinates",    &Interpolant3D::setSampleCoordinates, py::arg("X"), py::arg("Y"), py::arg("Z"))
        .def("condition",               &Interpolant3D::condition, py::arg("U"), py::arg("V"), py::arg("W"))
        .def("interpolate",             &Interpolant3D::interpolate);
}
