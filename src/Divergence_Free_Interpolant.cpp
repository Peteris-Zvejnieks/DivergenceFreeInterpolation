#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "RBF_kernels/RBF_kernels.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>

//#define EIGEN_USE_BLAS
#define EIGEN_USE_LAPACKE_STRICT
namespace py = pybind11;
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Map;
using Eigen::Ref;
using Eigen::Dynamic;
using Eigen::Matrix;
using Eigen::all;

/*MatrixXd generate_array_3D(const Ref<const Matrix<double, Dynamic, Dynamic>> &positions, double support_radii = 1) {
    const int dim = 3;
    const int N = positions.cols();
    MatrixXd A(dim * N, dim * N);
    py::gil_scoped_release release;
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        Matrix<double, Dynamic, 1> vi = positions.col(i);
        for (int j = i; j < N; j++) {
            Matrix<double, dim, 1> vj = positions.col(j);
            Matrix<double, dim, 1> dv = (vi - vj) / support_radii;
            Matrix<double, dim, dim> kernel = kernel_5_3_3D(dv);

            for(int m = 0; m < dim; m++){
                for(int n = 0; n < dim; n++){
                    A(dim*i + m, dim*j + n) = kernel(m, n);
                }
            }
        }
    }
    A = A.selfadjointView<Eigen::Upper>();
    return A;
}*/


class Interpolant3D {
    const int dim = 3;
    int nu, k;
    double support_radii;
    int N;

    Matrix<double, Dynamic, 1> solution;
    Matrix<double, Dynamic, Dynamic> A;
    Matrix<double, Dynamic, 3> positions;
public:


    Interpolant3D(int nu_ = 5, int k_ = 3) : nu(nu_), k(k_) {}

    void setCoordinates(const Matrix<double, Dynamic, Dynamic> &pos){
        N = pos.rows();
        positions.resize(N, 3);
        positions = pos;
        A.resize(dim*N, dim*N);
        solution.resize(dim*N, 1);
    }

    void generateArray(double support_radii_) {
        support_radii = support_radii_;
        py::gil_scoped_release release;
        #pragma omp parallel for

        for (int i = 0; i < N; i++) {
            Matrix<double, 3, 1> vi = positions.row(i);
            for (int j = i; j < N; j++) {
                Matrix<double, 3, 1> vj = positions.row(j);
                Matrix<double, 3, 1> dv = (vi - vj) / support_radii;
                Matrix<double, 3, 3> kernel = pow(support_radii, -dim)*kernel_5_3_3D(dv);

                for (int m = 0; m < dim; m++) {
                    for (int n = 0; n < dim; n++) {
                        A(dim * i + m, dim * j + n) = kernel(m, n);
                    }
                }
            }
        }
//        A = A.selfadjointView<Eigen::Upper>();
    }

    void condition(const Matrix<double, Dynamic, Dynamic> &field){
        py::gil_scoped_release release;
//        VectorXd b(dim*N);
//        for(int i = 0; i < N; i++){
//            for(int m = 0; m < 3; m++){
//                b(3*i+m) = field(i, m);
//            }
//        }
//        solution = A.lu().solve(field.reshaped(dim*N,1));
        solution =  A.selfadjointView<Eigen::Upper>().llt().solve(field.reshaped(dim*N,1));
    }

    MatrixXd interpolate(const Ref<const Matrix<double, Dynamic, 3>> &pos) {
        int n = pos.rows();
        MatrixXd interpolationKernels(3*n, 3*N);

        py::gil_scoped_release release;
        #pragma omp parallel for

        for(int i = 0; i < n; i++) {
            Matrix<double, 3, 1> vi = pos.row(i);
            for(int j = 0; j < N; j++) {
                Matrix<double, 3, 1> vj = positions.row(j);
                Matrix<double, 3, 1> dv = (vi - vj) / support_radii;
                Matrix<double, 3, 3> kernel = kernel_5_3_3D(dv);

                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        interpolationKernels(dim * i + m, dim * j + n) = pow(support_radii, -dim) * kernel(m, n);
                    }
                }
            }
        }

        MatrixXd result_ = interpolationKernels * solution;

//        MatrixXd result(n, dim);
//        for(int i = 0; i < n; i++){
//            for(int m = 0; m < dim; m++){
//                result(i, m) = result_(dim*i + m);
//            }
//        }

        return result_.reshaped(n, 3);

    }

    MatrixXd getArray(){
        return A.selfadjointView<Eigen::Upper>();
    }
};

PYBIND11_MODULE(Divergence_Free_Interpolant_alt, m) {
    py::class_<Interpolant3D>(m,  "Interpolant3D")
        .def(py::init<int, int>(), py::arg("nu") = 5, py::arg("k") = 3)
        .def("setCoordinates",  &Interpolant3D::setCoordinates, py::arg("postitions"))
        .def("generateArray",   &Interpolant3D::generateArray, py::arg("support_radii") = 1)
        .def("condition",       &Interpolant3D::condition, py::arg("vectors"))
        .def("interpolate",     &Interpolant3D::interpolate, py::arg("postitions"))
        .def("getArray",        &Interpolant3D::getArray);
}

//int main(){
//    const int dim = 3;
//    const int N = 15;
//
//    MatrixXd positions = MatrixXd::Random(dim, N);
//
//    MatrixXd A = generate_array_3D(positions);
//    cout << A << endl;
//
//
//    return 0;
//}



//PYBIND11_MODULE(Divergence_Free_Interpolant, m) {
//    m.doc()  = "I'm a docstring hehe";
//    m.def("generate_array", &generate_array, py::arg("positions"), py::arg("support_radii"));
//    m.def("kernel_5_3", &kernel_5_3);
//}
