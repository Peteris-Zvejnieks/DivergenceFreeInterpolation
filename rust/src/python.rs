//! Python bindings: the `divergence_free_interpolant._core` extension module.
//!
//! Arrays come in as C-contiguous float64 arrays (the Python side ensures this) and results go
//! back as flat numpy arrays that Python reshapes. Heavy work runs without the GIL.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyMemoryError, PyValueError};
use pyo3::prelude::*;

use crate::assembly;
use crate::cg;
use crate::cholesky;
use crate::evaluate as evaluation;
use crate::kernel::{Kind, PairKernel, RadialKernel};
use crate::pool;
use crate::sparse::CsrMatrix;
use crate::thinning;

/// A divergence-free or curl-free kernel from the ascending coefficients of A(r) and H(r).
#[pyclass(
    name = "RadialKernel",
    module = "divergence_free_interpolant._core",
    frozen
)]
struct PyRadialKernel {
    inner: RadialKernel,
}

#[pymethods]
impl PyRadialKernel {
    #[new]
    fn new(kind: &str, dim: usize, a: Vec<f64>, h: Vec<f64>) -> PyResult<Self> {
        let kind = match kind {
            "divergence_free" => Kind::DivergenceFree,
            "curl_free" => Kind::CurlFree,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown kernel kind {other:?}"
                )));
            }
        };
        RadialKernel::new(kind, dim, a, h)
            .map(|inner| Self { inner })
            .map_err(PyValueError::new_err)
    }

    #[getter]
    fn dim(&self) -> usize {
        self.inner.dim()
    }
}

/// Borrows a C-contiguous (n, dim) array as a flat row-major slice.
fn rows<'a>(array: &'a PyReadonlyArray2<'_, f64>, dim: usize, name: &str) -> PyResult<&'a [f64]> {
    let shape = array.shape();
    if shape[1] != dim {
        return Err(PyValueError::new_err(format!(
            "{name} must have {dim} columns, got shape {shape:?}"
        )));
    }
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a C-contiguous array")))
}

type CsrArrays<'py> = (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
);

/// The sparse interpolation matrix as CSR arrays (indptr, indices, data).
#[pyfunction]
fn gram_matrix<'py>(
    py: Python<'py>,
    kernel: &Bound<'py, PyRadialKernel>,
    points: PyReadonlyArray2<'py, f64>,
    support_radius: f64,
) -> PyResult<CsrArrays<'py>> {
    let kernel = &kernel.get().inner;
    let points = rows(&points, kernel.dim(), "points")?;
    let csr = py.detach(|| pool::install(|| assembly::gram_matrix(kernel, points, support_radius)));
    Ok((
        PyArray1::from_vec(py, csr.indptr),
        PyArray1::from_vec(py, csr.indices),
        PyArray1::from_vec(py, csr.data),
    ))
}

/// Shared argument checks for the evaluation functions.
fn evaluation_inputs<'a>(
    kernel: &RadialKernel,
    nodes: &'a PyReadonlyArray2<'_, f64>,
    coefficients: &'a PyReadonlyArray2<'_, f64>,
    queries: &'a PyReadonlyArray2<'_, f64>,
) -> PyResult<(&'a [f64], &'a [f64], &'a [f64])> {
    let d = kernel.dim();
    let nodes = rows(nodes, d, "nodes")?;
    let coefficients = rows(coefficients, d, "coefficients")?;
    if nodes.len() != coefficients.len() {
        return Err(PyValueError::new_err(
            "nodes and coefficients must have the same shape",
        ));
    }
    Ok((nodes, coefficients, rows(queries, d, "queries")?))
}

/// The interpolant at the queries, flattened (M * dim values).
#[pyfunction]
fn evaluate<'py>(
    py: Python<'py>,
    kernel: &Bound<'py, PyRadialKernel>,
    nodes: PyReadonlyArray2<'py, f64>,
    coefficients: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    support_radius: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kernel = &kernel.get().inner;
    let (nodes, coefficients, queries) =
        evaluation_inputs(kernel, &nodes, &coefficients, &queries)?;
    let values = py.detach(|| {
        pool::install(|| evaluation::evaluate(kernel, nodes, coefficients, queries, support_radius))
    });
    Ok(PyArray1::from_vec(py, values))
}

/// The scalar potential (curl-free) or stream function (2D divergence-free) at the queries.
#[pyfunction]
fn evaluate_potential<'py>(
    py: Python<'py>,
    kernel: &Bound<'py, PyRadialKernel>,
    nodes: PyReadonlyArray2<'py, f64>,
    coefficients: PyReadonlyArray2<'py, f64>,
    queries: PyReadonlyArray2<'py, f64>,
    support_radius: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let kernel = &kernel.get().inner;
    if kernel.kind() == Kind::DivergenceFree && kernel.dim() != 2 {
        return Err(PyValueError::new_err(
            "a scalar stream function only exists in 2D",
        ));
    }
    let (nodes, coefficients, queries) =
        evaluation_inputs(kernel, &nodes, &coefficients, &queries)?;
    let values = py.detach(|| {
        pool::install(|| {
            evaluation::evaluate_potential(kernel, nodes, coefficients, queries, support_radius)
        })
    });
    Ok(PyArray1::from_vec(py, values))
}

type OrderingArrays<'py> = (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>);

/// Greedy farthest-point ordering from `start`: (indices, insertion radii).
#[pyfunction]
fn farthest_point_ordering<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f64>,
    start: usize,
    stop_radius: f64,
) -> PyResult<OrderingArrays<'py>> {
    let dim = points.shape()[1];
    let flat = rows(&points, dim, "points")?;
    if dim == 0 || start >= flat.len() / dim {
        return Err(PyValueError::new_err("start must index one of the points"));
    }
    let (order, radii) = py.detach(|| {
        pool::install(|| thinning::farthest_point_ordering(flat, dim, start, stop_radius))
    });
    let order: Vec<i64> = order.into_iter().map(|i| i as i64).collect();
    Ok((PyArray1::from_vec(py, order), PyArray1::from_vec(py, radii)))
}

/// A CSR matrix from Python as (indptr, indices, data), int64 indices.
type CsrInput<'py> = (
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, i64>,
    PyReadonlyArray1<'py, f64>,
);

fn slice<'a, T: numpy::Element>(
    array: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    array
        .as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a contiguous array")))
}

fn csr<'a>(input: &'a CsrInput<'_>, columns: usize, name: &str) -> PyResult<CsrMatrix<'a>> {
    CsrMatrix::new(
        slice(&input.0, name)?,
        slice(&input.1, name)?,
        slice(&input.2, name)?,
        columns,
    )
    .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))
}

/// Preconditioned conjugate gradients from x = 0 for a square CSR matrix. Returns
/// (solution, iterations, info) with scipy's convention: info 0 converged, info > 0 the
/// iteration limit was reached, info < 0 breakdown.
#[pyfunction]
fn conjugate_gradients<'py>(
    py: Python<'py>,
    matrix: CsrInput<'py>,
    preconditioner: CsrInput<'py>,
    rhs: PyReadonlyArray1<'py, f64>,
    rtol: f64,
    max_iterations: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, i64)> {
    let rhs = slice(&rhs, "rhs")?;
    let n = rhs.len();
    let matrix = csr(&matrix, n, "matrix")?;
    let preconditioner = csr(&preconditioner, n, "preconditioner")?;
    if matrix.rows() != n || preconditioner.rows() != n {
        return Err(PyValueError::new_err(
            "matrix, preconditioner and rhs sizes differ",
        ));
    }
    let solution = py.detach(|| {
        pool::install(|| {
            cg::conjugate_gradients(&matrix, &preconditioner, rhs, rtol, max_iterations)
        })
    });
    let info = match solution.status {
        cg::Status::Converged => 0,
        cg::Status::MaxIterations => solution.iterations as i64,
        cg::Status::Breakdown => -1,
    };
    Ok((
        PyArray1::from_vec(py, solution.x),
        solution.iterations,
        info,
    ))
}

/// Sparse Cholesky solve of a symmetric positive definite CSR matrix (column indices sorted and
/// unique within each row; only the upper triangle is read). Returns (solution, "") or
/// (None, reason) if the factorisation fails; raises MemoryError if the factor does not fit.
#[pyfunction]
fn cholesky_solve<'py>(
    py: Python<'py>,
    matrix: CsrInput<'py>,
    rhs: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Option<Bound<'py, PyArray1<f64>>>, String)> {
    let rhs = slice(&rhs, "rhs")?;
    let n = rhs.len();
    let checked = csr(&matrix, n, "matrix")?;
    if checked.rows() != n {
        return Err(PyValueError::new_err("matrix and rhs sizes differ"));
    }
    let (indptr, indices, data) = (
        slice(&matrix.0, "matrix")?,
        slice(&matrix.1, "matrix")?,
        slice(&matrix.2, "matrix")?,
    );
    let result = py.detach(|| {
        pool::install(|| {
            let pointers: Vec<usize> = indptr.iter().map(|&v| v as usize).collect();
            let rows: Vec<usize> = indices.iter().map(|&v| v as usize).collect();
            cholesky::solve(n, &pointers, &rows, data, rhs)
        })
    });
    Ok(match result {
        Ok(solution) => (Some(PyArray1::from_vec(py, solution)), String::new()),
        Err(cholesky::CholeskyError::NotPositiveDefinite) => {
            (None, "the matrix is not positive definite".to_string())
        }
        Err(cholesky::CholeskyError::OutOfMemory) => {
            return Err(PyMemoryError::new_err(
                "the sparse Cholesky factor does not fit in memory",
            ));
        }
        Err(cholesky::CholeskyError::InvalidInput(reason)) => {
            return Err(PyValueError::new_err(format!("matrix: {reason}")));
        }
        Err(cholesky::CholeskyError::Other(reason)) => (None, reason),
    })
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyRadialKernel>()?;
    module.add_function(wrap_pyfunction!(gram_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_potential, module)?)?;
    module.add_function(wrap_pyfunction!(farthest_point_ordering, module)?)?;
    module.add_function(wrap_pyfunction!(conjugate_gradients, module)?)?;
    module.add_function(wrap_pyfunction!(cholesky_solve, module)?)?;
    Ok(())
}
