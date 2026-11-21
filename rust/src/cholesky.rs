//! Sparse Cholesky factorisation (faer) for the direct solves.
//!
//! The interpolation matrices are symmetric positive definite, so a supernodal Cholesky
//! factorisation with a fill-reducing ordering does about half the work of an LU factorisation
//! and needs no pivoting. A matrix that is not positive definite (e.g. numerically singular
//! because of nearly coincident points) is reported, not silently factorised.

use faer::linalg::solvers::Solve;
use faer::sparse::linalg::LltError;
use faer::sparse::linalg::solvers::{Llt, SymbolicLlt};
use faer::sparse::{FaerError, SparseColMatRef, SymbolicSparseColMatRef};
use faer::{Mat, Side};

/// Why a factorisation failed.
#[derive(Debug, PartialEq, Eq)]
pub enum CholeskyError {
    /// The arrays do not describe an n x n matrix with sorted, unique row indices per column,
    /// or the right-hand side has the wrong length.
    InvalidInput(String),
    /// A non-positive pivot: the matrix is not (numerically) positive definite.
    NotPositiveDefinite,
    /// The factor does not fit in memory.
    OutOfMemory,
    /// Anything else, with a description.
    Other(String),
}

fn generic(error: FaerError) -> CholeskyError {
    match error {
        FaerError::OutOfMemory => CholeskyError::OutOfMemory,
        other => CholeskyError::Other(format!("{other:?}")),
    }
}

/// Checks the compressed sparse column arrays of an n x n matrix and the right-hand side.
fn validate(
    n: usize,
    column_pointers: &[usize],
    row_indices: &[usize],
    values: &[f64],
    rhs: &[f64],
) -> Result<(), String> {
    if column_pointers.len() != n + 1 || column_pointers[0] != 0 {
        return Err(format!("expected {} column pointers starting at 0", n + 1));
    }
    if column_pointers.windows(2).any(|w| w[0] > w[1]) {
        return Err("column pointers must be nondecreasing".into());
    }
    let nnz = column_pointers[n];
    if row_indices.len() != nnz || values.len() != nnz {
        return Err(format!(
            "expected {nnz} row indices and values, got {} and {}",
            row_indices.len(),
            values.len()
        ));
    }
    let sorted = column_pointers.windows(2).all(|w| {
        let column = &row_indices[w[0]..w[1]];
        column.windows(2).all(|p| p[0] < p[1]) && column.last().is_none_or(|&row| row < n)
    });
    if !sorted {
        return Err("row indices must be in range, sorted and unique within each column".into());
    }
    if rhs.len() != n {
        return Err(format!(
            "the right-hand side has {} entries, expected {n}",
            rhs.len()
        ));
    }
    Ok(())
}

/// Solves A x = b for a symmetric positive definite A given in compressed sparse column form
/// (row indices sorted and unique within each column). Only the lower triangle is read, which
/// for CSR arrays of a symmetric matrix (the CSC arrays of its transpose) is the upper one.
pub fn solve(
    n: usize,
    column_pointers: &[usize],
    row_indices: &[usize],
    values: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, CholeskyError> {
    validate(n, column_pointers, row_indices, values, rhs).map_err(CholeskyError::InvalidInput)?;
    let symbolic = SymbolicSparseColMatRef::new_checked(n, n, column_pointers, None, row_indices);
    let matrix = SparseColMatRef::new(symbolic, values);
    let analysis = SymbolicLlt::try_new(symbolic, Side::Lower).map_err(generic)?;
    let factor =
        Llt::try_new_with_symbolic(analysis, matrix, Side::Lower).map_err(|error| match error {
            LltError::Numeric(_) => CholeskyError::NotPositiveDefinite,
            LltError::Generic(error) => generic(error),
        })?;
    let mut solution = Mat::from_fn(n, 1, |i, _| rhs[i]);
    factor.solve_in_place(solution.as_mut());
    Ok((0..n).map(|i| solution[(i, 0)]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::gram_matrix;
    use crate::kernel::Kind;
    use crate::sparse::{CsrMatrix, add_scaled, norm};
    use crate::testing::{Rng, wendland_like_kernel};

    #[test]
    fn solves_an_interpolation_system() {
        let mut rng = Rng::new(61);
        let points = rng.points(500, 2);
        let kernel = wendland_like_kernel(Kind::CurlFree, 2);
        let csr = gram_matrix(&kernel, &points, 0.2);
        let size = 1000;
        let indptr: Vec<usize> = csr.indptr.iter().map(|&v| v as usize).collect();
        let indices: Vec<usize> = csr.indices.iter().map(|&v| v as usize).collect();
        let b: Vec<f64> = (0..size).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let x = solve(size, &indptr, &indices, &csr.data, &b).unwrap();
        let matrix = CsrMatrix::new(&csr.indptr, &csr.indices, &csr.data, size).unwrap();
        let mut residual = vec![0.0; size];
        matrix.multiply_into(&x, &mut residual);
        add_scaled(&mut residual, -1.0, &b);
        assert!(norm(&residual) <= 1e-10 * norm(&b));
    }

    #[test]
    fn reports_indefinite_matrices() {
        // diag(1, -2, 1)
        let result = solve(3, &[0, 1, 2, 3], &[0, 1, 2], &[1.0, -2.0, 1.0], &[1.0; 3]);
        assert_eq!(result, Err(CholeskyError::NotPositiveDefinite));
    }

    #[test]
    fn rejects_invalid_input() {
        let invalid = |pointers: &[usize], rows: &[usize], values: &[f64], rhs: &[f64]| {
            matches!(
                solve(2, pointers, rows, values, rhs),
                Err(CholeskyError::InvalidInput(_))
            )
        };
        let values = [2.0, 1.0, 1.0, 2.0];
        assert!(solve(2, &[0, 2, 4], &[0, 1, 0, 1], &values, &[1.0; 2]).is_ok());
        // Short right-hand side, unsorted, duplicate and out-of-range rows, wrong lengths
        assert!(invalid(&[0, 2, 4], &[0, 1, 0, 1], &values, &[1.0]));
        assert!(invalid(&[0, 2, 4], &[1, 0, 0, 1], &values, &[1.0; 2]));
        assert!(invalid(&[0, 2, 4], &[0, 0, 0, 1], &values, &[1.0; 2]));
        assert!(invalid(&[0, 2, 4], &[0, 1, 0, 2], &values, &[1.0; 2]));
        assert!(invalid(&[0, 2, 4], &[0, 1, 0], &values[..3], &[1.0; 2]));
        assert!(invalid(&[0, 3, 2], &[0, 1, 0, 1], &values, &[1.0; 2]));
        assert!(invalid(&[0, 2], &[0, 1], &values[..2], &[1.0; 2]));
    }
}
