//! Preconditioned conjugate gradients for the symmetric positive definite interpolation systems.

use rayon::prelude::*;

use crate::sparse::{CsrMatrix, add_scaled, dot, norm};

/// How a conjugate gradients run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// The relative residual reached the tolerance.
    Converged,
    /// The iteration limit was reached first.
    MaxIterations,
    /// A non-positive curvature or preconditioned residual: the matrix or the preconditioner
    /// is not positive definite (or the computation produced NaN).
    Breakdown,
}

pub struct Solution {
    pub x: Vec<f64>,
    pub iterations: usize,
    pub status: Status,
}

/// Solves A x = b by conjugate gradients from x = 0, with `preconditioner` approximating the
/// inverse of A, until ||b - A x|| <= rtol ||b|| (the recursively updated residual, as scipy).
pub fn conjugate_gradients(
    matrix: &CsrMatrix,
    preconditioner: &CsrMatrix,
    b: &[f64],
    rtol: f64,
    max_iterations: usize,
) -> Solution {
    let n = b.len();
    let mut x = vec![0.0; n];
    let b_norm = norm(b);
    if b_norm == 0.0 {
        return Solution {
            x,
            iterations: 0,
            status: Status::Converged,
        };
    }
    let tolerance = rtol * b_norm;
    let mut r = b.to_vec();
    let mut z = vec![0.0; n];
    preconditioner.multiply_into(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let mut q = vec![0.0; n];

    for iteration in 1..=max_iterations {
        matrix.multiply_into(&p, &mut q);
        let curvature = dot(&p, &q);
        // Comparisons with NaN are false, so NaN counts as a breakdown too
        let positive = curvature > 0.0 && rz > 0.0;
        if !positive {
            return Solution {
                x,
                iterations: iteration,
                status: Status::Breakdown,
            };
        }
        let alpha = rz / curvature;
        add_scaled(&mut x, alpha, &p);
        add_scaled(&mut r, -alpha, &q);
        if norm(&r) <= tolerance {
            return Solution {
                x,
                iterations: iteration,
                status: Status::Converged,
            };
        }
        preconditioner.multiply_into(&r, &mut z);
        let rz_next = dot(&r, &z);
        let beta = rz_next / rz;
        rz = rz_next;
        // p = z + beta p
        p.par_iter_mut()
            .zip(z.par_iter())
            .for_each(|(p, z)| *p = z + beta * *p);
    }
    Solution {
        x,
        iterations: max_iterations,
        status: Status::MaxIterations,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assembly::gram_matrix;
    use crate::kernel::Kind;
    use crate::testing::{Rng, wendland_like_kernel};

    /// Jacobi preconditioner of a CSR matrix, itself as CSR arrays.
    fn jacobi(indptr: &[i64], indices: &[i64], data: &[f64]) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let rows = indptr.len() - 1;
        let mut diagonal = vec![0.0; rows];
        for row in 0..rows {
            for k in indptr[row] as usize..indptr[row + 1] as usize {
                if indices[k] as usize == row {
                    diagonal[row] = data[k];
                }
            }
        }
        (
            (0..=rows as i64).collect(),
            (0..rows as i64).collect(),
            diagonal.iter().map(|d| 1.0 / d).collect(),
        )
    }

    #[test]
    fn solves_an_interpolation_system() {
        let mut rng = Rng::new(51);
        let points = rng.points(400, 2);
        let kernel = wendland_like_kernel(Kind::DivergenceFree, 2);
        let csr = gram_matrix(&kernel, &points, 0.2);
        let size = 800;
        let matrix = CsrMatrix::new(&csr.indptr, &csr.indices, &csr.data, size).unwrap();
        let (pi, px, pd) = jacobi(&csr.indptr, &csr.indices, &csr.data);
        let preconditioner = CsrMatrix::new(&pi, &px, &pd, size).unwrap();
        let b: Vec<f64> = (0..size).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let solution = conjugate_gradients(&matrix, &preconditioner, &b, 1e-10, 10_000);
        assert_eq!(solution.status, Status::Converged);
        let mut residual = vec![0.0; size];
        matrix.multiply_into(&solution.x, &mut residual);
        add_scaled(&mut residual, -1.0, &b);
        assert!(norm(&residual) <= 1e-9 * norm(&b));
    }

    #[test]
    fn reports_the_iteration_limit_and_breakdown() {
        let identity = (
            (0..=3).collect::<Vec<i64>>(),
            (0..3).collect::<Vec<i64>>(),
            vec![1.0; 3],
        );
        let id = CsrMatrix::new(&identity.0, &identity.1, &identity.2, 3).unwrap();
        // An ill-conditioned diagonal matrix needs more than one iteration
        let diagonal = vec![1.0, 10.0, 100.0];
        let spread = CsrMatrix::new(&identity.0, &identity.1, &diagonal, 3).unwrap();
        let limited = conjugate_gradients(&spread, &id, &[1.0, 1.0, 1.0], 1e-12, 1);
        assert_eq!(limited.status, Status::MaxIterations);
        // An indefinite matrix breaks down
        let negative = vec![1.0, -2.0, 1.0];
        let indefinite = CsrMatrix::new(&identity.0, &identity.1, &negative, 3).unwrap();
        let broken = conjugate_gradients(&indefinite, &id, &[0.0, 1.0, 0.0], 1e-12, 10);
        assert_eq!(broken.status, Status::Breakdown);
        // A zero right-hand side is solved by x = 0
        let zero = conjugate_gradients(&spread, &id, &[0.0; 3], 1e-12, 10);
        assert_eq!((zero.status, zero.iterations), (Status::Converged, 0));
    }
}
