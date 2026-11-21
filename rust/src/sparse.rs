//! Compressed sparse row matrices in scipy's layout, and parallel vector operations.
//!
//! Reductions (dot products, norms) sum fixed-size chunks in parallel and then add the partial
//! sums in order, so results do not depend on the number of threads and runs are reproducible.

use rayon::prelude::*;

/// Vector length per parallel chunk.
const CHUNK: usize = 4096;

/// A borrowed CSR matrix (scipy layout, int64 indices).
pub struct CsrMatrix<'a> {
    indptr: &'a [i64],
    indices: &'a [i64],
    data: &'a [f64],
    columns: usize,
}

impl<'a> CsrMatrix<'a> {
    /// Checks the structure: monotone row pointers and column indices within `columns`.
    pub fn new(
        indptr: &'a [i64],
        indices: &'a [i64],
        data: &'a [f64],
        columns: usize,
    ) -> Result<Self, String> {
        if indptr.is_empty() || indptr[0] != 0 {
            return Err("indptr must start with 0".to_string());
        }
        if indices.len() != data.len() || indptr[indptr.len() - 1] as usize != data.len() {
            return Err("indptr, indices and data are inconsistent".to_string());
        }
        if indptr.windows(2).any(|w| w[1] < w[0]) {
            return Err("indptr must be non-decreasing".to_string());
        }
        if indices.iter().any(|&j| j < 0 || j as usize >= columns) {
            return Err("column index out of range".to_string());
        }
        Ok(Self {
            indptr,
            indices,
            data,
            columns,
        })
    }

    pub fn rows(&self) -> usize {
        self.indptr.len() - 1
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    /// y = A x, rows in parallel.
    pub fn multiply_into(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.columns);
        assert_eq!(y.len(), self.rows());
        y.par_iter_mut()
            .enumerate()
            .with_min_len(256)
            .for_each(|(row, out)| {
                let (start, end) = (self.indptr[row] as usize, self.indptr[row + 1] as usize);
                *out = self.data[start..end]
                    .iter()
                    .zip(&self.indices[start..end])
                    .map(|(value, &column)| value * x[column as usize])
                    .sum();
            });
    }
}

/// Dot product, reproducible regardless of the thread count.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let partial: Vec<f64> = x
        .par_chunks(CHUNK)
        .zip(y.par_chunks(CHUNK))
        .map(|(a, b)| a.iter().zip(b).map(|(p, q)| p * q).sum())
        .collect();
    partial.iter().sum()
}

pub fn norm(x: &[f64]) -> f64 {
    dot(x, x).sqrt()
}

/// y += alpha x.
pub fn add_scaled(y: &mut [f64], alpha: f64, x: &[f64]) {
    y.par_chunks_mut(CHUNK)
        .zip(x.par_chunks(CHUNK))
        .for_each(|(a, b)| {
            a.iter_mut().zip(b).for_each(|(p, q)| *p += alpha * q);
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    #[test]
    fn product_matches_dense() {
        let mut rng = Rng::new(41);
        let (rows, columns) = (300, 200);
        let mut dense = vec![0.0; rows * columns];
        let (mut indptr, mut indices, mut data) = (vec![0i64], vec![], vec![]);
        for row in 0..rows {
            for column in 0..columns {
                if rng.next() < 0.05 {
                    let value = rng.uniform(-1.0, 1.0);
                    dense[row * columns + column] = value;
                    indices.push(column as i64);
                    data.push(value);
                }
            }
            indptr.push(indices.len() as i64);
        }
        let matrix = CsrMatrix::new(&indptr, &indices, &data, columns).unwrap();
        let x: Vec<f64> = (0..columns).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let mut y = vec![0.0; rows];
        matrix.multiply_into(&x, &mut y);
        for row in 0..rows {
            let expected: f64 = (0..columns).map(|c| dense[row * columns + c] * x[c]).sum();
            assert!((y[row] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn rejects_malformed_matrices() {
        assert!(CsrMatrix::new(&[1, 1], &[0], &[1.0], 1).is_err());
        assert!(CsrMatrix::new(&[0, 2], &[0], &[1.0], 1).is_err());
        assert!(CsrMatrix::new(&[0, 1], &[3], &[1.0], 2).is_err());
        assert!(CsrMatrix::new(&[0, 1, 0], &[0], &[1.0], 1).is_err());
    }

    #[test]
    fn reductions_are_reproducible() {
        let mut rng = Rng::new(42);
        let x: Vec<f64> = (0..100_000).map(|_| rng.uniform(-1.0, 1.0)).collect();
        let first = dot(&x, &x);
        for _ in 0..5 {
            assert_eq!(dot(&x, &x).to_bits(), first.to_bits());
        }
        let sequential: f64 = x.iter().map(|v| v * v).sum();
        assert!((first - sequential).abs() < 1e-9 * sequential);
    }
}
