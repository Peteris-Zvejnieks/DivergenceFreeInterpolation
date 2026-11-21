//! Sparse interpolation matrix: the d x d kernel blocks of all point pairs within the support.

use rayon::prelude::*;

use crate::kernel::PairKernel;
use crate::tree::KdTree;

/// A sparse matrix in compressed sparse row form, with scipy's index conventions.
pub struct Csr {
    pub indptr: Vec<i64>,
    pub indices: Vec<i64>,
    pub data: Vec<f64>,
}

/// The (N d) x (N d) interpolation matrix with blocks K(x_i, x_j), rows of point i at
/// `d * i .. d * i + d`. Blocks of every pair within the support are stored in full, explicit
/// zeros included, with sorted column indices; rows are computed in parallel.
pub fn gram_matrix<K: PairKernel>(kernel: &K, points: &[f64], support_radius: f64) -> Csr {
    let d = kernel.dim();
    let n = points.len() / d;
    let tree = KdTree::new(points, d);
    let block_size = d * d;

    // Per point: its neighbours (sorted) and their blocks
    let rows: Vec<(Vec<usize>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let x = &points[i * d..(i + 1) * d];
            let mut neighbours = Vec::new();
            tree.for_each_within(x, support_radius, |j, _| neighbours.push(j));
            neighbours.sort_unstable();
            let mut kept = Vec::with_capacity(neighbours.len());
            let mut blocks = vec![0.0; neighbours.len() * block_size];
            for j in neighbours {
                let offset = kept.len() * block_size;
                let y = &points[j * d..(j + 1) * d];
                if kernel.block(
                    x,
                    y,
                    support_radius,
                    &mut blocks[offset..offset + block_size],
                ) {
                    kept.push(j);
                }
            }
            blocks.truncate(kept.len() * block_size);
            (kept, blocks)
        })
        .collect();

    let mut indptr = Vec::with_capacity(n * d + 1);
    indptr.push(0i64);
    for (kept, _) in &rows {
        for _ in 0..d {
            indptr.push(indptr[indptr.len() - 1] + (kept.len() * d) as i64);
        }
    }
    let nnz = indptr[indptr.len() - 1] as usize;
    let mut indices = Vec::with_capacity(nnz);
    let mut data = Vec::with_capacity(nnz);
    for (kept, blocks) in &rows {
        for a in 0..d {
            for (k, &j) in kept.iter().enumerate() {
                for b in 0..d {
                    indices.push((d * j + b) as i64);
                    data.push(blocks[k * block_size + a * d + b]);
                }
            }
        }
    }
    Csr {
        indptr,
        indices,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kind;
    use crate::testing::{Rng, wendland_like_kernel};

    fn dense(csr: &Csr, size: usize) -> Vec<f64> {
        let mut matrix = vec![0.0; size * size];
        for row in 0..size {
            for k in csr.indptr[row] as usize..csr.indptr[row + 1] as usize {
                matrix[row * size + csr.indices[k] as usize] += csr.data[k];
            }
        }
        matrix
    }

    #[test]
    fn matches_brute_force_and_is_symmetric() {
        let mut rng = Rng::new(11);
        for kind in [Kind::DivergenceFree, Kind::CurlFree] {
            for d in [2, 3] {
                let kernel = wendland_like_kernel(kind, d);
                let n = 150;
                let points = rng.points(n, d);
                let delta = 0.3;
                let csr = gram_matrix(&kernel, &points, delta);
                let size = n * d;
                let matrix = dense(&csr, size);
                // Products like (H u_a) u_b and (H u_b) u_a round differently: compare relative
                // to the largest entry
                let tolerance = 1e-13 * matrix.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let mut block = vec![0.0; d * d];
                for i in 0..n {
                    for j in 0..n {
                        let (x, y) = (&points[i * d..(i + 1) * d], &points[j * d..(j + 1) * d]);
                        let inside = kernel.block(x, y, delta, &mut block);
                        for a in 0..d {
                            for b in 0..d {
                                let expected = if inside { block[a * d + b] } else { 0.0 };
                                let entry = matrix[(d * i + a) * size + d * j + b];
                                assert!((entry - expected).abs() <= tolerance);
                                let mirror = matrix[(d * j + b) * size + d * i + a];
                                assert!((entry - mirror).abs() <= tolerance);
                            }
                        }
                    }
                }
                for row in 0..size {
                    let columns =
                        &csr.indices[csr.indptr[row] as usize..csr.indptr[row + 1] as usize];
                    assert!(
                        columns.windows(2).all(|w| w[0] < w[1]),
                        "columns must be sorted"
                    );
                }
            }
        }
    }
}
