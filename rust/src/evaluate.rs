//! Evaluation of interpolants `sum_j K(x, x_j) c_j` and their scalar potentials, in parallel
//! over the query points.

use rayon::prelude::*;

use crate::kernel::PairKernel;
use crate::tree::KdTree;

/// The interpolant with kernels centred at `nodes` (N x d) with `coefficients` (N x d), at
/// `queries` (M x d); returns M x d values, row-major.
pub fn evaluate<K: PairKernel>(
    kernel: &K,
    nodes: &[f64],
    coefficients: &[f64],
    queries: &[f64],
    support_radius: f64,
) -> Vec<f64> {
    let d = kernel.dim();
    let tree = KdTree::new(nodes, d);
    let mut out = vec![0.0; queries.len()];
    out.par_chunks_mut(d).enumerate().for_each_init(
        || vec![0.0; d * d],
        |block, (q, value)| {
            let x = &queries[q * d..(q + 1) * d];
            tree.for_each_within(x, support_radius, |j, _| {
                if kernel.block(x, &nodes[j * d..(j + 1) * d], support_radius, block) {
                    let c = &coefficients[j * d..(j + 1) * d];
                    for (a, entry) in value.iter_mut().enumerate() {
                        let row = &block[a * d..(a + 1) * d];
                        *entry += row.iter().zip(c).map(|(k, c)| k * c).sum::<f64>();
                    }
                }
            });
        },
    );
    out
}

/// The scalar potential (see [`crate::kernel::RadialKernel`]) at `queries`; returns M values.
pub fn evaluate_potential<K: PairKernel>(
    kernel: &K,
    nodes: &[f64],
    coefficients: &[f64],
    queries: &[f64],
    support_radius: f64,
) -> Vec<f64> {
    let d = kernel.dim();
    let tree = KdTree::new(nodes, d);
    let mut out = vec![0.0; queries.len() / d];
    out.par_iter_mut().enumerate().for_each(|(q, value)| {
        let x = &queries[q * d..(q + 1) * d];
        tree.for_each_within(x, support_radius, |j, _| {
            let y = &nodes[j * d..(j + 1) * d];
            let c = &coefficients[j * d..(j + 1) * d];
            *value += kernel.potential_term(x, y, c, support_radius);
        });
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kind;
    use crate::testing::{Rng, wendland_like_kernel};

    #[test]
    fn matches_direct_summation() {
        let mut rng = Rng::new(21);
        for kind in [Kind::DivergenceFree, Kind::CurlFree] {
            let d = 2;
            let kernel = wendland_like_kernel(kind, d);
            let (n, m, delta) = (200, 300, 0.25);
            let nodes = rng.points(n, d);
            let coefficients: Vec<f64> = (0..n * d).map(|_| rng.uniform(-1.0, 1.0)).collect();
            let queries: Vec<f64> = (0..m * d).map(|_| rng.uniform(-0.1, 1.1)).collect();
            let values = evaluate(&kernel, &nodes, &coefficients, &queries, delta);
            let potentials = evaluate_potential(&kernel, &nodes, &coefficients, &queries, delta);
            let mut block = vec![0.0; d * d];
            for q in 0..m {
                let x = &queries[q * d..(q + 1) * d];
                let mut expected = vec![0.0; d];
                let mut expected_potential = 0.0;
                for j in 0..n {
                    let (y, c) = (
                        &nodes[j * d..(j + 1) * d],
                        &coefficients[j * d..(j + 1) * d],
                    );
                    if kernel.block(x, y, delta, &mut block) {
                        for a in 0..d {
                            expected[a] += (0..d).map(|b| block[a * d + b] * c[b]).sum::<f64>();
                        }
                    }
                    expected_potential += kernel.potential_term(x, y, c, delta);
                }
                for a in 0..d {
                    assert!((values[q * d + a] - expected[a]).abs() < 1e-12);
                }
                assert!((potentials[q] - expected_potential).abs() < 1e-12);
            }
        }
    }
}
