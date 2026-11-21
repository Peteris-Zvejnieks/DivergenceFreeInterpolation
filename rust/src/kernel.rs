//! Matrix-valued divergence-free and curl-free kernels built from Wendland functions.
//!
//! For a radial function psi(r), with A(r) = psi'(r)/r and H(r) = psi''(r) - psi'(r)/r,
//!
//! ```text
//! Hess psi(x) = A(r) I + H(r) xhat xhat^T,            xhat = x / r
//! divergence-free:  (grad grad^T - Laplacian I) psi = (-(d-1) A - H) I + H xhat xhat^T
//! curl-free:        -grad grad^T psi                = -A I - H xhat xhat^T
//! ```
//!
//! A and H are polynomials in r for Wendland functions (k >= 1), given by their ascending
//! coefficients, which the Python package derives with sympy. With a support radius delta the
//! kernel is `delta^-d Phi((x - y) / delta)`, as in the Python implementation.

/// Which of the two matrix kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    DivergenceFree,
    CurlFree,
}

/// A matrix-valued kernel evaluated on point pairs.
///
/// Kernels take the two points (x, y) rather than their difference, so that kernels that are
/// not translation invariant, such as the boundary-weighted kernels of
/// `docs/tier2_boundary_conditions.md`, can implement the same interface.
pub trait PairKernel: Sync {
    /// Dimension of the points and of the (dim x dim) kernel blocks.
    fn dim(&self) -> usize;

    /// Writes the block K(x, y) row-major into `out` (dim * dim values) and returns true, or
    /// returns false without touching `out` when the pair is outside the support (zero block).
    fn block(&self, x: &[f64], y: &[f64], support_radius: f64, out: &mut [f64]) -> bool;

    /// The term `c . (potential contribution)` of the scalar potential at x for the
    /// coefficient vector `c` of the kernel centred at y (see [`RadialKernel`]).
    fn potential_term(&self, x: &[f64], y: &[f64], c: &[f64], support_radius: f64) -> f64;
}

/// Horner evaluation of a polynomial with ascending coefficients.
pub(crate) fn polynomial(coefficients: &[f64], r: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |sum, &c| sum * r + c)
}

/// The translation-invariant kernels of Fuselier (2008) in radial form.
///
/// Potentials: the curl-free interpolant is grad P with
/// `P(x) = -delta^(1-d) sum_j A(|u_j|) u_j . c_j`, and in 2D the divergence-free interpolant is
/// `(d/dy, -d/dx) psi` with `psi(x) = -delta^(1-d) sum_j A(|u_j|) perp(u_j) . c_j`, where
/// `u_j = (x - x_j) / delta` and `perp(u) = (u_y, -u_x)`.
#[derive(Clone, Debug)]
pub struct RadialKernel {
    kind: Kind,
    dim: usize,
    a: Vec<f64>,
    h: Vec<f64>,
}

impl RadialKernel {
    /// A kernel from the ascending coefficients of A(r) and H(r).
    pub fn new(kind: Kind, dim: usize, a: Vec<f64>, h: Vec<f64>) -> Result<Self, String> {
        if dim < 2 {
            return Err(format!(
                "dimension {dim} is not supported, the data must be at least 2D"
            ));
        }
        if a.is_empty() || h.is_empty() {
            return Err("the radial coefficients must not be empty".to_string());
        }
        Ok(Self { kind, dim, a, h })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    /// Squared norm of the scaled difference u = (x - y) / delta.
    fn scaled_norm2(&self, x: &[f64], y: &[f64], support_radius: f64) -> f64 {
        (0..self.dim)
            .map(|k| {
                let u = (x[k] - y[k]) / support_radius;
                u * u
            })
            .sum()
    }
}

impl PairKernel for RadialKernel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn block(&self, x: &[f64], y: &[f64], support_radius: f64, out: &mut [f64]) -> bool {
        let d = self.dim;
        let r2 = self.scaled_norm2(x, y, support_radius);
        if r2 > 1.0 {
            return false;
        }
        let r = r2.sqrt();
        let a = polynomial(&self.a, r);
        let h = polynomial(&self.h, r);
        // xhat xhat^T = u u^T / r^2; H(0) = 0, so the outer part vanishes at r = 0
        let inverse_r2 = if r2 > 0.0 { 1.0 / r2 } else { 0.0 };
        let (identity_part, outer_part) = match self.kind {
            Kind::DivergenceFree => (-((d - 1) as f64) * a - h, h * inverse_r2),
            Kind::CurlFree => (-a, -h * inverse_r2),
        };
        let scale = support_radius.powi(-(d as i32));
        for i in 0..d {
            let ui = (x[i] - y[i]) / support_radius;
            for j in 0..d {
                let uj = (x[j] - y[j]) / support_radius;
                let diagonal = if i == j { identity_part } else { 0.0 };
                out[i * d + j] = scale * (outer_part * ui * uj + diagonal);
            }
        }
        true
    }

    fn potential_term(&self, x: &[f64], y: &[f64], c: &[f64], support_radius: f64) -> f64 {
        let d = self.dim;
        let r2 = self.scaled_norm2(x, y, support_radius);
        if r2 > 1.0 {
            return 0.0;
        }
        let a = polynomial(&self.a, r2.sqrt());
        let u = |k: usize| (x[k] - y[k]) / support_radius;
        let dot = match self.kind {
            Kind::CurlFree => (0..d).map(|k| u(k) * c[k]).sum::<f64>(),
            // The stream function exists in 2D only; the caller checks the dimension
            Kind::DivergenceFree => u(1) * c[0] - u(0) * c[1],
        };
        -support_radius.powi(1 - d as i32) * a * dot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{Rng, wendland_like_kernel};

    fn block(kernel: &RadialKernel, x: &[f64], y: &[f64], delta: f64) -> Vec<f64> {
        let d = kernel.dim();
        let mut out = vec![0.0; d * d];
        kernel.block(x, y, delta, &mut out);
        out
    }

    /// Column `column` of K(x, 0) as a vector field of x.
    fn column(kernel: &RadialKernel, x: &[f64], column: usize) -> Vec<f64> {
        let d = kernel.dim();
        let zero = vec![0.0; d];
        let values = block(kernel, x, &zero, 1.0);
        (0..d).map(|row| values[row * d + column]).collect()
    }

    /// Central-difference Jacobian J[a][b] = d f_a / d x_b of a vector field.
    fn jacobian(f: impl Fn(&[f64]) -> Vec<f64>, x: &[f64]) -> Vec<Vec<f64>> {
        let h = 1e-5;
        let d = x.len();
        let mut jac = vec![vec![0.0; d]; d];
        for b in 0..d {
            let mut forward = x.to_vec();
            let mut backward = x.to_vec();
            forward[b] += h;
            backward[b] -= h;
            let (fp, fm) = (f(&forward), f(&backward));
            for a in 0..d {
                jac[a][b] = (fp[a] - fm[a]) / (2.0 * h);
            }
        }
        jac
    }

    fn random_point(rng: &mut Rng, d: usize, radius: f64) -> Vec<f64> {
        (0..d).map(|_| rng.uniform(-radius, radius)).collect()
    }

    #[test]
    fn blocks_are_symmetric_even_and_compact() {
        let mut rng = Rng::new(1);
        for kind in [Kind::DivergenceFree, Kind::CurlFree] {
            for d in [2, 3, 4] {
                let kernel = wendland_like_kernel(kind, d);
                for _ in 0..50 {
                    let x = random_point(&mut rng, d, 0.4);
                    let y = random_point(&mut rng, d, 0.4);
                    let forward = block(&kernel, &x, &y, 1.0);
                    let backward = block(&kernel, &y, &x, 1.0);
                    for i in 0..d {
                        for j in 0..d {
                            let value = forward[i * d + j];
                            assert!((value - forward[j * d + i]).abs() < 1e-14);
                            assert!((value - backward[i * d + j]).abs() < 1e-14);
                        }
                    }
                }
                let far = vec![2.0; d];
                let mut out = vec![7.0; d * d];
                assert!(!kernel.block(&far, &vec![0.0; d], 1.0, &mut out));
                assert!(out.iter().all(|&v| v == 7.0));
            }
        }
    }

    #[test]
    fn divergence_free_columns() {
        let mut rng = Rng::new(2);
        for d in [2, 3, 4] {
            let kernel = wendland_like_kernel(Kind::DivergenceFree, d);
            for _ in 0..20 {
                let x = random_point(&mut rng, d, 0.4);
                for c in 0..d {
                    let jac = jacobian(|p| column(&kernel, p, c), &x);
                    let divergence: f64 = (0..d).map(|a| jac[a][a]).sum();
                    assert!(divergence.abs() < 1e-6, "divergence {divergence}");
                }
            }
        }
    }

    #[test]
    fn curl_free_columns_have_symmetric_jacobians() {
        let mut rng = Rng::new(3);
        for d in [2, 3, 4] {
            let kernel = wendland_like_kernel(Kind::CurlFree, d);
            for _ in 0..20 {
                let x = random_point(&mut rng, d, 0.4);
                for c in 0..d {
                    let jac = jacobian(|p| column(&kernel, p, c), &x);
                    for a in 0..d {
                        for b in 0..d {
                            assert!((jac[a][b] - jac[b][a]).abs() < 1e-6);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn potential_gradient_is_the_field() {
        let mut rng = Rng::new(4);
        let cases = [
            (Kind::CurlFree, 2),
            (Kind::CurlFree, 3),
            (Kind::DivergenceFree, 2),
        ];
        for (kind, d) in cases {
            let kernel = wendland_like_kernel(kind, d);
            let y = random_point(&mut rng, d, 0.3);
            let c = random_point(&mut rng, d, 1.0);
            let delta = 0.9;
            let potential = |p: &[f64]| vec![kernel.potential_term(p, &y, &c, delta)];
            for _ in 0..20 {
                let x = random_point(&mut rng, d, 0.3);
                let values = block(&kernel, &x, &y, delta);
                let field: Vec<f64> = (0..d)
                    .map(|a| (0..d).map(|b| values[a * d + b] * c[b]).sum())
                    .collect();
                // The potential is scalar: differentiate it along each axis
                let gradient: Vec<f64> = (0..d)
                    .map(|b| {
                        let h = 1e-6;
                        let mut forward = x.clone();
                        let mut backward = x.clone();
                        forward[b] += h;
                        backward[b] -= h;
                        (potential(&forward)[0] - potential(&backward)[0]) / (2.0 * h)
                    })
                    .collect();
                let expected = match kind {
                    Kind::CurlFree => gradient,
                    Kind::DivergenceFree => vec![gradient[1], -gradient[0]],
                };
                for a in 0..d {
                    assert!(
                        (field[a] - expected[a]).abs() < 1e-5,
                        "{field:?} vs {expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_parameters() {
        assert!(RadialKernel::new(Kind::CurlFree, 1, vec![1.0], vec![1.0]).is_err());
        assert!(RadialKernel::new(Kind::CurlFree, 2, vec![], vec![1.0]).is_err());
    }
}
