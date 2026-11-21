//! Helpers shared by the unit tests.

use crate::kernel::{Kind, RadialKernel};

/// A small deterministic random number generator (xorshift64*), to avoid a dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Uniform in [0, 1).
    pub fn next(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn uniform(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.next()
    }

    /// `n` random points in [0, 1)^d, row-major.
    pub fn points(&mut self, n: usize, d: usize) -> Vec<f64> {
        (0..n * d).map(|_| self.next()).collect()
    }
}

/// Ascending coefficients of a polynomial product.
fn multiply(p: &[f64], q: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; p.len() + q.len() - 1];
    for (i, a) in p.iter().enumerate() {
        for (j, b) in q.iter().enumerate() {
            out[i + j] += a * b;
        }
    }
    out
}

/// A kernel from the Wendland-type function psi(r) = (1 - r)^6 (35 r^2 + 18 r + 3), which is
/// smooth and even at the origin. The structural properties tested (symmetry, divergence- or
/// curl-freeness, potentials) hold for any such radial function.
pub fn wendland_like_kernel(kind: Kind, dim: usize) -> RadialKernel {
    let (_, a, h) = wendland_like_psi();
    RadialKernel::new(kind, dim, a, h).unwrap()
}

/// The ascending coefficients of that psi(r), A(r) = psi'(r)/r and H(r) = psi''(r) - psi'(r)/r.
pub fn wendland_like_psi() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut psi = vec![3.0, 18.0, 35.0];
    for _ in 0..6 {
        psi = multiply(&psi, &[1.0, -1.0]);
    }
    assert!(psi[1].abs() < 1e-12, "psi must be even at the origin");
    // Coefficient of r^(j-2) from c_j
    let a = (2..psi.len()).map(|j| j as f64 * psi[j]).collect();
    let h = (2..psi.len())
        .map(|j| (j * (j - 2)) as f64 * psi[j])
        .collect();
    (psi, a, h)
}
