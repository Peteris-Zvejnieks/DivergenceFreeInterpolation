//! Rust core of the `divergence_free_interpolant` Python package.
//!
//! The Python package keeps the API, the linear algebra (scipy) and the multilevel and boundary
//! condition logic; this crate does the heavy, embarrassingly parallel parts:
//!
//! - [`kernel`]: the divergence-free and curl-free matrix kernels, evaluated on point pairs;
//! - [`tree`]: a kd-tree for radius queries in any dimension;
//! - [`assembly`]: the sparse interpolation matrix, in parallel;
//! - [`evaluate`]: interpolants and their scalar potentials at query points, in parallel;
//! - [`thinning`]: the greedy farthest-point ordering behind the multilevel levels;
//! - [`sparse`] and [`cg`]: parallel CSR products and preconditioned conjugate gradients.
//!
//! Points are passed as flat row-major slices (`n * dim` values).

// Index loops mirror the formulas in the tests; keep them readable there
#![cfg_attr(test, allow(clippy::needless_range_loop))]

pub mod assembly;
pub mod cg;
pub mod evaluate;
pub mod kernel;
pub mod sparse;
pub mod thinning;
pub mod tree;

#[cfg(feature = "python")]
mod python;

#[cfg(test)]
pub(crate) mod testing;
