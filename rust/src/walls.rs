//! Exact wall conditions through a boundary-weighted divergence-free kernel ("tier 2", see
//! `docs/tier2_boundary_conditions.md`).
//!
//! A divergence-free field is the curl of a vector potential. Modelling the potential with the
//! scalar kernel `k(x, y) = s g(x) g(y) psi(|x - y| / delta)`, where the wall function g vanishes
//! on the walls and `s = delta^(2-d)`, gives the matrix kernel
//!
//! ```text
//! K(x, y) = tr(M) I - M^T,      M = grad_x grad_y^T k
//! M = s [grad g(x) grad g(y)^T psi - g(y) grad g(x) grad psi^T + g(x) grad psi grad g(y)^T
//!        - g(x) g(y) Hess psi]                       (psi and its derivatives at x - y)
//! ```
//!
//! Its columns are divergence-free in any dimension, and on a wall (g = 0) every interpolant is
//! tangent to it: `n . u = 0` everywhere on the wall. Squaring a wall's factor of g also makes
//! the tangential part vanish there (no slip). With g = 1 the kernel is the divergence-free
//! [`RadialKernel`](crate::kernel::RadialKernel), scale included.
//!
//! Each wall contributes the factor `q(dist(x) / band)^power` to g, where dist is the signed
//! distance to the wall (positive on the flow side) and `q(t) = 1 - (1 - t)^3` for t < 1, 1
//! beyond: zero on the wall with slope 3 / band, and C^2-flat to 1 at distance `band`. Several
//! walls multiply.

use crate::kernel::{PairKernel, polynomial};

/// The geometry of a wall, given by its signed distance (positive on the flow side).
#[derive(Clone, Debug)]
pub enum Shape {
    /// A circle (2D) or sphere: `|x - center| - radius`, negated when the flow is inside.
    Ball {
        center: Vec<f64>,
        radius: f64,
        inside: bool,
    },
    /// A line (2D) or plane through `point` with unit `normal`, pointing into the flow.
    HalfSpace { point: Vec<f64>, normal: Vec<f64> },
    /// An infinite cylinder (3D) around the line through `point` along the unit `axis`.
    Cylinder {
        point: Vec<f64>,
        axis: Vec<f64>,
        radius: f64,
        inside: bool,
    },
}

/// One wall: a shape, the width of the band in which it affects the kernel, and the power of
/// its factor (1: no penetration, 2: no slip).
#[derive(Clone, Debug)]
pub struct Wall {
    pub shape: Shape,
    pub band: f64,
    pub power: u32,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

impl Shape {
    fn dim(&self) -> usize {
        match self {
            Shape::Ball { center, .. } => center.len(),
            Shape::HalfSpace { point, .. } | Shape::Cylinder { point, .. } => point.len(),
        }
    }

    /// Signed distance at x; writes its gradient into `gradient` when `with_gradient` is set.
    /// At the centre of a ball or on the axis of a cylinder the gradient is set to zero.
    fn distance(&self, x: &[f64], gradient: &mut [f64], with_gradient: bool) -> f64 {
        // Distance from a centre (ball) or axis (cylinder), given the offset w from it
        let radial = |w: &mut [f64], radius: f64, inside: bool, gradient: &mut [f64]| {
            let rho = dot(w, w).sqrt();
            let sign = if inside { -1.0 } else { 1.0 };
            if with_gradient {
                for (g, w) in gradient.iter_mut().zip(w.iter()) {
                    *g = if rho > 0.0 { sign * w / rho } else { 0.0 };
                }
            }
            sign * (rho - radius)
        };
        match self {
            Shape::Ball {
                center,
                radius,
                inside,
            } => {
                let mut w: Vec<f64> = x.iter().zip(center).map(|(x, c)| x - c).collect();
                radial(&mut w, *radius, *inside, gradient)
            }
            Shape::HalfSpace { point, normal } => {
                if with_gradient {
                    gradient.copy_from_slice(normal);
                }
                x.iter()
                    .zip(point)
                    .zip(normal)
                    .map(|((x, p), n)| (x - p) * n)
                    .sum()
            }
            Shape::Cylinder {
                point,
                axis,
                radius,
                inside,
            } => {
                let v: Vec<f64> = x.iter().zip(point).map(|(x, p)| x - p).collect();
                let along = dot(&v, axis);
                let mut w: Vec<f64> = v.iter().zip(axis).map(|(v, a)| v - along * a).collect();
                radial(&mut w, *radius, *inside, gradient)
            }
        }
    }
}

impl Wall {
    /// This wall's factor f of g at x and its derivative df/d(dist) (grad f is that times the
    /// gradient of the signed distance). `scratch` holds dim values.
    fn factor(&self, x: &[f64], scratch: &mut [f64]) -> (f64, f64) {
        let t = self.shape.distance(x, scratch, false) / self.band;
        if t >= 1.0 {
            return (1.0, 0.0);
        }
        let q = 1.0 - (1.0 - t).powi(3);
        let slope = 3.0 * (1.0 - t).powi(2) / self.band;
        match self.power {
            1 => (q, slope),
            power => (
                q.powi(power as i32),
                power as f64 * q.powi(power as i32 - 1) * slope,
            ),
        }
    }
}

/// The product of the walls' factors: g(x) and grad g(x).
#[derive(Clone, Debug)]
pub struct WallFunction {
    dim: usize,
    walls: Vec<Wall>,
}

impl WallFunction {
    /// Checks the walls against the dimension and normalises their normals and axes.
    pub fn new(dim: usize, mut walls: Vec<Wall>) -> Result<Self, String> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let finite = |values: &[f64]| values.iter().all(|v| v.is_finite());
        // A direction of the data's dimension, scaled to unit length
        let unit = |vector: &mut Vec<f64>, name: &str| {
            let norm = dot(vector, vector).sqrt();
            if vector.len() != dim || !positive(norm) {
                return Err(format!(
                    "a wall {name} must be {dim} finite numbers, not zero"
                ));
            }
            vector.iter_mut().for_each(|v| *v /= norm);
            Ok(())
        };
        for wall in &mut walls {
            if wall.shape.dim() != dim {
                return Err(format!(
                    "a {}D wall cannot bound {dim}D data",
                    wall.shape.dim()
                ));
            }
            if !positive(wall.band) || wall.power == 0 {
                return Err("walls need a positive band and power".to_string());
            }
            match &mut wall.shape {
                Shape::Ball { center, radius, .. } => {
                    if !finite(center) || !positive(*radius) {
                        return Err("a ball needs a finite centre and positive radius".into());
                    }
                }
                Shape::HalfSpace { point, normal } => {
                    if !finite(point) {
                        return Err("a wall point must be finite".into());
                    }
                    unit(normal, "normal")?;
                }
                Shape::Cylinder {
                    point,
                    axis,
                    radius,
                    ..
                } => {
                    if dim != 3 {
                        return Err("cylinders are 3D walls".to_string());
                    }
                    if !finite(point) || !positive(*radius) {
                        return Err("a cylinder needs a finite point and positive radius".into());
                    }
                    unit(axis, "axis")?;
                }
            }
        }
        Ok(Self { dim, walls })
    }

    /// g(x), with grad g(x) written into `gradient` (dim values).
    pub fn value(&self, x: &[f64], gradient: &mut [f64]) -> f64 {
        let mut g = 1.0;
        gradient.fill(0.0);
        let mut normal = [0.0; 8];
        let mut heap;
        let normal: &mut [f64] = if self.dim <= normal.len() {
            &mut normal[..self.dim]
        } else {
            heap = vec![0.0; self.dim];
            &mut heap
        };
        for wall in &self.walls {
            let (f, slope) = wall.factor(x, normal);
            if f == 1.0 && slope == 0.0 {
                continue;
            }
            wall.shape.distance(x, normal, true);
            // Product rule: grad(g f) = f grad g + g grad f, with grad f = slope * grad dist
            for (out, n) in gradient.iter_mut().zip(normal.iter()) {
                *out = f * *out + g * slope * n;
            }
            g *= f;
        }
        g
    }
}

/// The boundary-weighted divergence-free kernel; see the module documentation.
#[derive(Clone, Debug)]
pub struct WallKernel {
    dim: usize,
    psi: Vec<f64>,
    a: Vec<f64>,
    h: Vec<f64>,
    walls: WallFunction,
}

impl WallKernel {
    /// A kernel from the ascending coefficients of psi(r), A(r) = psi'(r)/r and
    /// H(r) = psi''(r) - psi'(r)/r.
    pub fn new(
        dim: usize,
        psi: Vec<f64>,
        a: Vec<f64>,
        h: Vec<f64>,
        walls: Vec<Wall>,
    ) -> Result<Self, String> {
        if dim < 2 {
            return Err(format!(
                "dimension {dim} is not supported, the data must be at least 2D"
            ));
        }
        if psi.is_empty() || a.is_empty() || h.is_empty() {
            return Err("the radial coefficients must not be empty".to_string());
        }
        let walls = WallFunction::new(dim, walls)?;
        Ok(Self {
            dim,
            psi,
            a,
            h,
            walls,
        })
    }

    /// Scaled difference u = (x - y) / delta and r^2 = |u|^2.
    fn scaled(&self, x: &[f64], y: &[f64], delta: f64, u: &mut [f64]) -> f64 {
        for (k, u) in u.iter_mut().enumerate() {
            *u = (x[k] - y[k]) / delta;
        }
        dot(u, u)
    }
}

/// Stack storage for small dimensions, heap beyond.
const STACK_DIM: usize = 4;

impl PairKernel for WallKernel {
    fn dim(&self) -> usize {
        self.dim
    }

    fn block(&self, x: &[f64], y: &[f64], support_radius: f64, out: &mut [f64]) -> bool {
        let d = self.dim;
        let mut storage = [0.0; 4 * STACK_DIM];
        let mut heap;
        let storage: &mut [f64] = if d <= STACK_DIM {
            &mut storage[..4 * d]
        } else {
            heap = vec![0.0; 4 * d];
            &mut heap
        };
        let (u, rest) = storage.split_at_mut(d);
        let (grad_psi, rest) = rest.split_at_mut(d);
        let (dgx, dgy) = rest.split_at_mut(d);

        let r2 = self.scaled(x, y, support_radius, u);
        if r2 > 1.0 {
            return false;
        }
        let r = r2.sqrt();
        let psi = polynomial(&self.psi, r);
        let a = polynomial(&self.a, r);
        let h = polynomial(&self.h, r);
        let delta = support_radius;
        // grad psi = A u / delta; Hess psi = (A I + H u u^T / r^2) / delta^2 (H(0) = 0)
        for (g, u) in grad_psi.iter_mut().zip(u.iter()) {
            *g = a * u / delta;
        }
        let outer = if r2 > 0.0 { h / r2 } else { 0.0 };
        let gx = self.walls.value(x, dgx);
        let gy = self.walls.value(y, dgy);
        let scale = delta.powi(2 - d as i32);
        let hessian_scale = gx * gy / (delta * delta);

        // M (row-major, in `out`), then K = tr(M) I - M^T in place
        for i in 0..d {
            for j in 0..d {
                let hessian = if i == j { a } else { 0.0 } + outer * u[i] * u[j];
                out[i * d + j] = scale
                    * (dgx[i] * dgy[j] * psi - gy * dgx[i] * grad_psi[j]
                        + gx * grad_psi[i] * dgy[j]
                        - hessian_scale * hessian);
            }
        }
        let trace: f64 = (0..d).map(|i| out[i * d + i]).sum();
        for i in 0..d {
            for j in i + 1..d {
                let (upper, lower) = (out[i * d + j], out[j * d + i]);
                out[i * d + j] = -lower;
                out[j * d + i] = -upper;
            }
            out[i * d + i] = trace - out[i * d + i];
        }
        true
    }

    /// The 2D stream function term `c . perp(grad_y k)` with perp(v) = (v_y, -v_x), so that the
    /// interpolant is (d/dy, -d/dx) of the sum of these terms.
    fn potential_term(&self, x: &[f64], y: &[f64], c: &[f64], support_radius: f64) -> f64 {
        let d = self.dim;
        let mut u = [0.0; STACK_DIM];
        let mut dgx = [0.0; STACK_DIM];
        let mut dgy = [0.0; STACK_DIM];
        // The stream function exists in 2D only; the caller checks the dimension
        let r2 = self.scaled(x, y, support_radius, &mut u[..d]);
        if r2 > 1.0 {
            return 0.0;
        }
        let r = r2.sqrt();
        let psi = polynomial(&self.psi, r);
        let a = polynomial(&self.a, r);
        let gx = self.walls.value(x, &mut dgx[..d]);
        let gy = self.walls.value(y, &mut dgy[..d]);
        let delta = support_radius;
        // grad_y k = s g(x) [grad g(y) psi - g(y) grad psi], grad psi = A u / delta
        let perp = |v: [f64; 2]| v[1] * c[0] - v[0] * c[1];
        let wall_part = perp([dgy[0], dgy[1]]) * psi;
        let radial_part = perp([u[0], u[1]]) * a / delta;
        delta.powi(2 - d as i32) * gx * (wall_part - gy * radial_part)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kind;
    use crate::testing::{Rng, wendland_like_kernel, wendland_like_psi};

    fn kernel(d: usize, walls: Vec<Wall>) -> WallKernel {
        let (psi, a, h) = wendland_like_psi();
        WallKernel::new(d, psi, a, h, walls).unwrap()
    }

    fn block(kernel: &impl PairKernel, x: &[f64], y: &[f64], delta: f64) -> Vec<f64> {
        let d = kernel.dim();
        let mut out = vec![0.0; d * d];
        kernel.block(x, y, delta, &mut out);
        out
    }

    fn sphere(d: usize, power: u32) -> Wall {
        Wall {
            shape: Shape::Ball {
                center: vec![0.1; d],
                radius: 0.3,
                inside: false,
            },
            band: 0.4,
            power,
        }
    }

    fn walls(d: usize) -> Vec<Vec<Wall>> {
        let mut cases = vec![
            vec![sphere(d, 1)],
            vec![sphere(d, 2)],
            vec![Wall {
                shape: Shape::Ball {
                    center: vec![0.0; d],
                    radius: 1.0,
                    inside: true,
                },
                band: 0.5,
                power: 1,
            }],
            vec![
                sphere(d, 1),
                Wall {
                    shape: Shape::HalfSpace {
                        point: vec![-0.2; d],
                        normal: {
                            let mut n = vec![0.0; d];
                            n[0] = 0.6;
                            n[1] = 0.8;
                            n
                        },
                    },
                    band: 0.3,
                    power: 1,
                },
            ],
        ];
        if d == 3 {
            cases.push(vec![Wall {
                shape: Shape::Cylinder {
                    point: vec![0.0, 0.1, 0.0],
                    axis: vec![0.0, 0.6, 0.8],
                    radius: 0.2,
                    inside: false,
                },
                band: 0.3,
                power: 1,
            }]);
        }
        cases
    }

    fn random_point(rng: &mut Rng, d: usize) -> Vec<f64> {
        (0..d).map(|_| rng.uniform(-0.6, 0.6)).collect()
    }

    #[test]
    fn without_walls_it_is_the_radial_kernel() {
        let mut rng = Rng::new(31);
        for d in [2, 3] {
            let plain = wendland_like_kernel(Kind::DivergenceFree, d);
            let weighted = kernel(d, vec![]);
            for _ in 0..30 {
                let (x, y) = (random_point(&mut rng, d), random_point(&mut rng, d));
                let delta = 0.9;
                let (expected, actual) = (
                    block(&plain, &x, &y, delta),
                    block(&weighted, &x, &y, delta),
                );
                for (e, a) in expected.iter().zip(&actual) {
                    assert!((e - a).abs() < 1e-12 * (1.0 + e.abs()), "{e} vs {a}");
                }
                let c = random_point(&mut rng, d);
                if d == 2 {
                    let (e, a) = (
                        plain.potential_term(&x, &y, &c, delta),
                        weighted.potential_term(&x, &y, &c, delta),
                    );
                    assert!((e - a).abs() < 1e-12 * (1.0 + e.abs()));
                }
            }
        }
    }

    #[test]
    fn wall_function_gradient() {
        let mut rng = Rng::new(32);
        for d in [2, 3] {
            for walls in walls(d) {
                let function = WallFunction::new(d, walls).unwrap();
                for _ in 0..30 {
                    let x = random_point(&mut rng, d);
                    let mut gradient = vec![0.0; d];
                    function.value(&x, &mut gradient);
                    let mut scratch = vec![0.0; d];
                    for k in 0..d {
                        let step = 1e-6;
                        let (mut forward, mut backward) = (x.clone(), x.clone());
                        forward[k] += step;
                        backward[k] -= step;
                        let numeric = (function.value(&forward, &mut scratch)
                            - function.value(&backward, &mut scratch))
                            / (2.0 * step);
                        assert!(
                            (numeric - gradient[k]).abs() < 1e-6,
                            "{numeric} vs {gradient:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn blocks_are_transposes_and_columns_divergence_free() {
        let mut rng = Rng::new(33);
        for d in [2, 3] {
            for walls in walls(d) {
                let kernel = kernel(d, walls);
                let delta = 0.8;
                for _ in 0..20 {
                    let (x, y) = (random_point(&mut rng, d), random_point(&mut rng, d));
                    let (forward, backward) =
                        (block(&kernel, &x, &y, delta), block(&kernel, &y, &x, delta));
                    for i in 0..d {
                        for j in 0..d {
                            let (a, b) = (forward[i * d + j], backward[j * d + i]);
                            assert!((a - b).abs() < 1e-12 * (1.0 + a.abs()));
                        }
                    }
                    // Divergence of column c of K(., y) at x, by central differences
                    for c in 0..d {
                        let divergence: f64 = (0..d)
                            .map(|a| {
                                let step = 1e-5;
                                let (mut forward, mut backward) = (x.clone(), x.clone());
                                forward[a] += step;
                                backward[a] -= step;
                                (block(&kernel, &forward, &y, delta)[a * d + c]
                                    - block(&kernel, &backward, &y, delta)[a * d + c])
                                    / (2.0 * step)
                            })
                            .sum();
                        let scale = block(&kernel, &x, &y, delta)
                            .iter()
                            .fold(0.0f64, |m, v| m.max(v.abs()));
                        // Central differences of entries of size ~100: relative tolerance
                        assert!(
                            divergence.abs() <= 1e-6 * (scale + 1.0),
                            "divergence {divergence}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fields_are_tangent_to_walls_and_vanish_for_no_slip() {
        let mut rng = Rng::new(34);
        for d in [2, 3] {
            for power in [1, 2] {
                let wall = sphere(d, power);
                let kernel = kernel(d, vec![wall]);
                for _ in 0..30 {
                    // A point on the sphere around (0.1, ...) of radius 0.3
                    let direction = random_point(&mut rng, d);
                    let norm = dot(&direction, &direction).sqrt();
                    let normal: Vec<f64> = direction.iter().map(|v| v / norm).collect();
                    let x: Vec<f64> = normal.iter().map(|n| 0.1 + 0.3 * n).collect();
                    let y = random_point(&mut rng, d);
                    let values = block(&kernel, &x, &y, 0.9);
                    let largest = block(&kernel, &y, &y, 0.9)
                        .iter()
                        .fold(0.0f64, |m, v| m.max(v.abs()));
                    for c in 0..d {
                        let column: Vec<f64> = (0..d).map(|a| values[a * d + c]).collect();
                        let flux = dot(&column, &normal);
                        assert!(flux.abs() < 1e-12 * largest, "u . n = {flux}");
                        if power == 2 {
                            assert!(column.iter().all(|v| v.abs() < 1e-12 * largest));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stream_function_generates_the_field() {
        let mut rng = Rng::new(35);
        for walls in walls(2) {
            let kernel = kernel(2, walls);
            let delta = 0.9;
            for _ in 0..20 {
                let (x, y) = (random_point(&mut rng, 2), random_point(&mut rng, 2));
                let c = random_point(&mut rng, 2);
                let values = block(&kernel, &x, &y, delta);
                let field = [
                    values[0] * c[0] + values[1] * c[1],
                    values[2] * c[0] + values[3] * c[1],
                ];
                let derivative = |k: usize| {
                    let step = 1e-6;
                    let (mut forward, mut backward) = (x.clone(), x.clone());
                    forward[k] += step;
                    backward[k] -= step;
                    (kernel.potential_term(&forward, &y, &c, delta)
                        - kernel.potential_term(&backward, &y, &c, delta))
                        / (2.0 * step)
                };
                let expected = [derivative(1), -derivative(0)];
                for k in 0..2 {
                    assert!(
                        (field[k] - expected[k]).abs() < 1e-5,
                        "{field:?} vs {expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn rejects_bad_walls() {
        let (psi, a, h) = wendland_like_psi();
        let make = |d: usize, wall: Wall| {
            WallKernel::new(d, psi.clone(), a.clone(), h.clone(), vec![wall])
        };
        assert!(make(3, sphere(2, 1)).is_err());
        let mut flat = sphere(2, 1);
        flat.band = 0.0;
        assert!(make(2, flat).is_err());
        let cylinder = Wall {
            shape: Shape::Cylinder {
                point: vec![0.0; 2],
                axis: vec![1.0, 0.0],
                radius: 0.2,
                inside: false,
            },
            band: 0.3,
            power: 1,
        };
        assert!(make(2, cylinder).is_err());
        // Directions of the wrong length or zero, and non-positive radii
        let half_space = |normal: Vec<f64>| Wall {
            shape: Shape::HalfSpace {
                point: vec![0.0; 2],
                normal,
            },
            band: 0.3,
            power: 1,
        };
        assert!(make(2, half_space(vec![1.0])).is_err());
        assert!(make(2, half_space(vec![0.0, 0.0])).is_err());
        let mut shrunk = sphere(2, 1);
        if let Shape::Ball { radius, .. } = &mut shrunk.shape {
            *radius = 0.0;
        }
        assert!(make(2, shrunk).is_err());
    }

    #[test]
    fn normals_are_normalised() {
        let wall = |length: f64| Wall {
            shape: Shape::HalfSpace {
                point: vec![0.0; 2],
                normal: vec![0.6 * length, 0.8 * length],
            },
            band: 0.3,
            power: 1,
        };
        let (unit, long) = (kernel(2, vec![wall(1.0)]), kernel(2, vec![wall(5.0)]));
        let (x, y) = ([0.1, 0.05], [0.0, 0.2]);
        assert_eq!(block(&unit, &x, &y, 0.5), block(&long, &x, &y, 0.5));
    }
}
