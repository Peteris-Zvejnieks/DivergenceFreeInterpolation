# Exact wall conditions through a boundary-weighted kernel ("tier 2")

**Status: implemented for analytic walls** (`walls=` with the shapes in
`divergence_free_interpolant.walls`; see the README). This document records the derivation and
the research spike that led to it, in
[`research/boundary_weighted_kernel.py`](../research/boundary_weighted_kernel.py)
(`uv run python research/boundary_weighted_kernel.py` reproduces every number below).

**What exists:**
- Tier 1 (`NormalComponent`, `VectorValue`) remains the way to impose conditions on general
  parametric curves and surfaces, and the only one for curl-free fields.
- Tier 2 covers circles, spheres, lines, planes and infinite cylinders, with no-penetration or
  no-slip conditions, in single-level and multilevel interpolants, on both backends (the Rust
  kernel in `rust/src/walls.rs`, the numpy reference in `_kernel.py` and `walls.py`).

## Why a second approach

Tier 1 imposes a condition in the mean over boundary segments, as extra linear functionals
(see the README). That works well. In the cylinder example the largest normal velocity on the
wall drops from 0.34 to about 10⁻⁵. But it has three inherent limits:

- the condition holds exactly only for the segment means, and approximately in between;
- every segment adds unknowns to the system;
- the segments have to be chosen: their length, and their resolution on small obstacles.

Tier 2 builds the condition into the kernel instead. Every field the interpolant can produce
satisfies it, everywhere on the wall, with no extra unknowns.

## Construction

A divergence-free field is the curl of a vector potential, $u = \nabla \times (gA)$. Choose a
smooth **wall function** $g$ that vanishes on the wall, and model $A$ with the scalar kernel

$$
k(x, y) = g(x)\, g(y)\, \psi(x - y)
$$

(times the identity), where $\psi$ is the Wendland function. The resulting matrix kernel is

$$
K(x, y) = \mathrm{curl}_x \mathrm{curl}_y^{T} \big[k(x, y)\, I\big]
        = \mathrm{tr}(M)\, I - M^{T},
\qquad M = \nabla_x \nabla_y^{T}\, k(x, y).
$$

The same formula holds in 2D (where curl is the perpendicular gradient) and in 3D. Writing out
$M$, with $\psi$ and its derivatives evaluated at $x - y$:

$$
M = \nabla g(x)\, \nabla g(y)^{T} \psi
  - g(y)\, \nabla g(x)\, \nabla\psi^{T}
  + g(x)\, \nabla\psi\, \nabla g(y)^{T}
  - g(x)\, g(y)\, \nabla\nabla^{T}\psi .
$$

**Properties:**

- **Reduces to the package's kernel.** For $g \equiv 1$, $K = (\nabla\nabla^{T} - \Delta I)\,\psi$
  is Fuselier's divergence-free kernel. The spike checks this to $3 \cdot 10^{-16}$.
- **Divergence-free.** Every column of $K$ is a curl, so every interpolant is exactly
  divergence-free.
- **Exact no-penetration.** An interpolant is $u = \nabla \times (gF)
  = g\, \nabla \times F + \nabla g \times F$ for some field $F$. On the wall $g = 0$, so
  $u = \nabla g \times F$. That is perpendicular to $\nabla g$, which is parallel to the normal,
  so $u \cdot n = 0$ **at every point of the wall**.
- **Exact no-slip with $g^2$.** Using $g^2$ instead of $g$ gives
  $u = 2g\, \nabla g \times F + g^2\, \nabla \times F = 0$ on the wall.
- **Positive semi-definite.** $K = L_x L_y^{T} k$ with $L = \mathrm{curl}$ and $k$
  positive definite wherever $g \neq 0$. The interpolation matrices were positive definite in
  every test, in 2D and 3D, but this is not proven.
- **Cheap ingredients.** It needs only $g$ and $\nabla g$ at both points, plus $\psi$,
  $\nabla\psi$ and $\nabla\nabla^{T}\psi$. The last two are the radial building blocks $A(r)$
  and $H(r)$ the package already uses; $\psi$ itself is its polynomial.

## Wall functions

The spike uses $g(x) = q\big(\mathrm{dist}(x)/\ell\big)$, where $\mathrm{dist}$ is the signed distance to the wall (positive in the flow), and

$$
q(s) = \begin{cases} 1 - (1 - s)^3 & s < 1 \\ 1 & s \geq 1 \end{cases}
$$

$q$ is zero on the wall with slope 3, and flattens smoothly ($C^2$) to 1 at distance $\ell$
(the **band**). Beyond the band, $g = 1$ and the kernel is the ordinary one. Inside an
obstacle, $q$ continues smoothly with negative values. The field there is meaningless but
harmless; it should simply not be evaluated.

| Wall | Signed distance |
|---|---|
| circle / sphere, centre $c$, radius $R$ | $\lvert x - c \rvert - R$ |
| plane / line, unit normal $n$, offset $b$ | $n \cdot x - b$ |
| infinite cylinder (3D), axis through $c$ along unit $a$ | $\lvert (x - c) - ((x - c)\cdot a)\, a \rvert - R$ |

Several walls can be combined with a product, $g = \prod_i g_i$, which vanishes on each wall,
or with R-functions (smooth minimum). For **general parametric curves and surfaces** it is
harder:

- The closest-point distance is smooth only within the boundary's reach (the distance to its
  medial axis), so the band must stay below that.
- Open curves have kinked distances around their end points.
- The unsigned distance has a kink across the curve.

## Results

Uniform potential flow past a cylinder (2D) and a sphere (3D) of radius 0.3, which is exactly
tangent to the wall. No samples lie within 0.25 of the wall, where the methods differ most.
2D uses 150 samples and support 1.5; 3D uses 1500 samples and support 0.9. Tier 2 uses band
$\ell = 0.5$. "Plain" is the interpolant without any wall condition.

| 2D | max $\lvert u \cdot n \rvert$ on wall | RMS error near wall | RMS error outside | smallest eigenvalue |
|---|---|---|---|---|
| plain | 0.40 | 0.38 | 0.155 | 6.8 · 10⁻⁹ |
| tier 1 | 8.4 · 10⁻⁶ | 0.22 | 0.110 | 2.9 · 10⁻⁹ |
| tier 2 | 7.5 · 10⁻¹⁵ | **0.10** | **0.086** | 6.8 · 10⁻⁹ |
| tier 2, $g^2$ (no-slip) | 6 · 10⁻³⁰ ($u = 0$) | 0.43 | 0.169 | 6.8 · 10⁻⁹ |

| 3D | max $\lvert u \cdot n \rvert$ on wall | RMS error near wall | RMS error outside | smallest eigenvalue |
|---|---|---|---|---|
| plain | 0.39 | 0.61 | 0.384 | 1.8 · 10⁻⁶ |
| tier 1 | 2.5 · 10⁻³ | 0.61 | 0.382 | 1.0 · 10⁻⁷ |
| tier 2 | 8.7 · 10⁻¹⁶ | 0.62 | 0.383 | 1.6 · 10⁻⁶ |

**What the numbers say:**

- **Exactness.** Tier 2 satisfies the wall condition to round-off everywhere on the wall. With
  $g^2$ the wall velocity vanishes. All methods are equally divergence-free.
- **2D accuracy.** Near the wall, tier 2 is clearly the most accurate in 2D. It depends on the
  band: RMS error 0.10 with $\ell = 0.5$ against 0.17 with $\ell = 0.25$. With no samples
  inside the band, tier 2 is the plain interpolant's potential multiplied by $g$ near the wall,
  so $\ell$ sets how far the wall's influence reaches.
- **No-slip** conflicts with this test flow, which slips along the wall, hence its larger
  errors. It does what it is asked to do.
- **3D accuracy.** All methods have about 0.6 RMS error near the wall. The single-level
  reconstruction of the near-uniform far flow dominates, so the 3D accuracy comparison is
  **inconclusive**.
- **Conditioning.** Tier 2 leaves the conditioning unchanged; tier 1 lowers it moderately.
  Before the closed-boundary fix, tier 1 lowered it by a factor of about 2000, which the
  divergence theorem explains (see the README).

**Not tested yet:**
- multilevel interpolation;
- the curl-free analogue (below);
- several walls, and general parametric boundaries;
- data near the wall;
- performance.

## Known issues

The implementation handles issues 1, 4 and 5 (see [Implementation](#implementation)); 2 and 3
remain open.

1. **Data on or near the wall.** At a wall point $x_0$, $n \cdot K(x_0, y) = 0$ for every $y$:
   the normal component is zero by construction. A sample exactly on the wall therefore has an
   all-zero row for its normal component, which makes the matrix singular, and samples very
   near the wall make it nearly singular. Such samples must be rejected, or their
   normal-component functional dropped. Dropping is consistent, since a no-penetration wall
   fixes that component to zero anyway.
2. **Curl-free fields get a different condition.** This is derived, not tested. The analogous
   kernel $\nabla_x \nabla_y^{T}[g(x) g(y) \psi]$ gives $u = \nabla(gF) = F\, \nabla g$ on
   the wall, which is *parallel* to the normal. So the tangential components vanish: the wall is
   an equipotential, like a conductor in electrostatics. That is not the zero normal derivative
   ($u \cdot n = 0$) that tier 1's `NormalComponent` imposes on curl-free fields.
3. **Choosing the band $\ell$.** It affects accuracy (see above) and has no automatic rule yet.
   Natural candidates: a multiple of the data spacing near the wall, capped by the boundary's
   reach.
4. **Potentials need re-deriving.** The stream function (2D) and the potential (curl-free)
   change with the weighting.
5. **Not translation-invariant.** $K(x, y)$ depends on $x$ and $y$ separately, not just on
   $x - y$. Today, assembly, evaluation, the cluster preconditioner and the potentials all
   assume the difference form.

## Implementation

1. **Kernels on point pairs (phase 6).** The Rust core evaluates kernels through an interface
   on point pairs $(x_i, x_j)$, so the weighted kernel is one more implementation of it,
   covering assembly, evaluation, the preconditioner blocks and the 2D stream function
   $\psi(x) = \sum_j c_j \cdot \nabla^{\perp}_y k(x, x_j)$. The kernel is scaled by
   $\delta^{2-d}$ so that it equals the package's kernel for $g = 1$.
2. **Analytic walls (phase 7).**
   - Wall functions for circles, spheres, lines, planes and infinite cylinders (also with the
     flow inside), with a `band` and a `condition`, combined by product for several walls.
   - Single-level and multilevel, with $g$ shared by every level.
   - Known issue 1 is handled by dropping functionals: a data point within $10^{-3}$ band of a
     wall keeps only the components orthogonal to the normals of the walls it touches (none
     for no slip, or at a corner of two walls in 2D). Data behind a wall is an error, and so is
     a tier-1 condition on a wall.
   - Tested: reduction to the plain kernel, symmetry, divergence-free columns, the wall
     function's gradient, $u \cdot n = 0$ (and $u = 0$ for no slip) on every shape to
     round-off, the stream function, Rust against numpy, CG against the direct solver, and
     the cylinder problem's near-wall accuracy.
3. **Later, if needed:**
   - general parametric boundaries, through closest-point distances within the reach and
     smooth combinations;
   - an automatic band;
   - the curl-free equipotential variant;
   - computing $g$ once per point instead of once per pair, which would halve the cost of
     evaluation with walls.
