//! A kd-tree for radius queries on points of any dimension.
//!
//! Built once over borrowed points (flat, row-major); queries visit every point within a radius
//! through a callback, so callers can accumulate without allocating per query. Each node stores
//! its bounding box, which prunes far subtrees tightly.

/// Points per leaf: small enough to prune well, large enough to keep the tree shallow.
const LEAF_SIZE: usize = 16;

struct Node {
    /// Range of `KdTree::order` covered by this node.
    start: usize,
    end: usize,
    /// Child node indices; `usize::MAX` for leaves.
    left: usize,
    right: usize,
}

pub struct KdTree<'a> {
    points: &'a [f64],
    dim: usize,
    /// Point indices, permuted so that every node covers a contiguous range.
    order: Vec<usize>,
    nodes: Vec<Node>,
    /// Bounding box of each node: `dim` lower bounds, then `dim` upper bounds.
    boxes: Vec<f64>,
}

impl<'a> KdTree<'a> {
    /// Builds the tree over `points.len() / dim` points.
    pub fn new(points: &'a [f64], dim: usize) -> Self {
        assert!(
            dim > 0 && points.len().is_multiple_of(dim),
            "points must have shape (n, dim)"
        );
        let n = points.len() / dim;
        let mut tree = Self {
            points,
            dim,
            order: (0..n).collect(),
            nodes: Vec::with_capacity(2 * n / LEAF_SIZE + 1),
            boxes: Vec::with_capacity((2 * n / LEAF_SIZE + 1) * 2 * dim),
        };
        if n > 0 {
            tree.build(0, n);
        }
        tree
    }

    fn coordinate(&self, point: usize, axis: usize) -> f64 {
        self.points[point * self.dim + axis]
    }

    /// Builds the subtree over `order[start..end]` and returns its node index.
    fn build(&mut self, start: usize, end: usize) -> usize {
        let d = self.dim;
        let mut lower = vec![f64::INFINITY; d];
        let mut upper = vec![f64::NEG_INFINITY; d];
        for &point in &self.order[start..end] {
            for axis in 0..d {
                let value = self.coordinate(point, axis);
                lower[axis] = lower[axis].min(value);
                upper[axis] = upper[axis].max(value);
            }
        }
        let index = self.nodes.len();
        self.nodes.push(Node {
            start,
            end,
            left: usize::MAX,
            right: usize::MAX,
        });
        self.boxes.extend_from_slice(&lower);
        self.boxes.extend_from_slice(&upper);
        if end - start <= LEAF_SIZE {
            return index;
        }

        // Split at the median of the widest axis
        let axis = (0..d)
            .max_by(|&p, &q| (upper[p] - lower[p]).total_cmp(&(upper[q] - lower[q])))
            .unwrap_or(0);
        let middle = start + (end - start) / 2;
        let points = self.points;
        self.order[start..end].select_nth_unstable_by(middle - start, |&p, &q| {
            points[p * d + axis].total_cmp(&points[q * d + axis])
        });
        let left = self.build(start, middle);
        let right = self.build(middle, end);
        self.nodes[index].left = left;
        self.nodes[index].right = right;
        index
    }

    /// Squared distance from `query` to the bounding box of `node` (0 inside it).
    fn box_distance2(&self, node: usize, query: &[f64]) -> f64 {
        let d = self.dim;
        let bounds = &self.boxes[node * 2 * d..(node + 1) * 2 * d];
        (0..d)
            .map(|axis| {
                let gap = (bounds[axis] - query[axis]).max(query[axis] - bounds[d + axis]);
                if gap > 0.0 { gap * gap } else { 0.0 }
            })
            .sum()
    }

    /// Calls `visit(index, squared_distance)` for every point within `radius` of `query`
    /// (distance <= radius), in no particular order.
    pub fn for_each_within(&self, query: &[f64], radius: f64, mut visit: impl FnMut(usize, f64)) {
        if self.nodes.is_empty() {
            return;
        }
        let radius2 = radius * radius;
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            if self.box_distance2(node, query) > radius2 {
                continue;
            }
            let Node {
                start,
                end,
                left,
                right,
            } = self.nodes[node];
            if left == usize::MAX {
                for &point in &self.order[start..end] {
                    let row = &self.points[point * self.dim..(point + 1) * self.dim];
                    let distance2: f64 =
                        row.iter().zip(query).map(|(p, q)| (p - q) * (p - q)).sum();
                    if distance2 <= radius2 {
                        visit(point, distance2);
                    }
                }
            } else {
                stack.push(left);
                stack.push(right);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    fn brute_force(points: &[f64], d: usize, query: &[f64], radius: f64) -> Vec<usize> {
        (0..points.len() / d)
            .filter(|&i| {
                let row = &points[i * d..(i + 1) * d];
                row.iter()
                    .zip(query)
                    .map(|(p, q)| (p - q) * (p - q))
                    .sum::<f64>()
                    <= radius * radius
            })
            .collect()
    }

    #[test]
    fn radius_queries_match_brute_force() {
        let mut rng = Rng::new(7);
        for d in [1, 2, 3, 5] {
            let points = rng.points(1000, d);
            let tree = KdTree::new(&points, d);
            for radius in [0.0, 0.05, 0.2, 0.7, 3.0] {
                for _ in 0..20 {
                    let query: Vec<f64> = (0..d).map(|_| rng.uniform(-0.2, 1.2)).collect();
                    let mut found = Vec::new();
                    tree.for_each_within(&query, radius, |i, _| found.push(i));
                    found.sort_unstable();
                    assert_eq!(found, brute_force(&points, d, &query, radius));
                }
            }
        }
    }

    #[test]
    fn handles_duplicates_and_degenerate_inputs() {
        let points = vec![0.5; 2 * 100];
        let tree = KdTree::new(&points, 2);
        let mut count = 0;
        tree.for_each_within(&[0.5, 0.5], 0.0, |_, distance2| {
            assert_eq!(distance2, 0.0);
            count += 1;
        });
        assert_eq!(count, 100);
        let empty = KdTree::new(&[], 3);
        empty.for_each_within(&[0.0, 0.0, 0.0], 1.0, |_, _| panic!("no points"));
    }
}
