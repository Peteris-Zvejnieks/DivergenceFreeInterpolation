//! Greedy farthest-point ordering, used to build the nested levels of multilevel interpolation.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::tree::KdTree;

/// Heap entry: the farthest point first, ties broken by the smaller index (as in the Python
/// reference implementation, which uses a min-heap of (-distance, index)).
struct Candidate {
    distance: f64,
    index: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| other.index.cmp(&self.index))
    }
}

fn distance(points: &[f64], dim: usize, i: usize, j: usize) -> f64 {
    let (p, q) = (
        &points[i * dim..(i + 1) * dim],
        &points[j * dim..(j + 1) * dim],
    );
    p.iter()
        .zip(q)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

/// Greedy farthest-point ordering starting from `start`: repeatedly picks the point farthest
/// from all points picked so far. Returns the picked indices and each one's distance to the
/// previously picked points when it was inserted (infinity for the first); the radii never
/// increase. Stops before the first point whose radius would be below `stop_radius`.
///
/// Only points within the current radius of a newly picked point can move closer to the picked
/// set, so with a kd-tree and a lazy heap the cost is about O(N log N).
pub fn farthest_point_ordering(
    points: &[f64],
    dim: usize,
    start: usize,
    stop_radius: f64,
) -> (Vec<usize>, Vec<f64>) {
    let n = points.len() / dim;
    let tree = KdTree::new(points, dim);
    let mut distances: Vec<f64> = (0..n).map(|i| distance(points, dim, i, start)).collect();
    distances[start] = 0.0;
    let mut picked = vec![false; n];
    picked[start] = true;
    let mut order = vec![start];
    let mut radii = vec![f64::INFINITY];

    let mut heap: BinaryHeap<Candidate> = distances
        .iter()
        .enumerate()
        .map(|(index, &distance)| Candidate { distance, index })
        .collect();
    while let Some(Candidate {
        distance: radius,
        index,
    }) = heap.pop()
    {
        // Skip points picked already and stale entries (superseded by a smaller distance)
        if picked[index] || radius != distances[index] {
            continue;
        }
        if radius < stop_radius {
            break;
        }
        picked[index] = true;
        distances[index] = 0.0;
        order.push(index);
        radii.push(radius);
        let x = &points[index * dim..(index + 1) * dim];
        tree.for_each_within(x, radius, |j, distance2| {
            let new = distance2.sqrt();
            if new < distances[j] {
                distances[j] = new;
                heap.push(Candidate {
                    distance: new,
                    index: j,
                });
            }
        });
    }
    (order, radii)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::Rng;

    fn naive(points: &[f64], dim: usize, start: usize) -> (Vec<usize>, Vec<f64>) {
        let n = points.len() / dim;
        let mut distances: Vec<f64> = (0..n).map(|i| distance(points, dim, i, start)).collect();
        distances[start] = -1.0;
        let mut order = vec![start];
        let mut radii = vec![f64::INFINITY];
        for _ in 1..n {
            let (best, &radius) = distances
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                .unwrap();
            order.push(best);
            radii.push(radius);
            for i in 0..n {
                if distances[i] >= 0.0 {
                    distances[i] = distances[i].min(distance(points, dim, i, best));
                }
            }
            distances[best] = -1.0;
        }
        (order, radii)
    }

    #[test]
    fn matches_the_naive_ordering() {
        let mut rng = Rng::new(31);
        for d in [2, 3] {
            let points = rng.points(500, d);
            let (order, radii) = farthest_point_ordering(&points, d, 17, 0.0);
            let (expected_order, expected_radii) = naive(&points, d, 17);
            assert_eq!(order, expected_order);
            for (r, e) in radii.iter().zip(&expected_radii).skip(1) {
                assert!((r - e).abs() < 1e-14);
            }
            assert!(radii.windows(2).all(|w| w[1] <= w[0]));
        }
    }

    #[test]
    fn stops_below_the_stop_radius() {
        let mut rng = Rng::new(32);
        let points = rng.points(400, 2);
        let (order, radii) = farthest_point_ordering(&points, 2, 0, 0.0);
        let stop = radii[100];
        let (short, short_radii) = farthest_point_ordering(&points, 2, 0, stop);
        let count = radii.iter().filter(|&&r| r >= stop).count();
        assert_eq!(short, order[..count].to_vec());
        assert_eq!(short_radii.len(), count);
    }
}
