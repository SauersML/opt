//! Second-order verdicts a bound-constrained outer search takes at a point.
//!
//! These are the definiteness verdict, the Newton decrement it travels with, and
//! the longest feasible step along a ray. A caller's criterion Hessian is often a
//! derivative of a quantity computed through an inner solve, so its error is not
//! bounded by `√ε·‖H‖`. What the caller can measure, from an identity that is
//! exactly zero in exact arithmetic, is passed as `measured_resolution`, and every
//! verdict here is taken at one owner of the shift that measurement implies,
//! [`certificate_curvature_shift`].

use faer::{Mat, Side};
use ndarray::{Array1, Array2};

use crate::project_to_box;

/// The shift a definiteness verdict on `hessian` is taken at: the larger of the
/// measured resolution and the arithmetic shift `√ε·max(max|H_ii|, 1)`.
///
/// `measured_resolution = 0.0` gives the arithmetic shift, which is what decides
/// wherever nothing could be measured. The two are combined by `max`, so a
/// measurement can only admit a point the arithmetic shift refused, never refuse
/// one it admitted. A verdict and the shift it was taken at travel together: a
/// second layer judging the same direction at a narrower shift applies a
/// strictly stronger test than the first.
#[must_use]
pub fn certificate_curvature_shift(hessian: &Array2<f64>, measured_resolution: f64) -> f64 {
    let n = hessian.nrows();
    let max_diag = (0..n).fold(0.0_f64, |acc, j| acc.max(hessian[[j, j]].abs()));
    let arithmetic_shift = f64::EPSILON.sqrt() * max_diag.max(1.0);
    if measured_resolution.is_finite() && measured_resolution > arithmetic_shift {
        measured_resolution
    } else {
        arithmetic_shift
    }
}

/// Whether `hessian + shift·I` factors, at [`certificate_curvature_shift`]: a
/// negative eigenvalue within the shift is PSD by this standard, and one below
/// it is not.
///
/// `None` when the matrix is empty, not square, or not finite.
#[must_use]
pub fn hessian_is_psd_at_resolution(
    hessian: &Array2<f64>,
    measured_resolution: f64,
) -> Option<bool> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || hessian.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let shift = certificate_curvature_shift(hessian, measured_resolution);
    let mut chol = hessian.clone();
    for j in 0..n {
        chol[[j, j]] += shift;
    }
    for j in 0..n {
        for k in 0..j {
            let l_jk = chol[[j, k]];
            for i in j..n {
                chol[[i, j]] -= chol[[i, k]] * l_jk;
            }
        }
        let pivot = chol[[j, j]];
        if !(pivot.is_finite() && pivot > 0.0) {
            return Some(false);
        }
        let inv_sqrt = 1.0 / pivot.sqrt();
        for i in j..n {
            chol[[i, j]] *= inv_sqrt;
        }
    }
    Some(true)
}

/// The Newton decrement `½·gᵀ(H + shift·I)⁻¹g` at the arithmetic shift.
///
/// `hessian` and `grad` are the analytic Hessian and the KKT-projected gradient
/// at the point. `None` when the shapes are malformed, an entry is not finite,
/// the shifted matrix does not factor, or the quadratic form is negative (which a
/// positive definite factor rules out; kept as a roundoff guard).
#[must_use]
pub fn newton_predicted_decrease(hessian: &Array2<f64>, grad: &Array1<f64>) -> Option<f64> {
    newton_predicted_decrease_at_resolution(hessian, grad, 0.0)
}

/// [`newton_predicted_decrease`] for a caller whose definiteness verdict was
/// taken at a measured resolution.
///
/// The decrement is taken at the arithmetic shift wherever that factors. Only
/// when it does not is it taken at
/// [`certificate_curvature_shift`]`(H, measured_resolution)`, the shift at which
/// [`hessian_is_psd_at_resolution`] judged the matrix, so a point that verdict
/// accepts always has a decrement. The arithmetic shift goes first because a
/// larger shift shrinks every positive direction's share `g_i²/(λ_i + shift)`
/// and could read real descent along a near-flat positive direction as none. A
/// zero resolution is [`newton_predicted_decrease`] bit for bit.
#[must_use]
pub fn newton_predicted_decrease_at_resolution(
    hessian: &Array2<f64>,
    grad: &Array1<f64>,
    measured_resolution: f64,
) -> Option<f64> {
    let n = hessian.nrows();
    if n == 0 || hessian.ncols() != n || grad.len() != n {
        return None;
    }
    if hessian.iter().any(|v| !v.is_finite()) || grad.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let arithmetic_shift = certificate_curvature_shift(hessian, 0.0);
    if let Some(decrease) = shifted_newton_predicted_decrease(hessian, grad, arithmetic_shift) {
        return Some(decrease);
    }
    let resolution_shift = certificate_curvature_shift(hessian, measured_resolution);
    if resolution_shift > arithmetic_shift {
        shifted_newton_predicted_decrease(hessian, grad, resolution_shift)
    } else {
        None
    }
}

/// `½·gᵀ(H + shift·I)⁻¹g` through a lower Cholesky factor, or `None` when the
/// shifted matrix is not positive definite.
fn shifted_newton_predicted_decrease(
    hessian: &Array2<f64>,
    grad: &Array1<f64>,
    shift: f64,
) -> Option<f64> {
    let n = hessian.nrows();
    let mut l = hessian.clone();
    for j in 0..n {
        l[[j, j]] += shift;
    }
    for j in 0..n {
        for k in 0..j {
            let l_jk = l[[j, k]];
            for i in j..n {
                l[[i, j]] -= l[[i, k]] * l_jk;
            }
        }
        let pivot = l[[j, j]];
        if !(pivot.is_finite() && pivot > 0.0) {
            return None;
        }
        let inv_sqrt = 1.0 / pivot.sqrt();
        for i in j..n {
            l[[i, j]] *= inv_sqrt;
        }
    }
    let mut y = grad.clone();
    for j in 0..n {
        let mut s = y[j];
        for k in 0..j {
            s -= l[[j, k]] * y[k];
        }
        y[j] = s / l[[j, j]];
    }
    let mut d = y;
    for j in (0..n).rev() {
        let mut s = d[j];
        for k in (j + 1)..n {
            s -= l[[k, j]] * d[k];
        }
        d[j] = s / l[[j, j]];
    }
    let quad = grad.dot(&d);
    if !quad.is_finite() || quad < 0.0 {
        return None;
    }
    Some(0.5 * quad)
}

/// The largest magnitude of a negative curvature that a criterion known only to
/// within `objective_resolution` cannot resolve over steps up to `alpha_max` along
/// its eigenvector: `2·objective_resolution / α_max²`.
///
/// A curvature `λ < 0` at or under it predicts at most `½|λ|·α_max² ≤
/// objective_resolution` of decrease at the largest step, which the criterion cannot
/// represent ([`negative_curvature_claim`]). It is the one number both a verdict on
/// a single claim and a definiteness test shifted to the criterion's resolution
/// read.
///
/// `None` when `alpha_max` is not a finite positive step or `objective_resolution`
/// is not a finite positive resolution.
#[must_use]
pub fn unresolvable_curvature_magnitude(alpha_max: f64, objective_resolution: f64) -> Option<f64> {
    if !(alpha_max.is_finite() && alpha_max > 0.0)
        || !(objective_resolution.is_finite() && objective_resolution > 0.0)
    {
        return None;
    }
    Some(2.0 * objective_resolution / (alpha_max * alpha_max))
}

/// What a criterion can say about a claim of negative curvature at a stationary
/// point ([`negative_curvature_claim`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NegativeCurvatureClaim {
    /// The claim predicts a decrease the criterion can represent. Steps from
    /// `alpha_max` down to `alpha_min`, where the prediction reaches the resolution,
    /// are its whole falsifiable range.
    Resolvable { alpha_min: f64 },
    /// The largest decrease the claim predicts, `predicted_at_largest = ½|λ_min|·α_max²`,
    /// is at or below the criterion's resolution. No allowed step can produce a
    /// decrease the criterion represents, so the criterion can neither confirm nor
    /// falsify the claim, and a curvature it cannot resolve cannot refuse the point.
    Unresolvable { predicted_at_largest: f64 },
}

/// Whether a negative eigenvalue `lambda_min` of a criterion Hessian at a
/// stationary point is resolvable by the criterion, over steps up to `alpha_max`
/// along its eigenvector.
///
/// Along the eigenvector the quadratic model of the claim is
/// `V(x ± αv) − V(x) ≈ ½λ_min α²`, so it predicts a decrease of at most
/// `½|λ_min|α_max²`. A criterion known only to within `objective_resolution` can
/// represent that decrease only when it is larger:
///
/// ```text
///     resolvable  ⟺  ½|λ_min|·α_max² > objective_resolution
///                 ⟺  |λ_min| > unresolvable_curvature_magnitude(α_max, objective_resolution),
///     α_min = √(2·objective_resolution / |λ_min|)  (< α_max when resolvable).
/// ```
///
/// This is a statement about the criterion, not the matrix: the matrix's own
/// error enters its verdict through [`certificate_curvature_shift`].
///
/// `None` when `lambda_min` is not a finite negative number, `alpha_max` is not a
/// finite positive step, or `objective_resolution` is not a finite positive
/// resolution.
#[must_use]
pub fn negative_curvature_claim(
    lambda_min: f64,
    alpha_max: f64,
    objective_resolution: f64,
) -> Option<NegativeCurvatureClaim> {
    if !(lambda_min.is_finite() && lambda_min < 0.0) {
        return None;
    }
    let unresolvable = unresolvable_curvature_magnitude(alpha_max, objective_resolution)?;
    if lambda_min.abs() <= unresolvable {
        return Some(NegativeCurvatureClaim::Unresolvable {
            predicted_at_largest: 0.5 * lambda_min.abs() * alpha_max * alpha_max,
        });
    }
    Some(NegativeCurvatureClaim::Resolvable {
        alpha_min: (2.0 * objective_resolution / lambda_min.abs()).sqrt(),
    })
}

/// The largest `α ≥ 0` keeping `point + α·ray` inside the box `[lower, upper]`.
///
/// A coordinate with no bound entry, or a zero ray component, sets no limit, so
/// an unbounded ray returns `f64::INFINITY`. A point already outside its box
/// along the ray returns `0.0`.
#[must_use]
pub fn max_feasible_step_along(
    point: &Array1<f64>,
    ray: &Array1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
) -> f64 {
    let mut alpha = f64::INFINITY;
    for i in 0..point.len() {
        let step = ray[i];
        let bound = if step > 0.0 {
            upper.get(i)
        } else if step < 0.0 {
            lower.get(i)
        } else {
            None
        };
        if let Some(&limit) = bound {
            alpha = alpha.min((limit - point[i]) / step);
        }
    }
    alpha.max(0.0)
}

/// A claim that a Hessian carries negative curvature at a point, stated on the
/// subspace the caller's certificate judged ([`adjudicate_negative_curvature`]).
#[derive(Clone, Copy, Debug)]
pub struct NegativeCurvatureQuery<'a> {
    /// The point, inside the box.
    pub point: &'a Array1<f64>,
    /// The objective's gradient there.
    pub gradient: &'a Array1<f64>,
    /// `n × m` orthonormal columns spanning the searched directions. Coordinates
    /// held fixed (railed at a bound, or a direction the objective is invariant
    /// along) are zero in every column.
    pub basis: &'a Array2<f64>,
    /// The Hessian compressed to the basis, `basisᵀ H basis` (`m × m`).
    pub reduced_hessian: &'a Array2<f64>,
    /// The largest `|H_kk|` over the searched coordinates: the scale of the
    /// assembly roundoff `√ε·max(1, scale)` a negative eigenvalue must clear.
    pub diagonal_scale: f64,
    pub lower: &'a Array1<f64>,
    pub upper: &'a Array1<f64>,
    /// The objective at `point`.
    pub baseline_cost: f64,
    /// The objective's resolution: a decrease at or below it is not resolved.
    pub objective_resolution: f64,
    /// The largest step the claim is tested at along the unit eigenvector.
    pub largest_step: f64,
}

/// Why [`adjudicate_negative_curvature`] established nothing about the point.
#[derive(Clone, Debug, PartialEq)]
pub enum NegativeCurvatureDecline {
    /// The reduced Hessian is not a square finite matrix matching the basis.
    Malformed,
    /// The subspace is empty: nothing is left to search.
    EmptySubspace,
    /// The reduced Hessian's eigendecomposition failed.
    Eigendecomposition(String),
    /// The most negative eigenvalue does not clear the roundoff margin.
    WithinRoundoff { lambda_min: f64, margin: f64 },
    /// The eigenvector's norm is not a finite positive number.
    UnusableDirection { lambda_min: f64, norm: f64 },
    /// No trial was evaluable: each was clamped back onto the point, failed, or
    /// returned a non-finite value, so nothing was measured.
    NoEvaluableTrial {
        lambda_min: f64,
        margin: f64,
        clamped_onto_point: usize,
        failed: usize,
        non_finite: usize,
        steps: usize,
    },
}

/// What the objective said about a claim of negative curvature
/// ([`adjudicate_negative_curvature`]).
#[derive(Clone, Debug, PartialEq)]
pub enum NegativeCurvatureVerdict {
    /// A feasible point along the eigenvector lowers the objective by more than its
    /// resolution: the point is a saddle, and this is the descent off it.
    Descended {
        point: Array1<f64>,
        cost: f64,
        lambda_min: f64,
        /// The ladder step that confirmed the descent.
        confirmed_step: f64,
        /// The step taken after doubling while the objective kept improving.
        step: f64,
        doublings: usize,
        /// Whether the step reached the box along the ray.
        on_box_face: bool,
    },
    /// Every evaluable step in the claim's whole falsifiable range, both signs,
    /// failed to lower the objective past its resolution.
    Contradicted {
        lambda_min: f64,
        probed: usize,
        smallest_step: f64,
        /// `½|λ_min|·α_min²`, the claim's prediction at the smallest step.
        predicted_at_smallest: f64,
        best_seen_cost: f64,
        /// The decrease floor the trials had to clear.
        decrease_floor: f64,
    },
    /// The claim's falsifiable range is empty: even the largest step predicts a
    /// decrease at or below the resolution, so no trial was evaluated.
    Unresolvable { lambda_min: f64, predicted_at_largest: f64 },
    Declined(NegativeCurvatureDecline),
}

/// Test a Hessian's claim of negative curvature at a first-order stationary point
/// against the objective itself.
///
/// Along the unit eigenvector `v` of the most negative eigenvalue `λ_min` of the
/// reduced Hessian, the quadratic model predicts `V(x ± αv) − V(x) ≈ ½λ_min α²`.
/// The decrease floor is the objective's resolution, but never below the
/// arithmetic's `16ε·max(|V|, 1)`: a decrease under it is not a decrease under any
/// reading. The claim predicts something the objective can represent only while
/// `½|λ_min|α² > floor`, so its falsifiable range runs from `largest_step` down to
/// `α_min = √(2·floor/|λ_min|)` ([`negative_curvature_claim`]); an empty range is
/// [`NegativeCurvatureVerdict::Unresolvable`] before any evaluation.
///
/// The ladder halves from `largest_step` to `α_min` (or to `ε`, where halving
/// stops moving the point), first with the sign that makes the linear term
/// non-positive, then the other, each trial projected onto the box. The first
/// trial that clears the floor confirms the saddle. A confirmed step is then
/// doubled while the objective keeps improving past the floor, up to the box
/// intersection along the ray: the model has no interior minimizer along negative
/// curvature, so the step length has to come from the objective and the box. The
/// incumbent is re-evaluated before the doubling so every comparison in it comes
/// from one state of an objective whose evaluation carries state. The doubling
/// ends at the box face, at the first trial that does not improve, or where the
/// step overflows, so it needs no count.
///
/// The last evaluation is at a trial point; a caller whose objective carries state
/// restores it.
pub fn adjudicate_negative_curvature<E>(
    query: &NegativeCurvatureQuery<'_>,
    mut cost: impl FnMut(&Array1<f64>) -> Result<f64, E>,
) -> NegativeCurvatureVerdict {
    use NegativeCurvatureVerdict as Verdict;
    let point = query.point;
    let n = point.len();
    let m = query.basis.ncols();
    let reduced = query.reduced_hessian;
    if query.basis.nrows() != n
        || reduced.nrows() != m
        || reduced.ncols() != m
        || query.gradient.len() != n
        || reduced.iter().any(|value| !value.is_finite())
    {
        return Verdict::Declined(NegativeCurvatureDecline::Malformed);
    }
    if m == 0 {
        return Verdict::Declined(NegativeCurvatureDecline::EmptySubspace);
    }
    if !(query.largest_step.is_finite() && query.largest_step > 0.0) {
        return Verdict::Declined(NegativeCurvatureDecline::Malformed);
    }
    let symmetric = Mat::<f64>::from_fn(m, m, |row, col| {
        0.5 * (reduced[[row, col]] + reduced[[col, row]])
    });
    let eigen = match symmetric.self_adjoint_eigen(Side::Lower) {
        Ok(eigen) => eigen,
        Err(error) => {
            return Verdict::Declined(NegativeCurvatureDecline::Eigendecomposition(format!(
                "{error:?}"
            )));
        }
    };
    let values = eigen.S().column_vector().as_mat();
    let vectors = eigen.U();
    let mut min_index = 0usize;
    for index in 1..m {
        if values[(index, 0)] < values[(min_index, 0)] {
            min_index = index;
        }
    }
    let lambda_min = values[(min_index, 0)];
    let margin = f64::EPSILON.sqrt() * query.diagonal_scale.abs().max(1.0);
    if !(lambda_min < -margin) {
        return Verdict::Declined(NegativeCurvatureDecline::WithinRoundoff { lambda_min, margin });
    }
    let sub_direction = Array1::from_shape_fn(m, |row| vectors[(row, min_index)]);
    let norm = sub_direction.dot(&sub_direction).sqrt();
    if !(norm.is_finite() && norm > 0.0) {
        return Verdict::Declined(NegativeCurvatureDecline::UnusableDirection { lambda_min, norm });
    }
    let direction = query.basis.dot(&sub_direction.mapv(|value| value / norm));
    let primary_sign = if query.gradient.dot(&direction) > 0.0 { -1.0 } else { 1.0 };

    let roundoff_floor = 16.0 * f64::EPSILON * query.baseline_cost.abs().max(1.0);
    let floor = if query.objective_resolution.is_finite() && query.objective_resolution > 0.0 {
        query.objective_resolution.max(roundoff_floor)
    } else {
        roundoff_floor
    };
    let largest = query.largest_step;
    let alpha_min = match negative_curvature_claim(lambda_min, largest, floor) {
        Some(NegativeCurvatureClaim::Resolvable { alpha_min }) => alpha_min,
        Some(NegativeCurvatureClaim::Unresolvable {
            predicted_at_largest,
        }) => {
            return Verdict::Unresolvable {
                lambda_min,
                predicted_at_largest,
            };
        }
        None => return Verdict::Declined(NegativeCurvatureDecline::Malformed),
    };
    let mut steps = Vec::new();
    let mut alpha = largest;
    loop {
        steps.push(alpha);
        if alpha <= alpha_min || alpha <= f64::EPSILON {
            break;
        }
        alpha *= 0.5;
    }
    let point_at = |ray: &Array1<f64>, alpha: f64| {
        project_to_box(&(point + &ray.mapv(|value| alpha * value)), query.lower, query.upper)
    };
    let same_point = |trial: &Array1<f64>| {
        trial
            .iter()
            .zip(point.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    };
    let (mut probed, mut clamped, mut failed, mut non_finite) = (0usize, 0usize, 0usize, 0usize);
    let mut best_seen = f64::INFINITY;
    let mut confirmed: Option<(Array1<f64>, f64, f64)> = None;
    'signs: for sign in [primary_sign, -primary_sign] {
        let ray = direction.mapv(|value| sign * value);
        for &alpha in &steps {
            let trial = point_at(&ray, alpha);
            if same_point(&trial) {
                clamped += 1;
                continue;
            }
            match cost(&trial) {
                Ok(value) if value.is_finite() => {
                    probed += 1;
                    best_seen = best_seen.min(value);
                    if value < query.baseline_cost - floor {
                        confirmed = Some((ray, alpha, value));
                        break 'signs;
                    }
                }
                Ok(_) => non_finite += 1,
                Err(_) => failed += 1,
            }
        }
    }
    let smallest_step = *steps.last().unwrap_or(&largest);
    let Some((ray, confirmed_step, confirmed_cost)) = confirmed else {
        if probed == 0 {
            return Verdict::Declined(NegativeCurvatureDecline::NoEvaluableTrial {
                lambda_min,
                margin,
                clamped_onto_point: clamped,
                failed,
                non_finite,
                steps: steps.len(),
            });
        }
        return Verdict::Contradicted {
            lambda_min,
            probed,
            smallest_step,
            predicted_at_smallest: 0.5 * lambda_min.abs() * smallest_step * smallest_step,
            best_seen_cost: best_seen,
            decrease_floor: floor,
        };
    };

    let alpha_box = max_feasible_step_along(point, &ray, query.lower, query.upper);
    let mut best_point = point_at(&ray, confirmed_step);
    let mut best_cost = confirmed_cost;
    let mut step = confirmed_step;
    let mut doublings = 0usize;
    if confirmed_step < alpha_box {
        if let Ok(value) = cost(&best_point)
            && value.is_finite()
        {
            best_cost = value;
        }
        loop {
            let next = (2.0 * step).min(alpha_box);
            if !(next.is_finite() && next > step) {
                break;
            }
            let trial = point_at(&ray, next);
            if same_point(&trial) {
                break;
            }
            match cost(&trial) {
                Ok(value) if value.is_finite() && value < best_cost - floor => {
                    best_point = trial;
                    best_cost = value;
                    step = next;
                    doublings += 1;
                }
                _ => break,
            }
        }
    }
    Verdict::Descended {
        point: best_point,
        cost: best_cost,
        lambda_min,
        confirmed_step,
        step,
        doublings,
        on_box_face: alpha_box.is_finite() && step >= alpha_box,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        NegativeCurvatureClaim, certificate_curvature_shift, hessian_is_psd_at_resolution,
        max_feasible_step_along, negative_curvature_claim, newton_predicted_decrease,
        newton_predicted_decrease_at_resolution, unresolvable_curvature_magnitude,
    };
    use ndarray::{Array1, array};

    use super::{
        NegativeCurvatureQuery, NegativeCurvatureVerdict, adjudicate_negative_curvature,
    };

    fn query<'a>(
        point: &'a Array1<f64>,
        gradient: &'a Array1<f64>,
        basis: &'a ndarray::Array2<f64>,
        reduced: &'a ndarray::Array2<f64>,
        lower: &'a Array1<f64>,
        upper: &'a Array1<f64>,
        baseline_cost: f64,
    ) -> NegativeCurvatureQuery<'a> {
        NegativeCurvatureQuery {
            point,
            gradient,
            basis,
            reduced_hessian: reduced,
            diagonal_scale: 2.0,
            lower,
            upper,
            baseline_cost,
            objective_resolution: 1e-10,
            largest_step: 1.0,
        }
    }

    /// `f = x² − y²` at the origin: the saddle descends along `y`, and the doubling
    /// runs to the box face at `|y| = 8`.
    #[test]
    fn a_saddle_descends_to_the_box_face_2900() {
        let point = array![0.0, 0.0];
        let gradient = array![0.0, 0.0];
        let basis = ndarray::Array2::<f64>::eye(2);
        let reduced = array![[2.0, 0.0], [0.0, -2.0]];
        let (lower, upper) = (array![-8.0, -8.0], array![8.0, 8.0]);
        let f = |x: &Array1<f64>| -> Result<f64, ()> { Ok(x[0] * x[0] - x[1] * x[1]) };
        let verdict = adjudicate_negative_curvature(
            &query(&point, &gradient, &basis, &reduced, &lower, &upper, 0.0),
            f,
        );
        let NegativeCurvatureVerdict::Descended { point, cost, step, on_box_face, confirmed_step, .. } =
            verdict
        else {
            panic!("the saddle must descend, got {verdict:?}");
        };
        assert_eq!(confirmed_step, 1.0);
        assert_eq!(step, 8.0);
        assert!(on_box_face);
        assert_eq!(point[0], 0.0);
        assert_eq!(point[1].abs(), 8.0);
        assert_eq!(cost, -64.0);
    }

    /// A matrix that reports negative curvature where the objective is convex is
    /// contradicted over the whole falsifiable range, down to
    /// `α_min = √(2·floor/|λ_min|)`.
    #[test]
    fn a_curvature_the_objective_does_not_have_is_contradicted_2900() {
        let point = array![0.0];
        let gradient = array![0.0];
        let basis = ndarray::Array2::<f64>::eye(1);
        let reduced = array![[-2.0]];
        let (lower, upper) = (array![-10.0], array![10.0]);
        let verdict = adjudicate_negative_curvature(
            &query(&point, &gradient, &basis, &reduced, &lower, &upper, 0.0),
            |x: &Array1<f64>| -> Result<f64, ()> { Ok(x[0] * x[0]) },
        );
        let NegativeCurvatureVerdict::Contradicted { smallest_step, probed, .. } = verdict else {
            panic!("a convex objective must contradict the claim, got {verdict:?}");
        };
        let alpha_min = (2.0 * 1e-10 / 2.0_f64).sqrt();
        assert!(smallest_step <= alpha_min && smallest_step > 0.5 * alpha_min);
        assert_eq!(probed, 2 * (1.0 / smallest_step).log2().round() as usize + 2);
    }

    /// A claim whose largest step predicts no more than the resolution evaluates
    /// nothing.
    #[test]
    fn an_unresolvable_claim_evaluates_nothing_2900() {
        let point = array![0.0];
        let gradient = array![0.0];
        let basis = ndarray::Array2::<f64>::eye(1);
        let reduced = array![[-1e-3]];
        let (lower, upper) = (array![-1.0], array![1.0]);
        let mut q = query(&point, &gradient, &basis, &reduced, &lower, &upper, 0.0);
        q.objective_resolution = 1e-3;
        q.diagonal_scale = 0.0;
        let mut calls = 0;
        let verdict = adjudicate_negative_curvature(&q, |_: &Array1<f64>| -> Result<f64, ()> {
            calls += 1;
            Ok(0.0)
        });
        assert!(matches!(verdict, NegativeCurvatureVerdict::Unresolvable { .. }), "{verdict:?}");
        assert_eq!(calls, 0);
    }

    /// Held coordinates are zero in the basis, so the descent never moves them.
    #[test]
    fn the_descent_stays_in_the_judged_subspace_2900() {
        let point = array![0.5, 0.0];
        let gradient = array![0.0, 0.0];
        let basis = array![[0.0], [1.0]];
        let reduced = array![[-2.0]];
        let (lower, upper) = (array![0.5, -1.0], array![0.5, 1.0]);
        let verdict = adjudicate_negative_curvature(
            &query(&point, &gradient, &basis, &reduced, &lower, &upper, 0.25),
            |x: &Array1<f64>| -> Result<f64, ()> { Ok(x[0] * x[0] - x[1] * x[1]) },
        );
        let NegativeCurvatureVerdict::Descended { point, .. } = verdict else {
            panic!("got {verdict:?}");
        };
        assert_eq!(point[0], 0.5);
        assert_eq!(point[1].abs(), 1.0);
    }

    #[test]
    fn the_shift_is_the_larger_of_the_measurement_and_the_arithmetic_band() {
        let hessian = array![[4.0, 0.0], [0.0, 1.0]];
        let arithmetic = f64::EPSILON.sqrt() * 4.0;
        assert_eq!(certificate_curvature_shift(&hessian, 0.0), arithmetic);
        assert_eq!(certificate_curvature_shift(&hessian, 1e-3), 1e-3);
        assert_eq!(certificate_curvature_shift(&hessian, f64::NAN), arithmetic);
        let small = array![[1e-3, 0.0], [0.0, 1e-4]];
        assert_eq!(
            certificate_curvature_shift(&small, 0.0),
            f64::EPSILON.sqrt()
        );
    }

    #[test]
    fn a_negative_eigenvalue_is_psd_only_inside_the_shift() {
        let within = array![[1.0, 0.0], [0.0, -1e-9]];
        assert_eq!(hessian_is_psd_at_resolution(&within, 0.0), Some(true));
        let saddle = array![[1.0, 0.0], [0.0, -1e-3]];
        assert_eq!(hessian_is_psd_at_resolution(&saddle, 0.0), Some(false));
        // Positive control: the same saddle is PSD at a measured resolution wider
        // than its negative eigenvalue.
        assert_eq!(hessian_is_psd_at_resolution(&saddle, 1e-2), Some(true));
        assert_eq!(hessian_is_psd_at_resolution(&array![[f64::NAN]], 0.0), None);
        assert_eq!(
            hessian_is_psd_at_resolution(&ndarray::Array2::<f64>::zeros((0, 0)), 0.0),
            None
        );
    }

    #[test]
    fn the_decrement_is_half_the_inverse_quadratic_form_and_travels_with_its_verdict() {
        let hessian = array![[2.0, 0.0], [0.0, 4.0]];
        let grad = array![2.0, 4.0];
        let decrease = newton_predicted_decrease(&hessian, &grad).expect("positive definite");
        let shift = f64::EPSILON.sqrt() * 4.0;
        let exact = 0.5 * (4.0 / (2.0 + shift) + 16.0 / (4.0 + shift));
        assert!((decrease - exact).abs() <= 1e-12 * exact);

        let indefinite = array![[1.0, 0.0], [0.0, -1.0]];
        let ones = array![1.0, 1.0];
        assert_eq!(newton_predicted_decrease(&indefinite, &ones), None);
        // Positive control: at a resolution the verdict accepts, the decrement exists
        // and is taken at that shift: diag(3, 1) against g = (1, 1).
        let at_resolution = newton_predicted_decrease_at_resolution(&indefinite, &ones, 2.0)
            .expect("factors at the resolution shift");
        assert!((at_resolution - 0.5 * (1.0 / 3.0 + 1.0)).abs() <= 1e-12);
        assert_eq!(
            newton_predicted_decrease_at_resolution(&indefinite, &ones, 0.0),
            newton_predicted_decrease(&indefinite, &ones)
        );
        assert_eq!(newton_predicted_decrease(&hessian, &array![1.0]), None);
    }

    #[test]
    fn a_negative_curvature_is_resolvable_only_above_the_criterions_resolution() {
        // gam#3036's Gaussian location-scale fit: ½|λ| = 6.47e-7 at α = 1 against a
        // resolution of 1.23e-5. No step up to one e-fold can resolve it.
        let lambda = -1.294787e-6;
        assert_eq!(
            negative_curvature_claim(lambda, 1.0, 1.228631e-5),
            Some(NegativeCurvatureClaim::Unresolvable {
                predicted_at_largest: 0.5 * lambda.abs()
            })
        );
        // Positive control: the same curvature over a longer ladder predicts
        // ½|λ|·16 = 1.04e-5 at α = 4, still under the resolution, and at α = 5 it
        // predicts 1.62e-5, which the criterion resolves down to α_min = 4.36.
        assert!(matches!(
            negative_curvature_claim(lambda, 4.0, 1.228631e-5),
            Some(NegativeCurvatureClaim::Unresolvable { .. })
        ));
        let Some(NegativeCurvatureClaim::Resolvable { alpha_min }) =
            negative_curvature_claim(lambda, 5.0, 1.228631e-5)
        else {
            panic!("½|λ|·25 exceeds the resolution");
        };
        assert!((alpha_min - (2.0 * 1.228631e-5 / lambda.abs()).sqrt()).abs() <= 1e-15 * alpha_min);
        assert!(alpha_min < 5.0);
        // A prediction exactly at the resolution is not resolvable.
        assert_eq!(
            negative_curvature_claim(-2.0, 1.0, 1.0),
            Some(NegativeCurvatureClaim::Unresolvable {
                predicted_at_largest: 1.0
            })
        );
        assert_eq!(negative_curvature_claim(1e-3, 1.0, 1e-5), None);
        assert_eq!(negative_curvature_claim(-1e-3, 0.0, 1e-5), None);
        assert_eq!(negative_curvature_claim(-1e-3, 1.0, 0.0), None);
        assert_eq!(negative_curvature_claim(f64::NAN, 1.0, 1e-5), None);
        // The magnitude both a single claim and a definiteness test shifted to the
        // criterion's resolution read: 2·res/α², and the claim is unresolvable exactly
        // at or under it.
        assert_eq!(
            unresolvable_curvature_magnitude(1.0, 1.228631e-5),
            Some(2.0 * 1.228631e-5)
        );
        assert_eq!(unresolvable_curvature_magnitude(2.0, 1.0), Some(0.5));
        assert_eq!(unresolvable_curvature_magnitude(0.0, 1.0), None);
        assert_eq!(unresolvable_curvature_magnitude(1.0, f64::NAN), None);
        assert!(matches!(
            negative_curvature_claim(-0.5, 2.0, 1.0),
            Some(NegativeCurvatureClaim::Unresolvable { .. })
        ));
        assert!(matches!(
            negative_curvature_claim(-0.5 - 1e-12, 2.0, 1.0),
            Some(NegativeCurvatureClaim::Resolvable { .. })
        ));
    }

    #[test]
    fn the_feasible_step_stops_at_the_first_bound_the_ray_meets() {
        let point = array![0.0, 0.0];
        let lower = array![-1.0, -1.0];
        let upper = array![1.0, 1.0];
        assert_eq!(
            max_feasible_step_along(&point, &array![1.0, -2.0], &lower, &upper),
            0.5
        );
        assert_eq!(
            max_feasible_step_along(&point, &array![0.0, 0.0], &lower, &upper),
            f64::INFINITY
        );
        // Positive control: a point already past its upper bound along the ray gets
        // no step, where the same ray from the centre gets one.
        let outside = array![2.0, 0.0];
        assert_eq!(
            max_feasible_step_along(&outside, &array![1.0, 0.0], &lower, &upper),
            0.0
        );
        assert_eq!(
            max_feasible_step_along(&point, &array![1.0, 0.0], &lower, &upper),
            1.0
        );
        let empty = Array1::<f64>::zeros(0);
        assert_eq!(
            max_feasible_step_along(&point, &array![1.0, 1.0], &empty, &empty),
            f64::INFINITY
        );
    }
}
