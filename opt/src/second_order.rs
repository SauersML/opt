//! Second-order verdicts a bound-constrained outer search takes at a point.
//!
//! These are the definiteness verdict, the Newton decrement it travels with, and
//! the longest feasible step along a ray. A caller's criterion Hessian is often a
//! derivative of a quantity computed through an inner solve, so its error is not
//! bounded by `√ε·‖H‖`. What the caller can measure, from an identity that is
//! exactly zero in exact arithmetic, is passed as `measured_resolution`, and every
//! verdict here is taken at one owner of the shift that measurement implies,
//! [`certificate_curvature_shift`].

use ndarray::{Array1, Array2};

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

#[cfg(test)]
mod tests {
    use super::{
        NegativeCurvatureClaim, certificate_curvature_shift, hessian_is_psd_at_resolution,
        max_feasible_step_along, negative_curvature_claim, newton_predicted_decrease,
        newton_predicted_decrease_at_resolution, unresolvable_curvature_magnitude,
    };
    use ndarray::{Array1, array};

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
