//! Caller-fed stall monitor for an outer optimization loop.
//!
//! [`StallMonitor`] owns the part of a stalled-search decision that belongs to
//! the optimizer and to no particular objective: the monotone incumbent, the
//! stall rule that judges each step on its own resolution, the escape granted
//! when a stall's incumbent still carries resolvable descent, the bit-identity
//! replay cut that ends escapes, the progress licence that ends a stall buying
//! nothing, and the evidence a stall reports.
//!
//! # The stall rule
//!
//! Two computed values imply `V_b > V_j` exactly when `V̂_b − V̂_j` exceeds the sum
//! of their resolutions ([`resolvably_below`]). So a step buys resolved progress
//! iff its decrease exceeds `R_b + R_j`. A step that buys none is **stalled** when
//! its model's predicted decrease is not resolvable either; otherwise the model
//! promised a decrease the function did not deliver, which the solver acts on
//! (its ratio test or its curvature update), and the step is **adapting** and
//! reaches no verdict. One stalled step reaches the verdict: its model decrease
//! bounds the solver's Cauchy decrease, so either the gradient is at resolution
//! relative to the curvature, or rejected trials drove the step length down at
//! this point, and the next step comes from the same model at the same scale. No
//! window of stalled steps is counted.
//!
//! The caller decides what it observes and when. A consumer that drives a solver
//! through an objective bridge observes at the bridge (every evaluation, every
//! refused trial, every accepted step it reconciles), so the sample stream is
//! the consumer's, not the solver's. What the monitor cannot know is supplied:
//!
//! * the stationarity band at a criterion value, as a closure;
//! * each value's resolution, and the predicted decrease of the step behind it;
//! * whether a sample is trustworthy (for example, whether an inner solve behind
//!   it converged);
//! * the incumbent's curvature verdict, and an opaque payload that travels with
//!   the incumbent.
//!
//! Every stop the monitor decides is published as a [`StallExit`] that the
//! caller reads through [`StallMonitor::exit`]; [`StallMonitor::exit_revision`]
//! counts publishes, so a caller that mirrors the exit elsewhere copies it only
//! when the monitor published a new one.

use ndarray::Array1;
use std::collections::VecDeque;

/// Whether `value`, computed to within `resolution`, is resolvably below
/// `reference`, computed to within `reference_resolution`.
///
/// Two computed values imply the exact ones are ordered exactly when they are
/// further apart than the sum of their resolutions: that is sufficient, since
/// `V_r − V_x ≥ V̂_r − V̂_x − R_r − R_x`, and necessary, since otherwise
/// `V_r = V̂_r − R_r` and `V_x = V̂_x + R_x` fit the data with `V_r ≤ V_x`. A value
/// that is not a number is never below anything.
#[must_use]
pub fn resolvably_below(
    reference: f64,
    reference_resolution: f64,
    value: f64,
    resolution: f64,
) -> bool {
    reference - value > reference_resolution + resolution
}

/// One evaluated sample offered to a [`StallMonitor`].
#[derive(Debug, Clone, Copy)]
pub struct StallSample<'a> {
    pub point: &'a Array1<f64>,
    pub value: f64,
    /// The resolution `value` was computed to.
    pub resolution: f64,
    /// Projected gradient norm at `point`.
    pub grad_norm: f64,
    /// Whether the value and gradient are trustworthy (for example, whether the
    /// inner solve behind them converged). An untrusted sample never becomes
    /// the incumbent and reaches no verdict.
    pub trusted: bool,
    /// The sample's curvature verdict: `Some(false)` is a strict saddle.
    pub curvature_psd: Option<bool>,
}

/// What folding one observation into a [`StallMonitor`] decided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StallVerdict {
    /// The search made resolved progress or is adapting, or an escape was
    /// granted at a strict saddle.
    Continue,
    /// A stall whose incumbent is inside the band at its value.
    Converged,
    /// A stall whose incumbent is above the band, with no escape left: the escape
    /// would replay the previous one from a bit-identical incumbent, or the run of
    /// refusals held only unescapable ones.
    Floor {
        /// Projected gradient norm at the incumbent.
        residual_grad_norm: f64,
    },
    /// A stall whose incumbent is above the band: the search continues so it can
    /// take the descent the residual says is available.
    Escape {
        /// Projected gradient norm at the incumbent.
        residual_grad_norm: f64,
        /// The band at the incumbent's value that the residual exceeded.
        band: f64,
    },
}

/// A run of refusals made only of unescapable ones, reported with a stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnescapableRefusalWindow {
    /// Unescapable refusals in the run.
    pub refused_trials: usize,
    /// The band at the incumbent's value.
    pub band: f64,
}

/// The incumbent a stall publishes.
#[derive(Debug, Clone, PartialEq)]
pub struct StallExit {
    pub point: Array1<f64>,
    pub value: f64,
    /// Projected gradient norm at `point`.
    pub grad_norm: f64,
    /// Trusted samples folded in when the exit was written.
    pub accepted: usize,
    /// `true` only when a stall's incumbent met the band. A running best-so-far
    /// snapshot is never a convergence claim.
    pub converged: bool,
    /// `(noise floor σ̂, probe radius Δ)` over the samples since the last
    /// resolved progress; see [`StallMonitor::window_probe_scale`]. Evidence only.
    pub probe_scale: Option<(f64, f64)>,
    /// Set when the stop was a run of unescapable refusals.
    pub unescapable_window: Option<UnescapableRefusalWindow>,
    /// Refused trials proposed from this incumbent whose model promised at most
    /// its resolution; see [`StallMonitor::wall_refusals`].
    pub wall_refusals: usize,
}

/// The search state an escape was granted from, in raw bits: the whole
/// incumbent, and the trial points observed since the escape before it. Only
/// bit identity supports the claim a replay cut makes: a deterministic
/// procedure replayed from an identical state returns an identical result. Raw
/// bits keep the comparison total, and both of its possible errors (distinct
/// NaN payloads, `-0.0` against `0.0`) grant the escape rather than cut it.
///
/// The incumbent alone is not the search's state. Trials that do not improve it
/// leave it bit-identical while the solver's own state moves: a rejected cubic
/// trial raises ARC's regularization, so the next steps go to new points. The
/// trials observed are the part of that state the monitor can see, so a replay
/// is a run that observed the same points, in the same order, from the same
/// incumbent.
#[derive(Debug, Clone, PartialEq, Eq)]
struct IncumbentBits {
    point: Vec<u64>,
    value: u64,
    grad_norm: u64,
    window_trials: Vec<Vec<u64>>,
}

impl IncumbentBits {
    fn new(point: &Array1<f64>, value: f64, grad_norm: f64, window_trials: &[Vec<u64>]) -> Self {
        Self {
            point: point_bits(point),
            value: value.to_bits(),
            grad_norm: grad_norm.to_bits(),
            window_trials: window_trials.to_vec(),
        }
    }
}

fn point_bits(point: &Array1<f64>) -> Vec<u64> {
    point
        .iter()
        .map(|coordinate| coordinate.to_bits())
        .collect()
}

/// Monotone incumbent, per-step stall rule, escapes, replay cut and progress
/// licence for an outer search. See the module docs.
pub struct StallMonitor<C> {
    band: Box<dyn Fn(f64) -> f64 + Send>,
    /// Set once a replay cut proved that continuing replays the previous
    /// escape. No licence continues past a proven replay.
    replay_proven: bool,
    best_value: f64,
    /// The resolution `best_value` was computed to.
    best_resolution: f64,
    best_point: Option<Array1<f64>>,
    best_grad_norm: f64,
    /// The incumbent's curvature verdict: `Some(false)` is a strict saddle.
    best_curvature_psd: Option<bool>,
    best_payload: Option<C>,
    staged_payload: Option<C>,
    strict_saddle_refusal: bool,
    /// Consecutive refused trials since the last finite observation. Evidence a
    /// stop reports, not a rule.
    refused_streak: usize,
    /// How many of the current refused streak were unescapable refusals.
    unescapable_streak: usize,
    /// Refused trials proposed since the incumbent was adopted whose step's model
    /// promised at most the incumbent's resolution. Each proves the objective's
    /// domain ends, along that trial's direction, within a step whose predicted
    /// decrease no evaluation could resolve, so the incumbent is pinned at a
    /// domain wall. Only a new incumbent clears it.
    wall_refusals: usize,
    accepted: usize,
    /// Consecutive escapes since the last resolved progress. A diagnostic:
    /// escapes are bounded by the replay cut and the progress licence.
    pub stuck_escapes: usize,
    incumbent_at_last_escape: Option<IncumbentBits>,
    /// Every trial point observed without resolved progress since the latest
    /// progress or escape grant: the half of the replay identity the incumbent
    /// cannot carry (see [`IncumbentBits`]).
    window_trials: Vec<Vec<u64>>,
    /// `(value, projected gradient norm, resolution)` of the incumbent when the
    /// previous stall was licensed to continue.
    continuation_incumbent: Option<(f64, f64, f64)>,
    /// The trusted samples since the last resolved progress, newest last.
    recent: VecDeque<(Array1<f64>, f64)>,
    exit: Option<StallExit>,
    exit_revision: u64,
}

impl<C: Clone> StallMonitor<C> {
    /// A monitor judging stationarity by `band(value)`.
    pub fn new(band: impl Fn(f64) -> f64 + Send + 'static) -> Self {
        Self {
            band: Box::new(band),
            replay_proven: false,
            best_value: f64::INFINITY,
            best_resolution: 0.0,
            best_point: None,
            best_grad_norm: f64::INFINITY,
            best_curvature_psd: None,
            best_payload: None,
            staged_payload: None,
            strict_saddle_refusal: false,
            refused_streak: 0,
            unescapable_streak: 0,
            wall_refusals: 0,
            accepted: 0,
            stuck_escapes: 0,
            incumbent_at_last_escape: None,
            window_trials: Vec::new(),
            continuation_incumbent: None,
            recent: VecDeque::new(),
            exit: None,
            exit_revision: 0,
        }
    }

    /// The stationarity band at criterion value `value`.
    pub fn band(&self, value: f64) -> f64 {
        (self.band)(value)
    }

    pub fn best_value(&self) -> f64 {
        self.best_value
    }

    /// The resolution [`Self::best_value`] was computed to.
    pub fn best_resolution(&self) -> f64 {
        self.best_resolution
    }

    pub fn best_point(&self) -> Option<&Array1<f64>> {
        self.best_point.as_ref()
    }

    pub fn best_grad_norm(&self) -> f64 {
        self.best_grad_norm
    }

    pub fn best_curvature_psd(&self) -> Option<bool> {
        self.best_curvature_psd
    }

    pub fn best_payload(&self) -> Option<&C> {
        self.best_payload.as_ref()
    }

    pub fn accepted(&self) -> usize {
        self.accepted
    }

    pub fn refused_streak(&self) -> usize {
        self.refused_streak
    }

    pub fn unescapable_streak(&self) -> usize {
        self.unescapable_streak
    }

    /// Refused trials at a domain wall since the incumbent was adopted.
    pub fn wall_refusals(&self) -> usize {
        self.wall_refusals
    }

    pub fn replay_proven(&self) -> bool {
        self.replay_proven
    }

    /// The published exit, if any.
    pub fn exit(&self) -> Option<&StallExit> {
        self.exit.as_ref()
    }

    /// Incremented on every publish of [`Self::exit`] and every revocation of its
    /// convergence; not by the wall-refusal count stamped onto it.
    pub fn exit_revision(&self) -> u64 {
        self.exit_revision
    }

    /// Whether an escape was just granted at a strict-saddle incumbent; reading
    /// it clears it.
    pub fn take_strict_saddle_refusal(&mut self) -> bool {
        std::mem::take(&mut self.strict_saddle_refusal)
    }

    /// Stage the payload of the sample about to be observed. It travels with
    /// the incumbent only if that sample becomes it.
    pub fn stage_payload(&mut self, payload: Option<C>) {
        self.staged_payload = payload;
    }

    fn publish(&mut self, exit: StallExit) {
        self.exit = Some(exit);
        self.exit_revision = self.exit_revision.wrapping_add(1);
    }

    /// The iterate a stall is judged at: the incumbent, falling back to the
    /// current point for each field that is unset.
    fn best_iterate_or(
        &self,
        point: &Array1<f64>,
        value: f64,
        grad_norm: f64,
    ) -> (Array1<f64>, f64, f64) {
        (
            self.best_point.clone().unwrap_or_else(|| point.clone()),
            if self.best_value.is_finite() {
                self.best_value
            } else {
                value
            },
            if self.best_grad_norm.is_finite() {
                self.best_grad_norm
            } else {
                grad_norm
            },
        )
    }

    /// Grant one escape unless it would replay the previous one from a
    /// bit-identical search state.
    fn grant_escape_unless_replay(&mut self, incumbent: IncumbentBits) -> bool {
        if self.incumbent_at_last_escape.as_ref() == Some(&incumbent) {
            return false;
        }
        self.stuck_escapes = self.stuck_escapes.saturating_add(1);
        self.incumbent_at_last_escape = Some(incumbent);
        self.window_trials.clear();
        true
    }

    /// The replay identity of the search as it stands (see [`IncumbentBits`]).
    fn escape_state(&self, point: &Array1<f64>, value: f64, grad_norm: f64) -> IncumbentBits {
        IncumbentBits::new(point, value, grad_norm, &self.window_trials)
    }

    fn record_window_trial(&mut self, point: &Array1<f64>) {
        self.window_trials.push(point_bits(point));
    }

    fn restart_recent(&mut self, point: &Array1<f64>, value: f64) {
        self.recent.clear();
        self.recent.push_back((point.clone(), value));
    }

    /// Adopt `sample` as the incumbent. Clears the wall evidence, which belongs
    /// to the incumbent it was gathered at.
    fn adopt(&mut self, sample: &StallSample<'_>, payload: Option<C>) {
        self.best_value = sample.value;
        self.best_resolution = sample.resolution;
        self.best_point = Some(sample.point.clone());
        self.best_grad_norm = sample.grad_norm;
        self.best_curvature_psd = sample.curvature_psd;
        self.best_payload = payload;
        self.wall_refusals = 0;
    }

    /// Whether a stall at a non-stationary incumbent may be followed by another.
    ///
    /// A stall is licensed when, since the previous licensed one, the search bought
    /// resolved descent (the incumbent is [`resolvably_below`] the previous one) or
    /// stationarity (the incumbent's projected gradient contracted), or when the
    /// incumbent is a strict saddle, whose negative curvature is descent the search
    /// can still take. The first stall is always licensed. A proven replay is never
    /// licensed.
    ///
    /// Termination needs no count: the criterion is bounded below, so resolved
    /// descent is bought finitely often, and every contraction strictly lowers a
    /// floating-point value bounded below by zero. A saddle licence ends at the
    /// replay cut.
    pub fn license_continuation(&mut self) -> bool {
        if self.replay_proven {
            return false;
        }
        let licensed = match self.continuation_incumbent {
            None => true,
            Some((previous_value, previous_grad_norm, previous_resolution)) => {
                resolvably_below(
                    previous_value,
                    previous_resolution,
                    self.best_value,
                    self.best_resolution,
                ) || self.best_grad_norm < previous_grad_norm
                    || self.best_curvature_psd == Some(false)
            }
        };
        if licensed {
            self.continuation_incumbent =
                Some((self.best_value, self.best_grad_norm, self.best_resolution));
        }
        licensed
    }

    /// The incumbent to stop at when a stall is not licensed to continue; `None`
    /// while continuing is licensed.
    pub fn unprogressing_stop(&mut self) -> Option<StallExit> {
        if self.license_continuation() {
            return None;
        }
        let point = self.best_point.clone()?;
        log::debug!(
            "[STALL] stopping at an unprogressing stall: the search stalled again at \
             value={:.6e} |Pg|={:.3e} after {} trusted sample(s), with no resolved descent and \
             no contraction of the projected gradient since the last one",
            self.best_value,
            self.best_grad_norm,
            self.accepted,
        );
        Some(StallExit {
            point,
            value: self.best_value,
            grad_norm: self.best_grad_norm,
            accepted: self.accepted,
            converged: false,
            probe_scale: None,
            unescapable_window: None,
            wall_refusals: self.wall_refusals,
        })
    }

    /// The evidence about the criterion's scatter since the last resolved
    /// progress, licensing nothing.
    ///
    /// `σ̂ = median_i |f_i − f_{i−1}|`, floored at the value's rounding
    /// `ε·(1 + |f_best|)`, and `Δ = max_i ‖x_i − x_{i−1}‖₂`, the radius the trusted
    /// samples probed. `σ̂/Δ` bounds the gradient only along the directions the
    /// steps spanned, so it is reported and never widens a band. `None` with fewer
    /// than three differences or a degenerate radius.
    pub fn window_probe_scale(&self) -> Option<(f64, f64)> {
        if self.recent.len() < 4 {
            return None;
        }
        let mut value_diffs: Vec<f64> = Vec::with_capacity(self.recent.len() - 1);
        let mut probe_radius = 0.0_f64;
        for (prev, next) in self.recent.iter().zip(self.recent.iter().skip(1)) {
            value_diffs.push((next.1 - prev.1).abs());
            let step = prev
                .0
                .iter()
                .zip(next.0.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            probe_radius = probe_radius.max(step);
        }
        if !probe_radius.is_finite() || probe_radius <= 0.0 {
            return None;
        }
        value_diffs.sort_by(|a, b| a.total_cmp(b));
        let mid = value_diffs.len() / 2;
        let median = if value_diffs.len() % 2 == 1 {
            value_diffs[mid]
        } else {
            0.5 * (value_diffs[mid - 1] + value_diffs[mid])
        };
        if !median.is_finite() {
            return None;
        }
        let best_scale = if self.best_value.is_finite() {
            self.best_value.abs()
        } else {
            0.0
        };
        let noise_floor = median.max(f64::EPSILON * (1.0 + best_scale));
        Some((noise_floor, probe_radius))
    }

    /// Fold the seed. A finite seed becomes the incumbent and is published as a
    /// non-converged snapshot, so a budget exit always has a feasible best.
    pub fn observe_seed(&mut self, sample: StallSample<'_>) {
        let staged_payload = self.staged_payload.take();
        if !sample.value.is_finite() {
            return;
        }
        self.adopt(&sample, staged_payload);
        self.refused_streak = 0;
        self.window_trials.clear();
        self.accepted = self.accepted.saturating_add(1);
        self.restart_recent(sample.point, sample.value);
        self.publish_best_so_far();
    }

    /// Fold one accepted sample, reached by a step whose own model predicted
    /// `predicted_decrease`: the cubic or quadratic model's for ARC and a trust
    /// region, the linear model's `−gᵀs` for a line search.
    ///
    /// Returns `Continue` while the search makes resolved progress or adapts (see
    /// the module docs), and otherwise decides the stall at the incumbent. A
    /// sample inside the band at its own value is stalled whatever its decrease:
    /// progress pinned at a bound is not feasible descent. The first finite
    /// trusted sample is the first incumbent and reaches no verdict.
    ///
    /// A stall at a strict-saddle incumbent grants an escape (flagged for
    /// [`Self::take_strict_saddle_refusal`]) unless it would replay the previous
    /// one.
    pub fn observe(&mut self, sample: StallSample<'_>, predicted_decrease: f64) -> StallVerdict {
        let staged_payload = self.staged_payload.take();
        let point = sample.point;
        let (value, resolution, grad_norm) = (sample.value, sample.resolution, sample.grad_norm);
        if !value.is_finite() {
            self.record_window_trial(point);
            return StallVerdict::Continue;
        }
        if !sample.trusted {
            self.refused_streak = 0;
            self.record_window_trial(point);
            return StallVerdict::Continue;
        }
        self.refused_streak = 0;
        self.accepted = self.accepted.saturating_add(1);
        let first = !self.best_value.is_finite();
        let resolution_pair = self.best_resolution + resolution;
        let resolved = resolvably_below(self.best_value, self.best_resolution, value, resolution);
        if value < self.best_value {
            self.adopt(&sample, staged_payload);
            self.publish_best_so_far();
        }
        if first {
            self.restart_recent(point, value);
            return StallVerdict::Continue;
        }
        let stationary_at_value = grad_norm.is_finite() && grad_norm <= self.band(value);
        if !stationary_at_value && resolved {
            self.window_trials.clear();
            self.stuck_escapes = 0;
            self.incumbent_at_last_escape = None;
            self.restart_recent(point, value);
            return StallVerdict::Continue;
        }
        self.recent.push_back((point.clone(), value));
        self.record_window_trial(point);
        if !stationary_at_value && predicted_decrease > resolution_pair {
            return StallVerdict::Continue;
        }
        if self.best_curvature_psd == Some(false) {
            let (best_point, best_value, best_grad_norm) =
                self.best_iterate_or(point, value, grad_norm);
            let incumbent = self.escape_state(&best_point, best_value, best_grad_norm);
            if self.grant_escape_unless_replay(incumbent) {
                log::debug!(
                    "[STALL] stall at a strict-saddle incumbent (value={:.6e}): refusing to stop \
                     there and returning control to the solver to take the negative curvature \
                     (escape {})",
                    self.best_value,
                    self.stuck_escapes,
                );
                self.strict_saddle_refusal = true;
                return StallVerdict::Continue;
            }
            self.replay_proven = true;
            log::debug!(
                "[STALL] strict-saddle refusal cut at escape {}: the previous refusal was followed \
                 by the same trials from a bit-identical incumbent (best={:.9e}, |g|={:.3e}); \
                 halting",
                self.stuck_escapes,
                best_value,
                best_grad_norm,
            );
        }
        self.publish_stall(point, value, grad_norm)
    }

    /// Fold one finite trial the solver's ratio test rejected.
    ///
    /// The iterate did not move, so this is not a step and reaches no verdict:
    /// the incumbent and the trusted-sample count are untouched. Its finite value
    /// ends a run of refused trials, and it is a point the search observed, so it
    /// belongs to the replay identity. The payload staged for it is dropped.
    pub fn observe_rejected_trial(&mut self, point: &Array1<f64>) {
        self.staged_payload = None;
        self.refused_streak = 0;
        self.record_window_trial(point);
    }

    /// A feasible trial the solver did not accept ends a refusal run. It grants
    /// no progress and is not a step.
    pub fn observe_feasible_probe(&mut self) {
        self.refused_streak = 0;
        self.unescapable_streak = 0;
    }

    /// Whether a refused trial whose step's model predicted `predicted_decrease`
    /// stalls. A refused trial has no value, so only the predicted half of the
    /// stall rule can be taken, and it can, since the model decrease is known
    /// before the trial is evaluated: a model that promised at most the incumbent's
    /// resolution could not have shown resolved progress even had the trial
    /// evaluated exactly. A decrease that is not a number decides nothing.
    fn refusal_stalls(&self, predicted_decrease: f64) -> bool {
        predicted_decrease <= self.best_resolution
    }

    /// Count one wall refusal on the incumbent and stamp it onto the published
    /// exit, which a budget stop recovers its checkpoint from. It publishes
    /// nothing new, so [`Self::exit_revision`] does not move.
    fn record_wall_refusal(&mut self) {
        self.wall_refusals = self.wall_refusals.saturating_add(1);
        if let Some(exit) = self.exit.as_mut() {
            exit.wall_refusals = self.wall_refusals;
        }
    }

    /// Fold one trial refused before it produced a finite criterion value,
    /// proposed by a step whose model predicted `predicted_decrease`.
    ///
    /// A refusal that stalls ([`Self::refusal_stalls`]) is decided at the
    /// incumbent exactly as a stalled step is; one that does not reaches no
    /// verdict, however many come. Before any incumbent there is nothing to stop
    /// at. At a strict-saddle incumbent the refusal says only that the solver's
    /// current step left the domain, so the search continues.
    pub fn observe_refused(
        &mut self,
        point: &Array1<f64>,
        predicted_decrease: f64,
    ) -> StallVerdict {
        if self.best_point.is_none() || !self.best_value.is_finite() {
            return StallVerdict::Continue;
        }
        self.refused_streak = self.refused_streak.saturating_add(1);
        self.unescapable_streak = 0;
        self.record_window_trial(point);
        if !self.refusal_stalls(predicted_decrease) {
            return StallVerdict::Continue;
        }
        self.record_wall_refusal();
        if self.best_curvature_psd == Some(false) {
            log::debug!(
                "[STALL] refused trial stalled at a strict-saddle incumbent; refusing the stall \
                 and returning control to the solver"
            );
            return StallVerdict::Continue;
        }
        let (best_value, best_grad_norm) = (self.best_value, self.best_grad_norm);
        self.publish_stall(point, best_value, best_grad_norm)
    }

    /// Fold one trial that evaluated to `value` (to within `resolution`) on a part
    /// of the domain this search cannot move onto, proposed by a step whose model
    /// predicted `predicted_decrease`.
    ///
    /// It shares [`Self::observe_refused`]'s streak. It stops the search at once,
    /// with no escape, when its value is resolvably below the incumbent's: that is
    /// descent this search cannot take and a caller that can move there can.
    /// Otherwise the other part of the domain is only a wall, and the trial is
    /// judged by its model decrease as any refusal is. A stalled run that also
    /// holds an ordinary refusal is decided exactly as that one is. A stalled run
    /// made only of these trials, at an incumbent outside the band, publishes the
    /// incumbent, not converged, and grants no escape: an escape continues for
    /// descent the solver may still reach, and these trials say continuing only
    /// proposes more of them. Inside the band it is published as an ordinary stall.
    pub fn observe_unescapable_refusal(
        &mut self,
        point: &Array1<f64>,
        value: f64,
        resolution: f64,
        predicted_decrease: f64,
    ) -> StallVerdict {
        if self.best_point.is_none() || !self.best_value.is_finite() {
            return StallVerdict::Continue;
        }
        self.refused_streak = self.refused_streak.saturating_add(1);
        self.record_window_trial(point);
        self.unescapable_streak = match self.refused_streak {
            1 => 1,
            _ => self.unescapable_streak.saturating_add(1),
        };
        let lower_elsewhere =
            resolvably_below(self.best_value, self.best_resolution, value, resolution);
        if !lower_elsewhere {
            if !self.refusal_stalls(predicted_decrease) {
                return StallVerdict::Continue;
            }
            self.record_wall_refusal();
            if self.unescapable_streak < self.refused_streak {
                let (best_value, best_grad_norm) = (self.best_value, self.best_grad_norm);
                return self.publish_stall(point, best_value, best_grad_norm);
            }
        }
        let (best_point, best_value, best_grad_norm) =
            self.best_iterate_or(point, self.best_value, self.best_grad_norm);
        let band = self.band(best_value);
        if best_grad_norm.is_finite() && best_grad_norm <= band {
            return self.publish_stall(point, best_value, best_grad_norm);
        }
        let probe_scale = self.window_probe_scale();
        let (refused_trials, accepted, wall_refusals) =
            (self.unescapable_streak, self.accepted, self.wall_refusals);
        self.publish(StallExit {
            point: best_point,
            value: best_value,
            grad_norm: best_grad_norm,
            accepted,
            converged: false,
            probe_scale,
            unescapable_window: Some(UnescapableRefusalWindow {
                refused_trials,
                band,
            }),
            wall_refusals,
        });
        StallVerdict::Floor {
            residual_grad_norm: best_grad_norm,
        }
    }

    /// Adopt a sample the caller has shown constrained-stationary and publish
    /// it, unless it regresses the incumbent.
    ///
    /// `grad_norm` must be the bound-projected residual at the sample, so the
    /// published verdict certifies only a sample stationary in its feasible
    /// subspace. An untrusted sample, or one resolvably above the incumbent (a
    /// spurious corner of the box), is folded as an ordinary sample instead, so
    /// the better incumbent is kept.
    pub fn observe_constrained_stationary(
        &mut self,
        sample: StallSample<'_>,
        predicted_decrease: f64,
    ) -> StallVerdict {
        if !sample.value.is_finite() {
            return StallVerdict::Continue;
        }
        let regresses = self.best_value.is_finite()
            && resolvably_below(
                sample.value,
                sample.resolution,
                self.best_value,
                self.best_resolution,
            );
        if !sample.trusted || regresses {
            return self.observe(sample, predicted_decrease);
        }
        let payload = self.staged_payload.take();
        self.refused_streak = 0;
        self.accepted = self.accepted.saturating_add(1);
        self.adopt(&sample, payload);
        self.publish_stall(sample.point, sample.value, sample.grad_norm)
    }

    /// Publish the incumbent and decide a stall's verdict.
    fn publish_stall(&mut self, point: &Array1<f64>, value: f64, grad_norm: f64) -> StallVerdict {
        let (best_point, best_value, best_grad_norm) =
            self.best_iterate_or(point, value, grad_norm);
        let band = self.band(best_value);
        let probe_scale = self.window_probe_scale();
        let converged = best_grad_norm.is_finite() && best_grad_norm <= band;
        let (accepted, wall_refusals) = (self.accepted, self.wall_refusals);
        if converged {
            self.publish(StallExit {
                point: best_point,
                value: best_value,
                grad_norm: best_grad_norm,
                accepted,
                converged,
                probe_scale,
                unescapable_window: None,
                wall_refusals,
            });
            return StallVerdict::Converged;
        }
        let non_stationary = best_grad_norm.is_finite() && best_grad_norm > band;
        let incumbent = self.escape_state(&best_point, best_value, best_grad_norm);
        if non_stationary && self.grant_escape_unless_replay(incumbent.clone()) {
            self.refused_streak = 0;
            return StallVerdict::Escape {
                residual_grad_norm: best_grad_norm,
                band,
            };
        }
        if non_stationary && self.incumbent_at_last_escape.as_ref() == Some(&incumbent) {
            self.replay_proven = true;
            log::debug!(
                "[STALL] escape streak cut at {}: the escape was followed by the same trials from \
                 a bit-identical incumbent (best={:.9e}, |g|={:.3e}); halting",
                self.stuck_escapes,
                best_value,
                best_grad_norm,
            );
        }
        self.publish(StallExit {
            point: best_point,
            value: best_value,
            grad_norm: best_grad_norm,
            accepted,
            converged,
            probe_scale,
            unescapable_window: None,
            wall_refusals,
        });
        StallVerdict::Floor {
            residual_grad_norm: best_grad_norm,
        }
    }

    /// Publish the incumbent as a non-converged snapshot without deciding
    /// anything, so a budget exit halts back to the best feasible iterate.
    fn publish_best_so_far(&mut self) {
        let Some(point) = self.best_point.clone() else {
            return;
        };
        if !self.best_value.is_finite() {
            return;
        }
        let (value, grad_norm, accepted, wall_refusals) = (
            self.best_value,
            self.best_grad_norm,
            self.accepted,
            self.wall_refusals,
        );
        self.publish(StallExit {
            point,
            value,
            grad_norm,
            accepted,
            converged: false,
            probe_scale: None,
            unescapable_window: None,
            wall_refusals,
        });
    }

    /// Mark the published exit not converged, keeping the incumbent and the
    /// evidence, for a caller that owns a stronger acceptance test.
    pub fn revoke_published_convergence(&mut self) {
        if let Some(exit) = self.exit.as_mut() {
            exit.converged = false;
            self.exit_revision = self.exit_revision.wrapping_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{StallMonitor, StallSample, StallVerdict, resolvably_below};
    use ndarray::{Array1, array};

    const BAND: f64 = 1e-2;
    const RESOLUTION: f64 = 1e-9;

    fn monitor() -> StallMonitor<()> {
        StallMonitor::new(|_| BAND)
    }

    fn sample(point: &Array1<f64>, value: f64, grad_norm: f64) -> StallSample<'_> {
        StallSample {
            point,
            value,
            resolution: RESOLUTION,
            grad_norm,
            trusted: true,
            curvature_psd: None,
        }
    }

    #[test]
    fn two_values_are_ordered_exactly_past_the_sum_of_their_resolutions() {
        assert!(resolvably_below(1.0, 0.1, 0.7, 0.1));
        assert!(!resolvably_below(1.0, 0.1, 0.8, 0.1));
        assert!(!resolvably_below(1.0, 0.1, f64::NAN, 0.1));
    }

    /// One stalled step reaches the verdict: no window is counted.
    #[test]
    fn one_stalled_step_decides_at_the_incumbent() {
        let mut stall = monitor();
        let (a, b) = (array![0.0], array![0.1]);
        assert_eq!(
            stall.observe(sample(&a, 1.0, 0.0), 0.0),
            StallVerdict::Continue
        );
        assert_eq!(
            stall.observe(sample(&b, 1.0, 0.0), 0.0),
            StallVerdict::Converged
        );
        let exit = stall.exit().expect("published");
        assert!(exit.converged);
        assert_eq!(exit.point, a);
    }

    /// A step that buys no resolved progress while its model promised a resolvable
    /// decrease is the solver adapting, and reaches no verdict.
    #[test]
    fn an_adapting_step_reaches_no_verdict() {
        let mut stall = monitor();
        let (a, b) = (array![0.0], array![0.1]);
        stall.observe(sample(&a, 1.0, 1.0), 0.0);
        assert_eq!(
            stall.observe(sample(&b, 1.0, 1.0), 1e-3),
            StallVerdict::Continue
        );
        assert!(stall.exit().is_some_and(|exit| !exit.converged));
    }

    #[test]
    fn resolved_progress_continues_and_an_untrusted_sample_is_never_the_incumbent() {
        let mut stall = monitor();
        let (a, b, c) = (array![0.0], array![0.1], array![0.2]);
        stall.observe(sample(&a, 1.0, 1.0), 0.0);
        assert_eq!(
            stall.observe(sample(&b, 0.5, 1.0), 0.0),
            StallVerdict::Continue
        );
        let untrusted = StallSample {
            trusted: false,
            ..sample(&c, 0.1, 1.0)
        };
        assert_eq!(stall.observe(untrusted, 0.0), StallVerdict::Continue);
        assert_eq!(stall.best_value(), 0.5);
        assert_eq!(stall.best_point(), Some(&b));
    }

    #[test]
    fn an_escape_is_cut_only_when_it_replays_the_same_trials_from_a_bit_identical_incumbent() {
        let mut stall = monitor();
        let (a, b, c) = (array![0.0], array![0.1], array![0.2]);
        stall.observe(sample(&a, 1.0, 1.0), 0.0);
        assert!(matches!(
            stall.observe(sample(&b, 1.0, 1.0), 0.0),
            StallVerdict::Escape { .. }
        ));
        assert_eq!(stall.stuck_escapes, 1);
        assert!(matches!(
            stall.observe(sample(&b, 1.0, 1.0), 0.0),
            StallVerdict::Floor { .. }
        ));
        assert!(stall.replay_proven());
        assert!(!stall.exit().expect("published").converged);

        // A run that observes a trial the previous one never saw is a search in a
        // different state, however still its incumbent, and earns another escape.
        let mut exploring = monitor();
        exploring.observe(sample(&a, 1.0, 1.0), 0.0);
        exploring.observe(sample(&b, 1.0, 1.0), 0.0);
        assert!(matches!(
            exploring.observe(sample(&c, 1.0, 1.0), 0.0),
            StallVerdict::Escape { .. }
        ));
        assert_eq!(exploring.stuck_escapes, 2);
        assert!(!exploring.replay_proven());
    }

    #[test]
    fn a_strict_saddle_stall_grants_an_escape_flagged_for_the_caller() {
        let mut stall = monitor();
        let (a, b) = (array![0.0], array![0.1]);
        let saddle = StallSample {
            curvature_psd: Some(false),
            ..sample(&a, 1.0, 0.0)
        };
        stall.observe(saddle, 0.0);
        assert_eq!(
            stall.observe(sample(&b, 1.0, 0.0), 0.0),
            StallVerdict::Continue
        );
        assert!(stall.take_strict_saddle_refusal());
        assert!(!stall.take_strict_saddle_refusal());
    }

    /// A refused trial whose model promised at most the incumbent's resolution is a
    /// domain wall: it stalls, and it is counted on the incumbent. One whose model
    /// promised more reaches no verdict.
    #[test]
    fn a_refusal_stalls_only_when_its_model_promised_no_resolvable_decrease() {
        let mut stall = monitor();
        let (a, b) = (array![0.0], array![0.1]);
        stall.observe(sample(&a, 1.0, 0.0), 0.0);
        assert_eq!(stall.observe_refused(&b, 1e-3), StallVerdict::Continue);
        assert_eq!(stall.wall_refusals(), 0);
        assert_eq!(stall.observe_refused(&b, 0.0), StallVerdict::Converged);
        assert_eq!(stall.wall_refusals(), 1);
        assert_eq!(stall.exit().expect("published").wall_refusals, 1);
    }

    /// A trial lower on a part of the domain the search cannot move onto stops the
    /// search at once, with no escape.
    #[test]
    fn an_unescapable_trial_below_the_incumbent_stops_without_an_escape() {
        let mut stall = monitor();
        let (a, b) = (array![0.0], array![0.1]);
        stall.observe(sample(&a, 1.0, 1.0), 0.0);
        let verdict = stall.observe_unescapable_refusal(&b, 0.5, RESOLUTION, 1.0);
        assert!(matches!(verdict, StallVerdict::Floor { .. }), "{verdict:?}");
        let exit = stall.exit().expect("published");
        assert!(!exit.converged);
        assert_eq!(exit.unescapable_window.expect("window").refused_trials, 1);
        assert_eq!(stall.stuck_escapes, 0);
    }

    /// The licence continues only while the search buys resolved descent, a
    /// smaller projected gradient, or sits at a strict saddle.
    #[test]
    fn a_stall_that_bought_nothing_since_the_last_is_not_licensed() {
        let mut stall = monitor();
        let a = array![0.0];
        stall.observe(sample(&a, 1.0, 1.0), 0.0);
        assert!(stall.license_continuation());
        assert!(stall.unprogressing_stop().is_some());
    }

    #[test]
    fn a_constrained_stationary_sample_is_adopted_unless_it_regresses_the_incumbent() {
        let mut stall = monitor();
        let (a, b, c) = (array![0.0], array![0.1], array![0.2]);
        stall.observe(sample(&a, 1.0, 1.0), 0.0);
        assert!(matches!(
            stall.observe_constrained_stationary(sample(&b, 2.0, 0.0), 0.0),
            StallVerdict::Escape { .. }
        ));
        assert_eq!(stall.best_point(), Some(&a));
        assert_eq!(
            stall.observe_constrained_stationary(sample(&c, 1.0, 0.0), 0.0),
            StallVerdict::Converged
        );
        assert_eq!(stall.best_point(), Some(&c));
    }
}
