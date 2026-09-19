//! Caller-fed stall monitor for an outer optimization loop.
//!
//! [`StallMonitor`] owns the part of a stalled-search decision that belongs to
//! the optimizer and to no particular objective: the monotone incumbent, the
//! no-improvement window over trusted samples, the window of refused trials, the
//! escape granted when a filled window's incumbent still carries resolvable
//! descent, the bit-identity replay cut that ends escapes, the progress licence
//! that ends a stall buying nothing, and the window evidence a stall reports.
//!
//! The caller decides what it observes and when. A consumer that drives a solver
//! through an objective bridge observes at the bridge (every evaluation, every
//! refused trial, every accepted step it reconciles), so the sample stream is
//! the consumer's, not the solver's. What the monitor cannot know is supplied:
//!
//! * the stationarity band at a criterion value, as a closure;
//! * whether a sample is trustworthy (for example, whether an inner solve behind
//!   it converged);
//! * the incumbent's curvature verdict, and an opaque payload that travels with
//!   the incumbent.
//!
//! Every stop the monitor decides is published as a [`StallExit`] that the
//! caller reads through [`StallMonitor::exit`]; [`StallMonitor::exit_revision`]
//! counts writes, so a caller that mirrors the exit elsewhere copies it only
//! when the monitor changed it.

use ndarray::Array1;
use std::collections::VecDeque;

/// What folding one observation into a [`StallMonitor`] decided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StallVerdict {
    /// The window has not filled, or an escape was granted at a strict saddle.
    Continue,
    /// A filled window whose incumbent is inside the band at its value.
    Converged,
    /// A filled window whose incumbent is above the band, with no escape left:
    /// the escape would replay the previous one from a bit-identical incumbent,
    /// or the window held only unescapable refusals.
    Floor {
        /// Projected gradient norm at the incumbent.
        residual_grad_norm: f64,
    },
    /// A filled window whose incumbent is above the band: the window reopens so
    /// the search can take the descent the residual says is available.
    Escape {
        /// Projected gradient norm at the incumbent.
        residual_grad_norm: f64,
        /// The band at the incumbent's value that the residual exceeded.
        band: f64,
    },
}

/// A filled window made only of unescapable refusals, reported with a stop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnescapableRefusalWindow {
    /// Refused trials in the window.
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
    /// `true` only when a filled window's incumbent met the band. A running
    /// best-so-far snapshot is never a convergence claim.
    pub converged: bool,
    /// `(noise floor σ̂, probe radius Δ)` over the window; see
    /// [`StallMonitor::window_probe_scale`]. Evidence only.
    pub probe_scale: Option<(f64, f64)>,
    /// Set when the stopping window held only unescapable refusals.
    pub unescapable_window: Option<UnescapableRefusalWindow>,
}

/// The search state an escape was granted from, in raw bits: the whole
/// incumbent, and the trial points the window that filled had observed. Only
/// bit identity supports the claim a replay cut makes: a deterministic
/// procedure replayed from an identical state returns an identical result. Raw
/// bits keep the comparison total, and both of its possible errors (distinct
/// NaN payloads, `-0.0` against `0.0`) grant the escape rather than cut it.
///
/// The incumbent alone is not the search's state. A window of trials that do not
/// improve it leaves it bit-identical while the solver's own state moves: a
/// rejected cubic trial raises ARC's regularization, so the next window proposes
/// shorter steps to new points. The trials the window observed are the part of
/// that state the monitor can see, so a replay is a window that observed the
/// same points, in the same order, from the same incumbent (gam 0282f2b560: a
/// saddle whose window had just evaluated three fresh trials was cut as a
/// replay).
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

/// The solver state behind one accepted step, kept over the window so a caller
/// can derive the window from what the solver was doing (#2817).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StallStep {
    /// Step's L2 norm.
    pub step_norm: f64,
    /// The cubic regularization `σ` the step was solved with (ARC).
    pub regularization: Option<f64>,
    /// The accepted line-search step length `α` (BFGS).
    pub line_search_step: Option<f64>,
}

/// Monotone incumbent, no-improvement window, refused-trial window, escapes,
/// replay cut and progress licence for an outer search. See the module docs.
pub struct StallMonitor<C> {
    /// A trusted sample counts as no improvement when
    /// `best - value <= rel_tol · (1 + |best|)`.
    rel_tol: f64,
    window: usize,
    band: Box<dyn Fn(f64) -> f64 + Send>,
    /// Set once a replay cut proved that reopening the window replays the
    /// previous one. No licence continues past a proven replay.
    replay_proven: bool,
    best_value: f64,
    best_point: Option<Array1<f64>>,
    best_grad_norm: f64,
    /// The incumbent's curvature verdict: `Some(false)` is a strict saddle.
    best_curvature_psd: Option<bool>,
    best_payload: Option<C>,
    staged_payload: Option<C>,
    strict_saddle_refusal: bool,
    no_improve_streak: usize,
    refused_streak: usize,
    /// How many of the current refused streak were unescapable refusals.
    unescapable_streak: usize,
    accepted: usize,
    /// Consecutive escapes since the last resolved improvement. A diagnostic:
    /// escapes are bounded by the replay cut and by the caller's budget.
    pub stuck_escapes: usize,
    incumbent_at_last_escape: Option<IncumbentBits>,
    /// Every trial point observed without improving the incumbent since the
    /// window last opened, at the latest improvement or escape grant: the half of
    /// the replay identity the incumbent cannot carry (see [`IncumbentBits`]).
    window_trials: Vec<Vec<u64>>,
    continuation_incumbent: Option<(f64, f64)>,
    recent: VecDeque<(Array1<f64>, f64)>,
    /// The solver state of the latest `window` accepted steps, oldest first.
    recent_steps: VecDeque<StallStep>,
    exit: Option<StallExit>,
    exit_revision: u64,
}

impl<C: Clone> StallMonitor<C> {
    /// A monitor with improvement floor `rel_tol` over `window` samples, judging
    /// stationarity by `band(value)`.
    pub fn new(rel_tol: f64, window: usize, band: impl Fn(f64) -> f64 + Send + 'static) -> Self {
        Self {
            rel_tol,
            window,
            band: Box::new(band),
            replay_proven: false,
            best_value: f64::INFINITY,
            best_point: None,
            best_grad_norm: f64::INFINITY,
            best_curvature_psd: None,
            best_payload: None,
            staged_payload: None,
            strict_saddle_refusal: false,
            no_improve_streak: 0,
            refused_streak: 0,
            unescapable_streak: 0,
            accepted: 0,
            stuck_escapes: 0,
            incumbent_at_last_escape: None,
            window_trials: Vec::new(),
            continuation_incumbent: None,
            recent: VecDeque::new(),
            recent_steps: VecDeque::new(),
            exit: None,
            exit_revision: 0,
        }
    }

    pub fn rel_tol(&self) -> f64 {
        self.rel_tol
    }

    pub fn window(&self) -> usize {
        self.window
    }

    /// The stationarity band at criterion value `value`.
    pub fn band(&self, value: f64) -> f64 {
        (self.band)(value)
    }

    /// Record the solver state of one accepted step, keeping the latest
    /// `window` of them. It decides nothing; it is the evidence a caller derives
    /// a window from.
    pub fn observe_step(&mut self, step: &crate::StepInfo) {
        self.recent_steps.push_back(StallStep {
            step_norm: step.step_norm,
            regularization: step.regularization,
            line_search_step: step.line_search_step,
        });
        while self.recent_steps.len() > self.window {
            self.recent_steps.pop_front();
        }
    }

    /// The solver state of the latest `window` accepted steps, oldest first.
    pub fn recent_steps(&self) -> impl ExactSizeIterator<Item = &StallStep> {
        self.recent_steps.iter()
    }

    pub fn best_value(&self) -> f64 {
        self.best_value
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

    pub fn replay_proven(&self) -> bool {
        self.replay_proven
    }

    /// The published exit, if any.
    pub fn exit(&self) -> Option<&StallExit> {
        self.exit.as_ref()
    }

    /// Incremented on every write to [`Self::exit`].
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
    /// bit-identical incumbent. A granted escape reopens the no-improvement
    /// window.
    fn grant_escape_unless_replay(&mut self, incumbent: IncumbentBits) -> bool {
        if self.incumbent_at_last_escape.as_ref() == Some(&incumbent) {
            return false;
        }
        self.stuck_escapes = self.stuck_escapes.saturating_add(1);
        self.incumbent_at_last_escape = Some(incumbent);
        self.window_trials.clear();
        self.no_improve_streak = 0;
        true
    }

    /// The replay identity of the search as it stands (see [`IncumbentBits`]).
    fn escape_state(&self, point: &Array1<f64>, value: f64, grad_norm: f64) -> IncumbentBits {
        IncumbentBits::new(point, value, grad_norm, &self.window_trials)
    }

    /// Record a trial that did not improve the incumbent as part of the open
    /// window's replay identity.
    fn record_window_trial(&mut self, point: &Array1<f64>) {
        self.window_trials.push(point_bits(point));
    }

    /// Whether a filled window at a non-stationary stall may be followed by
    /// another one.
    ///
    /// A window is licensed when, since the previous licensed window, the search
    /// bought resolved descent (the incumbent improved by more than
    /// `rel_tol · (1 + |V|)`) or stationarity (the incumbent's projected gradient
    /// contracted), or when the incumbent is a strict saddle, whose negative
    /// curvature is descent the search can still take. The first window is
    /// always licensed. A proven replay is never licensed.
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
            Some((previous_value, previous_grad_norm)) => {
                let resolution = self.rel_tol * (1.0 + self.best_value.abs());
                previous_value - self.best_value > resolution
                    || self.best_grad_norm < previous_grad_norm
                    || self.best_curvature_psd == Some(false)
            }
        };
        if licensed {
            self.continuation_incumbent = Some((self.best_value, self.best_grad_norm));
        }
        licensed
    }

    /// The incumbent to stop at when a filled window is not licensed to
    /// continue; `None` while continuing is licensed.
    pub fn unprogressing_stop(&mut self) -> Option<StallExit> {
        if self.license_continuation() {
            return None;
        }
        let point = self.best_point.clone()?;
        log::debug!(
            "[STALL] stopping at an unprogressing stall: the window filled again at value={:.6e} \
             |Pg|={:.3e} after {} trusted sample(s), with no resolved descent and no contraction \
             of the projected gradient since the last one",
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
        })
    }

    fn record_recent(&mut self, point: &Array1<f64>, value: f64) {
        self.recent.push_back((point.clone(), value));
        while self.recent.len() > self.window + 1 {
            self.recent.pop_front();
        }
    }

    /// The window's evidence about the criterion's scatter, licensing nothing.
    ///
    /// `σ̂ = median_i |f_i − f_{i−1}|`, floored at the value's rounding
    /// `ε·(1 + |f_best|)`, and `Δ = max_i ‖x_i − x_{i−1}‖₂`, the radius the
    /// trusted samples probed. A window that filled on microscopic steps and one
    /// that filled on a flat surface differ in `Δ`. `σ̂/Δ` bounds the gradient
    /// only along the directions the steps spanned, so it is reported and never
    /// widens a band. `None` with fewer than three differences or a degenerate
    /// radius.
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
    pub fn observe_seed(
        &mut self,
        point: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        curvature_psd: Option<bool>,
    ) {
        let staged_payload = self.staged_payload.take();
        if !value.is_finite() {
            return;
        }
        self.best_value = value;
        self.best_point = Some(point.clone());
        self.best_grad_norm = grad_norm;
        self.best_curvature_psd = curvature_psd;
        self.best_payload = staged_payload;
        self.no_improve_streak = 0;
        self.refused_streak = 0;
        self.window_trials.clear();
        self.accepted = self.accepted.saturating_add(1);
        self.record_recent(point, value);
        self.publish_best_so_far();
    }

    /// Fold one sample `(x, value, projected gradient norm)`.
    ///
    /// An untrusted sample never becomes the incumbent and never counts toward a
    /// window: it resets both streaks. A trusted sample that improves on the
    /// incumbent by more than the floor resets the window and ends an escape
    /// streak. A sample inside the band at its own value counts as no
    /// improvement whatever its raw decrease (drift pinned at a bound is not
    /// feasible descent). The first finite trusted sample is the first
    /// incumbent and never counts toward a window.
    ///
    /// A filled window at a strict-saddle incumbent grants an escape (flagged
    /// for [`Self::take_strict_saddle_refusal`]) unless it would replay the
    /// previous one; otherwise the window is published.
    pub fn observe(
        &mut self,
        point: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        trusted: bool,
        curvature_psd: Option<bool>,
    ) -> StallVerdict {
        let staged_payload = self.staged_payload.take();
        if !value.is_finite() {
            self.no_improve_streak = 0;
            self.record_window_trial(point);
            return StallVerdict::Continue;
        }
        if !trusted {
            self.refused_streak = 0;
            self.no_improve_streak = 0;
            self.record_window_trial(point);
            return StallVerdict::Continue;
        }
        self.refused_streak = 0;
        self.accepted = self.accepted.saturating_add(1);
        self.record_recent(point, value);
        let improvement = self.best_value - value;
        let floor = self.rel_tol * (1.0 + self.best_value.abs());
        if value < self.best_value {
            self.best_value = value;
            self.best_point = Some(point.clone());
            self.best_grad_norm = grad_norm;
            self.best_curvature_psd = curvature_psd;
            self.best_payload = staged_payload;
            self.publish_best_so_far();
        }
        let stationary_at_value = grad_norm.is_finite() && grad_norm <= self.band(value);
        if floor.is_finite() && (improvement <= floor || stationary_at_value) {
            self.no_improve_streak = self.no_improve_streak.saturating_add(1);
            self.record_window_trial(point);
        } else {
            self.no_improve_streak = 0;
            self.window_trials.clear();
            self.stuck_escapes = 0;
            self.incumbent_at_last_escape = None;
        }
        if self.no_improve_streak < self.window {
            return StallVerdict::Continue;
        }
        if self.best_curvature_psd == Some(false) {
            let (best_point, best_value, best_grad_norm) =
                self.best_iterate_or(point, value, grad_norm);
            let incumbent = self.escape_state(&best_point, best_value, best_grad_norm);
            if self.grant_escape_unless_replay(incumbent) {
                log::debug!(
                    "[STALL] window filled at a strict-saddle incumbent (value={:.6e}): refusing \
                     to stop there and returning control to the solver to take the negative \
                     curvature (escape {})",
                    self.best_value,
                    self.stuck_escapes,
                );
                self.strict_saddle_refusal = true;
                return StallVerdict::Continue;
            }
            self.replay_proven = true;
            log::debug!(
                "[STALL] strict-saddle refusal cut at escape {}: the previous refusal reopened a \
                 full {}-sample window that observed the same trials from a bit-identical incumbent \
                 (best={:.9e}, |g|={:.3e}); halting",
                self.stuck_escapes,
                self.window,
                best_value,
                best_grad_norm,
            );
        }
        self.publish_stall(point, value, grad_norm)
    }

    /// Fold one finite trial the solver's ratio test rejected.
    ///
    /// The iterate did not move, so this is not a step: the incumbent, the
    /// trusted-sample count and the no-improvement streak are untouched. What
    /// the trial does change: its finite value ends a run of refused trials,
    /// which are consecutive trials that did not evaluate, and it is a point the
    /// open window observed, so it belongs to the replay identity (see
    /// [`IncumbentBits`]). The payload staged for it is dropped, since it never
    /// becomes the incumbent.
    pub fn observe_rejected_trial(&mut self, point: &Array1<f64>) {
        self.staged_payload = None;
        self.refused_streak = 0;
        self.record_window_trial(point);
    }

    /// Fold one trial refused before it produced a finite criterion value.
    ///
    /// A window of `window` consecutive refusals after a finite incumbent is the
    /// same "no further progress" signal as a no-improvement window and is
    /// decided the same way at the incumbent. Before any incumbent there is
    /// nothing to stop at. At a strict-saddle incumbent the refused run says
    /// only that the solver's current step left the domain, so both streaks
    /// reset and the search continues.
    pub fn observe_refused(&mut self, point: &Array1<f64>) -> StallVerdict {
        if self.best_point.is_none() || !self.best_value.is_finite() {
            return StallVerdict::Continue;
        }
        self.refused_streak = self.refused_streak.saturating_add(1);
        self.unescapable_streak = 0;
        self.record_window_trial(point);
        if self.refused_streak < self.window {
            return StallVerdict::Continue;
        }
        if self.best_curvature_psd == Some(false) {
            self.refused_streak = 0;
            self.no_improve_streak = 0;
            log::debug!(
                "[STALL] refused-trial run reached a strict-saddle incumbent; refusing the stall \
                 and returning control to the solver"
            );
            return StallVerdict::Continue;
        }
        let (best_value, best_grad_norm) = (self.best_value, self.best_grad_norm);
        self.publish_stall(point, best_value, best_grad_norm)
    }

    /// Fold one refused trial that no escape can get past.
    ///
    /// It shares [`Self::observe_refused`]'s streak and window. A filled window
    /// that also holds an ordinary refusal is decided exactly as that one is. A
    /// window made only of unescapable refusals at an incumbent outside the band
    /// publishes the incumbent, not converged, with the window's evidence, and
    /// grants no escape: an escape reopens the window for descent the solver may
    /// still reach, and these trials say continuing only proposes more of them.
    /// Inside the band it is published as an ordinary stall.
    pub fn observe_unescapable_refusal(&mut self, point: &Array1<f64>) -> StallVerdict {
        if self.best_point.is_none() || !self.best_value.is_finite() {
            return StallVerdict::Continue;
        }
        self.refused_streak = self.refused_streak.saturating_add(1);
        self.record_window_trial(point);
        self.unescapable_streak = match self.refused_streak {
            1 => 1,
            _ => self.unescapable_streak.saturating_add(1),
        };
        if self.refused_streak < self.window {
            return StallVerdict::Continue;
        }
        if self.unescapable_streak < self.refused_streak {
            let (best_value, best_grad_norm) = (self.best_value, self.best_grad_norm);
            return self.publish_stall(point, best_value, best_grad_norm);
        }
        let (best_point, best_value, best_grad_norm) =
            self.best_iterate_or(point, self.best_value, self.best_grad_norm);
        let band = self.band(best_value);
        if best_grad_norm.is_finite() && best_grad_norm <= band {
            return self.publish_stall(point, best_value, best_grad_norm);
        }
        let probe_scale = self.window_probe_scale();
        let refused_trials = self.unescapable_streak;
        let accepted = self.accepted;
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
    /// subspace. An untrusted sample, or one whose value exceeds the incumbent
    /// by more than the floor (a spurious corner of the box), is folded as an
    /// ordinary sample instead, so the better incumbent is kept.
    pub fn observe_constrained_stationary(
        &mut self,
        point: &Array1<f64>,
        value: f64,
        grad_norm: f64,
        trusted: bool,
        curvature_psd: Option<bool>,
    ) -> StallVerdict {
        if !value.is_finite() {
            return StallVerdict::Continue;
        }
        if !trusted {
            return self.observe(point, value, grad_norm, trusted, curvature_psd);
        }
        let regresses = self.best_value.is_finite()
            && value > self.best_value + self.rel_tol * (1.0 + self.best_value.abs());
        if regresses {
            return self.observe(point, value, grad_norm, trusted, curvature_psd);
        }
        self.refused_streak = 0;
        self.accepted = self.accepted.saturating_add(1);
        self.best_value = value;
        self.best_point = Some(point.clone());
        self.best_grad_norm = grad_norm;
        self.best_curvature_psd = curvature_psd;
        self.best_payload = self.staged_payload.take();
        self.no_improve_streak = self.window;
        self.publish_stall(point, value, grad_norm)
    }

    /// Publish the incumbent and decide a filled window's verdict.
    fn publish_stall(&mut self, point: &Array1<f64>, value: f64, grad_norm: f64) -> StallVerdict {
        let (best_point, best_value, best_grad_norm) =
            self.best_iterate_or(point, value, grad_norm);
        let band = self.band(best_value);
        let probe_scale = self.window_probe_scale();
        let converged = best_grad_norm.is_finite() && best_grad_norm <= band;
        let accepted = self.accepted;
        if converged {
            self.publish(StallExit {
                point: best_point,
                value: best_value,
                grad_norm: best_grad_norm,
                accepted,
                converged,
                probe_scale,
                unescapable_window: None,
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
                "[STALL] escape streak cut at {}: escape {} reopened a full {}-sample window that \
                 observed the same trials from a bit-identical incumbent (best={:.9e}, |g|={:.3e}); \
                 halting",
                self.stuck_escapes,
                self.stuck_escapes,
                self.window,
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
        let (value, grad_norm, accepted) = (self.best_value, self.best_grad_norm, self.accepted);
        self.publish(StallExit {
            point,
            value,
            grad_norm,
            accepted,
            converged: false,
            probe_scale: None,
            unescapable_window: None,
        });
    }

    /// Defer a filled window to a caller that owns a stronger acceptance test:
    /// reopen the no-improvement window and revoke any convergence the exit
    /// claims, keeping the incumbent and the window evidence.
    pub fn defer_filled_window(&mut self) {
        self.no_improve_streak = 0;
        self.revoke_published_convergence();
    }

    /// Mark the published exit not converged.
    pub fn revoke_published_convergence(&mut self) {
        if let Some(exit) = self.exit.as_mut() {
            exit.converged = false;
            self.exit_revision = self.exit_revision.wrapping_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{StallMonitor, StallVerdict};
    use ndarray::{Array1, array};

    const BAND: f64 = 1e-2;

    fn monitor(window: usize) -> StallMonitor<()> {
        StallMonitor::new(1e-6, window, |_| BAND)
    }

    fn point(x: f64) -> Array1<f64> {
        array![x]
    }

    #[test]
    fn the_first_sample_is_the_incumbent_and_the_window_counts_after_it() {
        let mut stall = monitor(2);
        assert_eq!(
            stall.observe(&point(0.0), 1.0, 0.0, true, None),
            StallVerdict::Continue
        );
        assert_eq!(
            stall.observe(&point(0.1), 1.0, 0.0, true, None),
            StallVerdict::Continue
        );
        assert_eq!(
            stall.observe(&point(0.2), 1.0, 0.0, true, None),
            StallVerdict::Converged
        );
        let exit = stall.exit().expect("published");
        assert!(exit.converged);
        assert_eq!(exit.point, point(0.0));
    }

    #[test]
    fn an_untrusted_sample_is_never_the_incumbent_and_resets_the_window() {
        let mut stall = monitor(2);
        stall.observe(&point(0.0), 1.0, 0.0, true, None);
        stall.observe(&point(0.1), 1.0, 0.0, true, None);
        assert_eq!(
            stall.observe(&point(0.2), 0.5, 0.0, false, None),
            StallVerdict::Continue
        );
        assert_eq!(stall.best_value(), 1.0);
        // Without the untrusted sample the next trusted one would fill the window.
        // It reset the streak, so two more trusted samples are needed.
        assert_eq!(
            stall.observe(&point(0.3), 1.0, 0.0, true, None),
            StallVerdict::Continue
        );
        assert_eq!(
            stall.observe(&point(0.4), 1.0, 0.0, true, None),
            StallVerdict::Converged
        );
    }

    #[test]
    fn an_escape_is_cut_only_when_it_replays_the_same_window_from_a_bit_identical_incumbent() {
        let mut stall = monitor(2);
        stall.observe(&point(0.0), 1.0, 1.0, true, None);
        stall.observe(&point(0.1), 1.0, 1.0, true, None);
        assert!(matches!(
            stall.observe(&point(0.2), 1.0, 1.0, true, None),
            StallVerdict::Escape { .. }
        ));
        assert_eq!(stall.stuck_escapes, 1);
        // The reopened window observes the same trials from the same incumbent:
        // a deterministic procedure replayed from an identical state.
        stall.observe(&point(0.1), 1.0, 1.0, true, None);
        assert!(matches!(
            stall.observe(&point(0.2), 1.0, 1.0, true, None),
            StallVerdict::Floor { .. }
        ));
        assert!(stall.replay_proven());
        assert!(!stall.exit().expect("published").converged);

        // Positive control: a reopened window that observes trials the previous
        // one never saw is a search in a different state, however still its
        // incumbent, and earns another escape instead of the cut.
        let mut exploring = monitor(2);
        exploring.observe(&point(0.0), 1.0, 1.0, true, None);
        exploring.observe(&point(0.1), 1.0, 1.0, true, None);
        assert!(matches!(
            exploring.observe(&point(0.2), 1.0, 1.0, true, None),
            StallVerdict::Escape { .. }
        ));
        exploring.observe(&point(0.3), 1.0, 1.0, true, None);
        assert!(matches!(
            exploring.observe(&point(0.4), 1.0, 1.0, true, None),
            StallVerdict::Escape { .. }
        ));
        assert_eq!(exploring.stuck_escapes, 2);
        assert!(!exploring.replay_proven());

        // Positive control: an incumbent that moved between the windows (a lower
        // value by less than the floor) earns another escape instead of the cut.
        let mut moving = monitor(2);
        moving.observe(&point(0.0), 1.0, 1.0, true, None);
        moving.observe(&point(0.1), 1.0, 1.0, true, None);
        assert!(matches!(
            moving.observe(&point(0.2), 1.0, 1.0, true, None),
            StallVerdict::Escape { .. }
        ));
        moving.observe(&point(0.3), 1.0 - 1e-9, 1.0, true, None);
        assert!(matches!(
            moving.observe(&point(0.4), 1.0 - 1e-9, 1.0, true, None),
            StallVerdict::Escape { .. }
        ));
        assert!(!moving.replay_proven());
    }

    #[test]
    fn a_strict_saddle_incumbent_escapes_until_the_escape_replays() {
        let mut stall = monitor(2);
        stall.observe(&point(0.0), 1.0, 0.0, true, Some(false));
        stall.observe(&point(0.1), 1.0, 0.0, true, Some(false));
        assert_eq!(
            stall.observe(&point(0.2), 1.0, 0.0, true, Some(false)),
            StallVerdict::Continue
        );
        assert!(stall.take_strict_saddle_refusal());
        assert!(!stall.take_strict_saddle_refusal());
        // A window on new trials is the search still exploring the saddle, so it
        // escapes again.
        stall.observe(&point(0.3), 1.0, 0.0, true, Some(false));
        assert_eq!(
            stall.observe(&point(0.4), 1.0, 0.0, true, Some(false)),
            StallVerdict::Continue
        );
        assert!(stall.take_strict_saddle_refusal());
        assert!(!stall.replay_proven());
        // The same trials again from the same incumbent are the replay.
        stall.observe(&point(0.3), 1.0, 0.0, true, Some(false));
        assert_eq!(
            stall.observe(&point(0.4), 1.0, 0.0, true, Some(false)),
            StallVerdict::Converged
        );
        assert!(stall.replay_proven());

        // Positive control: the same trajectory at a PSD incumbent stops at the first
        // filled window.
        let mut minimum = monitor(2);
        minimum.observe(&point(0.0), 1.0, 0.0, true, Some(true));
        minimum.observe(&point(0.1), 1.0, 0.0, true, Some(true));
        assert_eq!(
            minimum.observe(&point(0.2), 1.0, 0.0, true, Some(true)),
            StallVerdict::Converged
        );
    }

    #[test]
    fn a_rejected_trial_joins_the_replay_identity_and_nothing_else() {
        let mut stall = monitor(2);
        stall.observe(&point(0.0), 1.0, 1.0, true, None);
        stall.observe(&point(0.1), 1.0, 1.0, true, None);
        assert!(matches!(
            stall.observe(&point(0.2), 1.0, 1.0, true, None),
            StallVerdict::Escape { .. }
        ));
        let accepted = stall.accepted();
        // A rejected trial between the windows: not a step, but a point the
        // search observed, so the window that follows is not a replay.
        stall.observe_rejected_trial(&point(0.9));
        assert_eq!(stall.accepted(), accepted, "a rejected trial is not a step");
        assert_eq!(
            stall.observe(&point(0.1), 1.0, 1.0, true, None),
            StallVerdict::Continue,
            "a rejected trial does not count toward the window"
        );
        assert!(matches!(
            stall.observe(&point(0.2), 1.0, 1.0, true, None),
            StallVerdict::Escape { .. }
        ));
        assert!(!stall.replay_proven());

        // Its finite value ends a run of refused trials.
        let mut refused = monitor(2);
        refused.observe_seed(&point(0.0), 1.0, 0.0, None);
        refused.observe_refused(&point(9.0));
        refused.observe_rejected_trial(&point(9.5));
        assert_eq!(refused.refused_streak(), 0);
        assert_eq!(refused.observe_refused(&point(9.0)), StallVerdict::Continue);
    }

    #[test]
    fn a_refused_window_halts_at_the_incumbent_and_not_before_one() {
        let mut none = monitor(2);
        assert_eq!(none.observe_refused(&point(9.0)), StallVerdict::Continue);
        assert_eq!(none.observe_refused(&point(9.0)), StallVerdict::Continue);
        assert!(none.exit().is_none());

        let mut stall = monitor(2);
        stall.observe_seed(&point(0.0), 1.0, 0.0, None);
        assert_eq!(stall.observe_refused(&point(9.0)), StallVerdict::Continue);
        assert_eq!(stall.observe_refused(&point(9.0)), StallVerdict::Converged);
        assert_eq!(stall.exit().expect("published").point, point(0.0));

        // Positive control: at a strict-saddle incumbent the refused run continues.
        let mut saddle = monitor(2);
        saddle.observe_seed(&point(0.0), 1.0, 0.0, Some(false));
        saddle.observe_refused(&point(9.0));
        assert_eq!(saddle.observe_refused(&point(9.0)), StallVerdict::Continue);
        assert_eq!(saddle.refused_streak(), 0);
    }

    #[test]
    fn an_all_unescapable_window_stops_without_an_escape_and_a_mixed_one_escapes() {
        let mut pinned = monitor(3);
        pinned.observe_seed(&point(0.0), 1.0, 1.0, None);
        pinned.observe_unescapable_refusal(&point(1.0));
        pinned.observe_unescapable_refusal(&point(1.0));
        assert!(matches!(
            pinned.observe_unescapable_refusal(&point(1.0)),
            StallVerdict::Floor { .. }
        ));
        assert_eq!(pinned.stuck_escapes, 0);
        let exit = pinned.exit().expect("published");
        assert!(!exit.converged);
        let window = exit.unescapable_window.expect("unescapable evidence");
        assert_eq!(window.refused_trials, 3);
        assert_eq!(window.band, BAND);

        // Positive control: one ordinary refusal in the window keeps the escape.
        let mut mixed = monitor(3);
        mixed.observe_seed(&point(0.0), 1.0, 1.0, None);
        mixed.observe_refused(&point(1.0));
        mixed.observe_unescapable_refusal(&point(1.0));
        assert!(matches!(
            mixed.observe_unescapable_refusal(&point(1.0)),
            StallVerdict::Escape { .. }
        ));
        assert_eq!(mixed.stuck_escapes, 1);
        assert!(mixed.exit().expect("snapshot").unescapable_window.is_none());
    }

    #[test]
    fn the_licence_continues_only_on_descent_contraction_or_a_saddle() {
        let mut stall = monitor(2);
        stall.observe_seed(&point(0.0), 1.0, 1.0, None);
        assert!(stall.license_continuation());
        assert!(!stall.license_continuation());
        assert!(stall.unprogressing_stop().is_some());

        let mut contracting = monitor(2);
        contracting.observe_seed(&point(0.0), 1.0, 1.0, None);
        assert!(contracting.license_continuation());
        contracting.observe(&point(0.1), 1.0 - 1e-9, 0.5, true, None);
        assert!(contracting.license_continuation());

        let mut saddle = monitor(2);
        saddle.observe_seed(&point(0.0), 1.0, 1.0, Some(false));
        assert!(saddle.license_continuation());
        assert!(saddle.license_continuation());
    }

    #[test]
    fn the_probe_scale_reports_the_median_difference_and_the_largest_step() {
        let mut stall = monitor(4);
        stall.observe_seed(&point(0.0), 10.0, 1.0, None);
        stall.observe(&point(0.5), 10.0, 1.0, true, None);
        assert!(stall.window_probe_scale().is_none());
        stall.observe(&point(0.7), 10.0 + 3e-3, 1.0, true, None);
        stall.observe(&point(0.8), 10.0 + 1e-3, 1.0, true, None);
        let (noise, radius) = stall.window_probe_scale().expect("three differences");
        assert!((noise - 2e-3).abs() < 1e-12);
        assert!((radius - 0.5).abs() < 1e-12);
    }

    #[test]
    fn a_regressing_constrained_stationary_sample_keeps_the_incumbent() {
        let mut stall = monitor(3);
        stall.observe_seed(&point(0.0), 1.0, 1.0, None);
        assert!(matches!(
            stall.observe_constrained_stationary(&point(5.0), 2.0, 0.0, true, None),
            StallVerdict::Continue
        ));
        assert_eq!(stall.best_point(), Some(&point(0.0)));

        // Positive control: a non-regressing one is adopted and published converged.
        assert_eq!(
            stall.observe_constrained_stationary(&point(5.0), 1.0, 0.0, true, None),
            StallVerdict::Converged
        );
        assert_eq!(stall.exit().expect("published").point, point(5.0));
    }

    #[test]
    fn the_window_keeps_the_latest_steps_solver_state() {
        let step = |sigma: f64, alpha: f64| crate::StepInfo {
            iter: 0,
            step_norm: alpha,
            predicted_decrease: f64::NAN,
            actual_decrease: 0.0,
            trust_radius: None,
            regularization: Some(sigma),
            line_search_step: Some(alpha),
        };
        let mut stall = monitor(2);
        stall.observe_step(&step(1.0, 0.5));
        stall.observe_step(&step(2.0, 0.25));
        // Positive control: before a third step, the oldest is still held.
        assert_eq!(
            stall
                .recent_steps()
                .next()
                .and_then(|kept| kept.regularization),
            Some(1.0)
        );
        stall.observe_step(&step(4.0, 0.125));
        let kept: Vec<_> = stall.recent_steps().copied().collect();
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].regularization, Some(2.0));
        assert_eq!(kept[1].regularization, Some(4.0));
        assert_eq!(kept[1].line_search_step, Some(0.125));
        assert_eq!(kept[1].step_norm, 0.125);
    }
}
