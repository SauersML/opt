//! Thin BFGS-only re-export crate layered on top of `opt`.

pub use opt::{
    AcceptedStep, BacktrackConfig, Bfgs, BfgsError, Bounds, BoundsError, ConfigError,
    CostStallConfig, FiniteDiffGradient, FirstOrderObjective, FirstOrderSample, FusedObjective,
    LineSearchFailureReason, MaxIterations, ObjectiveEvalError, ObjectiveEvalKind,
    OptimizationStatus, Problem, Profile, RidgeExhausted, RidgeSchedule, RidgeSuccess, Solution,
    StationarityEvidence, StationarityKind, StationarityNorm, StationarityScaling,
    TerminationReason, Tolerance, ZerothOrderObjective, armijo_roundoff_cushion,
    backtracking_line_search, constants, degenerate_trust_radius, escalate_ridge, optimize,
};
