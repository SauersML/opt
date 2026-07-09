//! Thin BFGS-only re-export crate layered on top of `opt`.

pub use opt::{
    AcceptedStep, BacktrackConfig, Bfgs, BfgsError, Bounds, BoundsError, ConfigError,
    FiniteDiffGradient, FirstOrderObjective, FirstOrderSample, FusedObjective,
    LineSearchFailureReason, MaxIterations, ObjectiveEvalError, Problem, Profile, RidgeExhausted,
    RidgeSchedule, RidgeSuccess, Solution, Tolerance, ZerothOrderObjective,
    armijo_roundoff_cushion, backtracking_line_search, constants, escalate_ridge, optimize,
};
