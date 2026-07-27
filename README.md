# opt

[![Build Status](https://github.com/SauersML/opt/actions/workflows/test.yml/badge.svg)](https://github.com/SauersML/opt/actions)

`opt` is a Rust library for dense nonlinear optimization: BFGS, Newton trust-region,
ARC, matrix-free trust-region, and fixed-point iteration, plus the shared policy
primitives (trust-region radius control, Levenberg-Marquardt damping, ridge
escalation, backtracking and bracketed root-finding) the solvers are built from.

### On `wolfe_bfgs`

`wolfe_bfgs` was a second crate in this repository that re-exported `opt`'s
BFGS-focused API under a narrower name. It has been removed. It carried no code of
its own — only a `pub use` list — and that list had to be kept in lockstep with
`opt`'s public surface by hand. It fell behind exactly as you would expect: the
0.6.0 release broke the workspace build on all three platforms because the
re-export crate still required `opt = "^0.5.15"`, and crates.io accepted the `opt`
publish anyway because it builds each package independently. A crate whose only
content is a list that can silently go stale is a liability, not a convenience.

Versions already on crates.io (up to `wolfe_bfgs` 0.7.0) are left published and
keep working. New code should depend on `opt` directly; every name `wolfe_bfgs`
exported is available from `opt` under the same path.

## Repository layout

```text
.
├── Cargo.toml
├── README.md
├── opt
│   ├── Cargo.toml
│   ├── README.md
│   ├── optimization_harness.py
│   └── src
│       └── lib.rs
└── .github
    └── workflows
```

## Common commands

Run the full workspace tests:

```bash
cargo test --workspace
```

Run only the main crate tests:

```bash
cargo test -p opt
```

Publish:

```bash
cargo publish -p opt
```

The publish workflow in [`.github/workflows/publish.yml`](./.github/workflows/publish.yml)
does the same on a `crate-*` tag, skipping the publish when the version is already
on crates.io.
