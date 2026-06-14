# ET weight initializers — design notes & required future work

Working notes (2026-06-14). Context: removing the `Lux` hard dependency from
ET core required replacing `Lux.glorot_normal` in the sparse readout
(`sparse_ace_layer.jl`). That prompted the broader question of whether ET
should own a small initializer library tailored to equivariant *tensor*
formats (analogous to `WeightInitializers.jl`, but format-aware).

## What landed (PR `restruct_rm_lux`)

- `src/utils/initializers.jl`: the **leaf samplers** `et_zeros`, `et_normal`
  (`Random`-only, `WeightInitializers`-style `init(rng, [T], dims...)`
  signature so they drop into LuxCore/Lux `init_*` kwargs and
  `initialparameters`). Dumb samplers only — no format knowledge.
- `SparseACElayer.initialparameters` now uses `et_normal` with an **interim**
  fan-in scaling `σ = 1/√lens[i]` (linear readout ⇒ keeps untrained output
  O(1)). This is a placeholder for the format-aware policy below.
- `Lux` dropped from ET core deps (now LuxCore-only); kept as a *test* extra
  (active tests build `Chain`/`Parallel`/`WrappedFunction` models).

## Why ET needs its own initializers (the core idea)

Glorot/He exist to stabilise variance across an **additive** sum of `fan_in`
terms through stacked nonlinear layers, and to break symmetry between identical
hidden units. ET tensor formats are different: they are **multiplicative /
contractive** in their parameters, so the variance-propagation law is
geometric, not `1/fan`. The contract an ET initializer should satisfy is:

> choose parameters so the represented coefficient tensor `c` (equivalently the
> untrained output `F(X)` over a reference input distribution) has a target
> norm/variance — with the per-parameter scale derived from the *format's
> contraction structure*.

Precedent: this is exactly `e3nn`'s "path normalization" (normalise each
tensor-product path to unit variance); ET should adopt the same philosophy,
specialised per format.

## Required future work

1. **Format-aware policy (Layer 2).** A small policy object the format
   consumes, e.g.
   ```julia
   struct ETInit{F}; leaf::F; target_std::Float64; end   # target_std = 0 ⇒ zero init
   ```
   Each format's `initialparameters` maps `target_std → per-parameter σ` via its
   structure. Scaling laws to implement:
   - **sparse linear readout**: `σ = target/√fan_in` (the interim choice above).
   - **CP / TRACE** (rank `R`, correlation order `N`): output ≈ `Σ_r Π_n f_{rn}`
     ⇒ `Var ~ R·σ^{2N}` ⇒ `σ ~ (target/√R)^{1/N}` — the **Nth-root law**.
   - **TT** (chain length `L`, bond dims `r`): per-core variance `~ target^{1/L}`
     modulated by the bond dims (contracted-chain norm control).
   - **Tucker**: semi-orthogonal factor matrices (QR/HOSVD-style); scale carried
     by the core.
2. **Irrep-awareness.** Equivariant-linear inits (sparse `W`, CP channel-mixing
   `Ā_klm = Σ_n W_kn A_nlm`) act on the multiplicity index `n` only (Schur), as
   identity on `(l,m)`. Build per-`l` blocks and pull irrep dims from `groups/`
   rather than inlining `2l+1` (matches restructure.md §9).
3. **Eltype (policy, CO).** Parameters are **`Float64` by default for all
   formats**, dropping to `Float32` (or other) **only when the user explicitly
   asks** — e.g. for GPU. The leaf samplers already follow this (default
   `Float64`, optional `T`), so `WLL` is now `Float64` (was incidentally
   `Float32` via the `WeightInitializers` default). Remaining work: expose the
   precision as a user-facing option on the format constructors /
   `initialparameters` (a `T = Float64` kwarg threaded to the leaf samplers) —
   an explicit, single switch, *not* auto-derived from the model.
4. **Empirical (data-dependent) escape hatch.** An LSUV-style
   `data_init!(rng, layer, Xbatch; target_std)` that forward-passes a batch,
   measures output variance and rescales — insurance where the analytic law is
   awkward (general carriers), and a correctness check against the analytic `σ`.
5. **API surface.** Decide exports (currently unexported `et_*`), the curried /
   `default_rng` Lux-style forms, and whether `et_glorot` is provided for
   MLP-parity. Keep `WeightInitializers`-compatible signatures so formats accept
   an `init` kwarg and compose with the Lux ecosystem rather than forking it.

These should land **with the CP format** (the first multiplicative format),
where the Nth-root law is actually needed and unit-testable
("untrained CP output std ≈ target across N, R"), rather than abstracted ahead
of a concrete consumer.
