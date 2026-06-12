# Dormant tests

Tests not included in `runtests.jl`, parked here pending careful
analysis before deletion (2026-06-12, during the restructure; see
`agents/restructure.md`).

- `test_ace_ka2_new.jl` — **keep**: newest iteration of the KA model
  test (hard-codes Metal, needs adapting before it can run in CI).
- `test_ace_ka2.jl` — predecessor of the above; was commented out of
  runtests "temporarily".
- `test_ace_basis.jl` — references `ET.EdgeEmbed1`, which no longer
  exists in src.
- `test_acemodel.jl` — pure Polynomials4ML test, no ET references.
- `test_O3_scratch.jl` — scratch file, no `@test`.
- `test_prodpool_mult.jl` — experimental batched pooled products,
  per its header "not really supported yet".
- `test_forwarddiff.jl` — per its header, duplicates the individual
  layer tests; the frule tests might be the only ones worth keeping.
- `test_agnesi.jl` — real coverage of the agnesi transform (incl.
  species); **goes with the upcoming agnesi/chemistry move to
  lib/ACEradials**, reactivate there.
- `test_lux_models.jl` (+ helpers `luxtestmodels.jl`, `diffutils.jl`)
  — **review**: only CPU-vs-GPU consistency tests for full Lux models
  incl. parameter gradients; this coverage exists nowhere else.
- `test_embed.jl` — transformed embedding layers + the only direct
  `SelectLinL` fdtest; partially overlaps test_splines.jl. **Revisit
  before finalizing the restructure.**
