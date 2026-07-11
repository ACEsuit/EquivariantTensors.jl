# Dormant tests

Tests not included in `runtests.jl`. The 2026-06-12 batch was triaged and
mostly resolved in the test-suite cleanup (2026-06-14; see `agents/tests.md`):
the dead/scratch files were deleted, and the end-to-end model + CPU-vs-GPU
consistency coverage (`test_ace_ka2_new.jl`, `test_lux_models.jl`) was folded
into the active `test/acemodels/test_model.jl` on the promoted helpers
`test/test_utils/{luxtestmodels,diffutils}.jl`. `test_embed.jl` was removed (it
was built on the removed `NTtransform`). Its `SelectLinL` coverage has now been
**re-ported** (2026-06-15): a standalone unit suite in
`test/utils/test_selectlinl.jl` plus an EmbedDP∘SelectLinL gradient/jacobian
test in `test/embed/test_transform.jl`. Its `BranchLayer(Rnl, Ylm)` parallel-
embedding test was *not* re-ported — that pattern has no active coverage and is
tracked as a follow-up in `agents/tests.md`.

What remains here:

- `test_forwarddiff.jl` — real `frule`/Dual tests for `PooledSparseProduct` /
  `SparseSymmProd`. Valid coverage of the forward-mode path. **Parked for a
  dedicated PR** that thoroughly analyses forward rules & jacobians; salvage the
  `frule` assertions there rather than piecemeal.
- `test_sparsesymmproddag.jl` — tests the `symmprod_dag` source, which is itself
  dormant (not `include`d) in `src/`. Parked here for consistency; revive
  together with the DAG source if/when it is wired back in. The DAG source is
  **not** being retired.
