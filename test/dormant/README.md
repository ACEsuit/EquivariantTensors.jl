# Dormant tests

Tests not included in `runtests.jl`. The 2026-06-12 batch was triaged and
mostly resolved in the test-suite cleanup (2026-06-14; see `agents/tests.md`):
the dead/scratch files were deleted, and the end-to-end model + CPU-vs-GPU
consistency coverage (`test_ace_ka2_new.jl`, `test_lux_models.jl`) was folded
into the active `test/acemodels/test_model.jl` on the promoted helpers
`test/test_utils/{luxtestmodels,diffutils}.jl`. `test_embed.jl` was removed (it
was built on the removed `NTtransform`); its forward `SelectLinL` coverage
already exists in active `test/embed/test_transform.jl`, but its `SelectLinL`
finite-diff (input-gradient) check is **not yet re-ported** — see the TODO in
`test/embed/test_transform.jl` / `agents/tests.md`.

What remains here:

- `test_forwarddiff.jl` — real `frule`/Dual tests for `PooledSparseProduct` /
  `SparseSymmProd`. Valid coverage of the forward-mode path; not yet wired into
  the active suite. **TODO:** salvage the `frule` assertions into the active
  pooling / sparse tests, then delete.
- `test_sparsesymmproddag.jl` — tests the `symmprod_dag` source, which is itself
  dormant (not `include`d) in `src/`. Parked here for consistency; revive
  together with the DAG source if/when it is wired back in. The DAG source is
  **not** being retired.
