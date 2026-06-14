# Test suite — cleanup & expansion plan

Working notes (2026-06-14). The restructure left the test tree reorganised to
mirror `src/` (`test/{utils,embed,pooling,formats/sparse,groups/O3,acemodels,
graphs}/`) with a `test/dormant/` holding parked tests. Before the new-format
work begins, the suite should be cleaned up and the coverage holes closed —
the upcoming A-storage / format-I/O-contract refactors are *behaviour-
preserving* and need a trustworthy guardrail.

This is **P0** of the post-restructure priorities (see the chat synthesis and
`restructure.md` §10 test-suite-cleanup item).

## Current state

**Active suite (`runtests.jl`) — solid at the unit level:**
- `utils/` (setproduct, invmap, static_prod), `embed/` (EmbedDP + SelectLinL via
  `test_transform.jl`; EmbedDP+PState via `test_decoratedparticles.jl`),
  `pooling/` (`PooledSparseProduct` incl. rrule + KA/GPU),
  `formats/sparse/` (`SparseSymmProd` incl. rrule + KA/GPU; `sparsemat_ka`),
  `groups/O3/` (8 tests — coupling, representation, quad_O3, …),
  `acemodels/` (`test_sparse_ace[_cplx]` = parameter-gradient FD-vs-Zygote on the
  raw `𝔹basis` + a `DotL` readout; `test_ace_ka` = position-gradient
  FD-vs-Zygote-vs-manual + basis Jacobian), `graphs/` (NeighbourListsExt).
- GPU is handled by `test/test_utils/utils_gpu.jl`, which sets `dev = identity`
  when no GPU is present — so KA/GPU tests run as CPU no-ops on CI.

**Coverage gaps (the important part):**
1. **`SparseACElayer` is never instantiated in the active suite.** The three
   `acemodels/` tests use the raw `sparse_equivariant_tensors` basis with a
   custom `DotL` readout, *not* the layer. So the layer's `initialparameters`
   (the `et_normal` init added in #121), `evaluate`, and the `_tupmul` /
   `_mul_scal` rrules are **untested**. This is the highest-value hole.
2. **`et_zeros` / `et_normal`** (`utils/initializers.jl`) have no direct unit
   test — only implicit use inside the (untested) layer.
3. **No end-to-end *model* test and no end-to-end GPU consistency** in the
   active suite — GPU coverage stops at the kernel level. The only full-model
   CPU-vs-GPU consistency lives in `dormant/` (`test_ace_ka2_new.jl`,
   `test_lux_models.jl`).
4. **Known-fragile complex path:** `acemodels/test_sparse_ace_cplx.jl` carries an
   in-file note that the Zygote gradient "fails currently" — resolve or mark.

## Dormant-folder triage (final decisions)

| File | Decision | Why |
|------|----------|-----|
| **`test_ace_ka2_new.jl`** | **REVIVE (split)** | The only end-to-end model test: builds `embed → SparseACElayer → readout`, checks CPU forward, Zygote-vs-FD on **positions and parameters**, and CPU-vs-GPU (Metal). Closes gaps 1 & 3. Blocked only by hard-coded `using Metal`, debug `@show`/`@error`, and a genuinely failing GPU position-gradient. |
| `test_lux_models.jl` (+ `luxtestmodels.jl` `LTM`, `diffutils.jl` `DIFF`) | **REVIVE → fold in** | Systematic CPU-vs-GPU consistency over 3 model variants (scalar / vector / mixed), backend-agnostic via `dev()`. Overlaps `test_ace_ka2_new`; **merge both into one** active model test built on the `LTM`+`DIFF` helpers. |
| `test_forwarddiff.jl` | **SALVAGE frules → else delete** | Real `frule`/Dual tests for `PooledSparseProduct` / `SparseSymmProd`; forward-mode is a supported path. Move the `frule` assertions into the active pooling / sparse tests (or a `test_frules.jl`); drop the rest. |
| `test_embed.jl` | **SALVAGE SelectLinL → delete** | The only direct `SelectLinL` finite-diff test, but uses a non-existent `NTtransform`; `EmbedDP`+`SelectLinL` is already in active `embed/test_transform.jl`. Lift the `SelectLinL` fdtest into an active `utils/test_selectlinl.jl` (or extend `test_transform.jl`); delete. |
| `test_ace_ka2.jl` | **DELETE** | Uses removed `ParallelEmbed` / `TransformedBasis`; superseded by `test_ace_ka2_new`. |
| `test_ace_basis.jl` | **DELETE** | One assertion, P4ML-focused, redundant with `test_forwarddiff`. |
| `test_acemodel.jl` | **DELETE** | No `@test`; uses obsolete `EdgeEmbed1` / `DPTransform` API. |
| `test_O3_scratch.jl` | **DELETE** | Scratch, no assertions. |
| `test_prodpool_mult.jl` | **DELETE** | Experimental "not supported yet"; calls obsolete `ACEcore`. |
| `luxtestmodels.jl`, `diffutils.jl` | **PROMOTE to `test/test_utils/`** | Working helpers for the revived model test (`LTM.build_model`, `DIFF.grad_zy/grad_fd[_ps]`). |

**Orphan:** `test/formats/sparse/test_sparsesymmproddag.jl` is in the active tree
but *not* in `runtests.jl` (its `symmprod_dag` source is dormant). **Move it to
`dormant/`** for consistency (or delete with `symmprod_dag` if that path is
being retired).

## New / expanded coverage to add

1. **End-to-end model test** — `test/acemodels/test_model_consistency.jl`
   (active, CPU lane): build `embed → SparseACElayer → readout` via the promoted
   `LTM` helper, for the scalar / vector / mixed output variants. Assert: forward
   runs; `DIFF.grad_zy` ≈ `DIFF.grad_fd` w.r.t. **positions**; and
   `grad_zy_ps` ≈ `grad_fd_ps` w.r.t. **parameters** (this exercises the
   `SparseACElayer` `WLL` init + `_tupmul`/`_mul_scal` rrules). Closes gaps 1 & 3
   on CPU.
2. **GPU consistency** — same models, gated on `dev` from `utils_gpu.jl` so it is
   a CPU no-op on CI and a real check on a GPU box: forward CPU≈GPU, and
   parameter-gradient CPU≈GPU. The **failing GPU position-gradient** from
   `test_ace_ka2_new` (CPU vs Metal disagree) becomes a `@test_broken` with a
   tracking note — surfaced, not hidden (see "Known bugs" below).
3. **Initializers unit test** — `test/utils/test_initializers.jl`: `et_zeros`/
   `et_normal` shape; default eltype `Float64`; explicit `Float32`; `σ` scaling
   (sample variance ≈ `σ²`); `rng` determinism. Closes gap 2.
4. **`SparseACElayer` focused unit test** (optional if 1 is thorough): the
   `_mul_scal` / `_tupmul` rrules directly via finite differences, independent of
   a full model.
5. **Complex path** — resolve gap 4: confirm whether
   `test_sparse_ace_cplx` Zygote actually passes; if not, `@test_broken` + a
   tracked issue so the suite is honest.

## Infrastructure

- **GPU CI lane.** `CUDA` + `Metal` are heavy default test extras (slow
  instantiate; can't actually execute on CI). The `dev=identity` fallback means
  GPU tests already run trivially on CPU, but the deps are still
  installed/precompiled every `Pkg.test()`. Consider an env-gated GPU group
  (e.g. `ET_TEST_GPU=cuda|metal`) and a separate CI lane, keeping the default
  lane lean while preserving KA-on-CPU coverage (the P4ML `ka=true`-on-CPU
  pattern). Decision to confirm with CO.
- Update `test/dormant/README.md` after the triage (remove deleted entries; note
  what was promoted/merged).
- Keep the test tree mirroring `src/`; new files land in the matching subdir.

## Known bugs surfaced by this work

- **GPU full-model position gradient is wrong.** In `test_ace_ka2_new.jl` the
  Zygote gradient of a full model w.r.t. edge positions disagrees between CPU and
  Metal (`@error("This test currently fails!")`). This is a real correctness bug
  in the GPU reverse path, currently hidden by the test being dormant. Reviving
  the test as `@test_broken` makes it visible; fixing it is its own task.
- **Complex Zygote gradient** flagged as "fails currently" in
  `test_sparse_ace_cplx.jl` — verify and track.

## Sequencing

1. Triage deletions + move the orphan + promote `LTM`/`DIFF` helpers (pure
   bookkeeping, no risk).
2. Add the **initializers** unit test and the **end-to-end CPU model test**
   (closes gaps 1–3 on CPU) — this is the guardrail the format-interface refactor
   needs.
3. Salvage the `SelectLinL` fdtest and the `frule` tests into active files.
4. Add the **GPU-gated** consistency test with the position-gradient bug as
   `@test_broken`; resolve/track the complex-Zygote flag.
5. (Separately) decide the GPU CI lane.

Steps 1–2 are the immediate prerequisite for the P1 format-interface work; 3–5
can follow.
