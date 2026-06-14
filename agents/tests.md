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
4. **Complex path is fragile and its future is undecided:**
   `acemodels/test_sparse_ace_cplx.jl` flags its Zygote gradient as "fails
   currently", and whether ET maintains a complex path at all is an **open
   decision (CO, to make soon)** — so the test is *parked to `dormant/`* rather
   than fixed in place (see below).

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

**Active tests to park in `dormant/`** (move the test only — no `src/` change):
- `test/formats/sparse/test_sparsesymmproddag.jl` — in the active tree but *not*
  in `runtests.jl` (its `symmprod_dag` source is dormant in `src/`). Move it to
  `dormant/` for consistency. **Keep the `symmprod_dag` source as-is** — the DAG
  path is *not* being retired now; only the test is parked.
  *(Complex stays active — CO decided to maintain complex for now;
  `test_sparse_ace_cplx.jl` is not parked. If its Zygote gradient is broken,
  `@test_broken` it rather than remove it.)*

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

(The complex path is *parked*, not added — see "Active tests to park" above and
the open decision below.)

## Infrastructure

- **Automatic backend detection in `utils_gpu.jl`.** The current helper probes
  `CUDA.functional()` → `AMDGPU.functional()` → `Metal.functional()`, which
  requires *all* backend packages to be loaded first (the reason `CUDA`+`Metal`
  are heavy default test extras). Replace it with a **system-level probe that
  picks the backend *before* loading any GPU package**, with a `TEST_BACKEND`
  override:
  ```julia
  function detect_gpu_backend()
      haskey(ENV, "TEST_BACKEND") && return ENV["TEST_BACKEND"]   # manual override
      if Sys.isapple() && Sys.ARCH == :aarch64
          return "Metal"
      elseif !isnothing(Sys.which("nvidia-smi")) && success(`nvidia-smi`)
          return "CUDA"
      elseif !isnothing(Sys.which("rocm-smi")) || isdir("/dev/kfd")
          return "AMDGPU"
      elseif !isnothing(Sys.which("sycl-ls"))   # crude oneAPI probe
          return "oneAPI"
      else
          return "CPU"
      end
  end
  ```
  When a GPU is detected, **install the matching backend into the (sandboxed)
  test env and use it** — `Pkg.add(backend)`, `@eval using …`, set the
  `gpu`/`dev` *transfer function* (recursive, Adapt/Functors-aware — moves ps/st
  NamedTuples and ETGraph, unlike a bare array type) — wrapped in `try … catch`
  so a detected-but-unusable backend degrades to CPU (`dev = identity`) with a
  warning rather than failing the suite:
  ```julia
  const backend = detect_gpu_backend()
  if backend != "CPU"
      Pkg.add(backend)            # into the sandboxed test env only
      @eval using $(Symbol(backend))
      # set gpu/dev = the transfer function (cu / mtl / gpu_device() / …)
  end
  ```
  Also expose `gpu_supports_f64` (false on F32-only backends, e.g. Metal): the
  GPU-consistency tests run **F32 always and F64 only when supported**. (`Pkg`
  must be a test dep for the `Pkg.add`.)
  Consequence: the **default CI runner resolves to `"CPU"`** (no `nvidia-smi`,
  not apple-aarch64) and installs no GPU package — so `CUDA`/`Metal`/`AMDGPU`/
  `oneAPI` are **dropped from the default test deps**, giving a fast, lean
  `Pkg.test()`; a GPU machine / lane detects its hardware and *installs* the one
  backend on the fly. The KA-on-CPU coverage in the active suite is unaffected
  (it never needed a GPU package).
  Notes on the probe: it spawns `nvidia-smi` at load (cheap, fine); Apple-Intel
  Macs (`x86_64`) fall through to `"CPU"` — relax to `Sys.isapple()` if
  Metal-on-Intel matters; the `AMDGPU`/`oneAPI` probes are heuristic and can be
  hardened when those backends are actually exercised.
- Update `test/dormant/README.md` after the triage (remove deleted entries; note
  what was promoted/merged).
- Keep the test tree mirroring `src/`; new files land in the matching subdir.

## Known bugs surfaced by this work

- **GPU full-model position gradient — Metal-specific.** The original report
  (`test_ace_ka2_new.jl`, `@error("This test currently fails!")`) was on Metal.
  **Verified on a CUDA A100 (2026-06-14): the CPU-vs-GPU position gradient is
  correct in both F32 and F64** (forward + parameter gradients too). So
  `test_model.jl` runs it as a real `@test` everywhere and `@test_broken` only
  on Metal. Fixing the Metal reverse path is its own task (needs Metal hardware).
- **Complex Zygote gradient** flagged "fails currently" in
  `test_sparse_ace_cplx.jl`. Parked to `dormant/` pending the maintain-complex
  decision (below), not fixed or `@test_broken` in the active suite.

## Decisions (CO, 2026-06-14)

- **Maintain a complex path?** → **yes, for now.** So
  `test_sparse_ace_cplx.jl` *stays active* (not parked); if its Zygote
  gradient turns out broken, `@test_broken` it (since complex is maintained)
  rather than removing it.
- **Drop the GPU backend packages from the default test deps?** → **yes.**
  `CUDA`/`Metal` removed from `[extras]` + the test target; `detect_gpu_backend()`
  resolves the default lane to `"CPU"` (loads no backend), and a GPU lane adds
  its one backend (`TEST_BACKEND=…` + `Pkg.add`).

## Sequencing

1. Triage deletions + move the two active tests to park
   (`test_sparse_ace_cplx`, `test_sparsesymmproddag`) + promote `LTM`/`DIFF`
   helpers (pure bookkeeping, no risk; `src/` untouched).
2. Add the **initializers** unit test and the **end-to-end CPU model test**
   (closes gaps 1–3 on CPU) — this is the guardrail the format-interface refactor
   needs.
3. Salvage the `SelectLinL` fdtest and the `frule` tests into active files.
4. Rework `utils_gpu.jl` to `detect_gpu_backend()`, add the **GPU-gated**
   consistency test with the position-gradient bug as `@test_broken`.
5. (Separately, after the deps decision) trim the GPU backends from the default
   lane and set up the GPU CI lane.

Steps 1–2 are the immediate prerequisite for the P1 format-interface work; 3–5
can follow.
