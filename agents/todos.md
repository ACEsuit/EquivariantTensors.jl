# Repo TODO sweep & catalogue

Working notes (2026-06-14). A pass over every `TODO/FIXME/HACK` in `src/` plus
the design questions in `agents/co_notes_for_later.md`, so they stop being
re-discovered. The handful of safe, conflict-free items were fixed in the same
PR; the rest are classified with a recommendation. **`groups/O3/` + the sparse
carrier are deliberately left untouched** — the CP/TRACE work lives there and a
separate agent owns it.

## Fixed in this sweep

- `src/pooling/sparseprodpool.jl:464` — `a += bR*bY` → `a = muladd(bR, bY, a)`
  (the TODO asked for it; FMA in the pooling pullback CPU loop).
- `src/groups/O3/yyvector.jl:60` — stale `# TODO: @boundscheck /
  @propagate_inbounds`: already implemented on the next lines
  (`@propagate_inbounds` + `@boundscheck checkbounds`); comment removed.

## Defer → CP / carrier agent (`groups/O3/`, `symmop`, sparse carrier)

These sit in the code the CP/TRACE agent is actively working; left as-is to
avoid conflicts. Most are type-stability / notation-consistency cleanups:
- `src/groups/O3/O3.jl:412` — `SetLl` not type stable (removable once `!PI`
  uses the new method).
- `src/groups/O3/O3.jl:464`, `:479` — `coupling_coeffs` vs `mm_generate` return
  different `MM` formats; may need unifying.
- `src/groups/O3/O3.jl:643`, `:867`, `:868` — notation (`mm`/`μμ`, `K` vs `L`
  both = equivariance order) + whether `ll`/`nn` should be `SVector{N,Int}`.
- `src/groups/O3/O3_utils.jl:47` — short `Ctran` variant possibly type-unstable.
- `src/groups/O3/O3_transformations.jl:2` — transforms should use static sparse
  arrays (perf).
- `src/groups/symmop.jl:45` — `HACK TO DISTINGUISH L=0 and L>0` (possible
  type-stability issue).
- `src/formats/sparse/sparse_ace_basis.jl:165` — `∂BB` eltype check (subtle, AD
  + SVectors).
- `src/formats/sparse/sparse_ace_ka.jl:29` — "undo the double-transpose!!!" in
  the carrier eval (perf/clarity; touches a hot path).
- `src/formats/sparse/sparse_ace_layer.jl:43` — empty TODO; the note is that
  Zygote can't pull back through the broadcasted `𝔹 .* WLL`, hence the manual
  `_tupmul` rrule. Known Zygote limitation; leave.

## Needs a decision (design / refactor)

- `src/utils/selectlinl.jl:30` — the iterate-vs-not case distinction is a hack;
  flagged "an argument to move to DecoratedParticles.jl" → ties into the EmbedDP/
  DP-reliance question in `co_notes`.
- `src/utils/selectlinl.jl:51` — now that the type-unstable-selector bug is
  fixed, maybe revert to the simpler impl and drop the custom rrule. Worth a try
  but touches a tested AD path — defer to a focused PR.
- `src/utils/selector.jl:6` — optional sorted-categories variant for large
  category sets (future feature).

## Real issue to confirm (flagged, not fixed)

- `src/formats/sparse/sparse_ace_utils.jl:129-142` — `_auto_Ylm_spec` calls
  `_get_natural_Ylm_spec`, but that function's **definition is commented out**
  (the `# # TODO: not clear this should be here?` block). So `_auto_Ylm_spec`
  would error if ever hit. Likely a dead path (callers pass `Ylm_spec`
  explicitly) — carrier owner should confirm and remove `_auto_Ylm_spec` or
  restore the definition.

## Dormant (not `include`d)

- `src/formats/sparse/symmprod_dag.jl:60`, `:214`,
  `symmprod_dag_kernels.jl:217` — the DAG symmetric-product path is dormant in
  `src/` (and its test is in `test/dormant/`). Revive or retire together; see
  `agents/tests.md`.

## `co_notes_for_later.md` design questions (findings)

- **"where is `setproduct` used?"** — answered: defined in
  `src/utils/setproduct.jl` (included via `EquivariantTensors.jl`) and used once,
  in `src/formats/sparse/sparse_ace_utils.jl::_auto_nnllmm_spec`. Live,
  single-consumer.
- **EmbedDP / `dp_transform` — really in ET? how much relies on DP?** Open. The
  structs are in core; their evaluation/diff methods are in
  `ext/DecoratedParticlesExt.jl` (per restructure.md §5.1). The `selectlinl.jl:30`
  TODO echoes this. Needs an audit + a rename/relocate decision.
- **`EdgeEmbed` per-edge + auto-broadcast, move to `graphs/`?** Open design Q.
- **pooling → embedding (message op vs system embedding)?** Open design Q.
- **which utils → `lib/ETUtils`?** Open; pairs with the EmbedDP/DP audit.
- **"go through all TODO notes"** — this file.
