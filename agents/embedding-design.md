# Embedding / transforms / pooling — design analysis

Working notes (2026-06-14). CO asked for a deeper look at three `co_notes`
questions about the embed/transform/pooling boundary. The recurring tension:
**simplify the layering and naming while keeping the one piece of functionality
that justifies these layers — differentiation through a structured particle
state (the jacobian path).** Where I'm confident I give a recommendation; where
not, it lands in the "Decisions to make" list at the end.

## The code as it actually is (grounded)

The pipeline is `edge states → [trans → basis → post] → reshape → pool → A`.

- **`DPTransform`** (`src/transforms/decpart.jl`, ~30 lines) — a struct holding
  just `f` + `refstate`. **DP-agnostic.** All DP coupling is in
  `ext/DecoratedParticlesExt.jl` (~50 lines): the forward dispatches on
  `XState`, and `evaluate_ed`/`_pb_ed` use `DecoratedParticles.grad_fd` to
  differentiate `f` w.r.t. the particle input (returning a `VState` tangent),
  plus `float32`/`float64` for `XState`.
- **`EmbedDP`** (`src/embed/embeddings.jl`) — a 3-stage container
  (`trans, basis, post`) that is essentially a Lux `Chain`, **plus** a custom
  `evaluate_ed` that differentiates through the (particle-state) input by
  pulling the basis derivative back through the transform (`_pb_ed`). Its
  docstring says this input-differentiation "is e.g. needed for jacobians" — it
  *is* the reason the layer exists rather than a plain `Chain`.
- **`EdgeEmbed`** (`src/embed/embeddings.jl`) — applies `l.layer` to
  `X.edge_data` and `reshape_embedding`s the result into the graph-aware
  3-tensor (and `evaluate_ed` for jacobians). It is the **adapter between a
  per-edge embedding and the `ETGraph`**; `reshape_embedding` is graph machinery.
  It is itself DP-free.
- **Pooling** (`PooledSparseProduct`, `src/pooling/`) — the edge→node
  aggregation `A_i = Σ_j ∏_t φ_t(x_j)`, consuming the reshaped 3-tensor.
- **Consumers** of the `DP*` names: `lib/ACEradials` (`agnesi_dp.jl`,
  `transsplines.jl`) — so renames are breaking and need coordinating there.

**Bottom line for Q1:** the *DP-reliance is small and already isolated in the
extension* (`grad_fd` for input-differentiation + `XState` dispatch). The
structs belong in ET core (they're LuxCore-only); the only thing that "is DP" is
*how you differentiate w.r.t. a particle state*, which is genuinely DP's domain.

## Q1 — `EmbedDP` / `dp_transform`: keep in ET? rename?

The structs should **stay in ET** (this matches restructure.md §5's decision:
EmbedDP/decpart as a DP-*triggered extension*, not a lib). The real question is
**naming**, because "DP" overclaims: the layers are general; only the
input-jacobian uses DP.

Options:
- **(A) Status quo.** Names keep `DP`. Pro: zero churn; the `DP` flags the
  extension dependency. Con: misleading — implies DP-specificity the structs
  don't have; mild barrier to non-DP users.
- **(B) Rename to a *state/particle* vocabulary** (recommended direction):
  `DPTransform → StateTransform` (or `ParticleTransform`), `dp_transform →
  state_transform`, `EmbedDP → StateEmbed` (or `ParticleEmbed`). Keeps the true
  semantics — *embed a particle/state and differentiate w.r.t. it* — without
  binding the name to one package. DP remains the provider of that
  differentiation (via the ext). Cost: breaking rename; keep `dp_transform`/
  `EmbedDP` as deprecated aliases for one cycle and update ACEradials.
- **(C) Generalise away the layer.** Strip the input-diff and let `EmbedDP` be a
  plain `Chain`. Rejected: that *removes* the jacobian capability, which is the
  whole point.

**Recommendation:** (B), but it's a naming/breaking-change call, so it's a
"decision" not a silent fix. If we do nothing, at least drop the misleading
implication in the docstrings (note the structs are general; DP only provides
input-differentiation).

## Q2 — per-edge `EdgeEmbed` + auto-broadcast; move to `graphs/`?

`EdgeEmbed` is doing two separable jobs: **(i)** "embed an edge" (general,
state→vector) and **(ii)** "apply that over a graph's edges and reshape into the
graph tensor" (graph machinery — `reshape_embedding`). CO's instinct (a simpler
per-edge version that auto-broadcasts, living in `graphs/`) is exactly this
split.

Options:
- **(A) Status quo** — one `EdgeEmbed` in `embed/` that takes the whole
  `X.edge_data` vector + reshapes. Simple but conflates the two jobs and forces
  every inner layer to handle a *vector* of edges.
- **(B) Split (recommended):** keep the **per-edge embedding general in
  `embed/`** (acts on one state, broadcasts), and move the **graph adapter**
  (apply-over-`edge_data` + `reshape_embedding`/`rev_reshape_embedding` + the
  `evaluate_ed` reshape) into `graphs/` as e.g. `embed_edges(graph, layer)`.
  This puts the reshape next to the `ETGraph` code it already belongs to, and
  lets inner layers be written per-edge. Cost: a refactor that must keep the
  `evaluate_ed`/jacobian threading intact; `EdgeEmbed` becomes a thin
  graphs-side wrapper.
- **(C) Move `EdgeEmbed` wholesale to `graphs/`** without the per-edge split.
  Cheaper than (B); puts the graph-coupled layer with the graph code, but keeps
  the vector-in convention.

**Recommendation:** (B) is the clean separation and matches CO's framing, but
it's a real refactor; (C) is the low-effort middle. Either way the **reshape
machinery wants to live in `graphs/`**. Decision: (B) vs (C) vs defer.

## Q3 — pooling → embedding? (message op vs system embedding)

Pooling is the edge→node sum. It can be read two ways: **(i)** the final step of
"embed the system" (edge features → node features) or **(ii)** a
message/gather op on the graph. restructure.md §4 already **decided pooling
stays its own `src/pooling/` stage** in ET, because the *transformation law of A
is what defines the constraint on the tensor `T`* — so pooling co-evolves with
the format/carrier, not with the per-edge embedding.

Options:
- **(A) Keep `pooling/` as its own stage (recommended).** Honours §4; keeps the
  clean ET contract ("give me per-particle embedding arrays; I pool and
  contract"). The dual interpretation is a *documentation* matter, not a move.
- **(B) Fold pooling into `embed/`.** Tidy if "embed the system" is the mental
  model, but conflates a *map* (per-edge embedding) with an *aggregation*, and
  pulls a format-coupled stage into the embedding namespace — against §4's
  rationale.
- **(C) Fold pooling into `graphs/`** (it is an edge→node gather). Conceptually
  defensible, but pooling's `aspec` is tied to the tensor format, not just the
  graph topology.

**Recommendation:** (A) keep `pooling/` separate; this is the most "decided" of
the three (§4). The useful action is to **document** that pooling is the
edge→node bridge (message/aggregation) — and note that *if* `embed/` is ever
reorganised (Q2), revisit whether the graph-coupled pieces (EdgeEmbed adapter,
reshape, pooling-as-gather) should cluster under `graphs/`.

## Cross-cutting observation

Q2 and Q3 are the same boundary seen twice: **the graph-coupled glue (reshape,
edge-apply, edge→node gather) arguably wants to live near `graphs/`, while the
content (per-edge embedding, the product/pool spec) stays general.** A coherent
move would be: *general per-edge/per-state pieces in `embed/`; graph adapters in
`graphs/`; pooling stays its own format-coupled stage.* Worth deciding as one
reorg rather than three independent nudges.

## Decisions to make (soon)

1. **Naming (Q1):** rename `DP*` → `State*`/`Particle*` (with deprecated
   aliases + ACEradials update), or keep `DP*` and just de-mislead the
   docstrings? *(Breaking if renamed.)*
2. **EdgeEmbed (Q2):** split into per-edge embedding (`embed/`) + graph adapter
   (`graphs/`) **(B)**, move wholesale to `graphs/` **(C)**, or defer?
3. **Reshape machinery:** confirm `reshape_embedding`/`rev_reshape_embedding`
   should live in `graphs/` (it's graph-structure code regardless of Q2).
4. **Pooling (Q3):** confirm "keep `pooling/` separate + document the dual
   interpretation" — i.e. *don't* move it.
5. **Scope:** do Q2/Q3/reshape as **one embed↔graph reorg** (recommended) or
   piecemeal? And fold in the `utils → lib/ETUtils` question
   (`co_notes`) at the same time?

None of these block the CP work; all are pure ET-core organisation.
