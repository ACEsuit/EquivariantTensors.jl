# `agents/` — working notes

Design notes, decision records and forward-looking roadmaps for the
EquivariantTensors restructuring and ongoing work. The *current* package
structure is documented in `docs/src/architecture.md`; the PR-by-PR history of
the restructuring is the checklist in umbrella PR #109. These notes keep only
what is still needed to proceed.

- **`restructure.md`** — ET core roadmap + decision records. Boundary cleanup
  is done (compressed to "DONE" stubs); live content is the new-format work
  (CP/Tucker/TT, A storage layout, format I/O contract) and the open questions.
- **`radials.md`** — `lib/ACEradials` design record + remaining future work
  (ACEpotentials adaptation, learnable envelopes/transforms, spec utilities,
  smoothness priors, parked spline Option A). The port and spline/Agnesi
  consolidation have landed.
- **`initializers.md`** — ET weight-initializer design: the leaf samplers that
  landed (#121) and the format-aware policy planned for the CP format.
