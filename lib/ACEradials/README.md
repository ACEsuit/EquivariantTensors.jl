# ACEradials.jl

Learnable and splined radial bases (`Rnl`) for ACE-style atomistic models:

- `LearnableRnlBasis` — `Rnl = Wnlq * (Pq(x) .* env(r, x))` with learnable
  `Wnlq` (Lux parameters), multi-species via atomic-number indexing
- `SplineRnlBasis` — frozen variant with the `x -> Rnl(x)` map replaced by
  cubic splines (`Polynomials4ML.CubicSplines`); no learnable parameters
- `splinify(learnable, ps)` — converts the former to the latter
- cutoff envelopes (`PolyEnvelope1sR`, `PolyEnvelope2sX`) and scalar Agnesi
  distance transforms
- `learnable_Rnl_basis(elements, rin0cuts; ...)` — high-level constructor

This code is chemistry-specific (atomic numbers, `(rin, r0, rcut)` cutoff
conventions) and therefore lives outside the application-agnostic
`EquivariantTensors` namespace. It is maintained as a separately versioned
**subdirectory package** of the
[EquivariantTensors.jl](https://github.com/ACEsuit/EquivariantTensors.jl)
repository, so that radials and ET internals can co-evolve in atomic PRs
while keeping independent release streams. See the
[Architecture](https://ACEsuit.github.io/EquivariantTensors.jl/dev/architecture/)
docs for where this fits in the ecosystem, and `agents/radials.md` for the
design notes, decision record and remaining future work.

The code originates from `ACEpotentials.jl/src/models/`; the original
`LearnableRnlrzzBasis` / `SplineRnlrzzBasis` types were renamed to
`LearnableRnlBasis` / `SplineRnlBasis`.
