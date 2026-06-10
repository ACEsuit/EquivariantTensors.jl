
# High-level constructor for a `LearnableRnlBasis`. Ported from ACEpotentials
# `ace_learnable_Rnlrzz` (in `src/models/ace_heuristics.jl`) and renamed
# `learnable_Rnl_basis`.
#
# Differences from the ACEpotentials version:
# - `elements` (a list of integer atomic numbers) and `rin0cuts` are required
#   arguments; the ACE1x-based `_default_rin0cuts` is not ported (it would pull
#   in a covalent-radius database / JuLIP dependency).
# - polynomial dispatch additionally supports `:chebyshev` and `(:jacobi, α, β)`.
# - the polynomial basis length defaults to `maxq` (defaulting in turn to the
#   actual maximum `n` in the spec).
# - notions of "level" (e.g. `TotalDegree`) live in ACEpotentials for now; pass
#   either a ready-made `spec` or a callable `level` together with `max_level`.

function learnable_Rnl_basis(elements, rin0cuts;
               max_level = nothing,
               level = nothing,
               maxl = nothing,
               maxn = nothing,
               maxq = nothing,
               spec = nothing,
               transforms = agnesi_transform.(rin0cuts, 2, 2),
               polys = :legendre,
               envelopes = :poly2sx )

   if (spec == nothing) && (level == nothing || max_level == nothing)
      error("Must specify either `spec` or `level, max_level`!")
   end

   zlist = _convert_zlist(elements)
   NZ = length(zlist)

   if spec == nothing
      spec = [ (n = n, l = l) for n = 1:maxn, l = 0:maxl
                              if level((n = n, l = l)) <= max_level ]
   end

   # the actual maxn is the maximum n in the spec
   actual_maxn = maximum(s.n for s in spec)

   if maxq === nothing
      maxq = actual_maxn
   end

   if polys isa Symbol
      if polys == :legendre
         polys = P4ML.legendre_basis(maxq)
      elseif polys == :chebyshev
         polys = P4ML.chebyshev_basis(maxq)
      else
         error("unknown polynomial type : $polys")
      end
   elseif polys isa Tuple && polys[1] == :jacobi
      polys = P4ML.jacobi_basis(maxq, polys[2], polys[3])
   end

   if transforms isa Tuple && transforms[1] == :agnesi
      p = transforms[2]
      q = transforms[3]
      transforms = agnesi_transform.(rin0cuts, p, q)
   end

   if envelopes == :poly2sx
      envelopes = PolyEnvelope2sX(-1.0, 1.0, 2, 2)
   elseif envelopes == :poly1sr
      envelopes = [ PolyEnvelope1sR(rin0cuts[iz, jz].rcut, 1)
                    for iz = 1:NZ, jz = 1:NZ ]
   end

   if actual_maxn > length(polys)
      error("actual_maxn > length of polynomial basis")
   end

   return LearnableRnlBasis(zlist, polys, transforms, envelopes, rin0cuts, spec)
end
