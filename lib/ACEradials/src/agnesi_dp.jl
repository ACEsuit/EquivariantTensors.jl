
# Species-pair-indexed Agnesi transform operating on edge / XState
# descriptors (moved here from EquivariantTensors, src/transforms/agnesi.jl).
#
# This is a thin, multi-species adapter over the scalar Agnesi transform in
# `transforms.jl` (`GeneralizedAgnesiTransform` / `NormalizedTransform`): for
# each species pair it stores a scalar `r -> [-1, 1]` transform, and on an
# edge / XState descriptor it evaluates that transform at `norm(x.𝐫)` with a
# species-pair parameter lookup. Previously this file carried its own copy of
# the Agnesi math (`agnesi_params` / `eval_agnesi`); that duplication has been
# removed (see agents/radials.md §4 and agents/radials_restructure.md §3.2).

using LinearAlgebra: norm
import EquivariantTensors: StateTransform, symidx, catcat2idx_sym


@doc raw"""
   agnesi_transform(categories, rins, reqs, rcuts, pin, pcut)

Construct a multi-species generalized Agnesi transform: a `StateTransform` layer
that, on an edge / XState descriptor `x`, evaluates a scalar Agnesi transform
at `r = norm(x.𝐫)`, selecting the per-species-pair parameters via `(x.z0,
x.z1)`. The scalar transform implements
```math
   y(r) = b_0 + \frac{b_1}{1 + a s^q / (1 + s^{q-p})},
   \qquad s = \frac{r - r_{\rm in}}{r_{\rm eq} - r_{\rm in}},
```
with `p = pcut`, `q = pin`, default `b_0, b_1` mapping `r ∈ [r_{\rm in},
r_{\rm cut}]` to `[-1, 1]`, and `a` chosen so that `|dy/dr|` is maximised at
`r = r_{\rm eq}`.

`rins`, `reqs`, `rcuts` may each be a single number (applied to all pairs), a
`Dict` keyed by `(zi, zj)`, or `nothing` to request the default. Typical
defaults for `p, q` are `pcut = 2, pin = 4`.
"""
function agnesi_transform(categories, rins, reqs, rcuts,
                          pin::Integer, pcut::Integer)

   function _dict_from_num(cats, r, rdef)
      if r isa Dict
         return r
      end
      if r == nothing
         r = rdef
      end
      if r == nothing
         error("cannot have r == rdef == nothing")
      end

      rd = Dict{Tuple, Real}()
      for (c1, c2) in Iterators.product(cats, cats)
         rd[(c1, c2)] = r
      end
      return rd
   end

   # defaults for rins
   rins = _dict_from_num(categories, rins, 0.0)
   # default for reqs
   reqs = _dict_from_num(categories, reqs, nothing)
   # default for rcuts
   rcuts = _dict_from_num(categories, rcuts, nothing)

   # build a scalar Agnesi transform `r -> [-1, 1]` for each species pair
   #     upper-triangular storage, flattened into a vector
   #     use catcat2idx_sym / symidx to access
   NZ = length(categories)
   transforms = Vector{Any}(undef, (NZ * (NZ+1)) ÷ 2)
   idx = 0
   for i in 1:NZ, j in i:NZ
      # confirm that the indexing is correct
      idx += 1
      @assert idx == symidx(i, j, NZ)
      # scalar Agnesi transform for the (zi, zj) pair; note the scalar
      # transform's convention is `agnesi_transform(r0, rcut, p, q)` with
      # `p = pcut`, `q = pin`, `r0 = req`.
      zi = categories[i]; zj = categories[j]
      transforms[idx] = agnesi_transform(reqs[zi, zj], rcuts[zi, zj],
                                         pcut, pin; rin = rins[zi, zj])
   end

   transforms = identity.(transforms)
   # the transforms should all be of the same type so can be stored in an
   # SVector for efficiency. This is the reference state for the StateTransform.
   st = ( zlist = categories,
          transforms = SVector{length(transforms)}(transforms), )

   # build the actual transform mapping
   #     TODO: allow specification of how to get r, z0, z1 from input x!!!
   f_agnesi = let
      (x, st) -> begin
         r = norm(x.𝐫)
         idx = catcat2idx_sym(st.zlist, x.z0, x.z1)
         return evaluate(st.transforms[idx], r)
      end
   end

   return StateTransform(f_agnesi, st)
end
