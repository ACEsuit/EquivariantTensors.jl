

"""
   struct SelectLinL <: AbstractLuxLayer

A Lux layer which acts as a simple linear layer, but using a categorical 
variable to select a weight matrix:
```
   P -> W[x] * P 
```
This layer is experimental and likely very inefficient. 
"""
struct SelectLinL{TSEL} <: AbstractLuxLayer
   in_dim::Int
   out_dim::Int
   ncat::Int
   selector::TSEL
end

LuxCore.initialstates(rng::AbstractRNG, l::SelectLinL) = NamedTuple()

function LuxCore.initialparameters(rng::AbstractRNG, l::SelectLinL) 
   W = randn(rng, l.out_dim, l.in_dim, l.ncat) * sqrt(2 / (l.in_dim + l.out_dim))
   return (W = W,)
end

(l::SelectLinL)( P_X, ps, st) = 
      _apply_selectlinl(l, P_X[1], P_X[2], ps, st)

# TODO: this case distinction is a hack; it would be better to 
#       use types one cannot iterate over. Something to look into. 
#       and an argument to move to DecoratedParticles.jl

function _apply_selectlinl(l, P, x::Union{NamedTuple, Number, StaticArray},
                           ps, st)
   B = (@view ps.W[:, :, l.selector(x)]) * P
   return B, st 
end


function _apply_selectlinl(l, P, X::AbstractArray, ps, st)
   TB = promote_type(eltype(P), eltype(ps.W))
   B = similar(P, TB, size(P, 1), l.out_dim)
   
   # Morally this should work, but it doesn't like the views it seems?!
   #       so we need write a kernel for it; fairly straightforward for this 
   #       case but unfortuntately doesn't leverage BLAS 
   # B = reduce(vcat, transpose( (@view ps.W[:, :, l.selector(x)]) * P[i, :])
   #                  for (i, x) in enumerate(X) )

   # TODO: there was a problem applying the selector when it was type unstable
   # now that this is fixed, maybe try to go back to the above implementation?
   # that way we don't have to write a custom rrule. 

   kernel! = _ka_apply_selectlinl!(KernelAbstractions.get_backend(X))
   kernel!(B, P, X, ps.W, l.selector; ndrange = size(B))

   return B, st 
end


@kernel function _ka_apply_selectlinl!(B, P, X, W, selector)
   # B[iB, jB] = sum P[iB, k] * W[jB, k, selector(x)]
   iB, jB = @index(Global, NTuple)
   i_x = selector(X[iB])
   B[iB, jB] = 0
   for k = 1:size(P, 2) 
      B[iB, jB] += W[jB, k, i_x] * P[iB, k]
   end
   nothing
end