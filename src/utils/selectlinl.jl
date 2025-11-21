

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

function _apply_selectlinl(l, P, X::AbstractArray, ps, st)
   B = reduce(vcat, transpose(ps.W[:, :, l.selector(x)] * P[i, :])
                    for (i, x) in enumerate(X) )
   return B, st 
end

function _apply_selectlinl(l, P, x::Union{NamedTuple, Number, StaticArray},
                           ps, st)
   B = ps.W[:, :, l.selector(x)] * P
   return B, st 
end
