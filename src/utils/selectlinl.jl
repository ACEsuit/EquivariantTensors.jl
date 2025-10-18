

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

function (l::SelectLinL)( P_X, ps, st)
   P, X = P_X
   B = reduce(vcat, transpose(ps.W[:, :, l.selector(x)] * P[i, :])
                    for (i, x) in enumerate(X) )
   return B, st 
end
