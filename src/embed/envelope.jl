
"""
To be used as part of TransformedBasis as follows: 
```julia 
y = transform(x) 
P = basis(y) * envelope(x, y)
```
Warning: P may be modified in-place!
"""
struct Envelope{TF} <: AbstractLuxLayer
   f::TF
   sym::Symbol
end

#  TODO: allow f to be parameterized


Envelope(f; sym = Symbol("")) = Envelope(f, Symbol(sym))

Base.show(io::IO, l::Envelope) = print(io, "Envelope($(l.sym))")

initialparameters(rng::AbstractRNG, l::NTtransform) = NamedTuple()
initialstates(rng::AbstractRNG, l::NTtransform) = NamedTuple()

(l::Envelope)(args...) = evaluate(l, args...)

function evaluate(l::Envelope, P, x, y, ps, st) 
   evaluate(l, P, x, y) 
   return P
end

# single input case
function evaluate(l::Envelope, P::AbstractVector, x, y)
   e = l.f(x, y) 
   map!(x -> x * e, P, P) # in-place multiplication
   return P 
end

# batched input case
function evaluate(l::Envelope, P::AbstractMatrix, x::AbstractVector, y::AbstractVector)
   @assert length(x) == length(y) == size(P, 1) 
   env = l.f.(x, y)
   return env .* P # broadcasting multiplication
end

