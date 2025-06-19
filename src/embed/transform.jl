

import ForwardDiff as FD 

"""
   struct TransformedBasis

Basically a three-stage chain, consisting of an input transformation, 
and basis evaluation. Constructor: 
```julia
TransformedBasis(; transin, basis, transout)
```      
defaults for `transin` and `transout` are `IDtrans()`, i.e. identity 
transformations.

The layer performs the following chain: 
```julia
y = evaluate(transin, x, ...)
P = evaluate(basis, y, ...) 
B = evaluate(transout, P, x, ...)
```
- It is assumed that `transout` only utilizes the categorical variables stored in 
`x` but not the continuous variables. This means that when differentiating, 
one only needs to differentiate `B(P(y), x)` with respect to `y` but not 
with respect to `x`.
- it is also assumed that `basis` has no parameters 
"""
struct TransformedBasis{TIN, BAS, TOUT} <: AbstractLuxLayer
   transin::TIN
   basis::BAS
   transout::TOUT 
end

function Base.show(io::IO, l::TransformedBasis)
   print(io, "TransformedBasis($(l.transin), $(l.basis), $(l.transout))")
end

TransformedBasis(; basis, 
                   transin = IDtrans(), 
                   transout = IDtrans()) = 
   TransformedBasis(transin, basis, transout)

initialparameters(rng::AbstractRNG, l::TransformedBasis) = 
      ( transin = initialparameters(rng, l.transin),
        basis = initialparameters(rng, l.basis), 
        transout = initialparameters(rng, l.transout), )

initialstates(rng::AbstractRNG, l::TransformedBasis) = 
      ( transin = initialstates(rng, l.transin),
        basis = initialstates(rng, l.basis), 
        transout = initialstates(rng, l.transout), )

# --------------------------------------------------------- 
# evaluation  
# 

(l::TransformedBasis)(x, ps, st) = evaluate(l, x, ps, st)

function evaluate(tbasis::TransformedBasis, X, ps, st)
   Y = map(x -> evaluate(tbasis.transin, x, ps.transin, st.transin), X) 
   P, _ = tbasis.basis(Y, ps.basis, st.basis)
   B = evaluate(tbasis.transout, P, X, ps.transout, st.transout)
   return B, st 
end



# --------------------------------------------------------- 
#  Some auxiliary transformations 

struct IDtrans <: AbstractLuxLayer
end 

function Base.show(io::IO, l::IDtrans)
   print(io, "ID")
end

initialparameters(rng::AbstractRNG, l::IDtrans) = NamedTuple()
initialstates(rng::AbstractRNG, l::IDtrans) = NamedTuple()  

(l::IDtrans)(x, ps, st) = x 

# calling convention for input transformations 
evaluate(l::IDtrans, x, ps, st) = x 

# calling convention for output transformations 
# here P is the basis, but the transformatino may utilize the 
# original input x. 
evaluate(l::IDtrans, P, x, ps, st) = P


# --------------------------------------------------------- 
#  wrapping a transfrom from a named tuple 

struct NTtransform{FT} <: AbstractLuxLayer
   f::FT 
   sym::Symbol 
end

NTtransform(f; sym = Symbol(""))  = NTtransform(f, Symbol(sym))

Base.show(io::IO, l::NTtransform) = print(io, "NTtransform($(l.sym))")

initialparameters(rng::AbstractRNG, l::NTtransform) = NamedTuple()
initialstates(rng::AbstractRNG, l::NTtransform) = NamedTuple()

(l::NTtransform)(x, ps, st) = l.f(x), st 
(l::NTtransform)(x) = l.f(x)

evaluate(l::NTtransform, x::NamedTuple, ps, st) = l.f(x)

evaluate_ed(l::NTtransform, x::NamedTuple, ps, st) = 
      (l.f(x), DiffNT.grad_fd(l.f, x))
