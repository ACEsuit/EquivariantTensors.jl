

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

TransformedBasis(transin, basis) = TransformedBasis(transin, basis, IDtrans())

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
   P = evaluate(tbasis.basis, Y, ps.basis, st.basis)
   B = evaluate(tbasis.transout, P, X, Y, ps.transout, st.transout)
   return B, st 
end



# --------------------------------------------------------- 
#  Some auxiliary transformations 

# WrappedFunction can be used as a wrapper for a "simple" transformation but 
# we need to overload `evaluate` to make it work with our calling convention 

import Lux: WrappedFunction 
evaluate(wf::WrappedFunction, x, ps, st) = LuxCore.apply(wf, x, ps, st)[1] 


# --------------------------------------------------------- 
#  identity transformation

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
evaluate(l::IDtrans, P, x, y, ps, st) = P


# --------------------------------------------------------- 
#  wrapping a transfrom from a named tuple 
#
#  TODO: allow f to be parameterized

"""
If a particle x is represented as a NamedTuple, e.g., `x = (r = SA[...], Z = 13)`
then an `NTtransform` can be used to embed this named tuple into ℝ in a differentiable 
way, e.g., 
```julia
x = (𝐫 = randn(StaticVector{3, Float64}), Z = rand(10:50))
r0 = Float64[ ... ]  # list of r0 values for rescaling r 
trans = NTtransform(x -> 1 / (1 + norm(x.𝐫)/r0[x.Z]))
```
We can then evaluate and differenitate 
```julia 
y = evaluate(trans, x, ps, st)
y, dy = evaluate_ed(trans, x, ps, st)
```
Here, `dy` is again a named-tuple with the derivative w.r.t. x.𝐫 stored as 
`dy.𝐫`. The derivative w.r.t. Z is not taken because `Z` is a categorical variable.
"""
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
