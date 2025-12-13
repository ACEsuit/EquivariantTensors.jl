#
# TODO: to be retired 
#       replace with simple Chain and SkipConnection layers.
#
# Only keep the NTtransform 
#


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
#  TODO: 
#    - allow f to have states 
#    - allow f to have parameters 

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

(l::NTtransform)(x::NamedTuple, ps, st) = l.f(x), st 
(l::NTtransform)(x::NamedTuple) = l.f(x)

(l::NTtransform)(x::AbstractVector{<: NamedTuple}, ps, st) = map(l.f, x), st 
(l::NTtransform)(x::AbstractVector{<: NamedTuple}) = map(l.f, x) 

evaluate(l::NTtransform, x::NamedTuple, ps, st) = l.f(x)

evaluate_ed(l::NTtransform, x::NamedTuple, ps, st) = 
      (l.f(x), DiffNT.grad_fd(l.f, x))


# function rrule(trans::typeof(NTtransform), X::AbstractVector{<: NamedTuple}, ps, st) 
#    @assert ps == NamedTuple() "NTtransform cannot have parameters"
#    Y = map(trans.f, X)
#    dY = map(x -> DiffNT.grad_fd(trans.f, x), X)

#    function _pb_X(∂Y)
#       ∂X = map( (∂y, dy) -> ∂y * dy, ∂Y, dY )
#       return NoTangent(), ∂X, NoTangent(), NoTangent()
#    end

#    return Y, _pb_X 
# end


# -------------------------------------------------------------- 
#   a variant of NTtransform that allows parameters and states 


"""
Experimental extension of NTtransform that allows states.
(to be extended to also allow paramters)

Maybe a better approach would be to have just wrap a callable, and let that 
callable take care of the parameters. otoh, what we are doing here is 
more convenient to build on-the-fly ...

NOT PART OF THE OFFICIAL API YET! CAN CHANGE WITHOUT NOTICE!
"""
struct NTtransformST{FT, ST} <: AbstractLuxLayer
   f::FT 
   refstate::ST
   sym::Symbol 
end

NTtransformST(f, refstate = NamedTuple(); 
              sym = Symbol(""))  = 
      NTtransformST(f, refstate, Symbol(sym))

Base.show(io::IO, l::NTtransformST) = print(io, "NTtransformST($(l.sym))")

initialparameters(rng::AbstractRNG, l::NTtransformST) = NamedTuple()
initialstates(rng::AbstractRNG, l::NTtransformST) = deepcopy(l.refstate)

(l::NTtransformST)(x::NamedTuple, ps, st) = l.f(x, st), st 

# this non-standard calling convention assumes that st is not changed 
(l::NTtransformST)(x::NamedTuple, st) = l.f(x, st)

(l::NTtransformST)(x::AbstractVector{<: NamedTuple}, ps, st) = 
         l(x, st), st 

(l::NTtransformST)(x::AbstractVector{<: NamedTuple}, st) = 
         broadcast(l.f, x, Ref(st))
         # map(x -> l.f(x, st), x)

evaluate(l::NTtransformST, x::NamedTuple, ps, st) = 
         l.f(x, st)

evaluate_ed(l::NTtransformST, x::NamedTuple, ps, st) = 
         (l.f(x, st), DiffNT.grad_fd(l.f, x, st))


# rrule for NTtransformST evaluation with AbstractVector input
# This fixes the ProjectTo error when Zygote tries to backprop through
# states containing complex NamedTuples with non-differentiable fields
import ChainRulesCore: rrule, NoTangent

function rrule(l::NTtransformST, X::AbstractVector{<:NamedTuple}, st)
   Y = l(X, st)
   # Compute derivatives using ForwardDiff through the NamedTuple
   dY = map(x -> DiffNT.grad_fd(l.f, x, st), X)

   function _pb_NTtransformST(∂Y)
      if ∂Y isa ChainRulesCore.ZeroTangent
         return NoTangent(), ChainRulesCore.ZeroTangent(), NoTangent()
      end
      # Multiply each cotangent by the Jacobian (for scalar outputs, this is just multiplication)
      # dY[i] is a NamedTuple of derivatives w.r.t. the fields of X[i]
      ∂X = map((∂y, dy) -> DiffNT.scale_nt(dy, ∂y), ∂Y, dY)
      return NoTangent(), ∂X, NoTangent()
   end

   return Y, _pb_NTtransformST
end

# Also need rrule for the (l, x, ps, st) signature used by Lux
function rrule(l::NTtransformST, X::AbstractVector{<:NamedTuple}, ps, st)
   Y, st_out = l(X, ps, st)
   # Compute derivatives using ForwardDiff through the NamedTuple
   dY = map(x -> DiffNT.grad_fd(l.f, x, st), X)

   function _pb_NTtransformST_lux(∂Y_st)
      ∂Y = ∂Y_st[1]
      if ∂Y isa ChainRulesCore.ZeroTangent
         return NoTangent(), ChainRulesCore.ZeroTangent(), NoTangent(), NoTangent()
      end
      ∂X = map((∂y, dy) -> DiffNT.scale_nt(dy, ∂y), ∂Y, dY)
      return NoTangent(), ∂X, NoTangent(), NoTangent()
   end

   return (Y, st_out), _pb_NTtransformST_lux
end
