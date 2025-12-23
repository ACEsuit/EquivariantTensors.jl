#
# NOTE: this is currently not part of the package 
# TODO: decide whether to keep it or remove it.
#

using DecoratedParticles: XState 
const NTorDP = Union{NamedTuple, XState}

struct Get{SYM} <: AbstractLuxLayer
end

(l::Get)(x, ps, st) = evaluate(l, x, ps, st)

evaluate(l::Get, x, ps, st) = evaluate(l, x), st 
evaluate(l::Get, x, st) = evaluate(l, x)
evaluate_ed(l::Get, x, ps, st) = evaluate_ed(l, x), st 
evaluate_ed(l::Get, x, st) = evaluate_ed(l, x)

evaluate(l::Get{SYM}, x::NTorDP) = getproperty(x, SYM)

evaluate(l::Get{SYM}, X::AbstractArray{<: NTorDP}) = getproperty.(x, Ref(SYM))

function _pb_ed(l::Get{SYM}, Δ::AbstractArray{TY}, 
                 X::AbstractArray{<: XState}) where {TY} 
                    
   T∂X = vstate_type(eltype(X))
   ∂X = similar(Δ, T∂X, size(Δ))
   for i in eachindex(X)
      ∂X[i] = zero(T∂X)
      setproperty!(∂X[i], SYM, Δ[i])
   end
   return ∂X
end


# --------------------------------------------------------- 
#  identity transformation
#  we probably don't need this anymore and can just 
#  remove it. 
#= 
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
=# 

