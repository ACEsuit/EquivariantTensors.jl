
import ForwardDiff as FD 
import DecoratedParticles as DP 
import DecoratedParticles: VState, PState, XState 

const NTorDP = Union{NamedTuple, XState}


# --------------------------------------------------------- 
#  wrapping a transfrom from a named tuple or decorated particle 


"""
   function nt_transform(f::Function)
   function nt_transform(f::Function, refstate::NamedTuple)

If a particle x is represented as a `PState` (DecoratedParticles.jl) 
or a `NamedTuple``, e.g., `x = PState(r = SA[...], Z = 13)`, 
then an `nt_transform` generates a type that incorporates 
broadcasting and differentiation. For example 
```julia
x = (𝐫 = randn(StaticVector{3, Float64}), Z = rand(10:50))
refstate = (; r0 = SA[ ... ])    # list of r0 values for rescaling r 
trans = dp_transform( (x, st) -> 1 / (1 + norm(x.𝐫)/ st.r0[x.Z]))
```
We can then evaluate and differenitate 
```julia 
y, _ = evaluate(trans, x, ps, st)
(y, dy), _ = evaluate_ed(trans, x, ps, st)
```
Here, `dy` is a `VState` or a named-tuple with the derivative w.r.t. x.𝐫 stored 
as `dy.𝐫`. The derivative w.r.t. Z is not taken because `Z` is a categorical 
variable.

The transform is assumed to be parameter-free, or with frozen parameters 
stored in the `refstate`. If a transform has no parameters at all then 
one can generate it via 
```julia 
trans = dp_transform( x -> 1 / (1 + norm(x.𝐫)))
```
"""
function dp_transform(f::Function)
   fst = (x, st) -> f(x) 
   return dp_transform(fst, NamedTuple())
end

function dp_transform(f::Function, refstate::NamedTuple)
   return NTtransformST(f, refstate)
end

# can call nt_transform as alias for dp_transform 
nt_transform = dp_transform 




"""
Implementation of `nt_transform` and `dp_transform`. For details 
see the docstring for those. 
"""
struct NTtransformST{FT, ST} <: AbstractLuxLayer
   f::FT 
   refstate::ST
end

Base.show(io::IO, l::NTtransformST) = print(io, "NTtransformST()")

initialparameters(rng::AbstractRNG, l::NTtransformST) = NamedTuple()
initialstates(rng::AbstractRNG, l::NTtransformST) = deepcopy(l.refstate)

(l::NTtransformST)(x::NTorDP, ps, st) = l.f(x, st), st 

# this non-standard calling convention assumes that st is not changed 
(l::NTtransformST)(x::NTorDP, st) = l.f(x, st)

(l::NTtransformST)(x::AbstractVector{<: NTorDP}, ps, st) = 
         l(x, st), st 

(l::NTtransformST)(x::AbstractVector{<: NTorDP}, st) = 
         broadcast(l.f, x, Ref(st))

evaluate(l::NTtransformST, x::NTorDP, ps, st) = 
         l.f(x, st)

evaluate_ed(l::NTtransformST, x::NTorDP, ps, st) = 
         (l.f(x, st), DiffNT.grad_fd(l.f, x, st))

function evaluate_ed(l::NTtransformST, x::AbstractVector{<: NTorDP}, ps, st)
   Y = broadcast(l.f, x, Ref(st))
   dY = broadcast(DiffNT.grad_fd, Ref(l.f), x, Ref(st))
   return (Y, dY), st 
end


function _pb_ed(l::NTtransformST, Δ::AbstractArray, 
                 X::AbstractVector{<: NTorDP}, ps, st)
   pb1(x, d) = DiffNT.grad_fd(_x -> dot(l.f(_x, st), d), x)
   # pb1(x, d) = d * DiffNT.grad_fd(l.f, x, st)
   return pb1.(X, Δ)
end 

# WORKAROUND FOR THE RADIAL AGNESI EMBEDDING WITH 
# PARAMETERS STORED INSIDE THE TYPE NOT JUST THE STATE 
function _pb_ed(l::NTtransformST, Δ::AbstractArray{TY}, 
                 X::AbstractVector{<: NTorDP}, ps, st
                 ) where {TY <: Number} 
   (Y, dY), _ = evaluate_ed(l, X, ps, st)
   return dY .* Δ                 
end 
