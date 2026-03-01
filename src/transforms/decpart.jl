
import ForwardDiff as FD

# ---------------------------------------------------------
#  wrapping a transfrom from a named tuple or decorated particle


"""
   function dp_transform(f::Function)
   function dp_transform(f::Function, refstate::NamedTuple)

If a particle x is represented as a `PState` (DecoratedParticles.jl)
or a `NamedTuple``, e.g., `x = PState(r = SA[...], Z = 13)`,
then a `dp_transform` generates a type that incorporates
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
   return DPTransform(f, refstate)
end




"""
Implementation of `dp_transform`. For details see the docstring for that.
"""
struct DPTransform{FT, ST} <: AbstractLuxLayer
   f::FT
   refstate::ST
end

Base.show(io::IO, l::DPTransform) = print(io, "DPTransform()")

initialparameters(rng::AbstractRNG, l::DPTransform) = NamedTuple()
initialstates(rng::AbstractRNG, l::DPTransform) = deepcopy(l.refstate)

(l::DPTransform)(x::NTorDP, ps, st) = l.f(x, st), st

# this non-standard calling convention assumes that st is not changed
(l::DPTransform)(x::NTorDP, st) = l.f(x, st)

(l::DPTransform)(x::AbstractVector{<: NTorDP}, ps, st) =
         l(x, st), st

(l::DPTransform)(x::AbstractVector{<: NTorDP}, st) =
         broadcast(l.f, x, Ref(st))

evaluate(l::DPTransform, x::NTorDP, ps, st) =
         l.f(x, st)

evaluate_ed(l::DPTransform, x::NTorDP, ps, st) =
         (l.f(x, st), DiffNT.grad_fd(l.f, x, st))

function evaluate_ed(l::DPTransform, x::AbstractVector{<: NTorDP}, ps, st)
   Y = broadcast(l.f, x, Ref(st))
   dY = broadcast(DiffNT.grad_fd, Ref(l.f), x, Ref(st))
   return (Y, dY), st
end


function _pb_ed(l::DPTransform, Δ::AbstractArray,
                 X::AbstractVector{<: NTorDP}, ps, st)
   # make sure the closure doesn't capture l, but only l.f
   # and l.f itself cannot capture anything that doesn't run on GPU.
   pb1 = let l_f = l.f, st = st
      (x, d) -> DiffNT.grad_fd(_x -> dot(l_f(_x, st), d), x)
   end
   return pb1.(X, Δ)
end
