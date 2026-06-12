
# ---------------------------------------------------------
#  wrapping a transfrom from a decorated particle
#
# NOTE: only the struct and constructor live in core; all evaluation
#       and differentiation methods live in ext/DecoratedParticlesExt.jl
#       and are available once DecoratedParticles is loaded. Particles
#       must be represented as XStates (PState); bare NamedTuples are
#       not supported: they lack the tangent arithmetic (zero, +, *)
#       needed in the gradient and pullback paths.


"""
   function dp_transform(f::Function)
   function dp_transform(f::Function, refstate::NamedTuple)

If a particle x is represented as a `PState` (DecoratedParticles.jl),
e.g., `x = PState(r = SA[...], Z = 13)`,
then a `dp_transform` generates a type that incorporates
broadcasting and differentiation. For example
```julia
x = PState(𝐫 = randn(StaticVector{3, Float64}), Z = rand(10:50))
refstate = (; r0 = SA[ ... ])    # list of r0 values for rescaling r
trans = dp_transform( (x, st) -> 1 / (1 + norm(x.𝐫)/ st.r0[x.Z]))
```
We can then evaluate and differenitate
```julia
y, _ = evaluate(trans, x, ps, st)
(y, dy), _ = evaluate_ed(trans, x, ps, st)
```
Here, `dy` is a `VState` with the derivative w.r.t. x.𝐫 stored
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

# pullback through a DPTransform w.r.t. the particle inputs; methods are
# provided by ext/DecoratedParticlesExt.jl
function _pb_ed end
