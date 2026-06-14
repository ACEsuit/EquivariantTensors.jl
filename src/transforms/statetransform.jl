
# ---------------------------------------------------------
#  StateTransform: wrap a function as a Lux transform layer that maps a particle
#  *state* to a number / SVector.
#
# NOTE: the struct + constructor are general (LuxCore-only); all evaluation and
#       differentiation methods live in ext/DecoratedParticlesExt.jl and are
#       available once DecoratedParticles is loaded. The *canonical* usage is to
#       transform a DecoratedParticles state (PState/XState) and differentiate
#       w.r.t. it. Particles must be XStates (PState); bare NamedTuples are not
#       supported — they lack the tangent arithmetic (zero, +, *) needed in the
#       gradient and pullback paths.
#
# `DPTransform` / `dp_transform` are deprecated aliases of `StateTransform` /
# `state_transform`.


"""
   state_transform(f::Function)
   state_transform(f::Function, refstate::NamedTuple)

Wrap a function `f` as a `StateTransform` Lux layer. The **canonical usage** is
to transform a particle *state* — a `PState` from DecoratedParticles.jl — into a
number or `SVector`, with automatic broadcasting and differentiation w.r.t. the
state. For example
```julia
x = PState(𝐫 = randn(StaticVector{3, Float64}), Z = rand(10:50))
refstate = (; r0 = SA[ ... ])    # frozen parameters, e.g. per-species r0
trans = state_transform( (x, st) -> 1 / (1 + norm(x.𝐫) / st.r0[x.Z]) )
```
We can then evaluate and differentiate
```julia
y, _ = evaluate(trans, x, ps, st)
(y, dy), _ = evaluate_ed(trans, x, ps, st)
```
Here `dy` is a `VState` with the derivative w.r.t. `x.𝐫` stored as `dy.𝐫`;
categorical fields (e.g. `Z`) are not differentiated. The transform is
parameter-free, or with frozen parameters stored in `refstate`. With no
parameters at all,
```julia
trans = state_transform( x -> 1 / (1 + norm(x.𝐫)) )
```
The evaluation / differentiation methods are provided by the DecoratedParticles
extension (loaded together with DecoratedParticles).
"""
function state_transform(f::Function)
   fst = (x, st) -> f(x)
   return state_transform(fst, NamedTuple())
end

function state_transform(f::Function, refstate::NamedTuple)
   return StateTransform(f, refstate)
end


"""
Implementation type for `state_transform`; see its docstring.
"""
struct StateTransform{FT, ST} <: AbstractLuxLayer
   f::FT
   refstate::ST
end

Base.show(io::IO, l::StateTransform) = print(io, "StateTransform()")

initialparameters(rng::AbstractRNG, l::StateTransform) = NamedTuple()
initialstates(rng::AbstractRNG, l::StateTransform) = deepcopy(l.refstate)

# pullback through a StateTransform w.r.t. the particle inputs; methods are
# provided by ext/DecoratedParticlesExt.jl
function _pb_ed end


# --- deprecated aliases (pre-2026-06 names) ---
Base.@deprecate_binding DPTransform StateTransform false
Base.@deprecate_binding dp_transform state_transform false
