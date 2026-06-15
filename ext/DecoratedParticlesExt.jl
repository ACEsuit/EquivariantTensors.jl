
#
# Glue between EquivariantTensors and DecoratedParticles: XState methods
# for the DP-coupled embedding layers, and all derivatives of transforms
# w.r.t. particle inputs (NamedTuple or XState), which require the
# differentiation tooling `DecoratedParticles.grad_fd`.
#

module DecoratedParticlesExt

import EquivariantTensors as ET
import EquivariantTensors: StateTransform, evaluate, evaluate_ed
using DecoratedParticles: XState, PState, grad_fd
using LinearAlgebra: dot

# ---------------------------------------------------------
# StateTransform methods; particles must be XStates (cf. the NOTE in
# src/transforms/decpart.jl — bare NamedTuples are not supported)

(l::StateTransform)(x::XState, ps, st) = l.f(x, st), st

# this non-standard calling convention assumes that st is not changed
(l::StateTransform)(x::XState, st) = l.f(x, st)

(l::StateTransform)(x::AbstractVector{<: XState}, ps, st) =
         l(x, st), st

(l::StateTransform)(x::AbstractVector{<: XState}, st) =
         broadcast(l.f, x, Ref(st))

evaluate(l::StateTransform, x::XState, ps, st) =
         l.f(x, st)

# ---------------------------------------------------------
# derivatives w.r.t. particle inputs

evaluate_ed(l::StateTransform, x::XState, ps, st) =
         (l.f(x, st), grad_fd(l.f, x, st))

function evaluate_ed(l::StateTransform, x::AbstractVector{<: XState}, ps, st)
   Y = broadcast(l.f, x, Ref(st))
   dY = broadcast(grad_fd, Ref(l.f), x, Ref(st))
   return (Y, dY), st
end

function ET._pb_ed(l::StateTransform, Δ::AbstractArray,
                   X::AbstractVector{<: XState}, ps, st)
   # make sure the closure doesn't capture l, but only l.f
   # and l.f itself cannot capture anything that doesn't run on GPU.
   pb1 = let l_f = l.f, st = st
      (x, d) -> grad_fd(_x -> dot(l_f(_x, st), d), x)
   end
   return pb1.(X, Δ)
end

# ---------------------------------------------------------
# float32 / float64 conversion (cf. src/utils/adapt.jl)

ET.float32(x::T) where {T <: XState} = T( ET.float32(getfield(x, :x)) )
ET.float64(x::T) where {T <: XState} = T( ET.float64(getfield(x, :x)) )

# ---------------------------------------------------------
# default edge data for ET.Testing.rand_graph

ET.Testing._default_randedge(rcut) =
      PState( 𝐫 = ET.Testing.rand_ball(rcut), )

end
