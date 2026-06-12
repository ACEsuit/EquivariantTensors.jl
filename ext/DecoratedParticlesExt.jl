
#
# Glue between EquivariantTensors and DecoratedParticles: XState methods
# for the DP-coupled embedding layers, and all derivatives of transforms
# w.r.t. particle inputs (NamedTuple or XState), which require the
# differentiation tooling `DecoratedParticles.grad_fd`.
#

module DecoratedParticlesExt

import EquivariantTensors as ET
import EquivariantTensors: DPTransform, evaluate, evaluate_ed
using DecoratedParticles: XState, PState, grad_fd
using LinearAlgebra: dot

const NTorDP = Union{NamedTuple, XState}

# ---------------------------------------------------------
# DPTransform methods for XState inputs
# (the NamedTuple analogues live in src/transforms/decpart.jl)

(l::DPTransform)(x::XState, ps, st) = l.f(x, st), st

# this non-standard calling convention assumes that st is not changed
(l::DPTransform)(x::XState, st) = l.f(x, st)

(l::DPTransform)(x::AbstractVector{<: XState}, ps, st) =
         l(x, st), st

(l::DPTransform)(x::AbstractVector{<: XState}, st) =
         broadcast(l.f, x, Ref(st))

evaluate(l::DPTransform, x::XState, ps, st) =
         l.f(x, st)

# ---------------------------------------------------------
# derivatives w.r.t. particle inputs

evaluate_ed(l::DPTransform, x::NTorDP, ps, st) =
         (l.f(x, st), grad_fd(l.f, x, st))

function evaluate_ed(l::DPTransform, x::AbstractVector{<: NTorDP}, ps, st)
   Y = broadcast(l.f, x, Ref(st))
   dY = broadcast(grad_fd, Ref(l.f), x, Ref(st))
   return (Y, dY), st
end

function ET._pb_ed(l::DPTransform, Δ::AbstractArray,
                   X::AbstractVector{<: NTorDP}, ps, st)
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
