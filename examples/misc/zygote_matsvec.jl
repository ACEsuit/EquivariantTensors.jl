# =============================================================================
# Robust pullbacks through a model that mixes manipulations, linear operations,
# and contractions on arrays of `SVector`s — on both CPU and GPU (Metal/CUDA).
#
# The model in this script is deliberately small and not physically meaningful:
#
#     X            ::Vector{SVector{3,Float32}}             # the input
#     r   = norm.(X)                                         # length N
#     P   = polys(r)                                         # N × nembed
#     Y   = X .* P                                           # N × nembed of SVec
#     Z   = Y * W                                            # N × nfeat  of SVec
#     e   = sum(abs2, sum(Z))                                # scalar
#
# All four primitive operations (norm-of-svec, broadcast of svec×scalar, matmul
# of mat-of-svec by mat-of-scalar, sum-of-svecs) are perfectly well-defined and
# evaluate fine on CPU and GPU. Differentiating them with Zygote, however, is
# fragile — every operation that crosses Zygote's auto-diff path on
# non-`<:Number` eltypes hits an issue, and the issues are different on CPU
# vs. GPU. The script below shows the smallest set of changes from the naive
# code that produces correct gradients (compared to ForwardDiff) for both
#   (a) parameters `ps`,
#   (b) inputs `X`,
# on both CPU and GPU.
#
# Why each wrapper exists
# -----------------------
# `_norms(X) = map(norm, X)`
#   Naive `r = map(norm, X)`. Zygote's `∇map` adjoint builds a closure that
#   captures the entire `Zygote.Context` (which holds an `IdDict` cache) and
#   broadcasts that closure over X on the backward pass. On GPU, the kernel
#   arg list then contains the closure and Metal/CUDA reject it as
#   non-isbits ("passing non-bitstype argument"). Wrapping `map(norm, X)` in
#   `_norms` and giving it a hand-written rrule stops Zygote from generating
#   the Context-capturing adjoint at all; the rrule body is plain runtime
#   code, so the closure never enters a kernel.
#
# `_broad_(X, y) = X .* y`
#   Naive `Y = X .* polys(r)`. On CPU, Zygote's generic broadcast adjoint
#   produces a working pullback. On GPU, that adjoint can't synthesise a
#   pullback for the X slot (eltype `MtlArray{SVector}`) and silently drops
#   the cotangent — `Zygote.gradient(... , X_dev)[1]` returns `nothing`.
#   Putting the broadcast behind `_broad_` with a hand-written rrule replaces
#   the dropped path with explicit `dropdims(sum(∂out .* y; dims=2); dims=2)`
#   and `dot.(X, ∂out)`, both of which are single-level broadcasts/reductions
#   that compile cleanly on GPU.
#
# `_mul_(A, B) = A * B`   (where `A::Matrix{<:SVector}`, `B::Matrix{<:Real}`)
#   Naive `Z = Y * ps.W`. ChainRules' `rrule(::typeof(*), …)` is gated on
#   `<:Number` eltypes, and `Matrix{SVector} * Matrix{Real}` falls through to
#   `LinearAlgebra.generic_matmatmul!`, which uses `setindex!` accumulation —
#   Zygote refuses to differentiate it ("Mutating arrays is not supported").
#   Wrapping in `_mul_` with a hand-written rrule provides closed-form
#   `∂A = ∂out * transpose(B)` and `∂B = transpose(A) * ∂out`, neither of
#   which is itself differentiated, so the mutation in their bodies doesn't
#   matter.
#
# Gradient checks
# ---------------
# All four cotangent paths are validated against `ForwardDiff.gradient`:
#   - `g.W ≈ g_fd.W` and `Array(g_dev.W) ≈ g_fd.W`     (params, CPU+GPU)
#   - `gX ≈ gX_fd`   and `Array(gX_dev) ≈ gX_fd`        (inputs, CPU+GPU)
#
# The CPU/GPU agreement is the load-bearing test: a custom rrule that "looks
# correct" can still silently drop or mis-route a slot on GPU, so verifying
# both sides against the same FD reference is what catches that.
# =============================================================================

using Zygote, ForwardDiff, StaticArrays, Optimisers, Random, LinearAlgebra
using ChainRulesCore, Lux, Polynomials4ML
import ChainRulesCore: rrule
using KernelAbstractions 

using Metal 
dev = mtl 

# using CUDA 
# dev = cu 

function grad_fd_p(f, x, ps, st)
   p_flat, rebuild = destructure(ps)
   _eval_p(p) = f(x, rebuild(p), st)[1]
   ∇p_flat = ForwardDiff.gradient(_eval_p, p_flat)
   return rebuild(∇p_flat)
end

function grad_fd_X(f, X::AbstractVector{SVector{D,T}}, ps, st) where {D,T}
   x_flat = collect(reinterpret(T, X))
   function _eval_x(x)
      Xd = [ SVector{D}(ntuple(d -> x[(i-1)*D + d], Val(D))) for i in eachindex(X) ]
      return f(Xd, ps, st)[1]
   end
   ∇x_flat = ForwardDiff.gradient(_eval_x, x_flat)
   return [ SVector{D,T}(ntuple(d -> ∇x_flat[(i-1)*D + d], Val(D))) for i in eachindex(X) ]
end

##

struct SVecModel <: AbstractLuxLayer
   nembed::Int 
   nfeat::Int 
end

Lux.initialparameters(rng::AbstractRNG, m::SVecModel) = 
         ( W = Float32.(randn(rng, m.nembed, m.nfeat) / sqrt(m.nfeat+m.nembed)), )



function (m::SVecModel)(X, ps, st)
   r = _norms(X)                                                # length-N
   polys = ChebBasis(m.nembed)
   Y = _broad_(X, polys(r))
   Z = _mul_(Y, ps.W)       
   e = sum(abs2, sum(Z))
   return e, st
end

# Private wrapper around `map(norm, X)` so we can attach a hand-written rrule.
# Without this, Zygote's `∇map` adjoint builds a closure that captures the
# Zygote `Context` (holding an `IdDict` cache) and broadcasts it over X — the
# captured Context is not isbits, so a Metal kernel launch fails with
# "passing non-bitstype argument".
_norms(X) = map(norm, X)

function rrule(::typeof(_norms), X::AbstractVector{<:SVector})
   r = _norms(X)
   function _norms_pb(∂r_)
      ∂r = unthunk(∂r_)
      ∂r isa AbstractZero && return (NoTangent(), ZeroTangent())
      # d/dX[i] norm(X[i]) = X[i] / norm(X[i])
      ∂X = (∂r ./ r) .* X
      return (NoTangent(), ∂X)
   end
   return r, _norms_pb
end


_broad_(X, y) = X .* y

function rrule(::typeof(_broad_), X, y)
   out = _broad_(X, y)
   function _broad_pb(∂out_)
      ∂out = unthunk(∂out_)
      ∂out isa AbstractZero && return (NoTangent(), ZeroTangent(), ZeroTangent())
      ∂X = dropdims(sum(∂out .* y; dims = 2); dims = 2)
      ∂y = dot.(X, ∂out)
      return (NoTangent(), ∂X, ∂y)
   end
   return out, _broad_pb
end



function _mul_(A::AbstractMatrix{SVector{D, T1}}, B::AbstractMatrix{T2}) where {D, T1, T2}
   return A * B 
end

function rrule(::typeof(_mul_), A::AbstractMatrix{SVector{D, T1}}, B::AbstractMatrix{T2}) where {D, T1, T2}
   out = _mul_(A, B)
   function _mul_scal_pullback(_∂out)
      ∂out = unthunk(_∂out)
      @assert eltype(∂out) <: SVector 
      ∂A = ∂out * transpose(B) 
      ∂B = transpose(A) * ∂out
      return (NoTangent(), ∂A, ∂B)
   end
   return out, _mul_scal_pullback
end



## 

model = SVecModel(3, 2)
rng = Random.MersenneTwister(1234)
ps, st = Lux.setup(rng, model)

X = [ randn(SVector{3, Float32})/3 for _ = 1:5 ]
e, _ = model(X, ps, st)

## 

X_dev = dev(X) 
ps_dev = dev(ps)
st_dev = dev(st)

e_dev, _ = model(X_dev, ps_dev, st_dev)
@show e ≈ e_dev 
##

g = Zygote.gradient(_p -> model(X, _p, st)[1], ps)[1]
g_fd = grad_fd_p(model, X, ps, st)
@show g.W ≈ g_fd.W

##

g_dev = Zygote.gradient(_p -> model(X_dev, _p, st_dev)[1], ps_dev)[1]
W_dev = Array(g_dev.W)
@show W_dev ≈ g_fd.W

## gradient w.r.t. X — CPU

gX    = Zygote.gradient(_X -> model(_X, ps, st)[1], X)[1]
gX_fd = grad_fd_X(model, X, ps, st)
@show all(gX .≈ gX_fd)

## gradient w.r.t. X — GPU

gX_dev     = Zygote.gradient(_X -> model(_X, ps_dev, st_dev)[1], X_dev)[1]
gX_dev_cpu = Array(gX_dev)
@show all(gX_dev_cpu .≈ gX_fd)



