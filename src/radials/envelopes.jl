
# Cutoff envelopes for the radial bases. Ported verbatim from
# ACEpotentials `src/models/radial_envelopes.jl`.
#
# The envelope interface is `evaluate(env, r, x) -> scalar`, where `r` is the
# physical distance and `x = transform(r)` is already computed.

abstract type AbstractEnvelope end

struct PolyEnvelope1sR{T}
   rcut::T
   p::Int
   # -------
   meta::Dict{String, Any}
end


PolyEnvelope1sR(rcut, p) =
   PolyEnvelope1sR(rcut, p, Dict{String, Any}())

# `r` and `x` may have different types, e.g. when differentiating w.r.t.
# only one of them via ForwardDiff; the output type is their promotion.
function evaluate(env::PolyEnvelope1sR, r::Real, x::Real)
   T = promote_type(typeof(r), typeof(x))
   if r >= env.rcut
      return zero(T)
   end
   p = env.p
   # return r^(-p) - env.rcut^(-p) - p*(env.rcut^(-p-1))*(r - env.rcut)
   return T( ( (r/env.rcut)^(-p) - 1) * (1 - r / env.rcut) )
end

# returns the pair of partial derivatives (∂/∂r, ∂/∂x); this envelope
# depends on r only. (the ACEpotentials original called a non-existent
# 2-argument `evaluate` here; fixed during the port.)
evaluate_d(env::PolyEnvelope1sR, r::T, x::T) where {T} =
      (ForwardDiff.derivative(r1 -> evaluate(env, r1, x), r),
       zero(T),)


# ----------------------------

struct PolyEnvelope2sX{T}
   x1::T
   x2::T
   p1::Int
   p2::Int
   s::T
   # -------
   meta::Dict{String, Any}
end

function PolyEnvelope2sX(x1, x2, p1, p2)
   if x1 == x2
      error("x1 and x2 must be different!")
   end
   if x1 > x2
      @warn("swapping x1, x2 to ensure x1 < x2")
      x1, x2 = x2, x1
      p1, p2 = p2, p1
   end
   s = 1 / (abs(x2 - x1)/2)^(p1+p2)
   PolyEnvelope2sX(x1, x2, p1, p2, s, Dict{String, Any}())
end


function evaluate(env::PolyEnvelope2sX, r::Real, x::Real)
   T = promote_type(typeof(r), typeof(x))
   x1, x2 = env.x1, env.x2
   p1, p2 = env.p1, env.p2
   s = env.s

   if !(x1 < x < x2)
      return zero(T)
   end

   return T( s * (x-x1)^p1 * (x2-x)^p2 )
end


# (∂/∂r, ∂/∂x); this envelope depends on x only. Same fix as above.
evaluate_d(env::PolyEnvelope2sX, r::T, x::T) where T =
    (zero(T), ForwardDiff.derivative(x1 -> evaluate(env, r, x1), x))
